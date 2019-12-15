import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gym
import random
from collections import deque, namedtuple
import logging
from tqdm import tqdm
import argparse
import os
import time
import yaml
import csv
from copy import deepcopy
from optionpricing import *

# Defining transition namedtuple here rather than within the class to ensure pickle functionality
transition = namedtuple('transition',
            ['old_state', 'action', 'reward', 'new_state', 'done'])


class Estimator(nn.Module):
    def __init__(self, nhidden, nunits, state_space_dim, action_space_dim):
        """
        Estimator class that returns Q-values
            ngpu: number of gpus
            state_space_dim: Dimension of the state space
            action_space_dim: Dimension of the action space
        """

        super(Estimator, self).__init__()
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim

        assert nhidden > 0, 'Number of hidden layers must be > 0'

        init_layer = nn.Linear(state_space_dim, nunits)
        self.final_layer = nn.Linear(nunits, action_space_dim)

        layers = [init_layer]
        for n in range(nhidden - 1):
            layers.append(nn.Linear(nunits, nunits))

        self.module_list = nn.ModuleList(layers)

        self.relu = nn.ReLU()

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
            x = self.relu(x)

        x = self.final_layer(x)

        return x


class Agent:
    def __init__(self, env, args):
        """
        Agent class to train the DQN
            env: Gym like environment object
            args: Training arguments | use --help flag to view
        """
        self.env = env
        self.args = args

        self.epsilon = args.epsilon
        self.decay = args.decay
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.replay_memory_size = args.replay_memory_size
        self.update_every = args.update_every
        self.epsilon_min = args.epsilon_min
        self.savedir = args.savedir
        self.scale = args.scale
        if args.clip == 0:
            self.clip = np.inf
        else:
            self.clip = args.clip
        self.best_reward_criteria = args.best_reward_criteria # If mean reward over last 'best_reward_critera' > best_reward, save model

        # Get valid actions
        try:
            self.valid_actions = list(range(env.action_space.n))
        except AttributeError as e:
            print(f'Action space is not Discrete, {e}')

        # Logging
        self.train_logger = logging.getLogger('train')
        self.train_logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s, %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(os.path.join('experiments', args.savedir, 'training.log'))
        file_handler.setFormatter(formatter)
        self.train_logger.addHandler(file_handler)
        self.train_logger.propagate = False

        # Tensorboard
        self.writer = SummaryWriter(log_dir = os.path.join('experiments', self.savedir), flush_secs = 5)

        # Initialize model
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        state_shape = env.observation_space.shape
        state_space_dim = state_shape[0] if len(state_shape) == 1 else state_shape

        self.estimator = Estimator(args.nhidden, args.nunits, state_space_dim, env.action_space.n).to(self.device)
        self.target = Estimator(args.nhidden, args.nunits, state_space_dim, env.action_space.n).to(self.device)

        # Optimization
        self.criterion = nn.SmoothL1Loss(reduction = 'mean')
        self.optimizer = optim.Adam(self.estimator.parameters(), lr = args.lr, betas = (args.beta1, 0.999))

        # If resume, load from checkpoint | otherwise initialize
        if args.resume:
            try:
                self.load_checkpoint(os.path.join('experiments', args.savedir, 'checkpoint.pt'))
                self.train_logger.info(f'INFO: Resuming from checkpoint; episode: {self.episode}')
            except FileNotFoundError:
                print('Checkpoint not found')

        else:
            self.replay_memory = deque(maxlen = args.replay_memory_size)

            # Initialize replay memory
            self.initialize_replay_memory(self.batch_size)

            # Set target = estimator
            self.target.load_state_dict(self.estimator.state_dict())

            # Training details
            self.episode = 0
            self.steps = 0
            self.best_reward = -self.clip * self.env.T * self.env.D


    def initialize_replay_memory(self, size):
        """
        Populate replay memory with initial experience
            size: Number of experiences to initialize (must be >= batch_size)
        """
        if self.replay_memory:
            self.train_logger.info('INFO: Replay memory already initialized')
            return

        assert size >= self.batch_size, "Initialize with size >= batch size"

        old_state = self.env.reset()
        for i in range(size):
            action = random.choice(self.valid_actions)
            new_state, reward, done, _ = self.env.step(action)
            reward = np.clip(self.scale * reward, -self.clip, self.clip)
            self.replay_memory.append(transition(old_state, action,
                reward, new_state, done))

            if done:
                old_state = self.env.reset()
            else:
                old_state = new_state

        self.train_logger.info(f'INFO: Replay memory initialized with {size} experiences')


    def train(self, nepisodes, episode_length):
        """
        Train the agent
        """
        train_rewards = []

        for episode in tqdm(range(nepisodes)):
            self.estimator.train()
            self.episode += 1
            episode_rewards = []
            episode_steps = 0
            episode_history = []
            losses = []
            done = False
            kind = None # Type of action taken

            old_state = self.env.reset()

            while not done:
                delta = self.env.delta
                stock_price = self.env.S
                call = self.env.call

                ####################################################
                # Select e-greedy action                           #
                ####################################################
                if random.random() <= self.epsilon:
                    action = random.choice(self.valid_actions)
                    kind = 'random'

                else:
                    with torch.no_grad():
                        old_state = torch.from_numpy(old_state.reshape(1, -1)).to(self.device)
                        action = np.argmax(self.estimator(old_state).cpu().numpy())
                        old_state = old_state.cpu().numpy().reshape(-1)
                    kind = 'policy'

                ####################################################
                # Env step and store experience in replay memory   #
                ####################################################
                new_state, reward, done, info = self.env.step(action)
                reward = np.clip(self.scale * reward, -self.clip, self.clip)

                self.replay_memory.append(transition(old_state, action,
                    reward, new_state, done))

                episode_history.append(transition(old_state, action,
                    reward, new_state, done))

                episode_rewards.append(reward)
                episode_steps += 1
                self.steps += 1

                ####################################################
                # Sample batch and fit to model                    #
                ####################################################
                batch = random.sample(self.replay_memory, self.batch_size)
                old_states, actions, rewards, new_states, is_done = map(np.array, zip(*batch))
                rewards = rewards.astype(np.float32)

                old_states = torch.from_numpy(old_states).to(self.device)
                new_states = torch.from_numpy(new_states).to(self.device)
                rewards = torch.from_numpy(rewards).to(self.device)
                is_not_done = torch.from_numpy(np.logical_not(is_done)).to(self.device)
                actions = torch.from_numpy(actions).long().to(self.device)

                with torch.no_grad():
                    q_target = self.target(new_states)
                    max_q, _ = torch.max(q_target, dim = 1)
                    q_target = rewards + self.gamma * is_not_done.float() * max_q

                # Gather those Q values for which action was taken | since the output is Q values for all possible actions
                q_values_expected = self.estimator(old_states).gather(1, actions.view(-1, 1)).view(-1)

                loss = self.criterion(q_values_expected, q_target)
                self.estimator.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

                if not self.steps % self.update_every:
                    self.target.load_state_dict(self.estimator.state_dict())

                old_state = new_state

                # Tensorboard
                self.writer.add_scalar('Transition/reward', reward, self.steps)
                self.writer.add_scalar('Transition/loss', loss, self.steps)

                # Log statistics
                self.train_logger.info(f'LOG: episode:{self.episode}, step:{episode_steps}, action:{action}, kind:{kind}, reward:{reward}, best_mean_reward:{self.best_reward}, loss:{losses[-1]}, epsilon:{self.epsilon}, S:{stock_price}, c:{call}, delta:{delta}, n:{self.env.n}, dn:{info["dn"]}, cost:{info["cost"]}, pnl:{info["pnl"]}, K:{self.env.K}, T:{self.env.T}')

                if episode_steps >= episode_length:
                    break

            # Epsilon decay
            self.epsilon *= self.decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

            train_rewards.append(sum(episode_rewards))
            mean_reward = np.mean(train_rewards[-self.best_reward_criteria:])

            self.writer.add_scalar('Episode/epsilon', self.epsilon, self.episode)
            self.writer.add_scalar('Episode/total_reward', sum(episode_rewards), self.episode)
            self.writer.add_scalar('Episode/mean_loss', np.mean(losses), self.episode)
            self.writer.add_histogram('Episode/reward', np.array(episode_rewards), self.episode)

            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.save_checkpoint(os.path.join('experiments', self.savedir, 'best.pt'))

            if not self.episode % self.args.checkpoint_every:
                self.save_checkpoint(os.path.join('experiments', self.args.savedir, 'checkpoint.pt'))


    def save_checkpoint(self, path):
        """
        Checkpoint the model
            path: Save path
        """
        checkpoint = {
            'episode': self.episode,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'estimator': self.estimator.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'replay_memory': self.replay_memory,
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'best_reward': self.best_reward
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        Load checkpoint
            path: Checkpoint (checkpoint.pt) path
        """
        checkpoint = torch.load(path)
        self.episode = checkpoint['episode']
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
        self.estimator.load_state_dict(checkpoint['estimator'])
        self.target.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.replay_memory = checkpoint['replay_memory']
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        self.best_reward = checkpoint['best_reward']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepisodes', type = int, default = 15000, help = 'number of episodes to train')
    parser.add_argument('--episode_length', type = int, default = 1000, help = 'maximum episode length')
    parser.add_argument('--epsilon', type = float, default = 1, help = 'starting e-greedy probability')
    parser.add_argument('--decay', type = float, default = 0.999, help = 'decay of epsilon per episode')
    parser.add_argument('--epsilon_min', type = float, default = 0.005, help = 'minumum value taken by epsilon')
    parser.add_argument('--gamma', type = float, default = 0.3, help = 'discount factor')
    parser.add_argument('--update_every', type = int, default = 500, help = 'update target model every [_] steps')
    parser.add_argument('--checkpoint_every', type = int, default = 1000, help = 'checkpoint model every [_] steps')
    parser.add_argument('--resume', action = 'store_true', help = 'resume from previous checkpoint from save directory')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'batch size')
    parser.add_argument('--replay_memory_size', type = int, default = 64000, help = 'replay memory size')
    parser.add_argument('--seed', type = int, help = 'random seed')
    parser.add_argument('--savedir', type = str, help = 'save directory')
    parser.add_argument('--nhidden', type = int, default = 2, help = 'number of hidden layers')
    parser.add_argument('--nunits', type = int, default = 128, help = 'number of units in a hidden layer')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
    parser.add_argument('--beta1', type = float, default = 0.9, help = 'beta1')
    parser.add_argument('--cuda', action = 'store_true', help = 'cuda')
    parser.add_argument('--scale', type = float, default = 1, help = 'scale reward by [_] | reward = [_] * reward | Takes priority over clip')
    parser.add_argument('--clip', type = float, default = 100, help = 'clip reward between [-clip, clip] | Pass in 0 for no clipping')
    parser.add_argument('--best_reward_criteria', type = int, default = 10, help = 'save model if mean reward over last [_] episodes greater than best reward')
    parser.add_argument('--trc_multiplier', type = float, default = 1, help = 'transaction cost multiplier')
    parser.add_argument('--trc_ticksize', type = float, default = 0.1, help = 'transaction cost ticksize')

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)

    if args.savedir is None:
        args.savedir = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())

    try:
        os.makedirs(os.path.join('experiments', args.savedir))
    except OSError:
        pass

    if not args.resume:
        with open(os.path.join('experiments', args.savedir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(args), f)

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    config = {
        'S': 100,
        'T': 10, # 10 days
        'L': 1,
        'm': 100, # L options for m stocks
        'n': 0,
        'K': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105],
        'D': 5,
        'mu': 0,
        'sigma': 0.01,
        'r': 0,
        'ss': 5,
        'kappa': 0.1,
        'multiplier': args.trc_multiplier,
        'ticksize': args.trc_ticksize
        }
    env = OptionPricingEnv(config)
    env.configure()

    agent = Agent(env, args)
    agent.train(args.nepisodes, args.episode_length)
