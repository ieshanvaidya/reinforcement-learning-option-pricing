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
from optionpricing import *

# Defining transition namedtuple here rather than within the class to ensure pickle functionality
transition = namedtuple('transition',
            ['old_state', 'action', 'reward', 'new_state', 'done'])


class Estimator(nn.Module):
    def __init__(self, device, ngpu, state_space_dim, action_space_dim):
        super(Estimator, self).__init__()
        self.device = device
        self.ngpu = ngpu
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim

        self.model = nn.Sequential(
            nn.Linear(self.state_space_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space_dim)
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.model, x, range(self.ngpu))
        else:
            output = self.model(x)

        return output


class Agent:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.epsilon = args.epsilon
        self.decay = args.decay
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.replay_memory_size = args.replay_memory_size
        self.update_every = args.update_every
        #self.record_every = args.record_every
        self.epsilon_min = args.epsilon_min
        self.savedir = args.savedir
        self.clip = args.clip
        self.best_reward_criteria = args.best_reward_criteria

        self.best_reward_criteria = 10 # If mean reward over last 'best_reward_critera' > best_reward, save model

        # Get valid actions
        try:
            self.valid_actions = list(range(env.action_space.n))
        except AttributeError as e:
            print(f'Action space is not Discrete, {e}')

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s, %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(os.path.join('experiments', args.savedir, 'training.log'))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

        # Tensorboard
        self.writer = SummaryWriter(log_dir = os.path.join('experiments', self.savedir), flush_secs = 5)

        # Initialize model
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.ngpu = args.ngpu
        state_shape = env.observation_space.shape
        state_space_dim = state_shape[0] if len(state_shape) == 1 else state_shape

        self.estimator = Estimator(self.device, self.ngpu, state_space_dim, env.action_space.n)
        self.target = Estimator(self.device, self.ngpu, state_space_dim, env.action_space.n)

        # Optimization
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.estimator.parameters(), lr = args.lr, betas = (args.beta1, 0.999))

        if args.resume:
            try:
                self.load_checkpoint(os.path.join('experiments', args.savedir, 'checkpoint.pth'))
                self.logger.info(f'INFO: Resuming from checkpoint; episode: {self.episode}')
            except FileNotFoundError:
                print('Checkpoint not found')

        else:
            self.replay_memory = deque(maxlen = args.replay_memory_size)

            # Initialize replay memory
            self.initialize_replay_memory(self.batch_size)

            # Copy estimator state_dict to target
            self.target.load_state_dict(self.estimator.state_dict())

            # Training details
            self.episode = 0
            self.steps = 0
            self.best_reward = -np.inf


    def initialize_replay_memory(self, size):
        """
        Populate replay memory with initial experience
        """
        if self.replay_memory:
            self.logger.info('INFO: Replay memory already initialized')
            return

        assert size >= self.batch_size, "Initialize with size >= batch size"

        old_state = self.env.reset()
        for i in range(size):
            action = random.choice(self.valid_actions)
            new_state, reward, done, _ = self.env.step(action)
            self.replay_memory.append(transition(old_state, action,
                reward, new_state, done))
            reward = np.clip(reward, -self.clip, self.clip)

            if done:
                old_state = self.env.reset()
            else:
                old_state = new_state

        self.logger.info(f'INFO: Replay memory initialized with {size} experiences')

    def train(self, n_episodes, episode_length):
        """
        Train the agent
        """
        train_rewards = []

        for episode in tqdm(range(n_episodes)):
            self.episode += 1
            episode_rewards = []
            episode_steps = 0
            episode_history = []
            losses = []
            done = False
            kind = None # Type of action taken

            old_state = self.env.reset()

            while not done:
                ####################################################
                # Select e-greedy action                           #
                ####################################################
                if random.random() <= self.epsilon:
                    action = random.choice(self.valid_actions)
                    kind = 'random'

                else:
                    with torch.no_grad():
                        old_state = torch.from_numpy(old_state.reshape(1, -1)).to(self.device)
                        action = np.argmax(self.estimator(old_state).numpy())
                        old_state = old_state.numpy().reshape(-1)
                    kind = 'policy'

                ####################################################
                # Env step and store experience in replay memory   #
                ####################################################
                new_state, reward, done, info = self.env.step(action)
                reward = np.clip(reward, -self.clip, self.clip)

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

                # Q_old = reward + discount * max[over actions](Q_new)
                # Old Q value = reward + discounted Q value of new state

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
                self.logger.info(f'LOG: episode:{self.episode}, step:{episode_steps}, action:{action}, kind:{kind}, epsilon:{self.epsilon}, reward:{reward}, best_mean_reward:{self.best_reward}, loss:{losses[-1]}')

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
                self.save_checkpoint(os.path.join('experiments', self.savedir, 'best.pth'))

            if not self.episode % self.args.checkpoint_every:
                self.save_checkpoint(os.path.join('experiments', self.args.savedir, 'checkpoint.pth'))

    def simulate(self):
        """
        Simulation of one episode
        """
        state = self.env.reset().reshape(1, -1).to(self.device)
        done = False
        while not done:
            self.env.render() # To be implemented
            action = np.argmax(self.estimator(state).item())
            state, reward, done, info = self.env.step(action)
            state = state.reshape(1, -1).to(self.device)

        self.env.close() # To be implemented


    def save_checkpoint(self, path):
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
    parser.add_argument('-ne', '--n_episodes', type = int, default = 100, help = 'number of episodes to train')
    parser.add_argument('-el', '--episode_length', type = int, default = 1000, help = 'maximum episode length')
    parser.add_argument('-e', '--epsilon', type = float, default = 1, help = 'e-greedy probability')
    parser.add_argument('-d', '--decay', type = float, default = 0.99, help = 'decay of epsilon per episode')
    parser.add_argument('-emin', '--epsilon_min', type = float, default = 0.05, help = 'minumum value taken by epsilon')
    parser.add_argument('-g', '--gamma', type = float, default = 0.99, help = 'discount factor')
    parser.add_argument('-ue', '--update_every', type = int, default = 500, help = 'number of steps after which to update the target model')
    parser.add_argument('-ce', '--checkpoint-every', type = int, default = 100, help = 'number of episodes after which to checkpoint')
    parser.add_argument('-r', '--resume', action = 'store_true', help = 'resume from previous checkpoint from save directory')
    parser.add_argument('-bs', '--batch_size', type = int, default = 128, help = 'batch size')
    parser.add_argument('-rms', '--replay_memory_size', type = int, default = 50000, help = 'replay memory size')
    parser.add_argument('-s', '--seed', type = int, help = 'random seed')
    parser.add_argument('-sd', '--savedir', type = str, help = 'save directory')
    parser.add_argument('-lr', '--lr', type = float, default = 0.001, help = 'learning rate')
    parser.add_argument('-b', '--beta1', type = float, default = 0.9, help = 'beta1')
    parser.add_argument('-cuda', '--cuda', action = 'store_true', help = 'cuda')
    parser.add_argument('-ngpu', '--ngpu', type = int, default = 0, help = 'number of gpu')
    parser.add_argument('-clip', '--clip', type = float, default = np.inf, help = 'cutoff reward between [-clip, clip]')
    parser.add_argument('-brc', '--best_reward_criteria', type = int, default = 1, help = 'save model if mean reward over last n episodes greater than best rewardwhere n=brc')

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
        'K': 100,
        'D': 5,
        'mu': 0,
        'sigma': 0.01,
        'r': 0,
        'ss': 5,
        'kappa': 0.1
        }
    env = OptionPricingEnv()
    env.configure(**config)

    agent = Agent(env, args)
    agent.train(args.n_episodes, args.episode_length)
    #agent.simulate()
