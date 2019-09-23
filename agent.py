import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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

"""
THINGS TO DO:
1. Add resume capability
"""

class Estimator(nn.Module):
    def __init__(self, device, ngpu, state_space_dim, action_space_dim):
        super(Estimator, self).__init__()
        self.device = device
        self.ngpu = ngpu
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim

        self.model = nn.Sequential(
            nn.Linear(self.state_space_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space_dim)
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.model, x, range(self.ngpu))
        else:
            output = self.model(x)

        return output


class Agent:
    def __init__(self, env, args):

        self.args = args

        self.epsilon = args.epsilon
        self.decay = args.decay
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.replay_memory_size = args.replay_memory_size
        self.update_every = args.update_every
        self.record_every = args.record_every
        self.epsilon_min = args.epsilon_min
        self.savedir = args.savedir

        self.replay_memory = deque(maxlen = args.replay_memory_size)

        self.transition = namedtuple('Transition',
            ['old_state', 'action', 'reward', 'new_state', 'done'])

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

        # Initialize replay memory
        self.initialize_replay_memory(self.batch_size)

        # Initialize model
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.ngpu = args.ngpu
        state_shape = env.observation_space.shape
        state_space_dim = state_shape[0] if len(state_shape) == 1 else state_shape

        self.estimator = Estimator(self.device, self.ngpu, state_space_dim, env.action_space.n)
        self.target = Estimator(self.device, self.ngpu, state_space_dim, env.action_space_dim)

        # Copy estimator state_dict to target
        self.target.load_state_dict(self.estimator.state_dict())

        # Optimization
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.estimator.parameters(), lr = args.lr, betas = (args.beta1, 0.999))

        # Training details
        self.episodes = 0
        self.steps = 0


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
            self.replay_memory.append(self.transition(old_state, action,
                reward, new_state, done))
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
        best_reward = -np.inf

        for episode in tqdm(range(n_episodes)):
            self.episodes += 1
            episode_reward = 0
            episode_steps = 0
            episode_history = []
            losses = []
            done = False

            old_state = self.env.reset()

            while not done:
                ####################################################
                # Select e-greedy action                           #
                ####################################################
                if random.random() <= self.epsilon:
                    action = random.choice(self.valid_actions)

                else:
                    old_state = torch.from_numpy(old_state.reshape(1, -1)).to(self.device)
                    action = np.argmax(self.estimator(old_state).item())

                ####################################################
                # Env step and store experience in replay memory   #
                ####################################################
                new_state, reward, done, info = self.env.step(action)

                self.replay_memory.append(self.transition(old_state, action,
                    reward, new_state, done))

                episode_history.append(self.transition(old_state, action,
                    reward, new_state, done))

                episode_reward += reward
                episode_steps += 1
                self.steps += 1

                ####################################################
                # Sample batch and fit to model                    #
                ####################################################
                batch = random.sample(self.replay_memory, self.batch_size)
                old_states, actions, rewards, new_states, is_done = map(np.array, zip(*batch))
                is_done = is_done.astype(np.uint8)

                old_states = torch.from_numpy(old_states).to(self.device)
                new_states = torch.from_numpy(new_states).to(self.device)
                rewards = torch.from_numpy(rewards).to(self.device)
                is_done = torch.from_numpy(is_done).to(self.device)
                actions = torch.from_numpy(actions).long().to(self.device)

                # Q_old = reward + discount * max[over actions](Q_new)
                # Old Q value = reward + discounted Q value of new state
                q_values = self.estimator(old_states)
                q_target = self.target(new_states)
                max_q, _ = torch.max(q_target, dim = 1)
                q_target = rewards + self.gamma * (~is_done) * max_q

                # Gather those Q values for which action was taken | since the output is Q values for all possible actions
                q_values_expected = q_values.gather(1, actions.view(-1, 1)).view(-1)

                loss = self.criterion(q_values_expected, q_target)
                self.estimator.zero_grad()
                loss.backward()
                self.optimizer.step()

                if not self.steps % self.update_every:
                    self.target.load_state_dict(self.estimator.state_dict())

                old_state = new_state

                if episode_steps >= episode_length:
                    break

            # Epsilon decay
            self.epsilon *= self.decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

            train_rewards.append(episode_reward)
            mean_reward = np.mean(train_rewards[-self.best_reward_criteria:])
            if mean_reward > best_reward:
                best_reward = mean_reward
                self.estimator.save(os.path.join('experiments', self.savedir))

            # Log statistics
            self.logger.info(f'LOG: episode:{self.episodes}, epsilon:{self.epsilon}, steps:{episode_steps}, reward:{episode_reward}, best_mean_reward:{best_reward}, average_loss:{np.mean(losses)}')


    def simulate(self):
        state = self.env.reset().reshape(1, -1).to(self.device)
        done = False
        while not done:
            self.env.render() # To be implemented
            action = np.argmax(self.estimator(state).item())
            state, reward, done, info = self.env.step(action)
            state = state.reshape(1, -1).to(self.device)

        self.env.close() # To be implemented
