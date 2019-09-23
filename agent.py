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
    def __init__(self, args):

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
