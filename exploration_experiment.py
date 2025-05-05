import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
import time
from collections import deque
import os
import pickle

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EnvWrapper, self).__init__(env)
        self.env = env
        self.env = RGBImgPartialObsWrapper(self.env, tile_size=8)
        self.env = ImgObsWrapper(self.env)
        
    def reset(self):
        state = self.env.reset()
        return self._process_state(state)
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return self._process_state(next_state), reward, done, info
        
    def _process_state(self, state):
        return state.astype(np.float32) / 255.0
        
    def render(self, mode='human'):
        return self.env.render(mode)