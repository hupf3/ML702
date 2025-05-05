import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DQNNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.action_size = action_size
        
        if len(input_shape) == 3: 
            min_input_size = 28 
            
            if input_shape[1] < min_input_size or input_shape[2] < min_input_size:
                self.features = nn.Sequential(
                    nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
                )
                
                def conv2d_size_out(size, kernel_size, stride, padding):
                    return ((size + 2*padding - kernel_size) // stride) + 1
                    
                h = conv2d_size_out(conv2d_size_out(conv2d_size_out(
                    input_shape[1], 3, 2, 1), 3, 2, 1), 3, 1, 1)
                w = conv2d_size_out(conv2d_size_out(conv2d_size_out(
                    input_shape[2], 3, 2, 1), 3, 2, 1), 3, 1, 1)
                
            else:
                self.features = nn.Sequential(
                    nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU()
                )
                
                def conv2d_size_out(size, kernel_size, stride):
                    return (size - (kernel_size - 1) - 1) // stride + 1
                    
                h = conv2d_size_out(conv2d_size_out(conv2d_size_out(
                    input_shape[1], 8, 4), 4, 2), 3, 1)
                w = conv2d_size_out(conv2d_size_out(conv2d_size_out(
                    input_shape[2], 8, 4), 4, 2), 3, 1)
            
            linear_input_size = h * w * 64
            
            if linear_input_size <= 0:
                print(f"警告: 计算得到的linear_input_size是 {linear_input_size}。使用默认值 256。")
                linear_input_size = 256
            
            self.fc = nn.Sequential(
                nn.Linear(linear_input_size, 512),
                nn.ReLU(),
                nn.Linear(512, action_size)
            )
        else: 
            input_size = np.prod(input_shape)
            self.features = None
            self.fc = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            )
            
    def forward(self, x):
        if self.features:
            if len(x.shape) == 3:
                x = x.unsqueeze(0) 
            batch_size = x.size(0)
            x = self.features(x)
            x = x.view(batch_size, -1)  
        else:
            x = x.view(x.size(0), -1) if len(x.shape) > 1 else x
            
        return self.fc(x)


class DQNAgent:
    def __init__(self, state_shape, action_size, exploration_strategy,
                learning_rate=0.001, gamma=0.99, batch_size=64, 
                memory_size=100000, target_update_freq=1000):
        self.state_shape = state_shape
        self.action_size = action_size
        self.exploration = exploration_strategy
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        
        self.memory = deque(maxlen=memory_size)
        
        self.q_network = DQNNetwork(state_shape, action_size)
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.total_steps = 0
        
        self.intrinsic_rewards = []
        
    def select_action(self, state):
        with torch.no_grad():
            try:
                if isinstance(state, np.ndarray):
                    if len(state.shape) == 3:  
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    else:
                        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
                elif isinstance(state, list):
                    state_np = np.array(state, dtype=np.float32)
                    if state_np.size == 0:
                        return np.random.randint(self.action_size)
                    state_tensor = torch.FloatTensor(state_np).unsqueeze(0)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    
                if 0 in state_tensor.shape:
                    return np.random.randint(self.action_size)
                    
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            except Exception as e:
                print(f"状态处理错误: {e}")
                return np.random.randint(self.action_size)
        
        action = self.exploration.get_action(state, q_values)
        
        return action
        
    def store_experience(self, state, action, reward, next_state, done, intrinsic_reward=0):
        total_reward = reward + intrinsic_reward
        self.memory.append((state, action, total_reward, next_state, done))
        
        if intrinsic_reward != 0:
            self.intrinsic_rewards.append(intrinsic_reward)
        
    def update(self):
        self.total_steps += 1
        
        if len(self.memory) < self.batch_size:
            return 0
            
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for experience in batch:
            s, a, r, ns, d = experience
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        if len(self.state_shape) == 3: 
            states_tensor = torch.FloatTensor(states)
            next_states_tensor = torch.FloatTensor(next_states)
        else: 
            states_tensor = torch.FloatTensor(states.reshape(self.batch_size, -1))
            next_states_tensor = torch.FloatTensor(next_states.reshape(self.batch_size, -1))
            
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)
        
        current_q = self.q_network(states_tensor).gather(1, actions_tensor)
        
        with torch.no_grad():
            next_q = self.target_network(next_states_tensor).max(1, keepdim=True)[0]
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q
            
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return loss.item()
        
    def update_exploration(self, episode=None, states=None):
        if episode is not None:
            self.exploration.update(episode)
            
        if hasattr(self.exploration, 'update') and states is not None and self.exploration.name == "RND":
            return self.exploration.update(states)
            
        if hasattr(self.exploration, 'adapt_noise_scale') and states is not None:
            return self.exploration.adapt_noise_scale(states)
            
        return None
        
    def get_intrinsic_reward(self, state):
        if hasattr(self.exploration, 'get_intrinsic_reward'):
            return self.exploration.get_intrinsic_reward(state)
        return 0
        
    def save(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_info': {
                'name': self.exploration.name,
                'value': self.exploration.get_exploration_value()
            }
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])