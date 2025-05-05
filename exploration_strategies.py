
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EpsilonGreedyExploration:
    def __init__(self, action_size, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.name = "EpsilonGreedy"
        
    def get_action(self, state, q_values):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(q_values)
        
    def update(self, episode):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def get_exploration_value(self):
        return self.epsilon
        
    def __str__(self):
        return f"EpsilonGreedy(ε={self.epsilon:.4f})"


class BoltzmannExploration:
    def __init__(self, temperature=1.0, temperature_decay=0.995, temperature_min=0.1):
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min
        self.name = "Boltzmann"
        
    def get_action(self, state, q_values):
        q_values = q_values - np.max(q_values)
        exp_values = np.exp(q_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        return np.random.choice(len(q_values), p=probabilities)
        
    def update(self, episode):
        self.temperature = max(self.temperature_min, 
                              self.temperature * self.temperature_decay)
        
    def get_exploration_value(self):
        return self.temperature
        
    def __str__(self):
        return f"Boltzmann(T={self.temperature:.4f})"


class RNDExploration:
    def __init__(self, state_dim, action_size, hidden_dim=128, reward_scale=0.1,
                intrinsic_weight=0.5, epsilon_fallback=0.01):
        self.action_size = action_size
        self.reward_scale = reward_scale
        self.intrinsic_weight = intrinsic_weight
        self.epsilon_fallback = epsilon_fallback 
        self.name = "RND"
        
        if isinstance(state_dim, tuple):
            self.state_dim = np.prod(state_dim)
        else:
            self.state_dim = state_dim
            
        self.target_network = self._build_network(self.state_dim, hidden_dim)
        for param in self.target_network.parameters():
            param.requires_grad = False
            
        self.predictor_network = self._build_network(self.state_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=0.001)
        
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=self.state_dim)
        
    def _build_network(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
    def get_action(self, state, q_values):
        if np.random.random() < self.epsilon_fallback:
            return np.random.randint(self.action_size)
        
        return np.argmax(q_values)
        
    def get_intrinsic_reward(self, state):
        if isinstance(state, np.ndarray):
            state_flat = state.flatten()
        else:
            state_flat = state
            
        state_norm = self._normalize_obs(state_flat)
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0)
        
        with torch.no_grad():
            target_feature = self.target_network(state_tensor)
        predicted_feature = self.predictor_network(state_tensor)
        
        intrinsic_reward = ((target_feature - predicted_feature) ** 2).sum(dim=1).item()
        
        intrinsic_reward = self._normalize_reward(intrinsic_reward)
        
        return self.reward_scale * intrinsic_reward
        
    def update(self, states):
        if isinstance(states, list):
            states_flat = [s.flatten() for s in states]
            states_flat = np.array(states_flat)
        else:
            states_flat = states.flatten().reshape(1, -1)
            
        self.obs_rms.update(states_flat)
        
        states_norm = self._normalize_obs(states_flat)
        states_tensor = torch.FloatTensor(states_norm)
        
        with torch.no_grad():
            target_features = self.target_network(states_tensor)
        predicted_features = self.predictor_network(states_tensor)
        
        loss = F.mse_loss(predicted_features, target_features)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def _normalize_obs(self, obs):
        return (obs - self.obs_rms.mean) / (np.sqrt(self.obs_rms.var) + 1e-8)
        
    def _normalize_reward(self, reward):
        self.reward_rms.update(np.array([reward]))
        return reward / (np.sqrt(self.reward_rms.var) + 1e-8)
        
    def get_exploration_value(self):
        return self.intrinsic_weight
        
    def __str__(self):
        return f"RND(scale={self.reward_scale}, weight={self.intrinsic_weight})"


class CountBasedExploration:
    def __init__(self, action_size, beta=0.05, hash_function=None, 
                epsilon_fallback=0.01):
        self.action_size = action_size
        self.beta = beta
        self.hash_function = hash_function 
        self.epsilon_fallback = epsilon_fallback
        self.visit_counts = {} 
        self.name = "CountBased"
        
    def get_action(self, state, q_values):
        if np.random.random() < self.epsilon_fallback:
            return np.random.randint(self.action_size)
        
        return np.argmax(q_values)
        
    def get_intrinsic_reward(self, state):
        state_hash = self._hash_state(state)
        
        count = self.visit_counts.get(state_hash, 0)
        self.visit_counts[state_hash] = count + 1
        
        if count == 0:
            intrinsic_reward = 1.0 
        else:
            intrinsic_reward = 1.0 / np.sqrt(count)
            
        return self.beta * intrinsic_reward
        
    def _hash_state(self, state):
        if self.hash_function:
            return self.hash_function(state)
        
        if isinstance(state, np.ndarray):
            if state.ndim == 3 and min(state.shape) > 10:
                downsampled = state[::8, ::8, :] if state.shape[2] <= 3 else state[:, ::8, ::8]
                quantized = (downsampled * 16).astype(int)
                return hash(quantized.tobytes())
            return hash(state.tobytes())
        
        return hash(str(state))
        
    def update(self, episode):
        pass
        
    def get_exploration_value(self):
        return len(self.visit_counts)
        
    def __str__(self):
        return f"CountBased(β={self.beta}, states={len(self.visit_counts)})"


class ParameterNoiseExploration:
    def __init__(self, model, action_size, initial_stddev=0.1, 
                target_divergence=0.2, adaptation_factor=1.01):
        self.model = model 
        self.noisy_model = copy.deepcopy(model)
        self.action_size = action_size
        self.stddev = initial_stddev  
        self.target_divergence = target_divergence 
        self.adaptation_factor = adaptation_factor  
        self.name = "ParameterNoise"
        
    def get_action(self, state, q_values):

        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
        with torch.no_grad():
            noisy_q_values = self.noisy_model(state_tensor).detach().numpy()[0]
            
        return np.argmax(noisy_q_values)
        
    def add_noise(self):
        with torch.no_grad():
            for param, noisy_param in zip(self.model.parameters(), 
                                         self.noisy_model.parameters()):
                noise = torch.randn_like(param) * self.stddev
                noisy_param.copy_(param + noise)
                
    def adapt_noise_scale(self, states):
        if not isinstance(states, list):
            states = [states]
            
        original_actions = []
        noisy_actions = []
        
        with torch.no_grad():
            for state in states:
                if isinstance(state, np.ndarray):
                    state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    
                original_q = self.model(state_tensor)
                noisy_q = self.noisy_model(state_tensor)
                
                original_action = torch.argmax(original_q, dim=1).item()
                noisy_action = torch.argmax(noisy_q, dim=1).item()
                
                original_actions.append(original_action)
                noisy_actions.append(noisy_action)
                
        distance = sum(o != n for o, n in zip(original_actions, noisy_actions)) / len(states)
        
        if distance < self.target_divergence:
            self.stddev *= self.adaptation_factor
        else:
            self.stddev /= self.adaptation_factor
            
        return distance
        
    def update(self, episode=None):
        self.add_noise() 
        
    def get_exploration_value(self):
        return self.stddev
        
    def __str__(self):
        return f"ParameterNoise(σ={self.stddev:.4f})"


class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count