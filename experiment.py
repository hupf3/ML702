import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from collections import defaultdict
import os
import pickle

from exploration_strategies import (
    EpsilonGreedyExploration, BoltzmannExploration,
    RNDExploration, CountBasedExploration, ParameterNoiseExploration
)
from agent import DQNAgent

class ExplorationExperiment:
    def __init__(self, env_name, exploration_strategies, agent_class=DQNAgent,
                episodes=300, max_steps=1000, runs=3, results_dir="results"):
        self.env_name = env_name
        self.exploration_strategies = exploration_strategies
        self.agent_class = agent_class
        self.episodes = episodes
        self.max_steps = max_steps
        self.runs = runs
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)
        
    def run_experiment(self):
        results = {}
        
        for strategy_name, strategy in self.exploration_strategies.items():
            print(f"\n{'='*50}")
            print(f"Evaluating strategy: {strategy_name}")
            print(f"{'='*50}")
            
            strategy_results = self.evaluate_strategy(strategy_name, strategy)
            results[strategy_name] = strategy_results
            
            self._save_strategy_results(strategy_name, strategy_results)
            
        return results
        
    def evaluate_strategy(self, strategy_name, strategy_creator):
        all_rewards = []
        all_steps = []
        all_success_rates = []
        all_state_coverages = []
        all_exploration_values = []
        all_intrinsic_rewards = []
        
        for run in range(self.runs):
            print(f"\nRun {run+1}/{self.runs}")
            
            env = gym.make(self.env_name)
            env = RGBImgPartialObsWrapper(env, tile_size=8)
            env = ImgObsWrapper(env)
            
            state_shape = env.observation_space.shape
            action_size = env.action_space.n
            
            if callable(strategy_creator):
                strategy = strategy_creator(action_size)
            else:
                strategy = strategy_creator
                
            agent = self.agent_class(
                state_shape=state_shape, 
                action_size=action_size,
                exploration_strategy=strategy
            )
            
            episode_rewards = []
            episode_steps = []
            success_count = 0
            visited_states = set()
            exploration_values = []
            episode_intrinsic_rewards = []
            
            for episode in range(self.episodes):
                state = env.reset()
                episode_reward = 0
                intrinsic_reward_sum = 0
                done = False
                steps = 0
                
                state_hash = self._hash_state(state)
                visited_states.add(state_hash)
                
                episode_states = [state]
                
                while not done and steps < self.max_steps:
                    action = agent.select_action(state)
                    
                    next_state, reward, done, info = env.step(action)
                    
                    intrinsic_reward = agent.get_intrinsic_reward(state)
                    intrinsic_reward_sum += intrinsic_reward
                    
                    next_state_hash = self._hash_state(next_state)
                    visited_states.add(next_state_hash)
                    
                    agent.store_experience(state, action, reward, next_state, done, intrinsic_reward)
                    
                    agent.update()
                    
                    state = next_state
                    episode_states.append(state)
                    episode_reward += reward
                    steps += 1
                
                agent.update_exploration(episode=episode, states=episode_states)
                
                episode_rewards.append(episode_reward)
                episode_steps.append(steps)
                episode_intrinsic_rewards.append(intrinsic_reward_sum)
                exploration_values.append(strategy.get_exploration_value())
                
                if 'success' in info and info['success']:
                    success_count += 1
                    
                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode+1}/{self.episodes}, "
                          f"Reward: {episode_reward:.2f}, "
                          f"Intrinsic: {intrinsic_reward_sum:.4f}, "
                          f"Steps: {steps}, "
                          f"States: {len(visited_states)}, "
                          f"Explore: {strategy.get_exploration_value():.4f}")
            
            success_rate = success_count / self.episodes
            state_coverage = len(visited_states)
            
            all_rewards.append(episode_rewards)
            all_steps.append(episode_steps)
            all_success_rates.append(success_rate)
            all_state_coverages.append(state_coverage)
            all_exploration_values.append(exploration_values)
            all_intrinsic_rewards.append(episode_intrinsic_rewards)
            
            env.close()
            
            model_dir = os.path.join(self.results_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            agent.save(os.path.join(model_dir, f"{self.env_name}_{strategy_name}_run{run}.pt"))
        
        rewards_array = np.array(all_rewards)
        steps_array = np.array(all_steps)
        coverages_array = np.array(all_state_coverages)
        
        return {
            'rewards': rewards_array,
            'steps': steps_array,
            'success_rates': all_success_rates,
            'state_coverages': coverages_array,
            'exploration_values': np.array(all_exploration_values),
            'intrinsic_rewards': np.array(all_intrinsic_rewards),
            'mean_reward': np.mean(rewards_array, axis=0),
            'std_reward': np.std(rewards_array, axis=0),
            'mean_steps': np.mean(steps_array, axis=0),
            'mean_success_rate': np.mean(all_success_rates),
            'mean_state_coverage': np.mean(coverages_array),
            'final_reward': np.mean(rewards_array[:, -100:]),  
            'early_reward': np.mean(rewards_array[:, :100])  
        }
        
    def _hash_state(self, state):
        if isinstance(state, np.ndarray):
            if state.ndim == 3 and min(state.shape) > 10:
                downsampled = state[::8, ::8, :] if state.shape[2] <= 3 else state[:, ::8, ::8]
                quantized = (downsampled * 16).astype(int)
                return hash(quantized.tobytes())
            return hash(state.tobytes())
        return hash(str(state))
        
    def _save_strategy_results(self, strategy_name, results):
        filename = f"{self.env_name}_{strategy_name}_results.pkl"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
            
        print(f"Results saved to {filepath}")
        
    def load_results(self):
        results = {}
        
        for strategy_name in self.exploration_strategies.keys():
            filename = f"{self.env_name}_{strategy_name}_results.pkl"
            filepath = os.path.join(self.results_dir, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    results[strategy_name] = pickle.load(f)
                    
        return results