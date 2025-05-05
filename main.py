import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import numpy as np
import torch
import os
import time

from exploration_strategies import (
    EpsilonGreedyExploration, BoltzmannExploration,
    RNDExploration, CountBasedExploration, ParameterNoiseExploration
)
from agent import DQNAgent, DQNNetwork
from experiment import ExplorationExperiment
from visualization import visualize_all_results

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    env_names = [
        "MiniGrid-Empty-8x8-v0",      
        "MiniGrid-FourRooms-v0",      
        "MiniGrid-KeyCorridorS3R3-v0"  
    ]
    
    for env_name in env_names:
        print(f"\n\n{'='*70}")
        print(f"Running experiments on {env_name}")
        print(f"{'='*70}\n")
        
        temp_env = gym.make(env_name)
        temp_env = RGBImgPartialObsWrapper(temp_env, tile_size=8)
        temp_env = ImgObsWrapper(temp_env)
        
        state_shape = temp_env.observation_space.shape
        action_size = temp_env.action_space.n
        
        temp_env.close()
        
        temp_model = DQNNetwork(state_shape, action_size)
        
        exploration_strategies = {
            "EpsilonGreedy": EpsilonGreedyExploration(
                action_size=action_size, 
                epsilon=0.1,           
                epsilon_decay=1.0,       
                epsilon_min=0.1
            ),
            "DecayingEpsilon": EpsilonGreedyExploration(
                action_size=action_size, 
                epsilon=1.0,             
                epsilon_decay=0.995,     
                epsilon_min=0.01        
            ),
            "Boltzmann": BoltzmannExploration(
                temperature=1.0,          
                temperature_decay=0.995,   
                temperature_min=0.1       
            ),
            "RND": RNDExploration(
                state_dim=np.prod(state_shape),
                action_size=action_size,
                reward_scale=0.1,         
                intrinsic_weight=0.5      
            ),
            "CountBased": CountBasedExploration(
                action_size=action_size,
                beta=0.1                  
            ),
            "ParameterNoise": ParameterNoiseExploration(
                model=temp_model,
                action_size=action_size,
                initial_stddev=0.1,      
                target_divergence=0.1   
            )
        }
        
        experiment = ExplorationExperiment(
            env_name=env_name,
            exploration_strategies=exploration_strategies,
            episodes=300,                 
            max_steps=500,              
            runs=3,                     
            results_dir=f"results/{env_name.replace('-', '_')}"
        )
        
        start_time = time.time()
        results = experiment.run_experiment()
        end_time = time.time()
        
        print(f"\nExperiment completed in {(end_time - start_time)/60:.2f} minutes")
        
        visualize_all_results(results, env_name)

if __name__ == "__main__":
    main()