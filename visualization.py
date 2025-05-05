import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

def visualize_learning_curves(results, env_name, save_dir="figures"):
    plt.figure(figsize=(12, 8))
    
    os.makedirs(save_dir, exist_ok=True)
    
    for strategy_name, strategy_results in results.items():
        mean_rewards = strategy_results['mean_reward']
        std_rewards = strategy_results['std_reward']
        
        episodes = range(1, len(mean_rewards) + 1)
        
        plt.plot(episodes, mean_rewards, label=strategy_name)
        plt.fill_between(episodes, 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards, 
                        alpha=0.2)
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'Learning Curves - {env_name}')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, f"{env_name}_learning_curves.png"))
    plt.close()

def visualize_success_rates(results, env_name, save_dir="figures"):
    plt.figure(figsize=(10, 6))
    
    strategy_names = list(results.keys())
    success_rates = [results[name]['mean_success_rate'] for name in strategy_names]
    
    sorted_indices = np.argsort(success_rates)
    sorted_names = [strategy_names[i] for i in sorted_indices]
    sorted_rates = [success_rates[i] for i in sorted_indices]
    
    bars = plt.bar(sorted_names, sorted_rates, color='skyblue')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Exploration Strategy')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate Comparison - {env_name}')
    plt.ylim(0, 1.1)
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{env_name}_success_rates.png"))
    plt.close()

def visualize_state_coverage(results, env_name, save_dir="figures"):
    plt.figure(figsize=(10, 6))
    
    strategy_names = list(results.keys())
    coverages = [results[name]['mean_state_coverage'] for name in strategy_names]
    
    sorted_indices = np.argsort(coverages)
    sorted_names = [strategy_names[i] for i in sorted_indices]
    sorted_coverages = [coverages[i] for i in sorted_indices]
    
    bars = plt.bar(sorted_names, sorted_coverages, color='lightgreen')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Exploration Strategy')
    plt.ylabel('Unique States Visited')
    plt.title(f'State Coverage Comparison - {env_name}')
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{env_name}_state_coverage.png"))
    plt.close()

def visualize_early_vs_final_performance(results, env_name, save_dir="figures"):
    plt.figure(figsize=(12, 8))
    
    strategy_names = list(results.keys())
    early_rewards = [results[name]['early_reward'] for name in strategy_names]
    final_rewards = [results[name]['final_reward'] for name in strategy_names]
    
    x = np.arange(len(strategy_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, early_rewards, width, label='Early (First 100 Episodes)')
    rects2 = ax.bar(x + width/2, final_rewards, width, label='Final (Last 100 Episodes)')
    
    ax.set_xlabel('Exploration Strategy')
    ax.set_ylabel('Average Reward')
    ax.set_title(f'Early vs Final Performance - {env_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, rotation=45)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),  
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{env_name}_early_vs_final.png"))
    plt.close()

def visualize_steps_to_completion(results, env_name, save_dir="figures"):
    plt.figure(figsize=(12, 8))
    
    for strategy_name, strategy_results in results.items():
        mean_steps = strategy_results['mean_steps']
        episodes = range(1, len(mean_steps) + 1)
        
        plt.plot(episodes, mean_steps, label=strategy_name)
    
    plt.xlabel('Episodes')
    plt.ylabel('Steps to Completion')
    plt.title(f'Steps to Completion - {env_name}')
    plt.legend()
    plt.grid(True)
    
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.savefig(os.path.join(save_dir, f"{env_name}_steps_to_completion.png"))
    plt.close()

def visualize_exploration_values(results, env_name, save_dir="figures"):
    plt.figure(figsize=(12, 8))
    
    for strategy_name, strategy_results in results.items():
        exploration_values = strategy_results['exploration_values'][0]
        episodes = range(1, len(exploration_values) + 1)
        
        plt.plot(episodes, exploration_values, label=strategy_name)
    
    plt.xlabel('Episodes')
    plt.ylabel('Exploration Parameter Value')
    plt.title(f'Exploration Parameter Evolution - {env_name}')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, f"{env_name}_exploration_values.png"))
    plt.close()

def visualize_intrinsic_rewards(results, env_name, save_dir="figures"):
    plt.figure(figsize=(12, 8))
    
    for strategy_name, strategy_results in results.items():
        if 'intrinsic_rewards' in strategy_results:
            intrinsic_rewards = strategy_results['intrinsic_rewards'][0]
            
            if np.sum(intrinsic_rewards) > 0: 
                episodes = range(1, len(intrinsic_rewards) + 1)
                plt.plot(episodes, intrinsic_rewards, label=strategy_name)
    
    plt.xlabel('Episodes')
    plt.ylabel('Intrinsic Reward')
    plt.title(f'Intrinsic Rewards - {env_name}')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, f"{env_name}_intrinsic_rewards.png"))
    plt.close()

def create_summary_table(results, env_name, save_dir="figures"):
    import pandas as pd
    
    data = {
        'Strategy': [],
        'Final Reward': [],
        'Success Rate': [],
        'State Coverage': [],
        'Early Learning': []
    }
    
    for strategy_name, strategy_results in results.items():
        data['Strategy'].append(strategy_name)
        data['Final Reward'].append(strategy_results['final_reward'])
        data['Success Rate'].append(strategy_results['mean_success_rate'])
        data['State Coverage'].append(strategy_results['mean_state_coverage'])
        data['Early Learning'].append(strategy_results['early_reward'])
    
    df = pd.DataFrame(data)
    
    csv_path = os.path.join(save_dir, f"{env_name}_summary.csv")
    df.to_csv(csv_path, index=False)
    
    html_path = os.path.join(save_dir, f"{env_name}_summary.html")
    df.to_html(html_path, index=False)
    
    return df

def visualize_all_results(results, env_name):
    save_dir = os.path.join("figures", env_name.replace('-', '_'))
    os.makedirs(save_dir, exist_ok=True)
    
    visualize_learning_curves(results, env_name, save_dir)
    visualize_success_rates(results, env_name, save_dir)
    visualize_state_coverage(results, env_name, save_dir)
    visualize_early_vs_final_performance(results, env_name, save_dir)
    visualize_steps_to_completion(results, env_name, save_dir)
    visualize_exploration_values(results, env_name, save_dir)
    visualize_intrinsic_rewards(results, env_name, save_dir)
    
    summary_df = create_summary_table(results, env_name, save_dir)
    
    print(f"All visualizations saved to {save_dir}")
    print("\nResults Summary:")
    print(summary_df)