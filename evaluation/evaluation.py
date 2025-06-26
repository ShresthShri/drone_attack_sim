import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_evaluation_data(eval_path="./models/evaluations.npz"):
    """
    Analyze the evaluation data from your SAC drone training.
    """
    if not os.path.exists(eval_path):
        print(f"Evaluation file not found at: {eval_path}")
        print("Available files in models folder:")
        if os.path.exists("./models/"):
            for file in os.listdir("./models/"):
                print(f"  - {file}")
        return
    
    # Load evaluation data
    eval_data = np.load(eval_path)
    
    print("📊 Evaluation Data Analysis")
    print("=" * 50)
    print(f"Available keys in evaluation data: {list(eval_data.keys())}")
    print()
    
    # Extract data
    timesteps = eval_data['timesteps']
    results = eval_data['results']  # Shape: (n_evals, n_eval_episodes)
    ep_lengths = eval_data['ep_lengths']  # Episode lengths
    
    print(f"Number of evaluation points: {len(timesteps)}")
    print(f"Episodes per evaluation: {results.shape[1]}")
    print(f"Training duration: {timesteps[0]:,} to {timesteps[-1]:,} timesteps")
    print()
    
    # Calculate statistics
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)
    mean_ep_lengths = ep_lengths.mean(axis=1)
    std_ep_lengths = ep_lengths.std(axis=1)
    
    # Find best performance
    best_eval_idx = np.argmax(mean_rewards)
    best_timestep = timesteps[best_eval_idx]
    best_reward = mean_rewards[best_eval_idx]
    best_ep_length = mean_ep_lengths[best_eval_idx]
    
    print("🏆 Best Performance:")
    print(f"  Timestep: {best_timestep:,}")
    print(f"  Mean Reward: {best_reward:.2f} ± {std_rewards[best_eval_idx]:.2f}")
    print(f"  Mean Episode Length: {best_ep_length:.1f} ± {std_ep_lengths[best_eval_idx]:.1f}")
    print()
    
    # Final performance
    final_reward = mean_rewards[-1]
    final_ep_length = mean_ep_lengths[-1]
    
    print("🎯 Final Performance:")
    print(f"  Mean Reward: {final_reward:.2f} ± {std_rewards[-1]:.2f}")
    print(f"  Mean Episode Length: {final_ep_length:.1f} ± {std_ep_lengths[-1]:.1f}")
    print()
    
    # Estimate success rate (assuming reward > 80 means success)
    success_threshold = 80.0
    final_eval_rewards = results[-1]  # Last evaluation episode rewards
    success_rate = (final_eval_rewards > success_threshold).mean() * 100
    
    print("📈 Success Analysis:")
    print(f"  Success threshold: {success_threshold}")
    print(f"  Final evaluation success rate: {success_rate:.1f}%")
    print(f"  Individual episode rewards: {final_eval_rewards}")
    print()
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Reward progression
    plt.subplot(2, 3, 1)
    plt.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
    plt.fill_between(timesteps, 
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards, 
                     alpha=0.3, color='blue')
    plt.axhline(y=success_threshold, color='red', linestyle='--', 
                label=f'Success Threshold ({success_threshold})')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Evaluation Rewards Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Episode length progression
    plt.subplot(2, 3, 2)
    plt.plot(timesteps, mean_ep_lengths, 'g-', linewidth=2, label='Mean Episode Length')
    plt.fill_between(timesteps, 
                     mean_ep_lengths - std_ep_lengths,
                     mean_ep_lengths + std_ep_lengths, 
                     alpha=0.3, color='green')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Length')
    plt.title('Episode Length Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Reward vs Episode Length scatter
    plt.subplot(2, 3, 3)
    plt.scatter(mean_ep_lengths, mean_rewards, c=timesteps, cmap='viridis', s=50)
    plt.colorbar(label='Timesteps')
    plt.xlabel('Mean Episode Length')
    plt.ylabel('Mean Reward')
    plt.title('Reward vs Episode Length')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Final evaluation episode rewards histogram
    plt.subplot(2, 3, 4)
    plt.hist(final_eval_rewards, bins=10, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(x=success_threshold, color='red', linestyle='--', 
                label=f'Success Threshold ({success_threshold})')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Final Evaluation Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Success rate over time
    plt.subplot(2, 3, 5)
    success_rates_over_time = []
    for i in range(len(timesteps)):
        eval_rewards = results[i]
        success_rate_at_time = (eval_rewards > success_threshold).mean() * 100
        success_rates_over_time.append(success_rate_at_time)
    
    plt.plot(timesteps, success_rates_over_time, 'r-', linewidth=2, marker='o')
    plt.xlabel('Timesteps')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Over Time')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Learning progress summary
    plt.subplot(2, 3, 6)
    improvement = mean_rewards[-1] - mean_rewards[0]
    efficiency_gain = mean_ep_lengths[0] - mean_ep_lengths[-1]
    
    metrics = ['Initial\nReward', 'Final\nReward', 'Episode Length\nReduction', 'Success Rate\n(%)']
    values = [mean_rewards[0], mean_rewards[-1], efficiency_gain, success_rate]
    colors = ['red', 'green', 'blue', 'purple']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Training Summary')
    plt.ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'timesteps': timesteps,
        'rewards': results,
        'episode_lengths': ep_lengths,
        'best_performance': {
            'timestep': best_timestep,
            'reward': best_reward,
            'episode_length': best_ep_length
        },
        'final_performance': {
            'reward': final_reward,
            'episode_length': final_ep_length,
            'success_rate': success_rate
        }
    }

if __name__ == "__main__":
    # Analyze the evaluation data
    results = analyze_evaluation_data()
    
    print("\n" + "="*50)
    print("💡 Analysis Complete!")
    print("Check the plots above for detailed insights into your drone's learning progress.")