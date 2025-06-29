#!/usr/bin/env python3
"""
Evaluate trained progressive difficulty drone navigation model and create visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import json
from datetime import datetime
import pandas as pd

# Import your environment - adjust import path as needed
from agents.train_sac import ProgressiveObstacleNavigationEnv, make_progressive_env


def load_trained_model(model_path, normalize_path=None):
    """Load the trained model and normalization."""
    print(f"📂 Loading model from: {model_path}")
    
    # Load model
    model = SAC.load(model_path)
    
    # Load normalization if available
    if normalize_path and os.path.exists(normalize_path):
        print(f"📂 Loading normalization from: {normalize_path}")
        # Create dummy env for normalization structure
        dummy_env = DummyVecEnv([make_progressive_env(gui=False)])
        normalize = VecNormalize.load(normalize_path, dummy_env)
        normalize.training = False
        normalize.norm_reward = False
        return model, normalize
    else:
        print("⚠️ No normalization file found, using raw environment")
        return model, None


def evaluate_model_at_difficulty(model, env, difficulty_level, n_episodes=5, render=False, record_trajectories=True):
    """Evaluate the model at a specific difficulty level."""
    print(f"🎯 Evaluating at difficulty level {difficulty_level} over {n_episodes} episodes...")
    
    # Set difficulty level for all environments
    if hasattr(env, 'envs'):
        for env_instance in env.envs:
            if hasattr(env_instance.env, 'difficulty_level'):
                env_instance.env.difficulty_level = difficulty_level
    elif hasattr(env.env, 'difficulty_level'):
        env.env.difficulty_level = difficulty_level
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    final_distances = []
    efficiency_scores = []
    
    # Trajectory tracking
    trajectories = []
    obstacle_data = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        # Track trajectory for this episode
        if record_trajectories:
            trajectory = []
            # Get environment instance
            if hasattr(env, 'envs'):
                env_instance = env.envs[0].env
            else:
                env_instance = env.env if hasattr(env, 'env') else env
            
            current_obstacles = []
            if hasattr(env_instance, 'obstacles'):
                current_obstacles = [(obs['position'], obs['size']) for obs in env_instance.obstacles]
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic for evaluation
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            episode_length += 1
            
            # Record trajectory
            if record_trajectories and hasattr(env_instance, 'pos'):
                current_pos = env_instance.pos[0].copy()
                trajectory.append(current_pos)
            
            if render:
                env.render()
        
        # Process episode results
        final_info = info[0] if isinstance(info, list) else info
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Count termination reasons
        if isinstance(final_info, dict):
            success = final_info.get('success', False)
            collision = final_info.get('collision', False)
            distance = final_info.get('distance_to_goal', np.inf)
            
            if success:
                success_count += 1
                print(f"  Episode {episode+1}: ✅ SUCCESS (reward: {episode_reward:.1f}, length: {episode_length})")
            elif collision:
                collision_count += 1
                print(f"  Episode {episode+1}: 💥 COLLISION (reward: {episode_reward:.1f}, length: {episode_length})")
            else:
                timeout_count += 1
                print(f"  Episode {episode+1}: ⏰ TIMEOUT (reward: {episode_reward:.1f}, length: {episode_length})")
            
            final_distances.append(distance)
            
            # Calculate efficiency
            if hasattr(env_instance, '_get_efficiency_metric'):
                efficiency = env_instance._get_efficiency_metric()
                efficiency_scores.append(efficiency)
        else:
            print(f"  Episode {episode+1}: COMPLETED (reward: {episode_reward:.1f}, length: {episode_length})")
        
        # Store trajectory and obstacles
        if record_trajectories:
            trajectories.append(trajectory)
            obstacle_data.append(current_obstacles)
    
    # Calculate statistics
    stats = {
        'difficulty_level': difficulty_level,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / n_episodes,
        'collision_rate': collision_count / n_episodes,
        'timeout_rate': timeout_count / n_episodes,
        'mean_final_distance': np.mean(final_distances) if final_distances else np.inf,
        'mean_efficiency': np.mean(efficiency_scores) if efficiency_scores else 0.0,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_distances': final_distances,
        'efficiency_scores': efficiency_scores,
        'trajectories': trajectories,
        'obstacles': obstacle_data
    }
    
    return stats


def evaluate_all_difficulties(model, env, n_episodes_per_level=5, render=False):
    """Evaluate model across all difficulty levels."""
    print("🚀 Evaluating across all difficulty levels...")
    
    all_stats = {}
    difficulty_levels = [0, 1, 2, 3]  # Based on your progressive environment
    
    for level in difficulty_levels:
        print(f"\n{'='*50}")
        print(f"🎯 DIFFICULTY LEVEL {level}")
        level_names = {0: "Beginner (No obstacles)", 1: "Easy (2 obstacles)", 
                      2: "Medium (4 obstacles)", 3: "Hard (6 obstacles)"}
        print(f"   {level_names.get(level, f'Level {level}')}")
        print(f"{'='*50}")
        
        stats = evaluate_model_at_difficulty(
            model, env, level, n_episodes_per_level, render, record_trajectories=True
        )
        all_stats[level] = stats
        
        # Print level summary
        print(f"\n📊 LEVEL {level} SUMMARY:")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   Mean Reward: {stats['mean_reward']:.1f} ± {stats['std_reward']:.1f}")
        print(f"   Mean Length: {stats['mean_length']:.1f} steps")
        print(f"   Collision Rate: {stats['collision_rate']:.1%}")
        if stats['efficiency_scores']:
            print(f"   Mean Efficiency: {stats['mean_efficiency']:.3f}")
    
    return all_stats


def plot_difficulty_comparison(all_stats, save_dir=None):
    """Create comprehensive comparison plots across difficulty levels."""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Extract data for plotting
    levels = sorted(all_stats.keys())
    success_rates = [all_stats[level]['success_rate'] for level in levels]
    mean_rewards = [all_stats[level]['mean_reward'] for level in levels]
    std_rewards = [all_stats[level]['std_reward'] for level in levels]
    mean_lengths = [all_stats[level]['mean_length'] for level in levels]
    collision_rates = [all_stats[level]['collision_rate'] for level in levels]
    mean_efficiencies = [all_stats[level]['mean_efficiency'] for level in levels]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Success Rate by Difficulty
    plt.subplot(3, 4, 1)
    bars = plt.bar(levels, success_rates, color=['green', 'lightgreen', 'orange', 'red'], alpha=0.7)
    plt.xlabel('Difficulty Level')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Difficulty Level')
    plt.ylim(0, 1.1)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{success_rates[i]:.1%}', ha='center', va='bottom', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. Mean Reward by Difficulty
    plt.subplot(3, 4, 2)
    plt.errorbar(levels, mean_rewards, yerr=std_rewards, marker='o', linewidth=3, markersize=8, capsize=5)
    plt.xlabel('Difficulty Level')
    plt.ylabel('Mean Episode Reward')
    plt.title('Performance by Difficulty Level')
    plt.grid(True, alpha=0.3)
    
    # 3. Episode Length by Difficulty
    plt.subplot(3, 4, 3)
    plt.plot(levels, mean_lengths, 'o-', linewidth=3, markersize=8, color='purple')
    plt.xlabel('Difficulty Level')
    plt.ylabel('Mean Episode Length')
    plt.title('Episode Length by Difficulty')
    plt.grid(True, alpha=0.3)
    
    # 4. Collision Rate by Difficulty
    plt.subplot(3, 4, 4)
    bars = plt.bar(levels, collision_rates, color=['darkgreen', 'green', 'orange', 'darkred'], alpha=0.7)
    plt.xlabel('Difficulty Level')
    plt.ylabel('Collision Rate')
    plt.title('Collision Rate by Difficulty Level')
    plt.ylim(0, max(collision_rates) * 1.2 if collision_rates else 1)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{collision_rates[i]:.1%}', ha='center', va='bottom', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5-8. Individual Difficulty Level Reward Distributions
    for i, level in enumerate(levels):
        plt.subplot(3, 4, 5 + i)
        rewards = all_stats[level]['episode_rewards']
        plt.hist(rewards, bins=max(3, len(rewards)//2), alpha=0.7, 
                color=['green', 'lightgreen', 'orange', 'red'][i], edgecolor='black')
        plt.axvline(np.mean(rewards), color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(rewards):.1f}')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title(f'Level {level} Reward Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 9. Efficiency by Difficulty
    plt.subplot(3, 4, 9)
    valid_efficiencies = [eff for eff in mean_efficiencies if eff > 0]
    valid_levels = [levels[i] for i, eff in enumerate(mean_efficiencies) if eff > 0]
    if valid_efficiencies:
        plt.plot(valid_levels, valid_efficiencies, 'o-', linewidth=3, markersize=8, color='teal')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Mean Efficiency')
        plt.title('Path Efficiency by Difficulty')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No efficiency data available', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 10. Performance Summary Table
    plt.subplot(3, 4, 10)
    plt.axis('off')
    
    table_data = []
    for level in levels:
        stats = all_stats[level]
        table_data.append([
            f"Level {level}",
            f"{stats['success_rate']:.1%}",
            f"{stats['mean_reward']:.0f}",
            f"{stats['collision_rate']:.1%}",
            f"{stats['mean_length']:.0f}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Level', 'Success', 'Reward', 'Collision', 'Length'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Performance Summary Table', pad=20)
    
    # 11. Success Rate Trend
    plt.subplot(3, 4, 11)
    plt.plot(levels, success_rates, 'o-', linewidth=4, markersize=10, color='darkgreen')
    plt.fill_between(levels, success_rates, alpha=0.3, color='green')
    plt.xlabel('Difficulty Level')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Trend')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # 12. Overall Performance Score
    plt.subplot(3, 4, 12)
    # Calculate overall performance score (weighted by difficulty)
    performance_scores = []
    for level in levels:
        stats = all_stats[level]
        # Score: success_rate * (1 + difficulty_weight) - collision_rate
        difficulty_weight = level * 0.2  # Higher levels get bonus
        score = stats['success_rate'] * (1 + difficulty_weight) - stats['collision_rate'] * 0.5
        performance_scores.append(score)
    
    bars = plt.bar(levels, performance_scores, color=['lightblue', 'blue', 'purple', 'darkblue'], alpha=0.7)
    plt.xlabel('Difficulty Level')
    plt.ylabel('Performance Score')
    plt.title('Overall Performance Score\n(Success Rate + Difficulty Bonus - Collision Penalty)')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{performance_scores[i]:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/difficulty_comparison.png", dpi=300, bbox_inches='tight')
        print(f"📊 Difficulty comparison plots saved to: {save_dir}/difficulty_comparison.png")
    
    plt.show()


def plot_trajectory_3d(stats, save_dir=None, episode_idx=0, difficulty_level=0):
    """Create 3D trajectory visualization with obstacles and waypoints."""
    
    if not stats['trajectories'] or episode_idx >= len(stats['trajectories']):
        print("⚠️ No trajectory data available for visualization")
        return
    
    trajectory = np.array(stats['trajectories'][episode_idx])
    obstacles = stats['obstacles'][episode_idx] if stats['obstacles'] else []
    
    # Environment parameters - get from environment
    start_waypoint = np.array([0.0, 0.0, 1.0])
    end_waypoints_by_level = [
        np.array([1.5, 1.5, 1.2]),  # Level 0
        np.array([2.0, 2.0, 1.3]),  # Level 1
        np.array([2.8, 2.8, 1.7]),  # Level 2
        np.array([3.5, 3.5, 2.0])   # Level 3
    ]
    end_waypoint = end_waypoints_by_level[difficulty_level]
    workspace_bounds = 4.0
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot workspace bounds (as a wireframe box)
    bounds = workspace_bounds
    # Define the vertices of the cube
    vertices = [
        [-bounds, -bounds, 0], [bounds, -bounds, 0], [bounds, bounds, 0], [-bounds, bounds, 0],  # bottom
        [-bounds, -bounds, bounds], [bounds, -bounds, bounds], [bounds, bounds, bounds], [-bounds, bounds, bounds]  # top
    ]
    # Define the 12 edges of the cube
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # top edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    for edge in edges:
        start, end = edge
        ax.plot3D([vertices[start][0], vertices[end][0]],
                 [vertices[start][1], vertices[end][1]],
                 [vertices[start][2], vertices[end][2]], 'k--', alpha=0.3)
    
    # Plot obstacles as 3D cubes
    for i, (obs_pos, obs_size) in enumerate(obstacles):
        # Create cube vertices
        x, y, z = obs_pos
        size = obs_size
        
        # Define cube faces
        cube_vertices = np.array([
            [x-size, y-size, z-size], [x+size, y-size, z-size], [x+size, y+size, z-size], [x-size, y+size, z-size],
            [x-size, y-size, z+size], [x+size, y-size, z+size], [x+size, y+size, z+size], [x-size, y+size, z+size]
        ])
        
        # Draw cube edges
        cube_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
        ]
        
        for edge in cube_edges:
            start, end = edge
            ax.plot3D([cube_vertices[start][0], cube_vertices[end][0]],
                     [cube_vertices[start][1], cube_vertices[end][1]],
                     [cube_vertices[start][2], cube_vertices[end][2]], 'r-', linewidth=2, alpha=0.8)
    
    # Plot waypoints
    ax.scatter(start_waypoint[0], start_waypoint[1], start_waypoint[2], 
              c='green', s=200, marker='o', label='Start', alpha=0.8, edgecolors='darkgreen', linewidth=2)
    ax.scatter(end_waypoint[0], end_waypoint[1], end_waypoint[2], 
              c='blue', s=200, marker='s', label='Goal', alpha=0.8, edgecolors='darkblue', linewidth=2)
    
    # Plot trajectory
    if len(trajectory) > 0:
        traj_x = trajectory[:, 0]
        traj_y = trajectory[:, 1]
        traj_z = trajectory[:, 2]
        
        # Color trajectory by time (progression)
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
        
        # Plot trajectory as connected line segments with color
        for i in range(len(trajectory) - 1):
            ax.plot3D([traj_x[i], traj_x[i+1]], [traj_y[i], traj_y[i+1]], [traj_z[i], traj_z[i+1]], 
                     color=colors[i], linewidth=3, alpha=0.8)
        
        # Mark start and end of trajectory
        ax.scatter(traj_x[0], traj_y[0], traj_z[0], c='lightgreen', s=100, alpha=0.7, label='Traj Start')
        ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], c='red', s=100, alpha=0.7, label='Traj End')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(f'Drone 3D Trajectory - Level {difficulty_level}, Episode {episode_idx + 1}')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = workspace_bounds
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range])
    
    # Add trajectory statistics
    if len(trajectory) > 0:
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        direct_distance = np.linalg.norm(end_waypoint - start_waypoint)
        efficiency = direct_distance / total_distance if total_distance > 0 else 0
        
        stats_text = f"""Trajectory Stats:
Total Distance: {total_distance:.2f}m
Direct Distance: {direct_distance:.2f}m
Path Efficiency: {efficiency:.2%}
Steps: {len(trajectory)}
Obstacles: {len(obstacles)}"""
        
        # Add text box (position it in the corner)
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    if save_dir:
        plt.savefig(f"{save_dir}/trajectory_3d_level_{difficulty_level}_episode_{episode_idx+1}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"🗺️ 3D trajectory plot saved to: {save_dir}/trajectory_3d_level_{difficulty_level}_episode_{episode_idx+1}.png")
    
    plt.show()


def main():
    """Main evaluation function."""
    
    # Configuration - UPDATE THESE PATHS
    model_dir = "models/progressive_drone_nav_20250629_145417"  # Update with your actual timestamp
    model_path = f"{model_dir}/final_model.zip"  # or best_model.zip if you prefer
    normalize_path = f"{model_dir}/vec_normalize.pkl"
    n_episodes_per_level = 5  # Episodes to run per difficulty level
    
    # Create evaluation directory
    eval_dir = f"evaluation_progressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(eval_dir, exist_ok=True)
    
    print("🚁 Progressive Drone Navigation Model Evaluation")
    print(f"📁 Model: {model_path}")
    print(f"📁 Results will be saved to: {eval_dir}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please update the model_dir path in the script")
        return
    
    # Load model
    try:
        model, normalize = load_trained_model(model_path, normalize_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create evaluation environment
    try:
        eval_env = DummyVecEnv([make_progressive_env(gui=False)])
        if normalize is not None:
            eval_env = normalize
        print("✅ Evaluation environment created!")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate model across all difficulty levels
    try:
        all_stats = evaluate_all_difficulties(
            model, eval_env, n_episodes_per_level=n_episodes_per_level, render=False
        )
        
        # Save comprehensive statistics
        stats_for_json = {}
        for level, stats in all_stats.items():
            stats_copy = stats.copy()
            # Remove non-serializable data for JSON
            stats_copy.pop('trajectories', None)
            stats_copy.pop('obstacles', None)
            stats_for_json[level] = stats_copy
        
        with open(f"{eval_dir}/evaluation_stats.json", 'w') as f:
            json.dump(stats_for_json, f, indent=2, default=str)
        
        # Create comprehensive comparison plots
        print("\n📊 Creating difficulty comparison plots...")
        plot_difficulty_comparison(all_stats, save_dir=eval_dir)
        
        # Create trajectory plots for best performing episodes at each level
        print("\n🗺️ Creating trajectory visualizations...")
        for level, stats in all_stats.items():
            if stats['trajectories']:
                # Find episode with highest reward for this level
                best_episode_idx = np.argmax(stats['episode_rewards'])
                plot_trajectory_3d(stats, save_dir=eval_dir, 
                                 episode_idx=best_episode_idx, difficulty_level=level)
        
        # Print comprehensive summary
        print(f"\n✅ Evaluation complete! Results saved to: {eval_dir}")
        print(f"\n{'='*60}")
        print("📈 COMPREHENSIVE SUMMARY")
        print(f"{'='*60}")
        
        for level in sorted(all_stats.keys()):
            stats = all_stats[level]
            level_names = {0: "Beginner", 1: "Easy", 2: "Medium", 3: "Hard"}
            print(f"\n🎯 LEVEL {level} ({level_names.get(level, 'Unknown')}):")
            print(f"   Success Rate: {stats['success_rate']:.1%}")
            print(f"   Mean Reward: {stats['mean_reward']:.1f} ± {stats['std_reward']:.1f}")
            print(f"   Collision Rate: {stats['collision_rate']:.1%}")
            print(f"   Mean Episode Length: {stats['mean_length']:.1f} steps")
            if stats['efficiency_scores']:
                print(f"   Mean Efficiency: {stats['mean_efficiency']:.3f}")
        
        # Overall assessment
        overall_success = np.mean([stats['success_rate'] for stats in all_stats.values()])
        hardest_level_success = all_stats[3]['success_rate']  # Level 3 (hardest)
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print(f"   Average Success Rate: {overall_success:.1%}")
        print(f"   Hardest Level Success: {hardest_level_success:.1%}")
        
        if hardest_level_success > 0.8:
            print("   🌟 EXCELLENT: Agent masters all difficulty levels!")
        elif hardest_level_success > 0.6:
            print("   ✅ GOOD: Agent handles most scenarios well")
        elif hardest_level_success > 0.4:
            print("   ⚠️ FAIR: Agent struggles with hardest scenarios")
        else:
            print("   ❌ NEEDS IMPROVEMENT: Agent fails on difficult scenarios")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        eval_env.close()


if __name__ == "__main__":
    main()