#!/usr/bin/env python3
"""
Evaluate trained progressive difficulty drone navigation model and create visualizations - ENHANCED WITH 3D
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
    """Evaluate the model at a specific difficulty level - FIXED VERSION."""
    print(f"🎯 Evaluating at difficulty level {difficulty_level} over {n_episodes} episodes...")
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    final_distances = []
    efficiency_scores = []
    trajectories = []
    obstacle_data = []
    
    for episode in range(n_episodes):
        # CRITICAL: Reset environment first, then set difficulty
        obs = env.reset()
        
        # Get the actual environment instance
        if hasattr(env, 'envs'):
            env_instance = env.envs[0].env if hasattr(env.envs[0], 'env') else env.envs[0]
        else:
            env_instance = env.env if hasattr(env, 'env') else env
        
        # Set difficulty level and force regeneration
        if hasattr(env_instance, 'difficulty_level'):
            env_instance.difficulty_level = difficulty_level
            print(f"  Setting difficulty to {difficulty_level}")
            
            # Force obstacle regeneration by resetting again
            obs = env.reset()
            
            # Verify obstacles were created
            if hasattr(env_instance, 'obstacles'):
                expected_obstacles = env_instance.num_obstacles_by_level[difficulty_level]
                actual_obstacles = len(env_instance.obstacles)
                print(f"  Expected {expected_obstacles} obstacles, got {actual_obstacles}")
                
                if actual_obstacles != expected_obstacles:
                    print(f"  ⚠️ Obstacle count mismatch! Regenerating...")
                    env_instance._clear_obstacles_safe()
                    env_instance._generate_progressive_obstacles()
                    actual_obstacles = len(env_instance.obstacles)
                    print(f"  After regeneration: {actual_obstacles} obstacles")
        
        # Record initial state
        episode_reward = 0.0
        episode_length = 0
        done = False
        trajectory = []
        collision_detected = False
        
        # Record obstacles for this episode
        current_obstacles = []
        if hasattr(env_instance, 'obstacles'):
            current_obstacles = [(obs_info['position'], obs_info['size']) for obs_info in env_instance.obstacles]
            print(f"  Episode {episode+1}: Starting with {len(current_obstacles)} obstacles")
        
        # Run episode
        step_count = 0
        max_steps = 1000  # Prevent infinite loops
        
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            episode_length += 1
            step_count += 1
            
            # Record trajectory
            if record_trajectories and hasattr(env_instance, 'pos'):
                current_pos = env_instance.pos[0].copy()
                trajectory.append(current_pos)
            
            # Check for collision detection debug
            if hasattr(env_instance, '_check_collision'):
                collision_status = env_instance._check_collision()
                if collision_status and not collision_detected:
                    print(f"    Step {step_count}: COLLISION DETECTED!")
                    collision_detected = True
            
            if render:
                env.render()
        
        # Process episode results
        final_info = info[0] if isinstance(info, list) else info
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Analyze termination reason
        termination_reason = "UNKNOWN"
        if isinstance(final_info, dict):
            success = final_info.get('success', False)
            collision = final_info.get('collision', False)
            distance = final_info.get('distance_to_goal', np.inf)
            
            if success:
                success_count += 1
                termination_reason = "SUCCESS ✅"
            elif collision:
                collision_count += 1
                termination_reason = "COLLISION 💥"
            else:
                timeout_count += 1
                termination_reason = "TIMEOUT/OOB ⏰"
            
            final_distances.append(distance)
            
            # Calculate efficiency
            if hasattr(env_instance, '_get_efficiency_metric'):
                efficiency = env_instance._get_efficiency_metric()
                efficiency_scores.append(efficiency)
        else:
            print(f"  Episode {episode+1}: No info dict available")
        
        # Manual trajectory analysis for obstacle violations
        obstacle_violations = 0
        if trajectory and current_obstacles:
            for obs_pos, obs_size in current_obstacles:
                violated = False
                for point in trajectory:
                    distance_to_obstacle = np.linalg.norm(point[:2] - obs_pos[:2])
                    if distance_to_obstacle < obs_size + 0.3:  # Including drone radius + collision threshold
                        violated = True
                        break
                if violated:
                    obstacle_violations += 1
        
        print(f"  Episode {episode+1}: {termination_reason}")
        print(f"    Reward: {episode_reward:.1f}, Length: {episode_length}, Obstacles: {len(current_obstacles)}")
        if obstacle_violations > 0:
            print(f"    ⚠️ MANUAL CHECK: Trajectory violated {obstacle_violations} obstacles!")
        
        # Store trajectory and obstacles
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
        level_names = {0: "Beginner (No obstacles)", 1: "Easy (1 obstacle)", 
                      2: "Medium (3 obstacles)", 3: "Hard (5 obstacles)"}  # FIXED obstacle counts
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


def plot_trajectory_2d(stats, save_dir=None, episode_idx=0, difficulty_level=0):
    """Create 2D trajectory visualization with obstacles and waypoints - FIXED VERSION."""
    
    if not stats['trajectories'] or episode_idx >= len(stats['trajectories']):
        print("⚠️ No trajectory data available for 2D visualization")
        return
    
    trajectory = np.array(stats['trajectories'][episode_idx])
    obstacles = stats['obstacles'][episode_idx] if stats['obstacles'] else []
    
    # CORRECT waypoints to match training script exactly
    start_waypoint = np.array([-1.0, -3.0, 1.0])
    end_waypoints_by_level = [
        np.array([0.0, 1.5, 1.2]),  # Level 0
        np.array([0.0, 2.0, 1.3]),  # Level 1
        np.array([0.0, 2.8, 1.7]),  # Level 2
        np.array([0.0, 3.5, 2.0])   # Level 3
    ]
    end_waypoint = end_waypoints_by_level[difficulty_level]
    workspace_bounds = 4.0
    
    # Create 2D plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot workspace bounds
    ax.plot([-workspace_bounds, workspace_bounds, workspace_bounds, -workspace_bounds, -workspace_bounds],
            [-workspace_bounds, -workspace_bounds, workspace_bounds, workspace_bounds, -workspace_bounds],
            'k--', linewidth=2, alpha=0.5, label='Workspace Bounds')
    
    # Plot obstacles
    for i, (obs_pos, obs_size) in enumerate(obstacles):
        obs_x, obs_y = obs_pos[0], obs_pos[1]
        square = plt.Rectangle((obs_x - obs_size, obs_y - obs_size), 
                              2 * obs_size, 2 * obs_size,
                              facecolor='red', alpha=0.6, edgecolor='darkred', linewidth=2)
        ax.add_patch(square)
        ax.text(obs_x, obs_y, f'{i+1}', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=12)
    
    # Plot waypoints
    ax.plot(start_waypoint[0], start_waypoint[1], 'go', markersize=15, 
            label='Start', markeredgecolor='darkgreen', markeredgewidth=3)
    ax.plot(end_waypoint[0], end_waypoint[1], 'bs', markersize=15, 
            label='Goal', markeredgecolor='darkblue', markeredgewidth=3)
    
    # Plot direct path
    ax.plot([start_waypoint[0], end_waypoint[0]], [start_waypoint[1], end_waypoint[1]], 
            'yellow', linestyle=':', linewidth=3, alpha=0.8, label='Direct Path')
    
    # Plot trajectory with time progression
    if len(trajectory) > 0:
        traj_x = trajectory[:, 0]
        traj_y = trajectory[:, 1]
        
        # Color by time
        colors = plt.cm.plasma(np.linspace(0, 1, len(trajectory)))
        
        for i in range(len(trajectory) - 1):
            ax.plot([traj_x[i], traj_x[i+1]], [traj_y[i], traj_y[i+1]], 
                   color=colors[i], linewidth=4, alpha=0.8)
        
        # Mark trajectory endpoints
        ax.plot(traj_x[0], traj_y[0], 'go', markersize=10, alpha=0.8, 
               markeredgecolor='darkgreen', markeredgewidth=2, label='Traj Start')
        ax.plot(traj_x[-1], traj_y[-1], 'ro', markersize=10, alpha=0.8, 
               markeredgecolor='darkred', markeredgewidth=2, label='Traj End')
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                                  norm=plt.Normalize(vmin=0, vmax=len(trajectory)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Time Step', rotation=270, labelpad=20)
        
        # Calculate and display statistics
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        direct_distance = np.linalg.norm(end_waypoint[:2] - start_waypoint[:2])
        efficiency = direct_distance / total_distance if total_distance > 0 else 0
        
        # Check if trajectory passes through obstacles
        obstacle_violations = 0
        min_distances = []
        for obs_pos, obs_size in obstacles:
            min_dist_to_obs = float('inf')
            for point in trajectory:
                distance_to_obstacle = np.linalg.norm(point[:2] - obs_pos[:2])
                min_dist_to_obs = min(min_dist_to_obs, distance_to_obstacle)
                if distance_to_obstacle < obs_size + 0.3:  # Including drone radius + collision threshold
                    obstacle_violations += 1
                    break
            min_distances.append(min_dist_to_obs)
        
        # Final distance to goal
        final_distance = np.linalg.norm(trajectory[-1][:2] - end_waypoint[:2])
        
        stats_text = f"""Trajectory Stats:
Total Distance: {total_distance:.2f}m
Direct Distance: {direct_distance:.2f}m  
Path Efficiency: {efficiency:.1%}
Steps: {len(trajectory)}
Obstacles: {len(obstacles)}
Max Altitude: {np.max(trajectory[:, 2]):.2f}m
Final Distance to Goal: {final_distance:.2f}m
Obstacle Violations: {obstacle_violations}"""
        
        if min_distances:
            stats_text += f"\nClosest to Obstacle: {min(min_distances):.2f}m"
        
        # Color code the text box based on obstacle violations
        box_color = "lightcoral" if obstacle_violations > 0 else "lightgreen"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor=box_color, alpha=0.9))
    
    ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax.set_title(f'Drone 2D Trajectory - Level {difficulty_level}, Episode {episode_idx + 1}\n'
                f'Top-Down View (X-Y Plane)', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/trajectory_2d_level_{difficulty_level}_episode_{episode_idx+1}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"🗺️ 2D trajectory plot saved")
    
    plt.show()


def create_3d_obstacle_wireframe(ax, position, size, color='red', alpha=0.3):
    """Create a 3D wireframe box for obstacle visualization."""
    x, y, z = position
    
    # Define the 8 vertices of the box
    vertices = [
        [x-size, y-size, z-size], [x+size, y-size, z-size],
        [x+size, y+size, z-size], [x-size, y+size, z-size],
        [x-size, y-size, z+size], [x+size, y-size, z+size],
        [x+size, y+size, z+size], [x-size, y+size, z+size]
    ]
    
    # Define the 6 faces of the box
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
        [vertices[4], vertices[7], vertices[3], vertices[0]]   # Left
    ]
    
    # Add faces to the plot
    face_collection = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='darkred', linewidth=2)
    ax.add_collection3d(face_collection)
    
    return face_collection


def plot_trajectory_3d(stats, save_dir=None, episode_idx=0, difficulty_level=0):
    """Create COMPREHENSIVE 3D trajectory visualization with obstacles and waypoints."""
    
    if not stats['trajectories'] or episode_idx >= len(stats['trajectories']):
        print("⚠️ No trajectory data available for 3D visualization")
        return
    
    trajectory = np.array(stats['trajectories'][episode_idx])
    obstacles = stats['obstacles'][episode_idx] if stats['obstacles'] else []
    
    # CORRECT waypoints to match training script exactly
    start_waypoint = np.array([-1.0, -3.0, 1.0])
    end_waypoints_by_level = [
        np.array([0.0, 1.5, 1.2]),  # Level 0
        np.array([0.0, 2.0, 1.3]),  # Level 1
        np.array([0.0, 2.8, 1.7]),  # Level 2
        np.array([0.0, 3.5, 2.0])   # Level 3
    ]
    end_waypoint = end_waypoints_by_level[difficulty_level]
    workspace_bounds = 4.0
    
    # Create 3D plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot workspace bounds (wireframe cube)
    wb = workspace_bounds
    # Bottom face
    ax.plot([-wb, wb, wb, -wb, -wb], [-wb, -wb, wb, wb, -wb], [0, 0, 0, 0, 0], 
            'k--', alpha=0.5, linewidth=2)
    # Top face
    ax.plot([-wb, wb, wb, -wb, -wb], [-wb, -wb, wb, wb, -wb], [3, 3, 3, 3, 3], 
            'k--', alpha=0.5, linewidth=2)
    # Vertical edges
    for x, y in [(-wb, -wb), (wb, -wb), (wb, wb), (-wb, wb)]:
        ax.plot([x, x], [y, y], [0, 3], 'k--', alpha=0.5, linewidth=2)
    
    # Plot ground plane
    xx, yy = np.meshgrid(np.linspace(-wb, wb, 10), np.linspace(-wb, wb, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
    
    # Plot 3D obstacles with enhanced visualization
    for i, (obs_pos, obs_size) in enumerate(obstacles):
        # Create solid obstacle
        create_3d_obstacle_wireframe(ax, obs_pos, obs_size, color='red', alpha=0.4)
        
        # Add obstacle label
        ax.text(obs_pos[0], obs_pos[1], obs_pos[2] + obs_size + 0.1, f'Obs {i+1}', 
                fontsize=12, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add danger zone visualization (larger transparent box)
        danger_size = obs_size + 0.2  # Drone radius + collision threshold
        create_3d_obstacle_wireframe(ax, obs_pos, danger_size, color='yellow', alpha=0.15)
    
    # Plot waypoints with enhanced 3D markers
    ax.scatter(start_waypoint[0], start_waypoint[1], start_waypoint[2], 
               c='green', s=200, marker='o', alpha=0.9, edgecolors='darkgreen', linewidth=3, label='Start')
    ax.scatter(end_waypoint[0], end_waypoint[1], end_waypoint[2], 
               c='blue', s=200, marker='s', alpha=0.9, edgecolors='darkblue', linewidth=3, label='Goal')
    
    # Add waypoint labels
    ax.text(start_waypoint[0], start_waypoint[1], start_waypoint[2] + 0.3, 'START', 
            fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.9))
    ax.text(end_waypoint[0], end_waypoint[1], end_waypoint[2] + 0.3, 'GOAL', 
            fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.9))
    
    # Plot direct path line
    ax.plot([start_waypoint[0], end_waypoint[0]], 
            [start_waypoint[1], end_waypoint[1]], 
            [start_waypoint[2], end_waypoint[2]], 
            'yellow', linestyle=':', linewidth=4, alpha=0.8, label='Direct Path')
    
    # Plot 3D trajectory with advanced visualization
    if len(trajectory) > 0:
        traj_x = trajectory[:, 0]
        traj_y = trajectory[:, 1]
        traj_z = trajectory[:, 2]
        
        # Create time-based color mapping
        time_steps = np.arange(len(trajectory))
        colors = plt.cm.plasma(np.linspace(0, 1, len(trajectory)))
        
        # Plot trajectory as connected line segments with varying colors
        for i in range(len(trajectory) - 1):
            ax.plot([traj_x[i], traj_x[i+1]], 
                   [traj_y[i], traj_y[i+1]], 
                   [traj_z[i], traj_z[i+1]], 
                   color=colors[i], linewidth=3, alpha=0.8)
        
        # Add trajectory start/end markers
        ax.scatter(traj_x[0], traj_y[0], traj_z[0], 
                  c='lime', s=150, marker='o', alpha=1.0, 
                  edgecolors='darkgreen', linewidth=2, label='Traj Start')
        ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], 
                  c='red', s=150, marker='X', alpha=1.0, 
                  edgecolors='darkred', linewidth=2, label='Traj End')
        
        # Add altitude projection (shadow on ground)
        ax.plot(traj_x, traj_y, np.zeros_like(traj_z), 
               color='gray', linewidth=2, alpha=0.5, linestyle='--', label='Ground Projection')
        
        # Sample trajectory points for detailed visualization
        sample_indices = np.linspace(0, len(trajectory)-1, min(20, len(trajectory)), dtype=int)
        for idx in sample_indices[::3]:  # Every 3rd sample point
            ax.scatter(traj_x[idx], traj_y[idx], traj_z[idx], 
                      c=colors[idx], s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add vertical line to ground for key points
            if idx % 6 == 0:  # Every 6th point
                ax.plot([traj_x[idx], traj_x[idx]], 
                       [traj_y[idx], traj_y[idx]], 
                       [0, traj_z[idx]], 
                       color='gray', linewidth=1, alpha=0.3, linestyle=':')
        
        # Calculate comprehensive statistics
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        direct_distance = np.linalg.norm(end_waypoint - start_waypoint)
        efficiency = direct_distance / total_distance if total_distance > 0 else 0
        
        # Altitude statistics
        min_altitude = np.min(traj_z)
        max_altitude = np.max(traj_z)
        avg_altitude = np.mean(traj_z)
        
        # Velocity analysis
        velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        avg_velocity = np.mean(velocities) if len(velocities) > 0 else 0
        max_velocity = np.max(velocities) if len(velocities) > 0 else 0
        
        # Obstacle proximity analysis
        obstacle_violations = 0
        min_distances = []
        closest_approach = float('inf')
        
        for obs_pos, obs_size in obstacles:
            min_dist_to_obs = float('inf')
            for point in trajectory:
                distance_to_obstacle = np.linalg.norm(point - obs_pos)
                min_dist_to_obs = min(min_dist_to_obs, distance_to_obstacle)
                closest_approach = min(closest_approach, distance_to_obstacle)
                if distance_to_obstacle < obs_size + 0.3:  # Including drone radius
                    obstacle_violations += 1
                    break
            min_distances.append(min_dist_to_obs)
        
        # Final distance to goal
        final_distance = np.linalg.norm(trajectory[-1] - end_waypoint)
        
        # Create comprehensive statistics text
        stats_text = f"""3D Trajectory Analysis:
═══════════════════════════
Path Metrics:
• Total Distance: {total_distance:.2f}m
• Direct Distance: {direct_distance:.2f}m
• Path Efficiency: {efficiency:.1%}
• Steps: {len(trajectory)}

Altitude Profile:
• Min Altitude: {min_altitude:.2f}m
• Max Altitude: {max_altitude:.2f}m
• Avg Altitude: {avg_altitude:.2f}m

Flight Dynamics:
• Avg Velocity: {avg_velocity:.3f}m/step
• Max Velocity: {max_velocity:.3f}m/step

Navigation:
• Final Distance to Goal: {final_distance:.2f}m
• Obstacles: {len(obstacles)}
• Violations: {obstacle_violations}"""
        
        if obstacles and min_distances:
            stats_text += f"\n• Closest Approach: {min(min_distances):.2f}m"
        
        # Color code the text box
        box_color = "lightcoral" if obstacle_violations > 0 else "lightgreen" if final_distance < 0.3 else "lightyellow"
        
        # Position text box in 3D space
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                 verticalalignment='top', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=box_color, alpha=0.95),
                 family='monospace')
    
    # Enhanced 3D plot formatting
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Z Position (m)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Set equal aspect ratio and limits
    max_range = workspace_bounds
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, 3])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Enhanced title
    title = f'3D Drone Trajectory - Level {difficulty_level}, Episode {episode_idx + 1}\n'
    level_names = {0: "Beginner (No obstacles)", 1: "Easy (1 obstacle)", 
                  2: "Medium (3 obstacles)", 3: "Hard (5 obstacles)"}
    title += f'{level_names.get(difficulty_level, f"Level {difficulty_level}")}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set optimal viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    
    if save_dir:
        # Save multiple views
        plt.savefig(f"{save_dir}/trajectory_3d_level_{difficulty_level}_episode_{episode_idx+1}_view1.png", 
                   dpi=300, bbox_inches='tight')
        
        # Different viewing angle
        ax.view_init(elev=60, azim=0)  # Top-down view
        plt.savefig(f"{save_dir}/trajectory_3d_level_{difficulty_level}_episode_{episode_idx+1}_topdown.png", 
                   dpi=300, bbox_inches='tight')
        
        # Side view
        ax.view_init(elev=0, azim=0)  # Side view
        plt.savefig(f"{save_dir}/trajectory_3d_level_{difficulty_level}_episode_{episode_idx+1}_side.png", 
                   dpi=300, bbox_inches='tight')
        
        # Reset to original view
        ax.view_init(elev=25, azim=45)
        
        print(f"🎯 3D trajectory plots saved (multiple views)")
    
    plt.show()


def plot_altitude_profile(stats, save_dir=None, episode_idx=0, difficulty_level=0):
    """Create detailed altitude profile visualization."""
    
    if not stats['trajectories'] or episode_idx >= len(stats['trajectories']):
        print("⚠️ No trajectory data available for altitude profile")
        return
    
    trajectory = np.array(stats['trajectories'][episode_idx])
    obstacles = stats['obstacles'][episode_idx] if stats['obstacles'] else []
    
    if len(trajectory) == 0:
        print("⚠️ Empty trajectory data")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    time_steps = np.arange(len(trajectory))
    
    # 1. Altitude over time
    ax1.plot(time_steps, trajectory[:, 2], 'b-', linewidth=3, alpha=0.8, label='Altitude')
    ax1.fill_between(time_steps, 0, trajectory[:, 2], alpha=0.3, color='skyblue')
    
    # Add obstacle altitude bands
    for i, (obs_pos, obs_size) in enumerate(obstacles):
        obs_bottom = obs_pos[2] - obs_size
        obs_top = obs_pos[2] + obs_size
        ax1.axhspan(obs_bottom, obs_top, alpha=0.3, color='red', 
                   label=f'Obstacle {i+1} Zone' if i == 0 else "")
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title(f'Altitude Profile - Level {difficulty_level}, Episode {episode_idx + 1}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 2D position over time
    ax2.plot(time_steps, trajectory[:, 0], 'r-', linewidth=2, label='X Position', alpha=0.8)
    ax2.plot(time_steps, trajectory[:, 1], 'g-', linewidth=2, label='Y Position', alpha=0.8)
    
    # Add waypoint lines
    start_waypoint = np.array([-1.0, -3.0, 1.0])
    end_waypoints_by_level = [
        np.array([0.0, 1.5, 1.2]),  # Level 0
        np.array([0.0, 2.0, 1.3]),  # Level 1
        np.array([0.0, 2.8, 1.7]),  # Level 2
        np.array([0.0, 3.5, 2.0])   # Level 3
    ]
    end_waypoint = end_waypoints_by_level[difficulty_level]
    
    ax2.axhline(y=start_waypoint[0], color='red', linestyle='--', alpha=0.5, label='Start X')
    ax2.axhline(y=start_waypoint[1], color='green', linestyle='--', alpha=0.5, label='Start Y')
    ax2.axhline(y=end_waypoint[0], color='red', linestyle=':', alpha=0.8, label='Goal X')
    ax2.axhline(y=end_waypoint[1], color='green', linestyle=':', alpha=0.8, label='Goal Y')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('X-Y Position Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Distance to goal over time
    distances_to_goal = []
    for point in trajectory:
        dist = np.linalg.norm(point - end_waypoint)
        distances_to_goal.append(dist)
    
    ax3.plot(time_steps, distances_to_goal, 'purple', linewidth=3, alpha=0.8)
    ax3.fill_between(time_steps, 0, distances_to_goal, alpha=0.3, color='purple')
    
    # Add success threshold
    ax3.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Success Threshold')
    
    # Mark final distance
    final_distance = distances_to_goal[-1]
    ax3.plot(len(trajectory)-1, final_distance, 'ro', markersize=10, 
            label=f'Final: {final_distance:.2f}m')
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Distance to Goal (m)')
    ax3.set_title('Distance to Goal Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/altitude_profile_level_{difficulty_level}_episode_{episode_idx+1}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"📈 Altitude profile saved")
    
    plt.show()


def main():
    """Main evaluation function - ENHANCED WITH 3D VISUALIZATION."""
    
    # Configuration - UPDATE THESE PATHS
    model_dir = "models/progressive_drone_nav_20250701_142313"  # Update with your actual timestamp
    model_path = f"{model_dir}/final_model.zip"  # or best_model.zip if you prefer
    normalize_path = f"{model_dir}/vec_normalize.pkl"
    n_episodes_per_level = 5  # Episodes to run per difficulty level
    
    # Create evaluation directory
    eval_dir = f"evaluation_progressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(eval_dir, exist_ok=True)
    
    print("🚁 Progressive Drone Navigation Model Evaluation - ENHANCED WITH 3D VISUALIZATION")
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
        for level, stats in all_stats.items():
            if stats['trajectories']:
                # Find episode with highest reward for this level
                best_episode_idx = np.argmax(stats['episode_rewards'])
                
                print(f"\n🎯 Creating visualizations for Level {level}, Episode {best_episode_idx + 1}...")
                
                # Create 2D plot
                plot_trajectory_2d(stats, save_dir=eval_dir, 
                                episode_idx=best_episode_idx, difficulty_level=level)
                
                # Create 3D plot (NEW!)
                plot_trajectory_3d(stats, save_dir=eval_dir, 
                                 episode_idx=best_episode_idx, difficulty_level=level)
                
                # Create altitude profile (NEW!)
                plot_altitude_profile(stats, save_dir=eval_dir,
                                    episode_idx=best_episode_idx, difficulty_level=level)
        
        # Print comprehensive summary
        print(f"\n✅ Evaluation complete! Results saved to: {eval_dir}")
        print(f"\n{'='*60}")
        print("📈 COMPREHENSIVE SUMMARY")
        print(f"{'='*60}")
        
        for level in sorted(all_stats.keys()):
            stats = all_stats[level]
            level_names = {0: "Beginner (0 obstacles)", 1: "Easy (1 obstacle)", 
                          2: "Medium (3 obstacles)", 3: "Hard (5 obstacles)"}
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
        
        # Check for collision detection issues
        collision_rates = [all_stats[level]['collision_rate'] for level in [1, 2, 3]]
        if all(rate == 0.0 for rate in collision_rates):
            print(f"\n⚠️ WARNING: Zero collision rates detected across obstacle levels!")
            print("   This suggests collision detection may not be working properly.")
        
        if hardest_level_success > 0.8:
            print("   🌟 EXCELLENT: Agent masters all difficulty levels!")
        elif hardest_level_success > 0.6:
            print("   ✅ GOOD: Agent handles most scenarios well")
        elif hardest_level_success > 0.4:
            print("   ⚠️ FAIR: Agent struggles with hardest scenarios")
        else:
            print("   ❌ NEEDS IMPROVEMENT: Agent fails on difficult scenarios")
        
        # Enhanced visualization summary
        print(f"\n🎨 VISUALIZATION SUMMARY:")
        print(f"   📊 Difficulty comparison plots: ✅")
        print(f"   🗺️ 2D trajectory plots: ✅ (per level)")
        print(f"   🎯 3D trajectory plots: ✅ (multiple views per level)")
        print(f"   📈 Altitude profiles: ✅ (per level)")
        print(f"   📁 All plots saved to: {eval_dir}")
        
        # Detailed collision analysis
        print(f"\n🔍 COLLISION ANALYSIS:")
        for level in sorted(all_stats.keys()):
            if level > 0:  # Only check levels with obstacles
                stats = all_stats[level]
                print(f"   Level {level}: {stats['collision_rate']:.1%} collision rate")
                if stats['collision_rate'] == 0.0:
                    print("     ⚠️ Zero collisions may indicate detection issues")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        eval_env.close()


if __name__ == "__main__":
    main()