import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pybullet as p
from stable_baselines3 import SAC
from agents.baseline_RL_new import AdvancedDroneEnv  # Import your environment

def evaluate_drone_with_trajectory(model_path="advanced_drone_navigation_sac", num_episodes=5):
    """Evaluate the trained drone model and plot 2D trajectories"""
    
    # Load model
    model = SAC.load(model_path)
    
    # Create test environment
    env = AdvancedDroneEnv()
    
    success_count = 0
    all_results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        # Store trajectory data
        trajectory = []
        rewards_over_time = []
        
        print(f"\nEpisode {episode + 1}:")
        print(f"Start: {env.start_pos}")
        print(f"Goal: {env.goal_pos}")
        print(f"Distance: {np.linalg.norm(env.goal_pos - env.start_pos):.2f}")
        
        while True:
            # Get drone position for trajectory
            drone_pos, _ = p.getBasePositionAndOrientation(env.drone)
            trajectory.append([drone_pos[0], drone_pos[1], drone_pos[2]])  # Store X, Y, Z coordinates
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            rewards_over_time.append(total_reward)
            steps += 1
            
            if done or truncated:
                break
        
        # Convert trajectory to numpy array
        trajectory = np.array(trajectory)
        
        print(f"Steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final distance: {info['distance_to_goal']:.2f}")
        print(f"Progress: {info['progress_ratio']*100:.1f}%")
        
        # Determine episode outcome
        if info['success']:
            print("SUCCESS!")
            success_count += 1
            outcome = "SUCCESS"
        elif info['collision']:
            print("COLLISION")
            outcome = "COLLISION"
        else:
            print("TIMEOUT")
            outcome = "TIMEOUT"
        
        # Store results for plotting
        result = {
            'episode': episode + 1,
            'start_pos': env.start_pos,
            'goal_pos': env.goal_pos,
            'trajectory': trajectory,
            'rewards_over_time': rewards_over_time,
            'total_reward': total_reward,
            'steps': steps,
            'final_distance': info['distance_to_goal'],
            'progress': info['progress_ratio'],
            'outcome': outcome,
            'obstacle_params': env.obstacle_params
        }
        all_results.append(result)
        
        # Plot trajectory for this episode
        plot_episode_trajectory(result)
    
    print(f"\nOverall success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    
    # Plot summary
    plot_summary(all_results)
    
    env.close()
    return all_results

def plot_episode_trajectory(result):
    """Plot 2D and 3D trajectory for a single episode"""
    fig = plt.figure(figsize=(20, 8))
    
    # Create subplots: 2D trajectory, 3D trajectory, and rewards over time
    ax1 = plt.subplot(1, 3, 1)  # 2D plot
    ax2 = plt.subplot(1, 3, 2, projection='3d')  # 3D plot
    ax3 = plt.subplot(1, 3, 3)  # Rewards plot
    
    # Left plot: 2D trajectory (top-down view)
    ax1.set_title(f"Episode {result['episode']} - {result['outcome']}\n"
                  f"Progress: {result['progress']*100:.1f}% | Reward: {result['total_reward']:.0f}")
    
    # Plot obstacles in 2D
    for (center, half_extents) in result['obstacle_params']:
        # Create rectangle for obstacle (top-down view)
        rect = patches.Rectangle(
            (center[0] - half_extents[0], center[1] - half_extents[1]),
            2 * half_extents[0], 2 * half_extents[1],
            linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.7
        )
        ax1.add_patch(rect)
    
    # Plot 2D trajectory
    trajectory = result['trajectory']
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    
    # Plot start and goal in 2D
    start = result['start_pos']
    goal = result['goal_pos']
    ax1.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax1.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    # Mark final position in 2D
    final_pos = trajectory[-1]
    ax1.plot(final_pos[0], final_pos[1], 'bx', markersize=12, markeredgewidth=3, label='Final')
    
    # Add distance annotation
    ax1.annotate(f'Final\n{result["final_distance"]:.2f}', 
                xy=final_pos[:2], xytext=(final_pos[0]+1, final_pos[1]+1),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, ha='center')
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Set reasonable axis limits for 2D - include all obstacles
    all_x = np.concatenate([trajectory[:, 0], [start[0], goal[0]]])
    all_y = np.concatenate([trajectory[:, 1], [start[1], goal[1]]])
    
    # Add obstacle positions to ensure they're all visible
    for (center, half_extents) in result['obstacle_params']:
        obstacle_x_range = [center[0] - half_extents[0], center[0] + half_extents[0]]
        obstacle_y_range = [center[1] - half_extents[1], center[1] + half_extents[1]]
        all_x = np.concatenate([all_x, obstacle_x_range])
        all_y = np.concatenate([all_y, obstacle_y_range])
    
    margin = 2
    ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax1.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    # Middle plot: 3D trajectory
    ax2.set_title('3D Trajectory View')
    
    # Plot obstacles in 3D as boxes
    for (center, half_extents) in result['obstacle_params']:
        # Create 3D box for obstacle
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Define the vertices of a box
        x_range = [center[0] - half_extents[0], center[0] + half_extents[0]]
        y_range = [center[1] - half_extents[1], center[1] + half_extents[1]]
        z_range = [center[2] - half_extents[2], center[2] + half_extents[2]]
        
        # Create vertices for the box
        vertices = []
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    vertices.append([x, y, z])
        
        # Define the 12 edges of the box
        edges = [
            [vertices[0], vertices[1]], [vertices[1], vertices[3]], [vertices[3], vertices[2]], [vertices[2], vertices[0]],  # bottom
            [vertices[4], vertices[5]], [vertices[5], vertices[7]], [vertices[7], vertices[6]], [vertices[6], vertices[4]],  # top
            [vertices[0], vertices[4]], [vertices[1], vertices[5]], [vertices[2], vertices[6]], [vertices[3], vertices[7]]   # sides
        ]
        
        # Plot box edges
        for edge in edges:
            points = np.array(edge)
            ax2.plot3D(points[:, 0], points[:, 1], points[:, 2], 'gray', alpha=0.6)
    
    # Plot 3D trajectory
    ax2.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2, alpha=0.8, label='Trajectory')
    
    # Plot start and goal in 3D
    ax2.scatter(start[0], start[1], start[2], color='green', s=100, label='Start')
    ax2.scatter(goal[0], goal[1], goal[2], color='red', s=100, label='Goal')
    
    # Mark final position in 3D
    ax2.scatter(final_pos[0], final_pos[1], final_pos[2], color='blue', s=100, marker='x', label='Final')
    
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_zlabel('Z Position (Altitude)')
    ax2.legend()
    
    # Set 3D axis limits
    all_z = np.concatenate([trajectory[:, 2], [start[2], goal[2]]])
    ax2.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax2.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax2.set_zlim(max(0, min(all_z) - 1), max(all_z) + 1)
    
    # Right plot: Rewards over time
    ax3.set_title('Rewards Over Time')
    ax3.plot(result['rewards_over_time'], 'g-', linewidth=2)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Cumulative Reward')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'episode_{result["episode"]}_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_summary(all_results):
    """Plot summary statistics across all episodes"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    episodes = [r['episode'] for r in all_results]
    progress = [r['progress'] * 100 for r in all_results]
    rewards = [r['total_reward'] for r in all_results]
    final_distances = [r['final_distance'] for r in all_results]
    outcomes = [r['outcome'] for r in all_results]
    
    # Color mapping for outcomes
    colors = {'SUCCESS': 'green', 'TIMEOUT': 'orange', 'COLLISION': 'red'}
    episode_colors = [colors[outcome] for outcome in outcomes]
    
    # Plot 1: Progress per episode
    ax1.bar(episodes, progress, color=episode_colors, alpha=0.7)
    ax1.set_title('Progress per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Progress (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rewards per episode
    ax2.bar(episodes, rewards, color=episode_colors, alpha=0.7)
    ax2.set_title('Total Reward per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final distance per episode
    ax3.bar(episodes, final_distances, color=episode_colors, alpha=0.7)
    ax3.axhline(y=2.0, color='r', linestyle='--', label='Goal Threshold (2.0)')
    ax3.set_title('Final Distance to Goal per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Final Distance')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Outcome distribution
    outcome_counts = {outcome: outcomes.count(outcome) for outcome in set(outcomes)}
    ax4.pie(outcome_counts.values(), labels=outcome_counts.keys(), autopct='%1.1f%%',
            colors=[colors[outcome] for outcome in outcome_counts.keys()])
    ax4.set_title('Episode Outcomes')
    
    plt.tight_layout()
    plt.savefig('summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Average Progress: {np.mean(progress):.1f}%")
    print(f"Average Reward: {np.mean(rewards):.1f}")
    print(f"Average Final Distance: {np.mean(final_distances):.2f}")
    print(f"Success Rate: {outcomes.count('SUCCESS')}/{len(outcomes)} ({outcomes.count('SUCCESS')/len(outcomes)*100:.1f}%)")
    print(f"Timeout Rate: {outcomes.count('TIMEOUT')}/{len(outcomes)} ({outcomes.count('TIMEOUT')/len(outcomes)*100:.1f}%)")
    print(f"Collision Rate: {outcomes.count('COLLISION')}/{len(outcomes)} ({outcomes.count('COLLISION')/len(outcomes)*100:.1f}%)")

if __name__ == "__main__":
    # Run evaluation with trajectory plotting
    results = evaluate_drone_with_trajectory(num_episodes=5)