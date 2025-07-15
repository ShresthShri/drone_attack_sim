import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pybullet as p
import pybullet_data
from stable_baselines3 import SAC
import time
import os
from collections import defaultdict
import seaborn as sns

# Import your drone environment - handle different import paths
import sys
import os

# Add current directory and agents directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'agents'))

try:
    from agents.baseline_RL_new import AdvancedDroneEnv
except ImportError:
    try:
        from baseline_RL_new import AdvancedDroneEnv
    except ImportError:
        # If still can't import, provide helpful error message
        print("Error: Cannot import AdvancedDroneEnv. Make sure baseline_RL_new.py is in:")
        print("1. agents/baseline_RL_new.py, or")
        print("2. same directory as this script")
        sys.exit(1)

class DroneEvaluator:
    def __init__(self, model_path="advanced_drone_navigation_sac"):
        """Initialize the drone evaluator with a trained model"""
        self.model = SAC.load(model_path)
        self.env = AdvancedDroneEnv()
        self.trajectories = []
        self.evaluation_metrics = defaultdict(list)
        
    def run_single_episode(self, episode_id=None, save_trajectory=True, render_live=False):
        """Run a single episode and collect trajectory data"""
        obs, info = self.env.reset()
        
        # Store episode info
        episode_data = {
            'episode_id': episode_id,
            'start_pos': self.env.start_pos.copy(),
            'goal_pos': self.env.goal_pos.copy(),
            'initial_distance': np.linalg.norm(self.env.goal_pos - self.env.start_pos),
            'trajectory': [],
            'actions': [],
            'rewards': [],
            'obstacles': self.env.obstacle_params.copy(),
            'timestamps': []
        }
        
        total_reward = 0
        step_count = 0
        start_time = time.time()
        
        # Set up live plotting if requested
        if render_live:
            fig, ax = self._setup_live_plot(episode_data)
            plt.ion()
            plt.show()
        
        while True:
            # Get drone position
            drone_pos, _ = p.getBasePositionAndOrientation(self.env.drone)
            episode_data['trajectory'].append(drone_pos)
            episode_data['timestamps'].append(time.time() - start_time)
            
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            episode_data['actions'].append(action)
            
            # Take step
            obs, reward, done, truncated, step_info = self.env.step(action)
            episode_data['rewards'].append(reward)
            total_reward += reward
            step_count += 1
            
            # Update live plot
            if render_live and step_count % 5 == 0:  # Update every 5 steps
                self._update_live_plot(ax, episode_data, step_count)
                plt.pause(0.01)
            
            if done or truncated:
                break
        
        # Calculate final metrics
        final_pos = episode_data['trajectory'][-1]
        final_distance = np.linalg.norm(np.array(final_pos) - episode_data['goal_pos'])
        path_length = self._calculate_path_length(episode_data['trajectory'])
        
        episode_data.update({
            'final_pos': final_pos,
            'final_distance': final_distance,
            'total_reward': total_reward,
            'steps': step_count,
            'success': step_info['success'],
            'collision': step_info['collision'],
            'progress_ratio': step_info['progress_ratio'],
            'path_length': path_length,
            'path_efficiency': self._calculate_path_efficiency(episode_data['initial_distance'], path_length),
            'obstacle_clearance': self._calculate_min_obstacle_clearance(episode_data)
        })
        
        if save_trajectory:
            self.trajectories.append(episode_data)
            
        if render_live:
            plt.ioff()
            
        return episode_data
    
    def _setup_live_plot(self, episode_data):
        """Set up the live plotting window"""
        fig, ax = plt.subplots(figsize=(12, 8))
        self._plot_environment(ax, episode_data)
        ax.set_title(f"Live Drone Navigation - Episode {episode_data['episode_id']}")
        return fig, ax
    
    def _update_live_plot(self, ax, episode_data, current_step):
        """Update the live plot with current drone position"""
        # Clear previous drone position
        for artist in ax.findobj(match=plt.Circle):
            if artist.get_facecolor()[0] == 1.0:  # Red drone
                artist.remove()
        
        # Plot current trajectory
        if len(episode_data['trajectory']) > 1:
            traj = np.array(episode_data['trajectory'])
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.7, linewidth=2, label='Trajectory')
        
        # Plot current drone position
        current_pos = episode_data['trajectory'][-1]
        drone_circle = plt.Circle((current_pos[0], current_pos[1]), 0.2, 
                                 color='red', alpha=0.8, zorder=10)
        ax.add_patch(drone_circle)
        
        plt.draw()
    
    def _plot_environment(self, ax, episode_data):
        """Plot the environment layout"""
        # Plot obstacles
        for (center, half_extents) in episode_data['obstacles']:
            rect = patches.Rectangle(
                (center[0] - half_extents[0], center[1] - half_extents[1]),
                2 * half_extents[0], 2 * half_extents[1],
                linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7
            )
            ax.add_patch(rect)
        
        # Plot start position
        start_circle = plt.Circle((episode_data['start_pos'][0], episode_data['start_pos'][1]), 
                                 0.3, color='green', alpha=0.8, label='Start')
        ax.add_patch(start_circle)
        
        # Plot goal position
        goal_circle = plt.Circle((episode_data['goal_pos'][0], episode_data['goal_pos'][1]), 
                                0.5, color='gold', alpha=0.8, label='Goal')
        ax.add_patch(goal_circle)
        
        # Set axis properties
        ax.set_xlim(-5, 20)
        ax.set_ylim(-7, 7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def plot_trajectory(self, episode_data, save_path=None):
        """Create a detailed trajectory plot for a single episode"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplots with different arrangements
        ax1 = plt.subplot(2, 3, 1)  # 2D X-Y view
        ax2 = plt.subplot(2, 3, 2)  # 2D X-Z view  
        ax3 = plt.subplot(2, 3, 3)  # 2D Y-Z view
        ax4 = plt.subplot(2, 3, (4, 5), projection='3d')  # 3D view
        ax5 = plt.subplot(2, 3, 6)  # Rewards over time
        
        trajectory = np.array(episode_data['trajectory'])
        final_pos = episode_data['final_pos']
        
        # 1. X-Y View (Top Down)
        self._plot_environment_2d(ax1, episode_data, 'xy')
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax1.scatter(final_pos[0], final_pos[1], color='red', s=100, zorder=10, label='Final')
        ax1.set_title('Top View (X-Y)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. X-Z View (Side View)
        self._plot_environment_2d(ax2, episode_data, 'xz')
        ax2.plot(trajectory[:, 0], trajectory[:, 2], 'b-', linewidth=2, label='Trajectory')
        ax2.scatter(episode_data['start_pos'][0], episode_data['start_pos'][2], 
                   color='green', s=100, label='Start', zorder=5)
        ax2.scatter(episode_data['goal_pos'][0], episode_data['goal_pos'][2], 
                   color='gold', s=150, marker='*', label='Goal', zorder=5)
        ax2.scatter(final_pos[0], final_pos[2], color='red', s=100, label='Final', zorder=5)
        ax2.set_title('Side View (X-Z)')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Y-Z View (Front View)
        self._plot_environment_2d(ax3, episode_data, 'yz')
        ax3.plot(trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2, label='Trajectory')
        ax3.scatter(episode_data['start_pos'][1], episode_data['start_pos'][2], 
                   color='green', s=100, label='Start', zorder=5)
        ax3.scatter(episode_data['goal_pos'][1], episode_data['goal_pos'][2], 
                   color='gold', s=150, marker='*', label='Goal', zorder=5)
        ax3.scatter(final_pos[1], final_pos[2], color='red', s=100, label='Final', zorder=5)
        ax3.set_title('Front View (Y-Z)')
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. TRUE 3D View
        self._plot_environment_3d(ax4, episode_data)
        ax4.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=3, label='Trajectory')
        ax4.scatter(episode_data['start_pos'][0], episode_data['start_pos'][1], episode_data['start_pos'][2], 
                   color='green', s=200, label='Start', alpha=0.8)
        ax4.scatter(episode_data['goal_pos'][0], episode_data['goal_pos'][1], episode_data['goal_pos'][2], 
                   color='gold', s=300, marker='*', label='Goal', alpha=0.8)
        ax4.scatter(final_pos[0], final_pos[1], final_pos[2], 
                   color='red', s=200, label='Final', alpha=0.8)
        
        ax4.set_title('3D Trajectory')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.legend()
        
        # 5. Rewards over time
        ax5.plot(episode_data['rewards'], 'g-', linewidth=1)
        ax5.set_title('Rewards Over Time')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Reward')
        ax5.grid(True, alpha=0.3)
        
        # Overall title
        status = "SUCCESS" if episode_data['success'] else ("COLLISION" if episode_data['collision'] else "TIMEOUT")
        fig.suptitle(f"Episode {episode_data.get('episode_id', 'N/A')} - {status}\n"
                    f"Progress: {episode_data['progress_ratio']*100:.1f}% | "
                    f"Efficiency: {episode_data['path_efficiency']:.2f} | "
                    f"Final Distance: {episode_data['final_distance']:.2f}m | "
                    f"Steps: {episode_data['steps']}", fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def _plot_environment_2d(self, ax, episode_data, view='xy'):
        """Plot 2D projections of the environment"""
        # Choose dimensions based on view
        if view == 'xy':
            dim1, dim2 = 0, 1
            ax.set_xlim(-5, 20)
            ax.set_ylim(-7, 7)
        elif view == 'xz':
            dim1, dim2 = 0, 2
            ax.set_xlim(-5, 20)
            ax.set_ylim(0, 6)
        elif view == 'yz':
            dim1, dim2 = 1, 2
            ax.set_xlim(-7, 7)
            ax.set_ylim(0, 6)
        
        # Plot obstacles
        for (center, half_extents) in episode_data['obstacles']:
            rect = patches.Rectangle(
                (center[dim1] - half_extents[dim1], center[dim2] - half_extents[dim2]),
                2 * half_extents[dim1], 2 * half_extents[dim2],
                linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7
            )
            ax.add_patch(rect)
        
        # Plot start and goal
        start_pos = episode_data['start_pos']
        goal_pos = episode_data['goal_pos']
        
        ax.scatter(start_pos[dim1], start_pos[dim2], color='green', s=100, label='Start', zorder=5)
        ax.scatter(goal_pos[dim1], goal_pos[dim2], color='gold', s=150, marker='*', label='Goal', zorder=5)
        
        ax.set_aspect('equal')
    
    def _plot_environment_3d(self, ax, episode_data):
        """Plot true 3D environment"""
        # Plot obstacles as 3D boxes
        for (center, half_extents) in episode_data['obstacles']:
            # Create 3D box vertices
            x_range = [center[0] - half_extents[0], center[0] + half_extents[0]]
            y_range = [center[1] - half_extents[1], center[1] + half_extents[1]]
            z_range = [center[2] - half_extents[2], center[2] + half_extents[2]]
            
            # Create box faces
            vertices = []
            for x in x_range:
                for y in y_range:
                    for z in z_range:
                        vertices.append([x, y, z])
            
            # Define the 12 edges of a cube
            edges = [
                [0, 1], [1, 3], [3, 2], [2, 0],  # bottom face
                [4, 5], [5, 7], [7, 6], [6, 4],  # top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
            ]
            
            # Draw box edges
            for edge in edges:
                points = np.array([vertices[edge[0]], vertices[edge[1]]])
                ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', alpha=0.6)
        
        # Set 3D axis properties
        ax.set_xlim(-5, 20)
        ax.set_ylim(-7, 7)
        ax.set_zlim(0, 6)
        ax.set_box_aspect([25, 14, 6])  # Aspect ratio for better visualization
    
    def create_3d_animation(self, episode_data, save_path=None):
        """Create an animated 3D visualization of the drone trajectory"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot environment
        self._plot_environment_3d(ax, episode_data)
        
        trajectory = np.array(episode_data['trajectory'])
        
        # Plot static elements
        ax.scatter(episode_data['start_pos'][0], episode_data['start_pos'][1], episode_data['start_pos'][2], 
                  color='green', s=200, label='Start', alpha=0.8)
        ax.scatter(episode_data['goal_pos'][0], episode_data['goal_pos'][1], episode_data['goal_pos'][2], 
                  color='gold', s=300, marker='*', label='Goal', alpha=0.8)
        
        # Initialize trajectory line and drone position
        line, = ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        drone_point, = ax.plot([], [], [], 'ro', markersize=8, label='Drone')
        
        ax.set_title(f'3D Drone Navigation Animation - Episode {episode_data.get("episode_id", "N/A")}')
        ax.legend()
        
        def animate(frame):
            # Update trajectory line up to current frame
            if frame > 0:
                line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
                line.set_3d_properties(trajectory[:frame, 2])
            
            # Update drone position
            if frame < len(trajectory):
                drone_point.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
                drone_point.set_3d_properties([trajectory[frame, 2]])
            
            return line, drone_point
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(trajectory), interval=50, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            print(f"3D animation saved to {save_path}")
        
        plt.show()
        return anim

    def create_comprehensive_3d_visualization(self, episode_data, save_path=None):
        """Create a comprehensive 3D visualization showing trajectory, obstacles, and navigation strategy"""
        fig = plt.figure(figsize=(16, 12))
        
        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot environment
        self._plot_environment_3d(ax, episode_data)
        
        trajectory = np.array(episode_data['trajectory'])
        
        # Plot trajectory with color gradient (blue to red over time)
        n_points = len(trajectory)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        
        for i in range(n_points - 1):
            ax.plot3D(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], 
                     color=colors[i], linewidth=2, alpha=0.7)
        
        # Plot key points
        ax.scatter(episode_data['start_pos'][0], episode_data['start_pos'][1], episode_data['start_pos'][2], 
                  color='green', s=300, label='Start', alpha=1.0, edgecolor='black')
        ax.scatter(episode_data['goal_pos'][0], episode_data['goal_pos'][1], episode_data['goal_pos'][2], 
                  color='gold', s=400, marker='*', label='Goal', alpha=1.0, edgecolor='black')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  color='red', s=300, label='Final', alpha=1.0, edgecolor='black')
        
        # Add altitude analysis
        min_altitude = np.min(trajectory[:, 2])
        max_altitude = np.max(trajectory[:, 2])
        avg_altitude = np.mean(trajectory[:, 2])
        
        # Plot altitude reference planes
        xx, yy = np.meshgrid(np.linspace(-5, 20, 10), np.linspace(-7, 7, 10))
        ax.plot_surface(xx, yy, np.full_like(xx, avg_altitude), alpha=0.1, color='blue', label='Avg Altitude')
        
        # Customize plot
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_xlim(-5, 20)
        ax.set_ylim(-7, 7)
        ax.set_zlim(0, 6)
        
        # Add text annotations
        status = "SUCCESS" if episode_data['success'] else ("COLLISION" if episode_data['collision'] else "TIMEOUT")
        ax.text2D(0.05, 0.95, f"Episode {episode_data.get('episode_id', 'N/A')} - {status}", 
                 transform=ax.transAxes, fontsize=12, weight='bold')
        ax.text2D(0.05, 0.90, f"Progress: {episode_data['progress_ratio']*100:.1f}%", 
                 transform=ax.transAxes, fontsize=10)
        ax.text2D(0.05, 0.85, f"Final Distance: {episode_data['final_distance']:.2f}m", 
                 transform=ax.transAxes, fontsize=10)
        ax.text2D(0.05, 0.80, f"Path Length: {episode_data['path_length']:.2f}m", 
                 transform=ax.transAxes, fontsize=10)
        ax.text2D(0.05, 0.75, f"Altitude: {min_altitude:.1f}m - {max_altitude:.1f}m (avg: {avg_altitude:.1f}m)", 
                 transform=ax.transAxes, fontsize=10)
        
        ax.legend(loc='upper right')
        ax.set_title('3D Drone Navigation Analysis', fontsize=14, weight='bold')
        
        # Set viewing angle for better perspective
        ax.view_init(elev=20, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D visualization saved to {save_path}")
            
        plt.show()
        return fig

    def run_evaluation(self, num_episodes=10, save_plots=True, render_live=False, create_3d_viz=True):
        """Run comprehensive evaluation"""
        print(f"Running evaluation with {num_episodes} episodes...")
        
        results = []
        
        for i in range(num_episodes):
            print(f"Episode {i+1}/{num_episodes}")
            
            episode_data = self.run_single_episode(
                episode_id=i+1, 
                save_trajectory=True,
                render_live=render_live and i < 3  # Only render first 3 episodes live
            )
            
            results.append(episode_data)
            
            # Save individual trajectory plots
            if save_plots:
                os.makedirs("evaluation_plots", exist_ok=True)
                fig = self.plot_trajectory(episode_data, 
                                         f"evaluation_plots/episode_{i+1}_trajectory.png")
                plt.close(fig)
                
                # Create 3D visualization for first few episodes
                if create_3d_viz and i < 3:
                    self.create_comprehensive_3d_visualization(
                        episode_data, 
                        f"evaluation_plots/episode_{i+1}_3d_viz.png"
                    )
                    
                    # Create animation for first episode
                    if i == 0:
                        self.create_3d_animation(
                            episode_data,
                            f"evaluation_plots/episode_{i+1}_3d_animation.gif"
                        )
            
            # Print episode summary
            status = "SUCCESS" if episode_data['success'] else ("COLLISION" if episode_data['collision'] else "TIMEOUT")
            print(f"  {status} | Distance: {episode_data['final_distance']:.2f}m | "
                  f"Progress: {episode_data['progress_ratio']*100:.1f}% | "
                  f"Steps: {episode_data['steps']}")
        
        # Generate summary statistics
        self._generate_evaluation_report(results, save_plots)
        
        return results
    
    def _generate_evaluation_report(self, results, save_plots=True):
        """Generate comprehensive evaluation report"""
        # Calculate metrics
        success_rate = np.mean([r['success'] for r in results])
        collision_rate = np.mean([r['collision'] for r in results])
        avg_progress = np.mean([r['progress_ratio'] for r in results])
        avg_efficiency = np.mean([r['path_efficiency'] for r in results])
        avg_final_distance = np.mean([r['final_distance'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        avg_reward = np.mean([r['total_reward'] for r in results])
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Episodes: {len(results)}")
        print(f"Success Rate: {success_rate*100:.1f}%")
        print(f"Collision Rate: {collision_rate*100:.1f}%")
        print(f"Average Progress: {avg_progress*100:.1f}%")
        print(f"Average Final Distance: {avg_final_distance:.2f}m")
        print(f"Average Path Efficiency: {avg_efficiency:.2f}")
        print(f"Average Steps: {avg_steps:.0f}")
        print(f"Average Reward: {avg_reward:.0f}")
        
        if save_plots:
            self._create_summary_plots(results)
    
    def _create_summary_plots(self, results):
        """Create summary visualization plots"""
        os.makedirs("evaluation_plots", exist_ok=True)
        
        # 1. Success/Failure distribution
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Success rate pie chart
        success_count = sum([r['success'] for r in results])
        collision_count = sum([r['collision'] for r in results])
        timeout_count = len(results) - success_count - collision_count
        
        ax1.pie([success_count, collision_count, timeout_count], 
               labels=['Success', 'Collision', 'Timeout'],
               colors=['green', 'red', 'orange'],
               autopct='%1.1f%%')
        ax1.set_title('Episode Outcomes')
        
        # Progress distribution
        progress_values = [r['progress_ratio'] * 100 for r in results]
        ax2.hist(progress_values, bins=10, alpha=0.7, color='blue')
        ax2.set_xlabel('Progress (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Progress Distribution')
        ax2.axvline(np.mean(progress_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(progress_values):.1f}%')
        ax2.legend()
        
        # Final distance vs Initial distance
        initial_distances = [r['initial_distance'] for r in results]
        final_distances = [r['final_distance'] for r in results]
        colors = ['green' if r['success'] else ('red' if r['collision'] else 'orange') 
                 for r in results]
        
        ax3.scatter(initial_distances, final_distances, c=colors, alpha=0.7)
        ax3.plot([0, max(initial_distances)], [0, max(initial_distances)], 'k--', alpha=0.5)
        ax3.set_xlabel('Initial Distance (m)')
        ax3.set_ylabel('Final Distance (m)')
        ax3.set_title('Final vs Initial Distance')
        ax3.grid(True, alpha=0.3)
        
        # Path efficiency distribution
        efficiencies = [r['path_efficiency'] for r in results]
        ax4.hist(efficiencies, bins=10, alpha=0.7, color='purple')
        ax4.set_xlabel('Path Efficiency')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Path Efficiency Distribution')
        ax4.axvline(np.mean(efficiencies), color='red', linestyle='--',
                   label=f'Mean: {np.mean(efficiencies):.2f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('evaluation_plots/summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. All trajectories overlay plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot environment (use first episode's obstacle layout)
        self._plot_environment(ax, results[0])
        
        # Plot all trajectories
        for i, result in enumerate(results):
            trajectory = np.array(result['trajectory'])
            color = 'green' if result['success'] else ('red' if result['collision'] else 'orange')
            alpha = 0.7 if result['success'] else 0.4
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, 
                   alpha=alpha, linewidth=1.5, label=f"Ep {i+1}")
        
        ax.set_title(f'All Trajectories Overlay (n={len(results)})')
        plt.savefig('evaluation_plots/all_trajectories_overlay.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _calculate_path_length(self, trajectory):
        """Calculate total path length"""
        if len(trajectory) < 2:
            return 0
        trajectory = np.array(trajectory)
        distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
        return np.sum(distances)
    
    def _calculate_path_efficiency(self, initial_distance, path_length):
        """Calculate path efficiency (straight line distance / actual path length)"""
        if path_length == 0:
            return 0
        return initial_distance / path_length
    
    def _calculate_min_obstacle_clearance(self, episode_data):
        """Calculate minimum clearance to obstacles during flight"""
        min_clearance = float('inf')
        trajectory = np.array(episode_data['trajectory'])
        
        for pos in trajectory:
            for (center, half_extents) in episode_data['obstacles']:
                # Calculate distance to obstacle surface
                closest_point = np.clip(pos, 
                                      np.array(center) - np.array(half_extents),
                                      np.array(center) + np.array(half_extents))
                distance = np.linalg.norm(pos - closest_point)
                min_clearance = min(min_clearance, distance)
        
        return min_clearance if min_clearance != float('inf') else 0

def main():
    """Main evaluation function"""
    print("Starting Drone Navigation Evaluation...")
    
    # Initialize evaluator
    evaluator = DroneEvaluator("advanced_drone_navigation_sac")
    
    # Run evaluation
    results = evaluator.run_evaluation(
        num_episodes=10,      # Number of test episodes
        save_plots=True,      # Save individual trajectory plots
        render_live=False,    # Show live navigation for first 3 episodes
        create_3d_viz=True    # Create 3D visualizations and animations
    )
    
    print("\nEvaluation complete! Check 'evaluation_plots' folder for detailed visualizations.")

if __name__ == "__main__":
    main()