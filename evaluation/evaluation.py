import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.drone_navigation_env import DroneNavigationEnv

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')

class DroneEvaluator:
    """
    Lean and clear evaluation system for drone navigation with trajectory plotting.
    """
    
    def __init__(self, model_path, normalization_path=None, output_dir="./evaluation_results"):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to the trained SAC model
            normalization_path (str): Path to VecNormalize parameters
            output_dir (str): Directory to save evaluation results
        """
        self.model_path = model_path
        self.normalization_path = normalization_path
        self.output_dir = output_dir
        self.model = None
        self.env = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.evaluation_results = {
            'episodes': [],
            'trajectories': [],
            'statistics': {},
            'metadata': {
                'model_path': model_path,
                'evaluation_date': datetime.now().isoformat(),
                'normalization_used': normalization_path is not None
            }
        }
    
    def load_model_and_env(self):
        """Load the trained model and setup evaluation environment."""
        print("🔄 Loading model and environment...")
        
        # Create environment
        env = DroneNavigationEnv(gui=False, record=False)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # Load normalization if available
        if self.normalization_path and os.path.exists(self.normalization_path):
            print(f"📊 Loading normalization from: {self.normalization_path}")
            env = VecNormalize.load(self.normalization_path, env)
            env.training = False
            env.norm_reward = False
        
        # Load model
        print(f"🤖 Loading model from: {self.model_path}")
        model = SAC.load(self.model_path, env=env)
        
        self.model = model
        self.env = env
        print("✅ Model and environment loaded successfully!")
    
    def run_evaluation(self, n_episodes=10, max_steps=1000, deterministic=True):
        """
        Run evaluation episodes and collect trajectory data.
        
        Args:
            n_episodes (int): Number of episodes to evaluate
            max_steps (int): Maximum steps per episode
            deterministic (bool): Use deterministic policy
        """
        if self.model is None or self.env is None:
            self.load_model_and_env()
        
        print(f"🎯 Running evaluation for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            episode_data = self._run_single_episode(episode, max_steps, deterministic)
            self.evaluation_results['episodes'].append(episode_data)
            self.evaluation_results['trajectories'].append(episode_data['trajectory'])
            
            # Print episode summary
            success_status = "✅ SUCCESS" if episode_data['success'] else "❌ FAILED"
            print(f"Episode {episode+1:2d}: {success_status} | "
                  f"Reward: {episode_data['total_reward']:6.1f} | "
                  f"Steps: {episode_data['steps']:3d} | "
                  f"Final Distance: {episode_data['final_distance']:.3f}")
        
        # Calculate statistics
        self._calculate_statistics()
        print(f"\n📈 Evaluation completed! Results saved to: {self.output_dir}")
    
    def _run_single_episode(self, episode_idx, max_steps, deterministic):
        """Run a single evaluation episode and collect data."""
        obs = self.env.reset()
        
        # Initialize episode data
        episode_data = {
            'episode': episode_idx,
            'trajectory': {'x': [], 'y': [], 'z': []},
            'actions': [],
            'rewards': [],
            'distances': [],
            'timestamps': [],
            'target_position': None,
            'start_position': None,
            'success': False,
            'total_reward': 0.0,
            'steps': 0,
            'final_distance': float('inf')
        }
        
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=deterministic)
            
            # Execute action
            obs, reward, done, info = self.env.step(action)
            
            # Extract environment info
            env_info = info[0] if isinstance(info, list) else info
            current_pos = env_info.get('current_position', [0, 0, 0])
            target_pos = env_info.get('target_position', [0, 0, 0])
            distance = env_info.get('distance_to_target', 0)
            
            # Store trajectory data
            episode_data['trajectory']['x'].append(current_pos[0])
            episode_data['trajectory']['y'].append(current_pos[1])
            episode_data['trajectory']['z'].append(current_pos[2])
            episode_data['actions'].append(action[0].tolist() if hasattr(action[0], 'tolist') else action[0])
            episode_data['rewards'].append(reward[0] if isinstance(reward, np.ndarray) else reward)
            episode_data['distances'].append(distance)
            episode_data['timestamps'].append(step)
            
            # Store initial positions
            if step == 0:
                episode_data['start_position'] = current_pos.copy()
                episode_data['target_position'] = target_pos.copy()
            
            episode_data['total_reward'] += reward[0] if isinstance(reward, np.ndarray) else reward
            step += 1
        
        # Final episode statistics
        episode_data['steps'] = step
        episode_data['success'] = env_info.get('success', False)
        episode_data['final_distance'] = episode_data['distances'][-1] if episode_data['distances'] else float('inf')
        
        return episode_data
    
    def _calculate_statistics(self):
        """Calculate evaluation statistics."""
        episodes = self.evaluation_results['episodes']
        
        rewards = [ep['total_reward'] for ep in episodes]
        steps = [ep['steps'] for ep in episodes]
        distances = [ep['final_distance'] for ep in episodes]
        successes = [ep['success'] for ep in episodes]
        
        stats = {
            'n_episodes': len(episodes),
            'success_rate': np.mean(successes) * 100,
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_min': np.min(rewards),
            'reward_max': np.max(rewards),
            'steps_mean': np.mean(steps),
            'steps_std': np.std(steps),
            'steps_min': np.min(steps),
            'steps_max': np.max(steps),
            'final_distance_mean': np.mean(distances),
            'final_distance_std': np.std(distances),
            'final_distance_min': np.min(distances),
            'final_distance_max': np.max(distances)
        }
        
        self.evaluation_results['statistics'] = stats
    
    def plot_trajectories_2d(self, save_individual=True, save_combined=True):
        """
        Create 2D trajectory plots.
        
        Args:
            save_individual (bool): Save individual episode plots
            save_combined (bool): Save combined trajectory plot
        """
        trajectories = self.evaluation_results['trajectories']
        episodes = self.evaluation_results['episodes']
        
        if save_combined:
            self._plot_combined_trajectories_2d(trajectories, episodes)
        
        if save_individual:
            self._plot_individual_trajectories_2d(trajectories, episodes)
    
    def _plot_combined_trajectories_2d(self, trajectories, episodes):
        """Plot all trajectories on a single 2D plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # X-Y plane plot
        successful_count = 0
        failed_count = 0
        
        for i, (traj, episode) in enumerate(zip(trajectories, episodes)):
            if episode['success']:
                color = 'green'
                alpha = 0.7
                label = 'Successful' if successful_count == 0 else ""
                successful_count += 1
            else:
                color = 'red'
                alpha = 0.5
                label = 'Failed' if failed_count == 0 else ""
                failed_count += 1
            
            ax1.plot(traj['x'], traj['y'], color=color, alpha=alpha, linewidth=2, label=label)
            
            # Mark start and end points
            ax1.plot(traj['x'][0], traj['y'][0], 'bo', markersize=8, alpha=0.7)  # Start (blue)
            ax1.plot(traj['x'][-1], traj['y'][-1], 'ro', markersize=8, alpha=0.7)  # End (red)
            
            # Mark target
            if episode['target_position'] is not None:
                target = episode['target_position']
                ax1.plot(target[0], target[1], 'g*', markersize=15, alpha=0.8)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('2D Trajectories (X-Y Plane)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # X-Z plane plot
        for i, (traj, episode) in enumerate(zip(trajectories, episodes)):
            color = 'green' if episode['success'] else 'red'
            alpha = 0.7 if episode['success'] else 0.5
            
            ax2.plot(traj['x'], traj['z'], color=color, alpha=alpha, linewidth=2)
            ax2.plot(traj['x'][0], traj['z'][0], 'bo', markersize=8, alpha=0.7)
            ax2.plot(traj['x'][-1], traj['z'][-1], 'ro', markersize=8, alpha=0.7)
            
            # Mark target
            if episode['target_position'] is not None:
                target = episode['target_position']
                ax2.plot(target[0], target[2], 'g*', markersize=15, alpha=0.8)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Z Position (m)')
        ax2.set_title('2D Trajectories (X-Z Plane)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'combined_trajectories_2d.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("📊 Combined 2D trajectory plot saved!")
    
    def _plot_individual_trajectories_2d(self, trajectories, episodes):
        """Plot individual trajectory plots."""
        individual_dir = os.path.join(self.output_dir, 'individual_trajectories')
        os.makedirs(individual_dir, exist_ok=True)
        
        for i, (traj, episode) in enumerate(zip(trajectories, episodes)):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Determine colors based on success
            traj_color = 'green' if episode['success'] else 'red'
            title_status = "SUCCESS" if episode['success'] else "FAILED"
            
            # X-Y plot
            ax1.plot(traj['x'], traj['y'], color=traj_color, linewidth=2, alpha=0.8)
            ax1.plot(traj['x'][0], traj['y'][0], 'bo', markersize=10, label='Start')
            ax1.plot(traj['x'][-1], traj['y'][-1], 'ro', markersize=10, label='End')
            
            if episode['target_position'] is not None:
                target = episode['target_position']
                ax1.plot(target[0], target[1], 'g*', markersize=20, label='Target')
                
                # Draw target tolerance circle
                circle = patches.Circle((target[0], target[1]), 0.1, 
                                      fill=False, color='green', linestyle='--', alpha=0.5)
                ax1.add_patch(circle)
            
            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')
            ax1.set_title(f'Episode {i+1} - {title_status} (X-Y)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.axis('equal')
            
            # X-Z plot
            ax2.plot(traj['x'], traj['z'], color=traj_color, linewidth=2, alpha=0.8)
            ax2.plot(traj['x'][0], traj['z'][0], 'bo', markersize=10, label='Start')
            ax2.plot(traj['x'][-1], traj['z'][-1], 'ro', markersize=10, label='End')
            
            if episode['target_position'] is not None:
                target = episode['target_position']
                ax2.plot(target[0], target[2], 'g*', markersize=20, label='Target')
            
            ax2.set_xlabel('X Position (m)')
            ax2.set_ylabel('Z Position (m)')
            ax2.set_title(f'Episode {i+1} - {title_status} (X-Z)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add episode statistics as text
            stats_text = (f'Reward: {episode["total_reward"]:.1f}\n'
                         f'Steps: {episode["steps"]}\n'
                         f'Final Dist: {episode["final_distance"]:.3f}m')
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(os.path.join(individual_dir, f'episode_{i+1:02d}_trajectory.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"📊 {len(trajectories)} individual trajectory plots saved!")
    
    def plot_performance_metrics(self):
        """Plot performance metrics and statistics."""
        episodes = self.evaluation_results['episodes']
        stats = self.evaluation_results['statistics']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Episode rewards
        rewards = [ep['total_reward'] for ep in episodes]
        colors = ['green' if ep['success'] else 'red' for ep in episodes]
        
        ax1.bar(range(1, len(rewards)+1), rewards, color=colors, alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards')
        ax1.grid(True, alpha=0.3)
        
        # Episode steps
        steps = [ep['steps'] for ep in episodes]
        ax2.bar(range(1, len(steps)+1), steps, color=colors, alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Duration (Steps)')
        ax2.grid(True, alpha=0.3)
        
        # Final distances
        distances = [ep['final_distance'] for ep in episodes]
        ax3.bar(range(1, len(distances)+1), distances, color=colors, alpha=0.7)
        ax3.axhline(y=0.1, color='green', linestyle='--', alpha=0.8, label='Success Threshold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Final Distance to Target (m)')
        ax3.set_title('Final Distance to Target')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Statistics summary
        ax4.axis('off')
        stats_text = f"""
Evaluation Summary:
{'='*30}
Episodes: {stats['n_episodes']}
Success Rate: {stats['success_rate']:.1f}%

Rewards:
  Mean: {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f}
  Range: [{stats['reward_min']:.1f}, {stats['reward_max']:.1f}]

Steps:
  Mean: {stats['steps_mean']:.1f} ± {stats['steps_std']:.1f}
  Range: [{stats['steps_min']}, {stats['steps_max']}]

Final Distance:
  Mean: {stats['final_distance_mean']:.3f} ± {stats['final_distance_std']:.3f}
  Range: [{stats['final_distance_min']:.3f}, {stats['final_distance_max']:.3f}]
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_metrics.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("📊 Performance metrics plot saved!")
    
    def save_results(self):
        """Save evaluation results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in self.evaluation_results.items():
            if key == 'episodes':
                json_results[key] = []
                for episode in value:
                    json_episode = {}
                    for ep_key, ep_value in episode.items():
                        if isinstance(ep_value, np.ndarray):
                            json_episode[ep_key] = ep_value.tolist()
                        elif isinstance(ep_value, dict):
                            json_episode[ep_key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                                  for k, v in ep_value.items()}
                        else:
                            json_episode[ep_key] = ep_value
                    json_results[key].append(json_episode)
            else:
                json_results[key] = value
        
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"💾 Evaluation results saved to: {results_path}")
    
    def generate_report(self):
        """Generate a comprehensive evaluation report."""
        stats = self.evaluation_results['statistics']
        
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("🚁 DRONE NAVIGATION EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"📅 Evaluation Date: {self.evaluation_results['metadata']['evaluation_date']}\n")
            f.write(f"🤖 Model Path: {self.evaluation_results['metadata']['model_path']}\n")
            f.write(f"📊 Normalization Used: {self.evaluation_results['metadata']['normalization_used']}\n\n")
            
            f.write("📈 PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Episodes Evaluated: {stats['n_episodes']}\n")
            f.write(f"Success Rate: {stats['success_rate']:.1f}%\n")
            f.write(f"Mean Reward: {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f}\n")
            f.write(f"Mean Steps: {stats['steps_mean']:.1f} ± {stats['steps_std']:.1f}\n")
            f.write(f"Mean Final Distance: {stats['final_distance_mean']:.3f} ± {stats['final_distance_std']:.3f} m\n\n")
            
            f.write("🎯 DETAILED STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Reward Range: [{stats['reward_min']:.1f}, {stats['reward_max']:.1f}]\n")
            f.write(f"Steps Range: [{stats['steps_min']}, {stats['steps_max']}]\n")
            f.write(f"Distance Range: [{stats['final_distance_min']:.3f}, {stats['final_distance_max']:.3f}] m\n\n")
            
            # Analysis
            if stats['success_rate'] >= 80:
                f.write("✅ EXCELLENT: High success rate achieved!\n")
            elif stats['success_rate'] >= 60:
                f.write("🟡 GOOD: Moderate success rate, room for improvement.\n")
            else:
                f.write("🔴 NEEDS IMPROVEMENT: Low success rate detected.\n")
            
            f.write(f"\n📊 Files Generated:\n")
            f.write(f"- combined_trajectories_2d.png\n")
            f.write(f"- performance_metrics.png\n")
            f.write(f"- individual_trajectories/ (folder)\n")
            f.write(f"- evaluation_results.json\n")
        
        print(f"📋 Evaluation report saved to: {report_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained SAC drone navigation model')
    parser.add_argument('--model_path', type=str, default='./models/best_model.zip',
                       help='Path to trained model')
    parser.add_argument('--norm_path', type=str, default='./models/vec_normalize.pkl',
                       help='Path to normalization parameters')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        print("Available models:")
        models_dir = os.path.dirname(args.model_path)
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.zip'):
                    print(f"  - {os.path.join(models_dir, file)}")
        return
    
    # Initialize evaluator
    evaluator = DroneEvaluator(
        model_path=args.model_path,
        normalization_path=args.norm_path if os.path.exists(args.norm_path) else None,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    print("🚀 Starting evaluation...")
    evaluator.run_evaluation(n_episodes=args.episodes, deterministic=args.deterministic)
    
    # Generate plots and reports
    print("📊 Generating plots...")
    evaluator.plot_trajectories_2d(save_individual=True, save_combined=True)
    evaluator.plot_performance_metrics()
    
    # Save results
    print("💾 Saving results...")
    evaluator.save_results()
    evaluator.generate_report()
    
    print("\n🎉 Evaluation completed successfully!")
    print(f"📁 All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()