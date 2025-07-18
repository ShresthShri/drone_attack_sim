#!/usr/bin/env python3
"""
Standalone evaluation script for curriculum learning drone model
Works with your existing curriculum_drone_training.py file
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from stable_baselines3 import SAC
import pandas as pd
from typing import Dict, List, Tuple, Any
import json
import os
import sys
from collections import defaultdict

# Try to import your curriculum classes
try:
    # Add current directory to path to find your training script
    sys.path.append('.')
    sys.path.append(os.getcwd())
    
    # Try different possible import names for your curriculum training file
    try:
        from agents.curriculum_drone_training import CurriculumConfig, AdaptiveDroneEnv
        print("✅ Successfully imported from curriculum_drone_training")
    except ImportError:
        try:
            from paste import CurriculumConfig, AdaptiveDroneEnv
            print("✅ Successfully imported from paste")
        except ImportError:
            # Try importing from the document you provided
            import importlib.util
            
            # Look for curriculum training files in current directory
            possible_files = [
                'curriculum_drone_training.py',
                'paste.py', 
                'drone_training.py',
                'curriculum_learning.py'
            ]
            
            curriculum_module = None
            for filename in possible_files:
                if os.path.exists(filename):
                    spec = importlib.util.spec_from_file_location("curriculum_module", filename)
                    curriculum_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(curriculum_module)
                    
                    if hasattr(curriculum_module, 'CurriculumConfig') and hasattr(curriculum_module, 'AdaptiveDroneEnv'):
                        CurriculumConfig = curriculum_module.CurriculumConfig
                        AdaptiveDroneEnv = curriculum_module.AdaptiveDroneEnv
                        print(f"✅ Successfully imported from {filename}")
                        break
            
            if curriculum_module is None:
                raise ImportError("Could not find curriculum training module")
    
    FULL_EVALUATION = True
except ImportError as e:
    print(f"⚠️  Could not import curriculum classes: {e}")
    print("Creating mock configuration for demonstration...")
    FULL_EVALUATION = False
    
    # Mock configuration for demonstration
    class CurriculumConfig:
        STAGES = {
            'basic_navigation': {'environment': {'goal_threshold': 1.0}},
            'obstacle_avoidance': {'environment': {'goal_threshold': 0.8}},
            'complex_navigation': {'environment': {'goal_threshold': 0.6}},
            'expert_level': {'environment': {'goal_threshold': 0.5}}
        }
        STAGE_ORDER = ['basic_navigation', 'obstacle_avoidance', 'complex_navigation', 'expert_level']

class DroneEvaluationVisualizer:
    """Visualizes drone evaluation results with 2D paths and summaries"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'success': '#2E8B57',      # Sea Green
            'collision': '#DC143C',     # Crimson
            'timeout': '#FF8C00',       # Dark Orange
            'drone_path': '#4169E1',    # Royal Blue
            'obstacles': '#696969',     # Dim Gray
            'start': '#32CD32',         # Lime Green
            'goal': '#FFD700'           # Gold
        }
        
    def plot_2d_paths(self, evaluation_results: Dict, save_path: str = None):
        """Create 2D path visualization for all curriculum stages"""
        
        stages = list(evaluation_results.keys())
        n_stages = min(len(stages), 4)  # Limit to 4 for 2x2 grid
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Drone Navigation Paths by Curriculum Stage', fontsize=16, fontweight='bold')
        
        for idx in range(4):
            ax = axes[idx // 2, idx % 2]
            
            if idx < len(stages):
                stage = stages[idx]
                stage_data = evaluation_results[stage]
                
                # Plot episodes (limit to first 10 for clarity)
                episodes = stage_data['episodes'][:10]
                
                for ep_idx, episode in enumerate(episodes):
                    path = np.array(episode['path'])
                    outcome = episode['outcome']
                    
                    # Path styling based on outcome
                    if outcome == 'success':
                        color = self.colors['success']
                        alpha = 0.8
                        linewidth = 2
                    elif outcome == 'collision':
                        color = self.colors['collision']
                        alpha = 0.6
                        linewidth = 1.5
                    else:  # timeout
                        color = self.colors['timeout']
                        alpha = 0.4
                        linewidth = 1
                    
                    # Plot path
                    if len(path) > 1:
                        ax.plot(path[:, 0], path[:, 1], color=color, alpha=alpha, 
                               linewidth=linewidth)
                    
                    # Mark start and goal
                    start_pos = episode['start_pos']
                    goal_pos = episode['goal_pos']
                    
                    ax.scatter(start_pos[0], start_pos[1], 
                              c=self.colors['start'], s=80, marker='o', 
                              edgecolors='black', linewidth=1, zorder=5, alpha=0.7)
                    ax.scatter(goal_pos[0], goal_pos[1], 
                              c=self.colors['goal'], s=100, marker='*', 
                              edgecolors='black', linewidth=1, zorder=5, alpha=0.8)
                
                # Plot obstacles if available
                obstacles = stage_data.get('obstacles', [])
                for obs_center, obs_size in obstacles:
                    if len(obs_center) >= 2 and len(obs_size) >= 2:
                        rect = patches.Rectangle(
                            (obs_center[0] - obs_size[0]/2, obs_center[1] - obs_size[1]/2),
                            obs_size[0], obs_size[1],
                            linewidth=2, edgecolor=self.colors['obstacles'],
                            facecolor=self.colors['obstacles'], alpha=0.7
                        )
                        ax.add_patch(rect)
                
                # Styling
                ax.set_title(f'{stage.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel('X Position (m)')
                ax.set_ylabel('Y Position (m)')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                
                # Add success rate annotation
                success_rate = stage_data['metrics']['success_rate']
                ax.text(0.02, 0.98, f'Success: {success_rate:.1%}', 
                       transform=ax.transAxes, fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       verticalalignment='top')
            else:
                # Empty subplot
                ax.set_visible(False)
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.colors['success'], linewidth=2, label='Success Path'),
            plt.Line2D([0], [0], color=self.colors['collision'], linewidth=2, label='Collision Path'),
            plt.Line2D([0], [0], color=self.colors['timeout'], linewidth=2, label='Timeout Path'),
            plt.scatter([], [], c=self.colors['start'], s=80, marker='o', 
                       edgecolors='black', linewidth=1, label='Start Position'),
            plt.scatter([], [], c=self.colors['goal'], s=100, marker='*', 
                       edgecolors='black', linewidth=1, label='Goal Position'),
            patches.Rectangle((0, 0), 1, 1, facecolor=self.colors['obstacles'], 
                            alpha=0.7, label='Obstacles')
        ]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                  ncol=3, fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 2D paths saved to: {save_path}")
        
        plt.show()
    
    def plot_performance_summary(self, evaluation_results: Dict, save_path: str = None):
        """Create comprehensive performance summary plots"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Curriculum Learning Performance Summary', fontsize=16, fontweight='bold')
        
        stages = list(evaluation_results.keys())
        
        # 1. Success rates by stage
        success_rates = [evaluation_results[stage]['metrics']['success_rate'] for stage in stages]
        colors = [self.colors['success'] if sr >= 0.7 else 
                 self.colors['timeout'] if sr >= 0.4 else 
                 self.colors['collision'] for sr in success_rates]
        
        bars1 = ax1.bar(range(len(stages)), success_rates, color=colors, alpha=0.8)
        ax1.set_title('Success Rate by Curriculum Stage', fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(range(len(stages)))
        ax1.set_xticklabels([s.replace('_', '\n') for s in stages], rotation=0)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Average rewards progression
        avg_rewards = [evaluation_results[stage]['metrics']['avg_reward'] for stage in stages]
        ax2.plot(range(len(stages)), avg_rewards, marker='o', linewidth=3, markersize=8, 
                color=self.colors['drone_path'])
        ax2.set_title('Average Episode Reward by Stage', fontweight='bold')
        ax2.set_ylabel('Average Reward')
        ax2.set_xticks(range(len(stages)))
        ax2.set_xticklabels([s.replace('_', '\n') for s in stages])
        ax2.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, reward in enumerate(avg_rewards):
            ax2.annotate(f'{reward:.0f}', (i, reward), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        
        # 3. Episode lengths
        avg_lengths = [evaluation_results[stage]['metrics']['avg_episode_length'] for stage in stages]
        bars3 = ax3.bar(range(len(stages)), avg_lengths, color=self.colors['timeout'], alpha=0.7)
        ax3.set_title('Average Episode Length by Stage', fontweight='bold')
        ax3.set_ylabel('Average Steps')
        ax3.set_xticks(range(len(stages)))
        ax3.set_xticklabels([s.replace('_', '\n') for s in stages])
        
        # Add value labels
        for bar, length in zip(bars3, avg_lengths):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{length:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Outcome distribution (stacked bar)
        outcome_data = []
        for stage in stages:
            metrics = evaluation_results[stage]['metrics']
            outcome_data.append([
                metrics['success_rate'],
                metrics['collision_rate'],
                metrics['timeout_rate']
            ])
        
        outcome_data = np.array(outcome_data).T
        bottom = np.zeros(len(stages))
        
        outcomes = ['Success', 'Collision', 'Timeout']
        outcome_colors = [self.colors['success'], self.colors['collision'], self.colors['timeout']]
        
        for i, (outcome, color) in enumerate(zip(outcomes, outcome_colors)):
            ax4.bar(range(len(stages)), outcome_data[i], bottom=bottom, 
                   label=outcome, color=color, alpha=0.8)
            
            # Add percentage labels for significant portions
            for j in range(len(stages)):
                if outcome_data[i][j] > 0.1:  # Only label if >10%
                    ax4.text(j, bottom[j] + outcome_data[i][j]/2, 
                            f'{outcome_data[i][j]:.0%}', 
                            ha='center', va='center', fontweight='bold', fontsize=9)
            
            bottom += outcome_data[i]
        
        ax4.set_title('Episode Outcome Distribution', fontweight='bold')
        ax4.set_ylabel('Proportion')
        ax4.set_xticks(range(len(stages)))
        ax4.set_xticklabels([s.replace('_', '\n') for s in stages])
        ax4.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance summary saved to: {save_path}")
        
        plt.show()

def run_curriculum_evaluation(model_path: str = "curriculum_drone_final", 
                             num_episodes: int = 15,
                             save_results: bool = True,
                             save_dir: str = "./evaluation_results/"):
    """
    Main evaluation function - works with or without full curriculum system
    """
    
    print("🚁 CURRICULUM DRONE MODEL EVALUATION")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the trained model
    try:
        model = SAC.load(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure you have trained a model first!")
        return None
    
    # Initialize visualizer
    visualizer = DroneEvaluationVisualizer()
    
    if FULL_EVALUATION:
        print("🔄 Running full curriculum evaluation...")
        evaluation_results = _run_full_evaluation(model, num_episodes)
    else:
        print("🔄 Running mock evaluation (full system not available)...")
        evaluation_results = _run_mock_evaluation(num_episodes)
    
    if not evaluation_results:
        print("❌ Evaluation failed")
        return None
    
    # Print summary
    _print_evaluation_summary(evaluation_results)
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # 2D Path plots
    path_plot_path = os.path.join(save_dir, "drone_paths_2d.png") if save_results else None
    visualizer.plot_2d_paths(evaluation_results, path_plot_path)
    
    # Performance summary plots
    summary_plot_path = os.path.join(save_dir, "performance_summary.png") if save_results else None
    visualizer.plot_performance_summary(evaluation_results, summary_plot_path)
    
    # Save detailed results
    if save_results:
        _save_detailed_results(evaluation_results, save_dir)
    
    print(f"\n🎉 Evaluation completed!")
    if save_results:
        print(f"📁 Results saved to: {save_dir}")
    
    return evaluation_results

def _run_full_evaluation(model, num_episodes: int) -> Dict:
    """Run full evaluation with actual curriculum environments"""
    
    evaluation_results = {}
    
    print(f"Testing on {len(CurriculumConfig.STAGE_ORDER)} curriculum stages...")
    print(f"Episodes per stage: {num_episodes}")
    print("-" * 50)
    
    for stage_idx, stage_name in enumerate(CurriculumConfig.STAGE_ORDER):
        print(f"\n📊 Stage {stage_idx + 1}/{len(CurriculumConfig.STAGE_ORDER)}: {stage_name.upper()}")
        
        # Initialize environment for this stage
        stage_config = CurriculumConfig.STAGES[stage_name]
        test_env = AdaptiveDroneEnv(stage_config)
        
        # Run episodes
        stage_results = _run_stage_episodes(model, test_env, num_episodes, stage_name)
        evaluation_results[stage_name] = stage_results
        
        # Print stage summary
        metrics = stage_results['metrics']
        print(f"  ✅ Success: {metrics['success_rate']:.1%}")
        print(f"  💥 Collision: {metrics['collision_rate']:.1%}")
        print(f"  ⏱️  Timeout: {metrics['timeout_rate']:.1%}")
        print(f"  📈 Avg Reward: {metrics['avg_reward']:.1f}")
        print(f"  🎯 Avg Progress: {metrics['avg_progress']:.1%}")
        
        test_env.close()
    
    return evaluation_results

def _run_stage_episodes(model, env, num_episodes: int, stage_name: str) -> Dict:
    """Run episodes for a specific curriculum stage"""
    
    episodes = []
    metrics_raw = defaultdict(list)
    
    # Get model's expected observation space
    model_obs_dim = model.observation_space.shape[0]
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        # Handle observation space mismatch
        if len(obs) != model_obs_dim:
            print(f"    ⚠️  Observation mismatch: env={len(obs)}D, model={model_obs_dim}D")
            if len(obs) > model_obs_dim:
                # Truncate observation to match model
                obs = obs[:model_obs_dim]
                print(f"    🔧 Truncating to {model_obs_dim}D")
            else:
                # Pad observation to match model
                padding = np.zeros(model_obs_dim - len(obs))
                obs = np.concatenate([obs, padding])
                print(f"    🔧 Padding to {model_obs_dim}D")
        
        # Episode tracking
        episode_data = {
            'path': [obs[:3].copy()],  # Always use first 3 dims for position
            'start_pos': env.start_pos.copy(),
            'goal_pos': env.goal_pos.copy(),
            'total_reward': 0,
            'episode_length': 0,
            'outcome': 'unknown'
        }
        
        done = False
        step_count = 0
        
        # Run episode
        while not done and step_count < 1000:  # Safety limit
            try:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                
                # Handle observation space mismatch for next step
                if len(obs) != model_obs_dim:
                    if len(obs) > model_obs_dim:
                        obs = obs[:model_obs_dim]
                    else:
                        padding = np.zeros(model_obs_dim - len(obs))
                        obs = np.concatenate([obs, padding])
                
                episode_data['path'].append(obs[:3].copy())
                episode_data['total_reward'] += reward
                step_count += 1
                
            except Exception as e:
                print(f"    ❌ Error during episode: {e}")
                done = True
                episode_data['outcome'] = 'error'
                break
        
        # Finalize episode data
        episode_data['episode_length'] = step_count
        episode_data['progress_ratio'] = info.get('progress_ratio', 0) if 'info' in locals() else 0
        
        # Determine outcome
        if episode_data['outcome'] != 'error':
            if info.get('success', False):
                episode_data['outcome'] = 'success'
            elif info.get('collision', False):
                episode_data['outcome'] = 'collision'
            else:
                episode_data['outcome'] = 'timeout'
        
        episodes.append(episode_data)
        
        # Update metrics
        metrics_raw['total_reward'].append(episode_data['total_reward'])
        metrics_raw['episode_length'].append(episode_data['episode_length'])
        metrics_raw['progress_ratio'].append(episode_data['progress_ratio'])
        metrics_raw['outcome'].append(episode_data['outcome'])
        
        # Progress indicator
        if (episode + 1) % 5 == 0:
            current_success = sum(1 for ep in episodes if ep['outcome'] == 'success') / len(episodes)
            print(f"    Episode {episode + 1:2d}/{num_episodes}: Success = {current_success:.1%}")
    
    # Calculate final metrics
    outcomes = metrics_raw['outcome']
    metrics = {
        'success_rate': outcomes.count('success') / len(outcomes),
        'collision_rate': outcomes.count('collision') / len(outcomes),
        'timeout_rate': outcomes.count('timeout') / len(outcomes),
        'error_rate': outcomes.count('error') / len(outcomes),
        'avg_reward': np.mean(metrics_raw['total_reward']),
        'avg_episode_length': np.mean(metrics_raw['episode_length']),
        'avg_progress': np.mean(metrics_raw['progress_ratio']),
        'std_reward': np.std(metrics_raw['total_reward']),
        'std_episode_length': np.std(metrics_raw['episode_length'])
    }
    
    return {
        'episodes': episodes,
        'metrics': metrics,
        'obstacles': getattr(env, 'obstacle_params', [])
    }

def _run_mock_evaluation(num_episodes: int) -> Dict:
    """Run mock evaluation for demonstration when full system unavailable"""
    
    print("⚠️  Running with mock data (full curriculum system not available)")
    
    evaluation_results = {}
    
    for stage_idx, stage_name in enumerate(CurriculumConfig.STAGE_ORDER):
        print(f"\n📊 Mock Stage {stage_idx + 1}: {stage_name.upper()}")
        
        # Generate realistic mock data
        stage_difficulty = stage_idx + 1
        base_success_rate = max(0.3, 0.9 - stage_difficulty * 0.15)
        
        episodes = []
        for i in range(num_episodes):
            # Simulate varying outcomes based on stage difficulty
            outcome_probs = [base_success_rate, 0.25, 1 - base_success_rate - 0.25]
            outcome = np.random.choice(['success', 'collision', 'timeout'], p=outcome_probs)
            
            # Generate mock path (simple straight line with noise)
            start_pos = np.random.uniform(-2, 2, 3)
            goal_pos = start_pos + np.random.uniform([5, -3, 0], [15, 3, 0])
            
            path_length = np.random.randint(20, 100)
            path = np.linspace(start_pos, goal_pos, path_length)
            path += np.random.normal(0, 0.5, path.shape)  # Add noise
            
            episode_data = {
                'outcome': outcome,
                'total_reward': np.random.normal(150 - stage_difficulty * 30, 40),
                'episode_length': path_length,
                'progress_ratio': np.random.uniform(0.4, 1.0 if outcome == 'success' else 0.8),
                'start_pos': start_pos,
                'goal_pos': goal_pos,
                'path': path
            }
            episodes.append(episode_data)
        
        # Calculate metrics
        outcomes = [ep['outcome'] for ep in episodes]
        metrics = {
            'success_rate': outcomes.count('success') / len(outcomes),
            'collision_rate': outcomes.count('collision') / len(outcomes),
            'timeout_rate': outcomes.count('timeout') / len(outcomes),
            'avg_reward': np.mean([ep['total_reward'] for ep in episodes]),
            'avg_episode_length': np.mean([ep['episode_length'] for ep in episodes]),
            'avg_progress': np.mean([ep['progress_ratio'] for ep in episodes]),
            'std_reward': np.std([ep['total_reward'] for ep in episodes]),
            'std_episode_length': np.std([ep['episode_length'] for ep in episodes])
        }
        
        evaluation_results[stage_name] = {
            'episodes': episodes,
            'metrics': metrics,
            'obstacles': []
        }
        
        print(f"  Success: {metrics['success_rate']:.1%} | "
              f"Collision: {metrics['collision_rate']:.1%} | "
              f"Timeout: {metrics['timeout_rate']:.1%}")
    
    return evaluation_results

def _print_evaluation_summary(evaluation_results: Dict):
    """Print comprehensive evaluation summary"""
    
    print(f"\n🎯 EVALUATION SUMMARY")
    print("=" * 60)
    
    # Overall statistics
    overall_success = np.mean([results['metrics']['success_rate'] 
                              for results in evaluation_results.values()])
    print(f"🏆 Overall Success Rate: {overall_success:.1%}")
    
    # Stage-by-stage breakdown
    print(f"\n📊 Stage-by-Stage Results:")
    print("-" * 40)
    
    for stage_name, results in evaluation_results.items():
        metrics = results['metrics']
        print(f"{stage_name.replace('_', ' ').title():<20} | "
              f"Success: {metrics['success_rate']:.1%} | "
              f"Avg Reward: {metrics['avg_reward']:.0f} | "
              f"Avg Steps: {metrics['avg_episode_length']:.0f}")
    
    # Curriculum analysis
    stages = list(evaluation_results.keys())
    if len(stages) >= 2:
        print(f"\n🧠 Curriculum Learning Analysis:")
        print("-" * 40)
        
        basic_success = evaluation_results[stages[0]]['metrics']['success_rate']
        expert_success = evaluation_results[stages[-1]]['metrics']['success_rate']
        
        print(f"Basic Skills Retention: {basic_success:.1%}")
        print(f"Expert Level Performance: {expert_success:.1%}")
        
        if basic_success > 0.6:
            print("✅ No catastrophic forgetting detected")
        else:
            print("⚠️  Potential skill degradation in basic tasks")
        
        # Check progression smoothness
        success_rates = [evaluation_results[stage]['metrics']['success_rate'] 
                        for stage in stages]
        
        smooth_progression = all(success_rates[i] >= success_rates[i+1] - 0.2 
                               for i in range(len(success_rates)-1))
        
        if smooth_progression:
            print("✅ Smooth learning progression across stages")
        else:
            print("⚠️  Irregular progression detected - may need curriculum tuning")

def _save_detailed_results(evaluation_results: Dict, save_dir: str):
    """Save detailed evaluation results to files"""
    
    # Save JSON results
    json_results = {}
    for stage, data in evaluation_results.items():
        json_results[stage] = {
            'metrics': data['metrics'],
            'episode_count': len(data['episodes']),
            'sample_episodes': [
                {
                    'outcome': ep['outcome'],
                    'total_reward': float(ep['total_reward']),
                    'episode_length': int(ep['episode_length']),
                    'progress_ratio': float(ep['progress_ratio']),
                    'start_pos': [float(x) for x in ep['start_pos']],
                    'goal_pos': [float(x) for x in ep['goal_pos']]
                }
                for ep in data['episodes'][:3]  # Save first 3 episodes
            ]
        }
    
    results_file = os.path.join(save_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save CSV summary
    summary_data = []
    for stage_name, stage_data in evaluation_results.items():
        metrics = stage_data['metrics']
        summary_data.append({
            'Stage': stage_name.replace('_', ' ').title(),
            'Success_Rate': metrics['success_rate'],
            'Collision_Rate': metrics['collision_rate'],
            'Timeout_Rate': metrics['timeout_rate'],
            'Avg_Reward': metrics['avg_reward'],
            'Avg_Episode_Length': metrics['avg_episode_length'],
            'Avg_Progress': metrics['avg_progress'],
            'Std_Reward': metrics['std_reward']
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_file = os.path.join(save_dir, 'evaluation_summary.csv')
    summary_df.to_csv(csv_file, index=False)
    
    print(f"💾 Detailed results saved to: {results_file}")
    print(f"💾 Summary CSV saved to: {csv_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate curriculum learning drone model')
    parser.add_argument('--model', type=str, default='curriculum_drone_final',
                       help='Path to trained model (default: curriculum_drone_final)')
    parser.add_argument('--episodes', type=int, default=15,
                       help='Number of episodes per stage (default: 15)')
    parser.add_argument('--save-dir', type=str, default='./evaluation_results/',
                       help='Directory to save results (default: ./evaluation_results/)')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save results to files')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_curriculum_evaluation(
        model_path=args.model,
        num_episodes=args.episodes,
        save_results=not args.no_save,
        save_dir=args.save_dir
    )
    
    if results:
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📈 Analyzed {sum(len(stage['episodes']) for stage in results.values())} total episodes")
    else:
        print("❌ Evaluation failed")
        exit(1)