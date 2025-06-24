# agents/train_sac.py (Fixed Version)

import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import pandas as pd
import matplotlib.pyplot as plt
import time

from envs.drone_navigation_env import DroneNavigationEnv

def make_env(rank=0, seed=0, gui=False, max_steps=300, obstacles=True, monitor=True):
    """
    Create a wrapped environment for training.
    
    Args:
        rank: The index of the subprocess
        seed: The initial seed for RNG
        gui: Whether to render GUI
        max_steps: Maximum steps per episode
        obstacles: Whether to include obstacles in the environment
        monitor: Whether to wrap with Monitor (avoid double wrapping)
    """
    def _init():
        env = DroneNavigationEnv(gui=gui, obstacles=obstacles)
        env = TimeLimit(env, max_episode_steps=max_steps)
        if monitor:
            env = Monitor(env)  # Only add Monitor if requested
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    # Create directories for saving models and logs
    model_dir = "models/sac_baseline_obstacles"
    log_dir = "results/logs_obstacles"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment for training
    NUM_ENVS = 4
    # Don't add Monitor wrapper here since VecMonitor will handle monitoring
    env = DummyVecEnv([make_env(rank=i, obstacles=True, monitor=False) for i in range(NUM_ENVS)])
    env = VecMonitor(env, log_dir)  # VecMonitor will handle logging

    # Initialize the SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200000,  # Increased buffer size
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=4,
        learning_starts=5000,
        verbose=1,
        ent_coef=0.01, 
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=log_dir
    )

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="sac_model"
    )
    
    # Create a separate environment for evaluation (no Monitor wrapping)
    eval_env = DummyVecEnv([make_env(rank=0, obstacles=True, monitor=False)])
    eval_env = VecMonitor(eval_env, log_dir)

    # Stop if eval reward > threshold
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=20,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback
    )

    # Start training
    print("Training SAC agent (parallel, headless)..")
    print(f"Tensorboard log: tensorboard --logdir {log_dir}")
    
    try:
        model.learn(
            total_timesteps=1_000_000,
            callback=[checkpoint_callback, eval_callback]
        )
        model.save(os.path.join(model_dir,"final_sac_model"))
        print("Training completed successfully.")
    except Exception as e:
        print(f"Training interrupted: {e}")
        # Save current model
        model.save(os.path.join(model_dir,"interrupted_sac_model"))
        print("Model saved despite interruption.")


def evaluate_model(model_path, num_episodes=5):
    """
    Evaluate a trained model and log detailed step-by-step data.
    """
    # Load the trained model
    model = SAC.load(model_path)
    
    # Create a new environment for evaluation WITH GUI and OBSTACLES
    env = DroneNavigationEnv(gui=True, obstacles=True, initial_xyzs=np.array([[0., 0., 1.]]))
    env = TimeLimit(env, max_episode_steps=500)
    
    all_episode_data = []

    print(f"\nEvaluating model: {model_path} for {num_episodes} episodes (with GUI)...")
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        episode_data = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            # Collect detailed step-by-step data
            step_info = {
                "episode": episode + 1,
                "step": step_count,
                "drone_x": info["drone_position"][0],
                "drone_y": info["drone_position"][1],
                "drone_z": info["drone_position"][2],
                "goal_x": info["goal_position"][0],
                "goal_y": info["goal_position"][1],
                "goal_z": info["goal_position"][2],
                "reward": info["true_reward"],
                "collision_occurred": info["collision_occurred"],
                "reached_goal": info["reached_goal"],
            }
            episode_data.append(step_info)
            
            # Small delay for visualization
            time.sleep(0.02)
            
        print(f"Episode {episode + 1} finished. Total reward: {episode_reward:.2f}, Steps: {step_count}")
        all_episode_data.extend(episode_data)

    env.close()
    
    # Save detailed logs to a CSV file
    df = pd.DataFrame(all_episode_data)
    log_file_path = os.path.join("results/evaluation_logs", "detailed_evaluation_log.csv")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    df.to_csv(log_file_path, index=False)
    print(f"Detailed evaluation logs saved to: {log_file_path}")

    # Plot trajectory for first episode
    if num_episodes > 0:
        first_episode_df = df[df['episode'] == 1]
        plt.figure(figsize=(8, 6))
        plt.plot(first_episode_df['drone_x'], first_episode_df['drone_y'], 
                label='Drone Trajectory', linewidth=2)
        plt.scatter(first_episode_df['goal_x'].iloc[0], first_episode_df['goal_y'].iloc[0], 
                   color='red', marker='*', s=200, label='Goal')
        plt.scatter(first_episode_df['drone_x'].iloc[0], first_episode_df['drone_y'].iloc[0], 
                   color='green', marker='o', s=100, label='Start')
        plt.title('Drone Trajectory (Episode 1)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plot_path = os.path.join("results/evaluation_logs", "episode_1_trajectory.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Trajectory plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
    
    # After training, evaluate the final model
    final_model_path = "models/sac_baseline_obstacles/final_sac_model.zip"
    best_model_path = "models/sac_baseline_obstacles/best_model.zip"
    
    # Try to find the best available model
    if os.path.exists(best_model_path):
        print("Found best model, evaluating...")
        evaluate_model(best_model_path, num_episodes=3)
    elif os.path.exists(final_model_path):
        print("Found final model, evaluating...")
        evaluate_model(final_model_path, num_episodes=3)
    else:
        print("No trained model found. Please check training logs.")