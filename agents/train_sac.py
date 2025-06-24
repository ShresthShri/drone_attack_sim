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
from datetime import datetime # Import datetime for timestamping

from envs.drone_navigation_env import DroneNavigationEnv

def make_env(rank=0, seed=0, gui=False, max_steps=300, obstacles=True, monitor=True):
    def _init():
        env = DroneNavigationEnv(gui=gui, obstacles=obstacles)
        env = TimeLimit(env, max_episode_steps=max_steps)
        if monitor:
            env = Monitor(env) 
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    # Generate a timestamp for the current run
    # Format: YYYY-MM-DD_HH-MM-SS (e.g., 2023-10-27_15-30-00)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Define base directories for models and logs
    base_model_dir = "models/sac_baseline_obstacles"
    base_log_dir = "results/logs_obstacles"

    # Create a unique directory name for this run using the timestamp
    # This will result in directories like:
    # models/sac_baseline_obstacles/2023-10-27_15-30-00_SAC
    # results/logs_obstacles/2023-10-27_15-30-00_SAC
    current_run_name = f"{timestamp}_SAC_run" # Added '_run' for clarity
    model_dir = os.path.join(base_model_dir, current_run_name)
    log_dir = os.path.join(base_log_dir, current_run_name)

    # Create the timestamped directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Results for this run will be saved in: {model_dir} and {log_dir}")

    # Create and wrap the environment for training
    NUM_ENVS = 4
    # Don't add Monitor wrapper here since VecMonitor will handle monitoring
    env = DummyVecEnv([make_env(rank=i, obstacles=True, monitor=False) for i in range(NUM_ENVS)])
    # VecMonitor will now log to the timestamped log_dir
    env = VecMonitor(env, log_dir) 

    # Initialize the SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=4,
        learning_starts=10000,
        verbose=1,
        ent_coef='auto', 
        policy_kwargs=dict(net_arch=[256, 256]),
        # Tensorboard logs will also go to the timestamped log_dir
        tensorboard_log=log_dir 
    )

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        # Checkpoints will be saved in the timestamped model_dir
        save_path=model_dir, 
        name_prefix="sac_model"
    )
    
    # Create a separate environment for evaluation (no Monitor wrapping)
    eval_env = DummyVecEnv([make_env(rank=0, obstacles=True, monitor=False)])
    # Eval logs will also go to the timestamped log_dir
    eval_env = VecMonitor(eval_env, log_dir) 

    # Stop if eval reward > threshold
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=150,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        # Best model will be saved in the timestamped model_dir
        best_model_save_path=model_dir, 
        # Eval results log will also go to the timestamped log_dir
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
            total_timesteps=5_000_000,
            callback=[checkpoint_callback, eval_callback]
        )
        # Final model will be saved in the timestamped model_dir
        model.save(os.path.join(model_dir,"final_sac_model"))
        print("Training completed successfully.")
    except Exception as e:
        print(f"Training interrupted: {e}")
        # Save current model in the timestamped model_dir
        model.save(os.path.join(model_dir,"interrupted_sac_model"))
        print("Model saved despite interruption.")
    
    # Return the timestamped model_dir and log_dir for evaluation
    return model_dir, log_dir


def evaluate_model(model_path, num_episodes=5, results_save_path=None):
    """
    Evaluate a trained model and log detailed step-by-step data.
    
    Args:
        model_path (str): Path to the trained model (.zip file).
        num_episodes (int): Number of episodes to run for evaluation.
        results_save_path (str, optional): Directory to save evaluation logs and plots.
                                           If None, a new timestamped directory will be created.
    """
    # Load the trained model
    model = SAC.load(model_path)
    
    # Create a new environment for evaluation WITH GUI and OBSTACLES
    env = DroneNavigationEnv(gui=False, obstacles=True, initial_xyzs=np.array([[0., 0., 1.]]))
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
    
    # Determine the save path for evaluation results
    if results_save_path is None:
        # If no specific path is provided, create a new timestamped dir for evaluation
        eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_save_path = os.path.join("results/evaluation_logs", f"eval_{eval_timestamp}")
    
    os.makedirs(results_save_path, exist_ok=True) # Ensure the directory exists

    # Save detailed logs to a CSV file
    df = pd.DataFrame(all_episode_data)
    log_file_path = os.path.join(results_save_path, "detailed_evaluation_log.csv")
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
        plot_path = os.path.join(results_save_path, "episode_1_trajectory.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Trajectory plot saved to: {plot_path}")


if __name__ == "__main__":
    # Run training and get the timestamped directories
    trained_model_dir, trained_log_dir = main()
    
    # After training, evaluate the final model
    final_model_path = os.path.join(trained_model_dir, "final_sac_model.zip")
    best_model_path = os.path.join(trained_model_dir, "best_model.zip") # This is typically generated by EvalCallback
    
    # Try to find the best available model within the current training run's directory
    if os.path.exists(best_model_path):
        print("Found best model, evaluating...")
        # Pass the training run's log directory for evaluation results
        evaluate_model(best_model_path, num_episodes=3, results_save_path=trained_log_dir)
    elif os.path.exists(final_model_path):
        print("Found final model, evaluating...")
        # Pass the training run's log directory for evaluation results
        evaluate_model(final_model_path, num_episodes=3, results_save_path=trained_log_dir)
    else:
        print("No trained model found. Please check training logs.") 