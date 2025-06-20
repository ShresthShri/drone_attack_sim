# agents/train_sac.py

import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit


from envs.drone_navigation_env import DroneNavigationEnv

def make_env(rank=0, seed=0):
    """
    Create a wrapped, monitored environment for training.
    
    Args:
        rank: The index of the subprocess
        seed: The inital seed for RNG
    """
    def _init():
        env = DroneNavigationEnv()
        env = TimeLimit(env, max_episode_steps=1000)  # Limit episode length
        env = Monitor(env, filename=None)  # Monitor wrapper for logging
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    # Create directories for saving models and logs
    model_dir = "models/sac_baseline"
    log_dir = "results/logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = DummyVecEnv([make_env(0)])
    env = VecMonitor(env, log_dir)  # Monitor wrapper for vectorized env

    # Initialize the SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
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
    
    # Create a separate environment for evaluation
    eval_env = DummyVecEnv([make_env(1)])
    eval_env = VecMonitor(eval_env, log_dir)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Train the agent
    print("Starting training...")
    print(f"Logging to {log_dir}")
    print(f"Models will be saved to {model_dir}")
    print("You can monitor the training progress using TensorBoard:")
    print(f"tensorboard --logdir {log_dir}")
    
    model.learn(
        total_timesteps=500_000,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save the final model
    model.save(os.path.join(model_dir, "final_sac_model"))
    print("Training completed!")
    print(f"Final model saved to {os.path.join(model_dir, 'final_sac_model')}")

def evaluate_model(model_path, num_episodes=5):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to evaluate
    """
    # Load the trained model
    model = SAC.load(model_path)
    
    # Create a new environment for evaluation
    env = DroneNavigationEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
        print(f"Episode {episode + 1} reward: {episode_reward}")

if __name__ == "__main__":
    main()
