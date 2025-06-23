# agents/train_sac.py

import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold

from gymnasium.wrappers import TimeLimit


from envs.drone_navigation_env import DroneNavigationEnv

def make_env(rank=0, seed=0,gui = False, max_steps = 1000):
    """
    Create a wrapped, monitored environment for training.
    
    Args:
        rank: The index of the subprocess
        seed: The inital seed for RNG
    """
    def _init():
        env = DroneNavigationEnv(gui = gui)
        env = TimeLimit(env, max_episode_steps=max_steps)  # Limit episode length
        env = Monitor(env)  # Monitor wrapper for logging
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
    NUM_ENVS = 4
    env = DummyVecEnv([make_env(rank = i, gui = False, max_steps = 300) for i in range(NUM_ENVS)])
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
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=log_dir
    )

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="sac_model"
    )
    
    # Create a separate environment for evaluation
    eval_env = DummyVecEnv([make_env(rank = 0, gui = False, max_steps=300)])
    eval_env = VecMonitor(eval_env, log_dir)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    # Stop if eval reward > threshold
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=3,  # <-- set to a meaningful reward level for your task
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback  # <- attach it here
    )

    # Start training
    print ("Training SAC agent (parallel,headless)..")
    print(f"Tensorboard log: tensorboard --logdir {log_dir}")
    model.learn(
        total_timesteps=500_000,
        callback=[checkpoint_callback, eval_callback]
    )
    model.save(os.path.join(model_dir,"final_sac_model"))
    print("Training done.")


    #=========================== EVALUATION==========================
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
    env = DroneNavigationEnv(gui = True)
    env = TimeLimit(env, max_episode_steps=1000)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
        print(f"Episode {episode + 1} reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
