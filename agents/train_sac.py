import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
from tqdm import tqdm

from envs.drone_navigation_env import DroneNavigationEnv


class ProgressBarCallback(BaseCallback):
    """Custom callback that displays a progress bar during training."""
    
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        
    def _on_training_start(self) -> None:
        """Initialize progress bar when training starts."""
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training SAC",
            unit="steps",
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
    def _on_step(self) -> bool:
        """Update progress bar on each step."""
        if self.pbar is not None:
            # Update progress bar
            self.pbar.update(1)
            
            # Update postfix with current metrics if available
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                postfix_dict = {}
                if 'rollout/ep_rew_mean' in self.model.logger.name_to_value:
                    postfix_dict['reward'] = f"{self.model.logger.name_to_value['rollout/ep_rew_mean']:.2f}"
                if 'rollout/ep_len_mean' in self.model.logger.name_to_value:
                    postfix_dict['ep_len'] = f"{self.model.logger.name_to_value['rollout/ep_len_mean']:.0f}"
                if 'train/learning_rate' in self.model.logger.name_to_value:
                    postfix_dict['lr'] = f"{self.model.logger.name_to_value['train/learning_rate']:.2e}"
                    
                if postfix_dict:
                    self.pbar.set_postfix(postfix_dict)
        
        return True
    
    def _on_training_end(self) -> None:
        """Close progress bar when training ends."""
        if self.pbar is not None:
            self.pbar.close()


class TrainingCallback:
    """Custom callback for monitoring training progress."""
    
    def __init__(self, eval_freq=1000, save_freq=5000, save_path="./models/"):
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
    def setup_callbacks(self, eval_env, total_timesteps):
        """Setup evaluation, checkpoint, and progress bar callbacks."""
        # Progress bar callback
        progress_callback = ProgressBarCallback(total_timesteps)
        
        # Stop training when reward threshold is reached
        reward_threshold_callback = StopTrainingOnRewardThreshold(
            reward_threshold=80.0,  # Adjust based on your reward scale
            verbose=1
        )
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.save_path,
            log_path=self.save_path,
            eval_freq=self.eval_freq,
            deterministic=True,
            render=False,
            callback_on_new_best=reward_threshold_callback,
            verbose=1
        )
        
        return [progress_callback, eval_callback]


def create_env(gui=False, record=False):
    """Create the drone navigation environment."""
    return DroneNavigationEnv(
        gui=gui,
        record=record,
        pyb_freq=240,  # Fixed frequency to avoid issues
        ctrl_freq=30   # Lower control frequency for stability
    )


def train_sac_drone():
    """Train SAC agent for drone navigation."""
    
    # Training parameters
    total_timesteps = 500000
    learning_rate = 3e-4
    buffer_size = 100000
    batch_size = 256
    tau = 0.005
    gamma = 0.99
    train_freq = 1
    gradient_steps = 1
    
    # Create training environment
    print("Creating training environment...")
    train_env = make_vec_env(
        lambda: Monitor(create_env(gui=False, record=False)),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    
    # Normalize observations and rewards
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        lambda: Monitor(create_env(gui=False, record=False)),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards for evaluation
        clip_obs=10.0,
        training=False
    )
    
    # Setup callbacks
    callback_manager = TrainingCallback(
        eval_freq=2000,
        save_freq=10000,
        save_path="./models/"
    )
    callbacks = callback_manager.setup_callbacks(eval_env, total_timesteps)
    
    # Create SAC model
    print("Creating SAC model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        learning_starts=1000,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device=device
    )
    
    print("Starting training...")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Device: {device}")
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name="SAC_DroneNavigation"
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Close progress bar if interrupted
        for callback in callbacks:
            if isinstance(callback, ProgressBarCallback):
                callback._on_training_end()
    
    # Save final model
    print("Saving final model...")
    model.save("./models/sac_drone_navigation_final")
    train_env.save("./models/vec_normalize.pkl")
    
    print("Training completed!")
    return model, train_env


def test_trained_model(model_path="./models/best_model.zip", 
                      norm_path="./models/vec_normalize.pkl",
                      episodes=5):
    """Test the trained model."""
    
    print(f"Loading model from {model_path}...")
    
    # Create test environment
    test_env = create_env(gui=False, record=False)
    test_env = Monitor(test_env)
    test_env = DummyVecEnv([lambda: test_env])
    
    # Load normalization parameters
    if os.path.exists(norm_path):
        test_env = VecNormalize.load(norm_path, test_env)
        test_env.training = False
        test_env.norm_reward = False
    
    # Load model
    model = SAC.load(model_path, env=test_env)
    
    # Test episodes
    total_rewards = []
    success_count = 0
    
    for episode in range(episodes):
        obs = test_env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        while not done and step_count < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            episode_reward += reward[0]
            step_count += 1
            
            if info[0].get('success', False):
                success_count += 1
                print(f"Success! Reached target in {step_count} steps")
                break
                
        total_rewards.append(episode_reward)
        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Distance to target: {info[0].get('distance_to_target', 'N/A'):.3f}")
    
    print(f"\nTest Results:")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    
    test_env.close()


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SAC agent for drone navigation')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                       help='Mode: train or test')
    parser.add_argument('--model_path', type=str, default='./models/best_model.zip',
                       help='Path to saved model for testing')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of test episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model, env = train_sac_drone()
        
        # Test the trained model
        print("\nTesting trained model...")
        test_trained_model(episodes=3)
        
    elif args.mode == 'test':
        if not os.path.exists(args.model_path):
            print(f"Model file {args.model_path} not found!")
            return
        
        test_trained_model(
            model_path=args.model_path,
            episodes=args.episodes
        )


if __name__ == "__main__":
    main()