#!/usr/bin/env python3
"""
Progressive difficulty training - start easy and increase challenge
"""

import os
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
from datetime import datetime


class ProgressiveDifficultyCallback(BaseCallback):
    """Callback to increase difficulty during training."""
    
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.last_difficulty_update = 0
        
    def _on_step(self) -> bool:
        # Increase difficulty every 200k steps
        if self.num_timesteps - self.last_difficulty_update >= 200000:
            if hasattr(self.env, 'envs'):
                for env in self.env.envs:
                    if hasattr(env.env, 'increase_difficulty'):
                        env.env.increase_difficulty()
            
            self.last_difficulty_update = self.num_timesteps
            if self.verbose > 0:
                print(f"\n🎯 Difficulty increased at step {self.num_timesteps}")
        
        return True


class ProgressiveObstacleNavigationEnv(VelocityAviary):
    """
    Progressive difficulty environment - starts easy, gets harder.
    """
    
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False):
        
        # Environment parameters
        self.workspace_bounds = 4.0
        self.obstacles = []
        self.obstacle_ids = []
        
        # Progressive difficulty parameters
        self.difficulty_level = 0  # Start at level 0 (easiest)
        self.max_difficulty = 3
        
        # Difficulty-dependent parameters
        self.num_obstacles_by_level = [0, 2, 4, 6]  # Start with NO obstacles
        self.obstacle_sizes_by_level = [
            [0.1, 0.15],  # Smaller obstacles at easier levels
            [0.15, 0.2], 
            [0.2, 0.3],
            [0.25, 0.35]
        ]
        
        # Waypoints - start closer, get further
        self.start_waypoint = np.array([-1.0, -3.0, 1.0])
        self.end_waypoints_by_level = [
            np.array([1.5, 1.5, 1.2]),  # Level 0: Very close
            np.array([2.0, 2.0, 1.3]),  # Level 1: Closer
            np.array([2.8, 2.8, 1.7]),  # Level 2: Medium
            np.array([3.5, 3.5, 2.0])   # Level 3: Far
        ]
        
        self.waypoint_threshold = 0.3
        
        # Safety parameters
        self.min_obstacle_distance = 0.4
        self.collision_threshold = 0.15
        self.max_episode_steps = 1000  # Shorter episodes for faster learning
        self.current_step = 0
        self.drone_radius = 0.1
        
        # OPTIMIZED REWARD PARAMETERS FOR EARLY LEARNING
        # Terminal rewards
        self.success_reward = 500.0  # VERY HIGH success reward
        self.collision_penalty = -20.0  # Low collision penalty
        self.oob_penalty = -15.0  # Low OOB penalty
        
        # Progress rewards (dominant signal)
        self.progress_reward_scale = 200.0  # MASSIVE progress reward
        self.retreat_penalty_scale = 50.0   # Moderate retreat penalty
        
        # Minimal penalties
        self.step_penalty = -0.0001  # Tiny step penalty
        self.distance_penalty_scale = 0.0  # No distance penalty
        
        # Obstacle avoidance
        self.obstacle_avoidance_scale = 10.0
        self.safe_distance_reward_scale = 1.0
        
        # Efficiency rewards
        self.velocity_reward_scale = 5.0
        self.efficiency_bonus_scale = 50.0
        
        # State tracking
        self.prev_distance_to_goal = None
        self.prev_pos = None
        self.initial_distance = None
        self.stagnation_counter = 0
        self.success_count = 0  # Track successes for difficulty progression
        
        # Initialize parent
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=np.array([self.start_waypoint]),
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record
        )
        
    def increase_difficulty(self):
        """Increase difficulty level."""
        if self.difficulty_level < self.max_difficulty:
            self.difficulty_level += 1
            print(f"🎯 Difficulty increased to level {self.difficulty_level}")
        
    def get_current_end_waypoint(self):
        """Get end waypoint for current difficulty."""
        return self.end_waypoints_by_level[self.difficulty_level]
    
    def _observationSpace(self):
        """Enhanced observation space."""
        parent_obs_space = super()._observationSpace()
        
        if isinstance(parent_obs_space, spaces.Box):
            original_dim = np.prod(parent_obs_space.shape)
        else:
            original_dim = 12
        
        additional_dim = 11
        total_dim = original_dim + additional_dim
        
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset with progressive difficulty."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Clear obstacles safely
        self._clear_obstacles_safe()
        
        # Generate obstacles based on current difficulty
        self._generate_progressive_obstacles()
        
        # Reset tracking
        self.current_step = 0
        current_end = self.get_current_end_waypoint()
        self.initial_distance = np.linalg.norm(self.pos[0] - current_end)
        self.prev_distance_to_goal = self.initial_distance
        self.prev_pos = self.pos[0].copy()
        self.stagnation_counter = 0
        
        return self._create_enhanced_observation(obs), info
    
    def step(self, action):
        """Execute step with improved reward structure."""
        obs, _, terminated, truncated, info = super().step(action)
        
        # Calculate reward
        reward = self._calculate_optimized_reward()
        
        # Check termination
        terminated = self._check_termination()
        
        # Track success for difficulty progression
        if self._is_success():
            self.success_count += 1
        
        # Check truncation
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        
        # Enhanced observation
        enhanced_obs = self._create_enhanced_observation(obs)
        
        # Update info
        current_end = self.get_current_end_waypoint()
        info.update({
            'success': self._is_success(),
            'collision': self._check_collision(),
            'distance_to_goal': np.linalg.norm(self.pos[0] - current_end),
            'episode_length': self.current_step,
            'difficulty_level': self.difficulty_level,
            'success_count': self.success_count
        })
        
        return enhanced_obs, reward, terminated, truncated, info
    
    def _create_enhanced_observation(self, original_obs):
        """Create enhanced observation."""
        obs_flat = original_obs.flatten() if hasattr(original_obs, 'flatten') else np.array(original_obs).flatten()
        
        current_pos = self.pos[0]
        current_vel = self.vel[0] if hasattr(self, 'vel') else np.zeros(3)
        current_end = self.get_current_end_waypoint()
        
        # Goal information
        goal_vector = current_end - current_pos
        goal_distance = np.linalg.norm(goal_vector)
        goal_direction = goal_vector / (goal_distance + 1e-8)
        
        # Progress metrics
        progress_ratio = self._get_progress_ratio()
        
        # Velocity toward goal
        velocity_toward_goal = np.dot(current_vel, goal_direction)
        
        # Closest obstacle info
        closest_obstacle_info = self._get_closest_obstacle_info(current_pos)
        
        # Efficiency metric
        efficiency = self._get_efficiency_metric()
        
        additional_features = np.array([
            # Goal direction (normalized)
            goal_direction[0], goal_direction[1], goal_direction[2],
            # Goal distance (normalized)
            goal_distance / self.initial_distance,
            # Progress ratio
            progress_ratio,
            # Velocity alignment with goal
            np.clip(velocity_toward_goal, -2.0, 2.0),
            # Closest obstacle info
            closest_obstacle_info[0], closest_obstacle_info[1],
            closest_obstacle_info[2], closest_obstacle_info[3],
            # Efficiency metric
            efficiency
        ], dtype=np.float32)
        
        return np.concatenate([obs_flat, additional_features])
    
    def _calculate_optimized_reward(self):
        """Optimized reward function for faster learning."""
        current_pos = self.pos[0]
        current_vel = self.vel[0] if hasattr(self, 'vel') else np.zeros(3)
        current_end = self.get_current_end_waypoint()
        total_reward = 0.0
        
        current_distance = np.linalg.norm(current_pos - current_end)
        
        # 1. TERMINAL REWARDS
        if self._is_success():
            # Massive success reward with difficulty bonus
            difficulty_bonus = self.difficulty_level * 100.0
            efficiency_bonus = self.efficiency_bonus_scale * self._get_efficiency_metric()
            total_reward = self.success_reward + difficulty_bonus + efficiency_bonus
            print(f"SUCCESS! Level {self.difficulty_level}, Base: {self.success_reward}, "
                  f"Difficulty: +{difficulty_bonus}, Efficiency: +{efficiency_bonus:.1f}, "
                  f"Total: {total_reward:.1f}")
            return total_reward
        
        if self._check_collision():
            return self.collision_penalty
        
        if self._check_out_of_bounds():
            return self.oob_penalty
        
        # 2. MASSIVE PROGRESS REWARD (main learning signal)
        if self.prev_distance_to_goal is not None:
            progress = self.prev_distance_to_goal - current_distance
            if progress > 0:
                # HUGE reward for any progress toward goal
                progress_reward = progress * self.progress_reward_scale
                total_reward += progress_reward
            elif progress < -0.01:  # Only penalize significant retreat
                retreat_penalty = progress * self.retreat_penalty_scale
                total_reward += retreat_penalty
        
        self.prev_distance_to_goal = current_distance
        
        # 3. VELOCITY ALIGNMENT REWARD (encourage right direction)
        goal_direction = (current_end - current_pos)
        goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
        velocity_alignment = np.dot(current_vel, goal_direction)
        velocity_reward = max(0, velocity_alignment) * self.velocity_reward_scale  # Only positive alignment
        total_reward += velocity_reward
        
        # 4. DISTANCE-BASED MOTIVATION (closer to goal = higher baseline reward)
        # Give higher baseline reward for being closer to goal
        max_distance = np.linalg.norm(self.get_current_end_waypoint() - self.start_waypoint)
        proximity_reward = (1.0 - current_distance / max_distance) * 2.0
        total_reward += proximity_reward
        
        # 5. OBSTACLE AVOIDANCE (only when obstacles exist)
        if self.obstacles:
            closest_dist = self._get_closest_obstacle_distance(current_pos)
            if closest_dist < self.min_obstacle_distance:
                danger_factor = (self.min_obstacle_distance - closest_dist) / self.min_obstacle_distance
                obstacle_penalty = -danger_factor * self.obstacle_avoidance_scale
                total_reward += obstacle_penalty
            elif closest_dist < self.min_obstacle_distance * 2.0:
                safety_factor = (closest_dist - self.min_obstacle_distance) / self.min_obstacle_distance
                safety_reward = safety_factor * self.safe_distance_reward_scale
                total_reward += safety_reward
        
        # 6. MINIMAL EFFICIENCY PENALTIES
        total_reward += self.step_penalty
        
        # 7. ANTI-STAGNATION
        if self.prev_pos is not None:
            movement = np.linalg.norm(current_pos - self.prev_pos)
            if movement < 0.003:
                self.stagnation_counter += 1
                if self.stagnation_counter > 20:
                    total_reward += -0.1
            else:
                self.stagnation_counter = 0
        
        self.prev_pos = current_pos.copy()
        
        return total_reward
    
    def _generate_progressive_obstacles(self):
        """Generate obstacles based on difficulty level."""
        self.obstacles = []
        self.obstacle_ids = []
        
        num_obstacles = self.num_obstacles_by_level[self.difficulty_level]
        if num_obstacles == 0:
            return  # No obstacles at difficulty 0
        
        current_end = self.get_current_end_waypoint()
        obstacle_size_range = self.obstacle_sizes_by_level[self.difficulty_level]
        
        # Simple obstacle placement - not blocking direct path initially
        for i in range(num_obstacles):
            max_attempts = 20
            
            for attempt in range(max_attempts):
                # Generate position away from direct path
                if self.difficulty_level <= 1:
                    # Easy placement - well away from direct path
                    x = np.random.uniform(-2.0, 2.0)
                    y = np.random.uniform(-2.0, 2.0)
                    z = np.random.uniform(0.5, 2.0)
                    pos = np.array([x, y, z])
                    
                    # Ensure not too close to waypoints
                    start_dist = np.linalg.norm(pos - self.start_waypoint)
                    end_dist = np.linalg.norm(pos - current_end)
                    
                    if start_dist > 1.0 and end_dist > 1.0:
                        # Check distance from direct path
                        path_vector = current_end - self.start_waypoint
                        path_length = np.linalg.norm(path_vector)
                        path_direction = path_vector / path_length
                        
                        start_to_pos = pos - self.start_waypoint
                        projection_length = np.dot(start_to_pos, path_direction)
                        projection_point = self.start_waypoint + projection_length * path_direction
                        path_distance = np.linalg.norm(pos - projection_point)
                        
                        if path_distance > 0.8:  # Well away from path
                            break
                else:
                    # Higher difficulty - can be closer to path
                    x = np.random.uniform(-self.workspace_bounds/2, self.workspace_bounds/2)
                    y = np.random.uniform(-self.workspace_bounds/2, self.workspace_bounds/2)
                    z = np.random.uniform(0.5, 2.5)
                    pos = np.array([x, y, z])
                    
                    start_dist = np.linalg.norm(pos - self.start_waypoint)
                    end_dist = np.linalg.norm(pos - current_end)
                    
                    if start_dist > 0.7 and end_dist > 0.7:
                        break
            else:
                # Fallback position
                angle = i * 2 * np.pi / num_obstacles
                pos = np.array([
                    1.5 * np.cos(angle),
                    1.5 * np.sin(angle),
                    1.0 + i * 0.2
                ])
            
            # Obstacle size
            size = np.random.uniform(*obstacle_size_range)
            
            # Create obstacle
            obstacle_id = self._create_obstacle(pos, size)
            if obstacle_id is not None:
                self.obstacle_ids.append(obstacle_id)
                self.obstacles.append({
                    'position': pos,
                    'size': size,
                    'id': obstacle_id
                })
    
    def _create_obstacle(self, pos, size):
        """Safely create obstacle."""
        try:
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size, size, size],
                physicsClientId=self.CLIENT
            )
            
            if self.GUI:
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[size, size, size],
                    rgbaColor=[0.8, 0.2, 0.2, 0.8],
                    physicsClientId=self.CLIENT
                )
                
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=pos,
                    physicsClientId=self.CLIENT
                )
            else:
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    basePosition=pos,
                    physicsClientId=self.CLIENT
                )
            
            return obstacle_id
            
        except Exception as e:
            print(f"Warning: Could not create obstacle: {e}")
            return None
    
    def _clear_obstacles_safe(self):
        """Safely clear obstacles."""
        for obstacle_id in self.obstacle_ids:
            try:
                p.removeBody(obstacle_id, physicsClientId=self.CLIENT)
            except Exception as e:
                # Ignore removal errors
                pass
        
        self.obstacles = []
        self.obstacle_ids = []
    
    def _get_progress_ratio(self):
        """Calculate progress ratio."""
        if self.initial_distance is None:
            return 0.0
        
        current_end = self.get_current_end_waypoint()
        current_distance = np.linalg.norm(self.pos[0] - current_end)
        progress = 1.0 - (current_distance / self.initial_distance)
        return np.clip(progress, 0.0, 1.0)
    
    def _get_efficiency_metric(self):
        """Calculate efficiency."""
        if self.current_step == 0:
            return 1.0
        
        current_end = self.get_current_end_waypoint()
        direct_distance = np.linalg.norm(current_end - self.start_waypoint)
        time_factor = self.current_step / 800.0  # Target 800 steps
        efficiency = direct_distance / (direct_distance + time_factor * 1.0)
        return np.clip(efficiency, 0.0, 1.0)
    
    def _get_closest_obstacle_info(self, position):
        """Get closest obstacle info."""
        if not self.obstacles:
            return np.array([10.0, 0.0, 0.0, 0.0])
        
        min_distance = float('inf')
        closest_relative_pos = np.zeros(3)
        
        for obstacle in self.obstacles:
            center_distance = np.linalg.norm(position - obstacle['position'])
            surface_distance = center_distance - obstacle['size'] - self.drone_radius
            
            if surface_distance < min_distance:
                min_distance = surface_distance
                closest_relative_pos = obstacle['position'] - position
        
        closest_relative_pos = closest_relative_pos / self.workspace_bounds
        
        return np.array([
            np.clip(min_distance / self.workspace_bounds, 0.0, 1.0),
            closest_relative_pos[0], closest_relative_pos[1], closest_relative_pos[2]
        ])
    
    def _get_closest_obstacle_distance(self, position):
        """Get distance to closest obstacle."""
        if not self.obstacles:
            return float('inf')
        
        min_distance = float('inf')
        for obstacle in self.obstacles:
            center_distance = np.linalg.norm(position - obstacle['position'])
            surface_distance = center_distance - obstacle['size'] - self.drone_radius
            min_distance = min(min_distance, surface_distance)
        
        return max(min_distance, 0.0)
    
    def _check_collision(self):
        """Check collision."""
        current_pos = self.pos[0]
        
        # Ground collision
        if current_pos[2] < 0.05:
            return True
        
        # Obstacle collision
        if self.obstacles:
            closest_distance = self._get_closest_obstacle_distance(current_pos)
            return closest_distance < self.collision_threshold
        
        return False
    
    def _check_out_of_bounds(self):
        """Check bounds."""
        pos = self.pos[0]
        bounds = self.workspace_bounds
        
        return (abs(pos[0]) > bounds or 
                abs(pos[1]) > bounds or 
                pos[2] > bounds or 
                pos[2] < 0.0)
    
    def _is_success(self):
        """Check success."""
        current_end = self.get_current_end_waypoint()
        distance = np.linalg.norm(self.pos[0] - current_end)
        return distance < self.waypoint_threshold
    
    def _check_termination(self):
        """Check termination."""
        return self._is_success() or self._check_collision() or self._check_out_of_bounds()


def make_progressive_env(gui=False):
    """Create progressive environment."""
    def _init():
        env = ProgressiveObstacleNavigationEnv(gui=gui)
        return Monitor(env)
    return _init


def train_progressive_drone_navigation():
    """Training with progressive difficulty."""
    
    config = {
        'total_timesteps': 1500000,  # Longer training for progression
        'learning_rate': 5e-4,  # Higher learning rate for faster convergence
        'buffer_size': 500000,  # Smaller buffer for faster turnover
        'batch_size': 128,  # Smaller batches for more frequent updates
        'learning_starts': 3000,  # Start learning very early
        'eval_freq': 10000,  # More frequent evaluation
        'n_eval_episodes': 5,  # Fewer eval episodes for speed
        'reward_threshold': 1500.0,  # Higher threshold for better performance
        'n_envs': 8
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/progressive_drone_nav_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("🚀 Starting Progressive Difficulty Drone Training")
    print(f"📁 Save directory: {save_dir}")
    print(f"🎯 Configuration: {config}")
    
    # Create environments
    train_env = make_vec_env(
        make_progressive_env(gui=False),
        n_envs=config['n_envs'],
        vec_env_cls=DummyVecEnv
    )
    
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )
    
    eval_env = make_vec_env(
        make_progressive_env(gui=False),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False
    )
    
    # Create SAC model optimized for fast learning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        learning_starts=config['learning_starts'],
        tau=0.01,  # Faster target updates
        gamma=0.99,  # Standard discount factor
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(
            net_arch=[256, 256],  # Smaller network for faster training
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        device=device,
        ent_coef='auto',
        target_update_interval=1,
        use_sde=True,
        sde_sample_freq=8  # Less frequent noise sampling
    )
    
    # Callbacks
    progressive_callback = ProgressiveDifficultyCallback(train_env, verbose=1)
    
    # stop_callback = StopTrainingOnRewardThreshold(
    #     reward_threshold=config['reward_threshold'],
    #     verbose=1
    # )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        deterministic=True,
        verbose=1,
        #callback_on_new_best=stop_callback
    )
    
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callback_list = CallbackList([progressive_callback, eval_callback])
   
    # Train
    print("\n🎓 Starting progressive training...")
    print("📊 Level 0: No obstacles, close goal")
    print("📊 Level 1: 2 obstacles, medium goal") 
    print("📊 Level 2: 4 obstacles, far goal")
    print("📊 Level 3: 6 obstacles, very far goal")
    
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callback_list,
            log_interval=4,
            progress_bar=True
        )
        print("✅ Training completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
    
    finally:
        model.save(f"{save_dir}/final_model")
        train_env.save(f"{save_dir}/vec_normalize.pkl")
        print(f"💾 Model saved to: {save_dir}")
        
        train_env.close()
        eval_env.close()
    
    return model, save_dir


if __name__ == "__main__":
    train_progressive_drone_navigation()