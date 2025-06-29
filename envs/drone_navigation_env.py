#!/usr/bin/env python3
"""
Improved navigation script for obstacle avoidance drone using trained SAC model
Loads trained model and demonstrates navigation through obstacles
"""

import os
import sys
import numpy as np
import torch
import time
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from typing import List, Tuple, Optional, Dict


class ObstacleNavigationEnv(VelocityAviary):
    """
    Same environment class as in training for consistency
    """
    
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 num_obstacles: int = 8,
                 workspace_bounds: float = 4.0):
        
        # Environment parameters
        self.workspace_bounds = workspace_bounds
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.obstacle_ids = []
        
        # Waypoint parameters
        self.start_waypoint = np.array([0.0, 0.0, 1.0])
        self.end_waypoint = np.array([3.0, 3.0, 1.5])
        self.waypoint_threshold = 0.3
        
        # Safety parameters
        self.min_obstacle_distance = 0.5
        self.collision_threshold = 0.2
        self.max_episode_steps = 2000
        self.current_step = 0
        
        # Reward parameters
        self.success_reward = 100.0
        self.collision_penalty = -50.0
        self.progress_reward_scale = 20.0
        self.obstacle_avoidance_reward_scale = 5.0
        self.step_penalty = -0.02
        self.stability_reward_scale = 0.5
        
        # State tracking
        self.prev_distance_to_goal = None
        self.episode_rewards = []
        self.trajectory = []  # For visualization
        
        # Initialize parent
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs if initial_xyzs is not None else np.array([self.start_waypoint]),
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record
        )
        
        # Setup observation space dimensions before parent initialization
        self._original_obs_dim = None
        self._additional_obs_dim = 11  # waypoints + distances + obstacle info
        
        # Initialize parent
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs if initial_xyzs is not None else np.array([self.start_waypoint]),
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record
        )
        
    def _observationSpace(self):
        """Return the custom observation space."""
        # Get original observation space from parent
        if self._original_obs_dim is None:
            # Get parent observation space
            parent_obs_space = super()._observationSpace()
            if isinstance(parent_obs_space, spaces.Dict):
                # Handle dict observation space
                self._original_obs_dim = sum(space.shape[0] for space in parent_obs_space.spaces.values())
            else:
                self._original_obs_dim = parent_obs_space.shape[0]
        
        # Total dimension including additional features
        total_dim = self._original_obs_dim + self._additional_obs_dim
        
        # Create combined observation space
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)
        
        self._clear_obstacles()
        self._generate_obstacles()
        
        self.current_step = 0
        self.prev_distance_to_goal = np.linalg.norm(self.pos[0] - self.end_waypoint)
        self.episode_rewards = []
        self.trajectory = []
        
        enhanced_obs = self._create_enhanced_observation(obs)
        return enhanced_obs, info
    
    def step(self, action):
        """Execute step."""
        obs, _, terminated, truncated, info = super().step(action)
        
        # Record trajectory
        self.trajectory.append(self.pos[0].copy())
        
        reward = self._calculate_reward()
        self.episode_rewards.append(reward)
        
        terminated = self._check_termination()
        
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        
        enhanced_obs = self._create_enhanced_observation(obs)
        
        info.update({
            'success': self._is_success(),
            'collision': self._check_collision(),
            'collision_type': self._get_collision_type(),
            'out_of_bounds': self._check_out_of_bounds(),
            'distance_to_goal': np.linalg.norm(self.pos[0] - self.end_waypoint),
            'closest_obstacle_distance': self._get_closest_obstacle_distance(self.pos[0]),
            'episode_reward_sum': sum(self.episode_rewards),
            'current_position': self.pos[0].copy(),
            'target_position': self.end_waypoint.copy(),
            'episode_length': self.current_step,
            
            # Detailed reward breakdown
            'reward_breakdown': self.get_reward_breakdown(),
            'reward_info': self.get_reward_info(),
            
            # Performance metrics
            'linear_velocity': np.linalg.norm(self.vel[0]),
            'angular_velocity': np.linalg.norm(self.ang_v[0]),
            'altitude': self.pos[0][2],
            'progress_made': self.prev_distance_to_goal is not None and 
                           (self.prev_distance_to_goal - np.linalg.norm(self.pos[0] - self.end_waypoint)) > 0
        })
        
        return enhanced_obs, reward, terminated, truncated, info
    
    def _create_enhanced_observation(self, original_obs):
        """Create enhanced observation."""
        if isinstance(original_obs, dict):
            obs_flat = np.concatenate([v.flatten() for v in original_obs.values()])
        else:
            obs_flat = original_obs.flatten()
        
        current_pos = self.pos[0]
        distance_to_goal = np.linalg.norm(current_pos - self.end_waypoint)
        closest_obstacle_distance = self._get_closest_obstacle_distance(current_pos)
        closest_obstacle_pos = self._get_closest_obstacle_position(current_pos)
        relative_obstacle_pos = closest_obstacle_pos - current_pos if closest_obstacle_pos is not None else np.zeros(3)
        
        additional_features = np.array([
            self.start_waypoint[0] / self.workspace_bounds,
            self.start_waypoint[1] / self.workspace_bounds,
            self.start_waypoint[2] / self.workspace_bounds,
            self.end_waypoint[0] / self.workspace_bounds,
            self.end_waypoint[1] / self.workspace_bounds,
            self.end_waypoint[2] / self.workspace_bounds,
            distance_to_goal / (2 * self.workspace_bounds),
            min(closest_obstacle_distance / self.workspace_bounds, 1.0),
            relative_obstacle_pos[0] / self.workspace_bounds,
            relative_obstacle_pos[1] / self.workspace_bounds,
            relative_obstacle_pos[2] / self.workspace_bounds
        ], dtype=np.float32)
        
        enhanced_obs = np.concatenate([obs_flat, additional_features])
        return enhanced_obs.astype(np.float32)
    
    def _calculate_reward(self):
        """Calculate reward with detailed breakdown for logging."""
        current_pos = self.pos[0]
        reward_breakdown = {
            'total': 0.0,
            'success': 0.0,
            'collision': 0.0,
            'out_of_bounds': 0.0,
            'progress': 0.0,
            'obstacle_penalty': 0.0,
            'obstacle_safety': 0.0,
            'stability': 0.0,
            'step_penalty': 0.0,
            'velocity_penalty': 0.0
        }
        
        # Store additional info for logging
        reward_info = {
            'distance_to_goal': 0.0,
            'closest_obstacle_dist': 0.0,
            'angular_velocity': 0.0,
            'linear_velocity': 0.0,
            'altitude': current_pos[2],
            'is_success': False,
            'is_collision': False,
            'is_out_of_bounds': False
        }
        
        # 1. SUCCESS REWARD (terminal)
        if self._is_success():
            reward_breakdown['success'] = self.success_reward
            reward_breakdown['total'] = self.success_reward
            reward_info['is_success'] = True
            self.last_reward_breakdown = reward_breakdown
            self.last_reward_info = reward_info
            return self.success_reward
        
        # 2. COLLISION PENALTY (terminal)
        collision_type = self._get_collision_type()
        if collision_type:
            reward_breakdown['collision'] = self.collision_penalty
            reward_breakdown['total'] = self.collision_penalty
            reward_info['is_collision'] = True
            reward_info['collision_type'] = collision_type
            self.last_reward_breakdown = reward_breakdown
            self.last_reward_info = reward_info
            return self.collision_penalty
        
        # 3. OUT OF BOUNDS PENALTY (terminal)
        if self._check_out_of_bounds():
            reward_breakdown['out_of_bounds'] = -25.0
            reward_breakdown['total'] = -25.0
            reward_info['is_out_of_bounds'] = True
            self.last_reward_breakdown = reward_breakdown
            self.last_reward_info = reward_info
            return -25.0
        
        # 4. PROGRESS REWARD (dense)
        current_distance = np.linalg.norm(current_pos - self.end_waypoint)
        reward_info['distance_to_goal'] = current_distance
        
        if self.prev_distance_to_goal is not None:
            progress = self.prev_distance_to_goal - current_distance
            progress_reward = progress * self.progress_reward_scale
            reward_breakdown['progress'] = progress_reward
            reward_breakdown['total'] += progress_reward
        self.prev_distance_to_goal = current_distance
        
        # 5. OBSTACLE AVOIDANCE REWARD/PENALTY
        closest_obstacle_dist = self._get_closest_obstacle_distance(current_pos)
        reward_info['closest_obstacle_dist'] = closest_obstacle_dist
        
        if closest_obstacle_dist < self.min_obstacle_distance:
            # Penalty for being too close to obstacles
            penalty = (self.min_obstacle_distance - closest_obstacle_dist) / self.min_obstacle_distance
            obstacle_penalty = -penalty * self.obstacle_avoidance_reward_scale
            reward_breakdown['obstacle_penalty'] = obstacle_penalty
            reward_breakdown['total'] += obstacle_penalty
        elif closest_obstacle_dist < self.min_obstacle_distance * 2:
            # Small reward for maintaining safe distance
            safety_factor = (closest_obstacle_dist - self.min_obstacle_distance) / self.min_obstacle_distance
            safety_reward = safety_factor * (self.obstacle_avoidance_reward_scale * 0.2)
            reward_breakdown['obstacle_safety'] = safety_reward
            reward_breakdown['total'] += safety_reward
        
        # 6. STABILITY REWARD
        angular_vel_magnitude = np.linalg.norm(self.ang_v[0])
        reward_info['angular_velocity'] = angular_vel_magnitude
        
        if angular_vel_magnitude < 1.0:
            reward_breakdown['stability'] = self.stability_reward_scale
            reward_breakdown['total'] += self.stability_reward_scale
        
        # 7. VELOCITY PENALTY (for excessive speed)
        linear_vel_magnitude = np.linalg.norm(self.vel[0])
        reward_info['linear_velocity'] = linear_vel_magnitude
        
        if linear_vel_magnitude > 5.0:  # Too fast
            velocity_penalty = -(linear_vel_magnitude - 5.0) * 0.1
            reward_breakdown['velocity_penalty'] = velocity_penalty
            reward_breakdown['total'] += velocity_penalty
        
        # 8. STEP PENALTY (encourage efficiency)
        reward_breakdown['step_penalty'] = self.step_penalty
        reward_breakdown['total'] += self.step_penalty
        
        # Store for logging
        self.last_reward_breakdown = reward_breakdown
        self.last_reward_info = reward_info
        
        return reward_breakdown['total']
    
    def _get_collision_type(self):
        """Get detailed collision type for logging."""
        current_pos = self.pos[0]
        
        # Ground collision
        if current_pos[2] < 0.05:
            return "ground"
        
        # Obstacle collision
        closest_distance = self._get_closest_obstacle_distance(current_pos)
        if closest_distance < self.collision_threshold:
            return "obstacle"
        
        # High velocity crash
        velocity_magnitude = np.linalg.norm(self.vel[0])
        if velocity_magnitude > 15.0:
            return "high_velocity"
        
        # High angular velocity (loss of control)
        angular_vel_magnitude = np.linalg.norm(self.ang_v[0])
        if angular_vel_magnitude > 10.0:
            return "loss_of_control"
        
        return None
    
    def get_reward_breakdown(self):
        """Get the last reward breakdown for logging."""
        return getattr(self, 'last_reward_breakdown', {})
    
    def get_reward_info(self):
        """Get additional reward info for logging."""
        return getattr(self, 'last_reward_info', {})
    
    def _generate_obstacles(self):
        """Generate obstacles."""
        self.obstacles = []
        self.obstacle_ids = []
        
        for i in range(self.num_obstacles):
            while True:
                x = np.random.uniform(-self.workspace_bounds/2, self.workspace_bounds/2)
                y = np.random.uniform(-self.workspace_bounds/2, self.workspace_bounds/2)
                z = np.random.uniform(0.5, 2.5)
                pos = np.array([x, y, z])
                
                if (np.linalg.norm(pos - self.start_waypoint) > 1.0 and 
                    np.linalg.norm(pos - self.end_waypoint) > 1.0):
                    break
            
            size = np.random.uniform(0.2, 0.5)
            
            if self.GUI:
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[size, size, size],
                    rgbaColor=[0.8, 0.2, 0.2, 0.8],
                    physicsClientId=self.CLIENT
                )
                
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[size, size, size],
                    physicsClientId=self.CLIENT
                )
                
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=pos,
                    physicsClientId=self.CLIENT
                )
                
                self.obstacle_ids.append(obstacle_id)
            
            self.obstacles.append({
                'position': pos,
                'size': size,
                'id': obstacle_id if self.GUI else None
            })
    
    def _clear_obstacles(self):
        """Clear obstacles."""
        if self.GUI:
            for obstacle_id in self.obstacle_ids:
                try:
                    p.removeBody(obstacle_id, physicsClientId=self.CLIENT)
                except:
                    pass
        self.obstacles = []
        self.obstacle_ids = []
    
    def _get_closest_obstacle_distance(self, position):
        """Get distance to closest obstacle."""
        if not self.obstacles:
            return float('inf')
        
        min_distance = float('inf')
        for obstacle in self.obstacles:
            distance = np.linalg.norm(position - obstacle['position']) - obstacle['size']
            min_distance = min(min_distance, distance)
        
        return max(min_distance, 0.0)
    
    def _get_closest_obstacle_position(self, position):
        """Get position of closest obstacle."""
        if not self.obstacles:
            return None
        
        min_distance = float('inf')
        closest_pos = None
        
        for obstacle in self.obstacles:
            distance = np.linalg.norm(position - obstacle['position'])
            if distance < min_distance:
                min_distance = distance
                closest_pos = obstacle['position']
        
        return closest_pos
    
    def _check_collision(self):
        """Check collision."""
        current_pos = self.pos[0]
        
        if current_pos[2] < 0.05:
            return True
        
        closest_distance = self._get_closest_obstacle_distance(current_pos)
        if closest_distance < self.collision_threshold:
            return True
        
        velocity_magnitude = np.linalg.norm(self.vel[0])
        if velocity_magnitude > 15.0:
            return True
        
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
        distance = np.linalg.norm(self.pos[0] - self.end_waypoint)
        return distance < self.waypoint_threshold
    
    def _check_termination(self):
        """Check termination."""
        return self._is_success() or self._check_collision() or self._check_out_of_bounds()
    
    def get_trajectory(self):
        """Get recorded trajectory."""
        return np.array(self.trajectory) if self.trajectory else np.array([])
    
    def get_obstacles(self):
        """Get obstacle information."""
        return self.obstacles
    
    def render(self, mode='human'):
        """Render with waypoints."""
        if self.GUI:
            self._draw_waypoint(self.start_waypoint, color=[0, 1, 0])
            self._draw_waypoint(self.end_waypoint, color=[0, 0, 1])
            
            if hasattr(self, 'pos') and self.pos is not None:
                p.addUserDebugLine(
                    lineFromXYZ=self.start_waypoint,
                    lineToXYZ=self.pos[0],
                    lineColorRGB=[0.5, 0.5, 0.5],
                    lineWidth=2,
                    lifeTime=0.1,
                    physicsClientId=self.CLIENT
                )
        
        return super().render(mode)
    
    def _draw_waypoint(self, position, color=[1, 0, 0]):
        """Draw waypoint marker."""
        size = 0.1
        p.addUserDebugLine(
            lineFromXYZ=position - [size, 0, 0],
            lineToXYZ=position + [size, 0, 0],
            lineColorRGB=color,
            lineWidth=3,
            lifeTime=0.1,
            physicsClientId=self.CLIENT
        )
        p.addUserDebugLine(
            lineFromXYZ=position - [0, size, 0],
            lineToXYZ=position + [0, size, 0],
            lineColorRGB=color,
            lineWidth=3,
            lifeTime=0.1,
            physicsClientId=self.CLIENT
        )
        p.addUserDebugLine(
            lineFromXYZ=position - [0, 0, size],
            lineToXYZ=position + [0, 0, size],
            lineColorRGB=color,
            lineWidth=3,
            lifeTime=0.1,
            physicsClientId=self.CLIENT
        )


def make_env(gui=False, record=False, num_obstacles=8):
    """Create environment."""
    def _init():
        env = ObstacleNavigationEnv(
            gui=gui,
            record=record,
            num_obstacles=num_obstacles,
            workspace_bounds=4.0
        )
        return Monitor(env)
    return _init


def load_trained_model(model_path: str, norm_path: str = None):
    """Load trained SAC model and normalization."""
    if not os.path.exists(model_path):
        if not model_path.endswith('.zip'):
            model_path += '.zip'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"📁 Loading model from: {model_path}")
    model = SAC.load(model_path)
    
    vec_normalize = None
    if norm_path and os.path.exists(norm_path):
        print(f"📁 Loading normalization from: {norm_path}")
        # We'll load this when creating the environment
    elif norm_path:
        print(f"⚠️ Normalization file not found: {norm_path}")
    
    print("✅ Model loaded successfully!")
    return model, vec_normalize


def evaluate_model(model_path: str, 
                  norm_path: str = None,
                  episodes: int = 10,
                  gui: bool = True,
                  record: bool = False,
                  num_obstacles: int = 8,
                  save_plots: bool = True):
    """Evaluate trained model."""
    
    print("🚀 Starting Model Evaluation")
    print(f"📁 Model: {model_path}")
    print(f"🔄 Normalization: {norm_path}")
    print(f"🎮 Episodes: {episodes}")
    print(f"🖥️ GUI: {gui}")
    print(f"🧱 Obstacles: {num_obstacles}")
    
    # Load model
    model, _ = load_trained_model(model_path, norm_path)
    
    # Create environment
    eval_env = make_vec_env(
        make_env(gui=gui, record=record, num_obstacles=num_obstacles),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    
    # Load normalization if available
    if norm_path and os.path.exists(norm_path):
        eval_env = VecNormalize.load(norm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    print(f"\n🎯 Starting evaluation for {episodes} episodes...")
    
    for episode in range(episodes):
        obs = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n--- Episode {episode + 1}/{episodes} ---")
        
        start_time = time.time()
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, info = eval_env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            # Optional: add delay for visualization
            if gui:
                time.sleep(0.01)
            
            # Print progress every 200 steps
            if episode_length % 200 == 0:
                current_info = info[0]
                print(f"  Step {episode_length}: Distance to goal: {current_info.get('distance_to_goal', 'N/A'):.2f}")
        
        episode_time = time.time() - start_time
        
        # Get final episode info
        final_info = info[0]
        success = final_info.get('success', False)
        collision = final_info.get('collision', False)
        
        # Update counters
        if success:
            success_count += 1
            print(f"  ✅ SUCCESS! Reached target in {episode_length} steps")
        elif collision:
            collision_count += 1
            print(f"  💥 COLLISION after {episode_length} steps")
        else:
            timeout_count += 1
            print(f"  ⏰ TIMEOUT after {episode_length} steps")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Time: {episode_time:.1f}s")
        print(f"  Final distance to goal: {final_info.get('distance_to_goal', 'N/A'):.2f}")
        
        # Print detailed reward breakdown if available
        if 'reward_breakdown' in final_info and final_info['reward_breakdown']:
            breakdown = final_info['reward_breakdown']
            print(f"  Reward breakdown:")
            for component, value in breakdown.items():
                if abs(value) > 0.001:  # Only show non-zero components
                    print(f"    {component}: {value:+.3f}")
        
        # Print collision details
        if collision and 'collision_type' in final_info:
            print(f"  💥 Collision type: {final_info['collision_type']}")
        
        # Print performance metrics
        if 'reward_info' in final_info:
            reward_info = final_info['reward_info']
            print(f"  📊 Performance metrics:")
            print(f"    Linear velocity: {reward_info.get('linear_velocity', 0):.2f} m/s")
            print(f"    Angular velocity: {reward_info.get('angular_velocity', 0):.2f} rad/s")
            print(f"    Min obstacle distance: {reward_info.get('closest_obstacle_dist', 'N/A'):.3f}m")
        
        # Save trajectory plot if requested
        if save_plots:
            # Get environment for trajectory
            unwrapped_env = eval_env.envs[0].env.env  # Unwrap Monitor and VecEnv
            trajectory = unwrapped_env.get_trajectory()
            obstacles = unwrapped_env.get_obstacles()
            
            if len(trajectory) > 0:
                save_trajectory_plot(
                    trajectory, 
                    obstacles,
                    unwrapped_env.start_waypoint,
                    unwrapped_env.end_waypoint,
                    episode + 1, 
                    success,
                    collision
                )
    
    # Calculate summary statistics
    success_rate = success_count / episodes
    collision_rate = collision_count / episodes
    timeout_rate = timeout_count / episodes
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    # Print summary
    print(f"\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"📋 Episodes completed: {episodes}")
    print(f"✅ Success rate: {success_count}/{episodes} ({success_rate:.1%})")
    print(f"💥 Collision rate: {collision_count}/{episodes} ({collision_rate:.1%})")
    print(f"⏰ Timeout rate: {timeout_count}/{episodes} ({timeout_rate:.1%})")
    print(f"🏆 Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"📏 Average episode length: {avg_length:.1f} ± {std_length:.1f}")
    print("="*60)
    
    # Cleanup
    eval_env.close()
    
    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'std_length': std_length
    }


def save_trajectory_plot(trajectory: np.ndarray, obstacles: List[Dict], 
                        start_waypoint: np.ndarray, end_waypoint: np.ndarray,
                        episode: int, success: bool, collision: bool):
    """Save 3D trajectory plot with obstacles."""
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    if len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'b-', linewidth=3, alpha=0.8, label='Drone trajectory')
        
        # Mark start and end positions
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                  c='green', s=200, marker='^', label='Start position', alpha=0.9)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                  c='orange', s=200, marker='v', label='End position', alpha=0.9)
    
    # Plot waypoints
    ax.scatter(start_waypoint[0], start_waypoint[1], start_waypoint[2],
              c='green', s=300, marker='s', label='Start waypoint', alpha=0.7)
    ax.scatter(end_waypoint[0], end_waypoint[1], end_waypoint[2],
              c='blue', s=300, marker='s', label='Target waypoint', alpha=0.7)
    
    # Plot obstacles
    for i, obstacle in enumerate(obstacles):
        pos = obstacle['position']
        size = obstacle['size']
        
        # Create a cube for each obstacle
        ax.scatter(pos[0], pos[1], pos[2], c='red', s=500, 
                  marker='o', alpha=0.6, label='Obstacles' if i == 0 else "")
        
        # Draw obstacle boundaries (simplified as sphere)
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = size * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y_sphere = size * np.outer(np.sin(u), np.sin(v)) + pos[1]
        z_sphere = size * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='red')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Determine title based on outcome
    if success:
        title = f'Episode {episode}: SUCCESS ✅'
        title_color = 'green'
    elif collision:
        title = f'Episode {episode}: COLLISION 💥'
        title_color = 'red'
    else:
        title = f'Episode {episode}: TIMEOUT ⏰'
        title_color = 'orange'
    
    ax.set_title(title, fontsize=16, color=title_color, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Save plot
    os.makedirs('evaluation_plots', exist_ok=True)
    outcome = 'success' if success else ('collision' if collision else 'timeout')
    filename = f'evaluation_plots/episode_{episode:02d}_{outcome}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Trajectory plot saved: {filename}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained SAC model for obstacle navigation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--norm_path', type=str, default=None,
                       help='Path to normalization file')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--no_gui', action='store_true',
                       help='Run without GUI')
    parser.add_argument('--record', action='store_true',
                       help='Record video')
    parser.add_argument('--num_obstacles', type=int, default=8,
                       help='Number of obstacles')
    parser.add_argument('--no_plots', action='store_true',
                       help='Don\'t save trajectory plots')
    
    args = parser.parse_args()
    
    # Auto-detect normalization path if not provided
    if args.norm_path is None:
        model_dir = os.path.dirname(args.model_path)
        potential_norm_path = os.path.join(model_dir, 'vec_normalize.pkl')
        if os.path.exists(potential_norm_path):
            args.norm_path = potential_norm_path
            print(f"🔍 Auto-detected normalization file: {args.norm_path}")
    
    # Run evaluation
    results = evaluate_model(
        model_path=args.model_path,
        norm_path=args.norm_path,
        episodes=args.episodes,
        gui=not args.no_gui,
        record=args.record,
        num_obstacles=args.num_obstacles,
        save_plots=not args.no_plots
    )
    
    print(f"\n🎉 Evaluation completed!")
    if results['success_rate'] > 0.7:
        print("🌟 Excellent performance!")
    elif results['success_rate'] > 0.5:
        print("👍 Good performance!")
    else:
        print("⚠️ Performance needs improvement")


if __name__ == "__main__":
    main()