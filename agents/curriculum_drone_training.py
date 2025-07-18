import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
import time
import json
import os
from collections import deque
from typing import Dict, List, Tuple, Any

class CurriculumConfig:
    """Centralized curriculum configuration"""
    
    STAGES = {
        'basic_navigation': {
            'environment': {
                'max_distance_range': (3.0, 8.0),
                'obstacles': [],
                'goal_threshold': 1.0,
                'episode_length': 500,
                'randomization_level': 0.3
            },
            'rewards': {
                'progress_weight': 100.0,
                'time_penalty': -0.01,
                'collision_penalty': -20.0,
                'proximity_weight': 0.5,
                'success_bonus': 200.0
            },
            'observations': {
                'components': ['position', 'velocity', 'goal_relative'],
                'normalize': True,
                'noise_level': 0.0
            },
            'hyperparams': {
                'learning_rate': 3e-4,
                'batch_size': 128,
                'buffer_size': 100000,
                'exploration_noise': 0.3,
                'tau': 0.005
            },
            'success_criteria': {
                'success_rate': 0.85,
                'stability_episodes': 200,
                'min_episodes': 5000
            }
        },
        
        'obstacle_avoidance': {
            'environment': {
                'max_distance_range': (8.0, 15.0),
                'obstacles': [
                    {'type': 'simple_wall', 'count': 1, 'complexity': 0.3}
                ],
                'goal_threshold': 0.8,
                'episode_length': 600,
                'randomization_level': 0.5
            },
            'rewards': {
                'progress_weight': 75.0,
                'time_penalty': -0.03,
                'collision_penalty': -40.0,
                'proximity_weight': 1.0,
                'success_bonus': 150.0
            },
            'observations': {
                'components': ['position', 'velocity', 'goal_relative', 'nearest_obstacle'],
                'normalize': True,
                'noise_level': 0.01
            },
            'hyperparams': {
                'learning_rate': 2e-4,
                'batch_size': 128,
                'buffer_size': 100000,
                'exploration_noise': 0.25,
                'tau': 0.01
            },
            'success_criteria': {
                'success_rate': 0.80,
                'stability_episodes': 300,
                'min_episodes': 8000
            }
        },
        
        'complex_navigation': {
            'environment': {
                'max_distance_range': (12.0, 20.0),
                'obstacles': [
                    {'type': 'corridor', 'count': 1, 'complexity': 0.6},
                    {'type': 'scattered', 'count': 2, 'complexity': 0.4}
                ],
                'goal_threshold': 0.6,
                'episode_length': 800,
                'randomization_level': 0.7
            },
            'rewards': {
                'progress_weight': 50.0,
                'time_penalty': -0.05,
                'collision_penalty': -60.0,
                'proximity_weight': 1.5,
                'success_bonus': 120.0
            },
            'observations': {
                'components': ['position', 'velocity', 'goal_relative', 'obstacle_map', 'collision_risk'],
                'normalize': True,
                'noise_level': 0.02
            },
            'hyperparams': {
                'learning_rate': 1e-4,
                'batch_size': 64,
                'buffer_size': 150000,
                'exploration_noise': 0.2,
                'tau': 0.02
            },
            'success_criteria': {
                'success_rate': 0.75,
                'stability_episodes': 400,
                'min_episodes': 12000
            }
        },
        
        'expert_level': {
            'environment': {
                'max_distance_range': (15.0, 25.0),
                'obstacles': [
                    {'type': 'maze', 'count': 1, 'complexity': 0.8},
                    {'type': 'dynamic', 'count': 1, 'complexity': 0.5}
                ],
                'goal_threshold': 0.5,
                'episode_length': 1000,
                'randomization_level': 1.0
            },
            'rewards': {
                'progress_weight': 30.0,
                'time_penalty': -0.08,
                'collision_penalty': -80.0,
                'proximity_weight': 2.0,
                'success_bonus': 100.0
            },
            'observations': {
                'components': ['position', 'velocity', 'goal_relative', 'obstacle_map', 
                             'collision_risk', 'path_history'],
                'normalize': True,
                'noise_level': 0.03
            },
            'hyperparams': {
                'learning_rate': 5e-5,
                'batch_size': 64,
                'buffer_size': 200000,
                'exploration_noise': 0.15,
                'tau': 0.03
            },
            'success_criteria': {
                'success_rate': 0.70,
                'stability_episodes': 500,
                'min_episodes': 15000
            }
        }
    }
    
    STAGE_ORDER = ['basic_navigation', 'obstacle_avoidance', 'complex_navigation', 'expert_level']

class PerformanceTracker:
    """Tracks performance metrics and determines progression readiness"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.success_history = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)
        self.episode_length_history = deque(maxlen=window_size)
        self.total_episodes = 0
        
    def update(self, success: bool, reward: float, episode_length: int):
        self.success_history.append(success)
        self.reward_history.append(reward)
        self.episode_length_history.append(episode_length)
        self.total_episodes += 1
        
    def get_metrics(self) -> Dict[str, float]:
        if len(self.success_history) == 0:
            return {'success_rate': 0.0, 'avg_reward': 0.0, 'stability': 0.0}
            
        success_rate = np.mean(self.success_history)
        avg_reward = np.mean(self.reward_history)
        
        # Calculate stability as inverse of variance in success rate
        if len(self.success_history) >= 20:
            recent_successes = list(self.success_history)[-20:]
            stability = 1.0 / (1.0 + np.var(recent_successes))
        else:
            stability = 0.0
            
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'stability': stability,
            'total_episodes': self.total_episodes
        }
        
    def should_advance(self, criteria: Dict[str, Any]) -> bool:
        metrics = self.get_metrics()
        
        # Check minimum episodes
        if metrics['total_episodes'] < criteria['min_episodes']:
            return False
            
        # Check success rate
        if metrics['success_rate'] < criteria['success_rate']:
            return False
            
        # Check stability (need sufficient data and stability)
        if len(self.success_history) < criteria['stability_episodes']:
            return False
            
        # Check recent performance stability
        recent_window = min(criteria['stability_episodes'], len(self.success_history))
        recent_successes = list(self.success_history)[-recent_window:]
        recent_success_rate = np.mean(recent_successes)
        
        return recent_success_rate >= criteria['success_rate']

class AdaptiveRewardSystem:
    """Dynamically adjusts rewards based on curriculum stage and progress"""
    
    def __init__(self, stage_config: Dict[str, Any]):
        self.config = stage_config['rewards']
        self.transition_factor = 0.0
        self.next_stage_config = None
        
    def set_transition(self, factor: float, next_config: Dict[str, Any]):
        self.transition_factor = factor
        self.next_stage_config = next_config['rewards'] if next_config else None
        
    def calculate_reward(self, state_info: Dict[str, Any]) -> float:
        """Calculate adaptive reward based on current stage and transition"""
        
        # Base reward components
        progress_reward = state_info.get('progress', 0) * self.config['progress_weight']
        time_penalty = self.config['time_penalty']
        collision_penalty = self.config['collision_penalty'] if state_info.get('collision', False) else 0
        success_bonus = self.config['success_bonus'] if state_info.get('success', False) else 0
        
        # Proximity reward (context-dependent)
        proximity_reward = self._calculate_proximity_reward(state_info)
        
        base_reward = progress_reward + time_penalty + collision_penalty + success_bonus + proximity_reward
        
        # Apply transition blending if approaching next stage
        if self.transition_factor > 0 and self.next_stage_config:
            next_progress = state_info.get('progress', 0) * self.next_stage_config['progress_weight']
            next_collision = self.next_stage_config['collision_penalty'] if state_info.get('collision', False) else 0
            next_success = self.next_stage_config['success_bonus'] if state_info.get('success', False) else 0
            next_proximity = self._calculate_proximity_reward(state_info, self.next_stage_config)
            
            next_reward = next_progress + self.next_stage_config['time_penalty'] + next_collision + next_success + next_proximity
            
            # Blend current and next stage rewards
            base_reward = (1 - self.transition_factor) * base_reward + self.transition_factor * next_reward
            
        return base_reward
        
    def _calculate_proximity_reward(self, state_info: Dict[str, Any], config: Dict = None) -> float:
        if config is None:
            config = self.config
            
        proximity_distance = state_info.get('min_obstacle_distance', float('inf'))
        weight = config['proximity_weight']
        
        if proximity_distance < 0.5:
            return -weight * 10 * (0.5 - proximity_distance)
        elif proximity_distance < 1.0:
            return -weight * 2 * (1.0 - proximity_distance)
        elif proximity_distance > 2.0:
            return weight * 1.0
        else:
            return 0

class AdaptiveDroneEnv(gym.Env):
    """Drone environment that adapts based on curriculum stage"""
    
    def __init__(self, stage_config: Dict[str, Any]):
        super(AdaptiveDroneEnv, self).__init__()
        
        self.stage_config = stage_config
        self.env_config = stage_config['environment']
        self.obs_config = stage_config['observations']
        
        # Physical parameters
        self.max_speed = 4.0
        self.max_force = 15.0
        self.gravity = 9.8
        self.mass = 1.0
        self.k_p = 5.0
        self.k_d = 1.0
        
        # Stage-specific parameters
        self.goal_threshold = self.env_config['goal_threshold']
        self.episode_length = self.env_config['episode_length']
        self.min_altitude = 0.3
        self.max_altitude = 8.0
        
        # Action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Adaptive observation space
        self._setup_observation_space()
        
        # PyBullet setup
        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(fixedTimeStep=1/120, numSubSteps=2)
        
        # Environment state
        self.drone = None
        self.obstacles = []
        self.obstacle_params = []
        self.current_step = 0
        self.start_pos = None
        self.goal_pos = None
        self.prev_distance = None
        self.prev_velocity = np.zeros(3)
        self.best_distance = None
        self.initial_distance = None
        self.path_history = deque(maxlen=10)
        
        # Reward system
        self.reward_system = AdaptiveRewardSystem(stage_config)
        
    def _setup_observation_space(self):
        """Setup observation space based on stage configuration"""
        components = self.obs_config['components']
        obs_dim = 0
        
        # Calculate observation dimension
        for component in components:
            if component in ['position', 'velocity', 'goal_relative']:
                obs_dim += 3
            elif component == 'nearest_obstacle':
                obs_dim += 3  # distance and direction to nearest obstacle
            elif component == 'obstacle_map':
                obs_dim += 8  # simplified 8-directional obstacle map
            elif component == 'collision_risk':
                obs_dim += 1  # single collision risk value
            elif component == 'path_history':
                obs_dim += 6  # last 2 positions
                
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        
    def update_stage_config(self, new_config: Dict[str, Any], transition_factor: float = 0.0, next_config: Dict = None):
        """Update environment configuration for new curriculum stage"""
        self.stage_config = new_config
        self.env_config = new_config['environment']
        self.obs_config = new_config['observations']
        
        # Update stage-specific parameters
        self.goal_threshold = self.env_config['goal_threshold']
        self.episode_length = self.env_config['episode_length']
        
        # Update observation space if needed
        self._setup_observation_space()
        
        # Update reward system
        self.reward_system = AdaptiveRewardSystem(new_config)
        if transition_factor > 0:
            self.reward_system.set_transition(transition_factor, next_config)
            
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        p.resetSimulation()
        p.setGravity(0, 0, -self.gravity)
        
        # Create ground plane
        ground = p.loadURDF("plane.urdf")
        
        # Generate positions based on stage configuration
        min_dist, max_dist = self.env_config['max_distance_range']
        randomization = self.env_config['randomization_level']
        
        # Start position
        start_variance = 2.0 * randomization
        self.start_pos = np.array([
            np.random.uniform(-start_variance, start_variance),
            np.random.uniform(-start_variance, start_variance),
            np.random.uniform(1.5, 2.5)
        ])
        
        # Goal position
        distance = np.random.uniform(min_dist, max_dist)
        angle = np.random.uniform(0, 2 * np.pi)
        goal_variance = 3.0 * randomization
        
        self.goal_pos = np.array([
            self.start_pos[0] + distance * np.cos(angle) + np.random.uniform(-goal_variance, goal_variance),
            self.start_pos[1] + distance * np.sin(angle) + np.random.uniform(-goal_variance, goal_variance),
            np.random.uniform(1.5, 2.5)
        ])
        
        # Create drone
        drone_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2)
        drone_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0, 0, 1])
        self.drone = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=drone_col,
            baseVisualShapeIndex=drone_vis,
            basePosition=self.start_pos
        )
        
        # Create goal marker
        goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.goal_threshold, rgbaColor=[0, 1, 0, 0.5])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_vis, basePosition=self.goal_pos)
        
        # Create stage-appropriate obstacles
        self._create_stage_obstacles()
        
        # Reset state variables
        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.start_pos - self.goal_pos)
        self.initial_distance = self.prev_distance
        self.best_distance = self.prev_distance
        self.prev_velocity = np.zeros(3)
        self.path_history.clear()
        self.path_history.append(self.start_pos.copy())
        
        return self._get_observation(), {}
        
    def _create_stage_obstacles(self):
        """Create obstacles based on current stage configuration"""
        self.obstacles = []
        self.obstacle_params = []
        
        obstacles_config = self.env_config.get('obstacles', [])
        
        for obs_config in obstacles_config:
            obs_type = obs_config['type']
            count = obs_config['count']
            complexity = obs_config['complexity']
            
            if obs_type == 'simple_wall':
                self._create_simple_walls(count, complexity)
            elif obs_type == 'corridor':
                self._create_corridor(complexity)
            elif obs_type == 'scattered':
                self._create_scattered_obstacles(count, complexity)
            elif obs_type == 'maze':
                self._create_maze(complexity)
                
    def _create_simple_walls(self, count: int, complexity: float):
        """Create simple wall obstacles"""
        mid_point = (self.start_pos + self.goal_pos) / 2
        
        for i in range(count):
            offset = (i - count/2) * 3.0
            wall_pos = [mid_point[0] + offset, mid_point[1] - 2 + 4 * complexity, mid_point[2]]
            wall_size = [0.5, 0.5 + complexity, 2.0]
            
            self._create_box_obstacle(wall_pos, wall_size)
            
    def _create_corridor(self, complexity: float):
        """Create corridor obstacles"""
        start_to_goal = self.goal_pos - self.start_pos
        corridor_length = np.linalg.norm(start_to_goal)
        corridor_direction = start_to_goal / corridor_length
        
        # Corridor width decreases with complexity
        width = 4.0 - 2.0 * complexity
        
        # Left and right walls
        perp_direction = np.array([-corridor_direction[1], corridor_direction[0], 0])
        
        for side in [-1, 1]:
            wall_center = self.start_pos + start_to_goal * 0.5 + perp_direction * side * (width / 2 + 0.5)
            wall_size = [corridor_length * 0.3, 0.5, 2.0]
            self._create_box_obstacle(wall_center, wall_size)
            
    def _create_scattered_obstacles(self, count: int, complexity: float):
        """Create scattered obstacles"""
        for i in range(count):
            # Random position between start and goal
            t = np.random.uniform(0.3, 0.7)
            base_pos = self.start_pos + t * (self.goal_pos - self.start_pos)
            
            # Add random offset
            offset_range = 2.0 + 2.0 * complexity
            offset = np.random.uniform(-offset_range, offset_range, 3)
            offset[2] = 0  # Keep same altitude
            
            obs_pos = base_pos + offset
            obs_size = [0.5 + complexity * 0.5, 0.5 + complexity * 0.5, 1.5 + complexity]
            
            self._create_box_obstacle(obs_pos, obs_size)
            
    def _create_maze(self, complexity: float):
        """Create maze-like obstacles"""
        # This is a simplified maze - in practice, you'd want more sophisticated maze generation
        maze_density = int(3 + complexity * 5)
        
        for i in range(maze_density):
            pos = [
                np.random.uniform(self.start_pos[0], self.goal_pos[0]),
                np.random.uniform(min(self.start_pos[1], self.goal_pos[1]) - 3,
                               max(self.start_pos[1], self.goal_pos[1]) + 3),
                2.0
            ]
            size = [0.8, 0.8, 2.0]
            
            # Ensure we don't block start or goal
            if (np.linalg.norm(np.array(pos) - self.start_pos) > 2.0 and
                np.linalg.norm(np.array(pos) - self.goal_pos) > 2.0):
                self._create_box_obstacle(pos, size)
                
    def _create_box_obstacle(self, position: List[float], size: List[float]):
        """Helper method to create box obstacles"""
        half_extents = [s/2 for s in size]
        obstacle_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        obstacle_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                         rgbaColor=[0.7, 0.7, 0.7, 1])
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=obstacle_col,
            baseVisualShapeIndex=obstacle_vis,
            basePosition=position
        )
        self.obstacles.append(obstacle_id)
        self.obstacle_params.append((position, half_extents))
        
    def _get_observation(self):
        """Get observation based on current stage configuration"""
        pos, _ = p.getBasePositionAndOrientation(self.drone)
        vel, _ = p.getBaseVelocity(self.drone)
        
        components = self.obs_config['components']
        obs_parts = []
        
        for component in components:
            if component == 'position':
                obs_parts.append(pos)
            elif component == 'velocity':
                obs_parts.append(vel)
            elif component == 'goal_relative':
                obs_parts.append(self.goal_pos - np.array(pos))
            elif component == 'nearest_obstacle':
                obs_parts.append(self._get_nearest_obstacle_info(pos))
            elif component == 'obstacle_map':
                obs_parts.append(self._get_obstacle_map(pos))
            elif component == 'collision_risk':
                obs_parts.append([self._get_collision_risk(pos)])
            elif component == 'path_history':
                obs_parts.append(self._get_path_history())
                
        obs = np.concatenate(obs_parts).astype(np.float32)
        
        # Apply normalization if configured
        if self.obs_config['normalize']:
            obs = self._normalize_observation(obs)
            
        # Add noise if configured
        noise_level = self.obs_config.get('noise_level', 0.0)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, obs.shape)
            obs += noise
            
        return obs
        
    def _get_nearest_obstacle_info(self, pos):
        """Get information about nearest obstacle"""
        if not self.obstacle_params:
            return [10.0, 0.0, 0.0]  # Far away, no direction
            
        min_distance = float('inf')
        nearest_direction = np.array([0.0, 0.0, 0.0])
        
        for center, half_extents in self.obstacle_params:
            distance = np.linalg.norm(np.array(pos) - np.array(center))
            if distance < min_distance:
                min_distance = distance
                direction = np.array(center) - np.array(pos)
                nearest_direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
                
        return [min_distance] + nearest_direction.tolist()[:2]  # distance + 2D direction
        
    def _get_obstacle_map(self, pos):
        """Get 8-directional obstacle map"""
        directions = [
            [1, 0], [1, 1], [0, 1], [-1, 1],
            [-1, 0], [-1, -1], [0, -1], [1, -1]
        ]
        
        obstacle_map = []
        for dx, dy in directions:
            # Cast ray in direction and find nearest obstacle
            ray_length = 5.0
            ray_end = np.array(pos) + np.array([dx, dy, 0]) * ray_length
            
            distance = ray_length
            for center, half_extents in self.obstacle_params:
                # Simplified distance calculation
                obs_distance = np.linalg.norm(np.array(center[:2]) - np.array(pos[:2]))
                if obs_distance < distance:
                    distance = obs_distance
                    
            obstacle_map.append(distance / ray_length)  # Normalize
            
        return obstacle_map
        
    def _get_collision_risk(self, pos):
        """Calculate collision risk based on proximity to obstacles"""
        if not self.obstacle_params:
            return 0.0
            
        min_distance = float('inf')
        for center, half_extents in self.obstacle_params:
            distance = np.linalg.norm(np.array(pos) - np.array(center))
            distance -= max(half_extents)  # Account for obstacle size
            min_distance = min(min_distance, distance)
            
        # Convert distance to risk (higher risk for closer obstacles)
        if min_distance < 0.5:
            return 1.0
        elif min_distance < 2.0:
            return (2.0 - min_distance) / 1.5
        else:
            return 0.0
            
    def _get_path_history(self):
        """Get simplified path history"""
        if len(self.path_history) < 2:
            return [0.0] * 6
            
        # Return last two positions (relative to current)
        current_pos = np.array(self.path_history[-1])
        prev_pos = np.array(self.path_history[-2])
        
        return (prev_pos - current_pos).tolist() + (current_pos - prev_pos).tolist()
        
    def _normalize_observation(self, obs):
        """Normalize observation components"""
        # This is a simplified normalization - in practice, you'd want component-specific normalization
        return np.clip(obs / 10.0, -1.0, 1.0)
        
    def _check_collision(self):
        """Check for collisions"""
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        
        # Altitude bounds
        if drone_pos[2] < self.min_altitude or drone_pos[2] > self.max_altitude:
            return True
            
        # Obstacle collisions
        for obstacle in self.obstacles:
            contact_points = p.getContactPoints(self.drone, obstacle)
            if contact_points:
                return True
                
        return False
        
    def step(self, action):
        # Apply action with PD control
        desired_velocity = action * self.max_speed
        current_velocity, _ = p.getBaseVelocity(self.drone)
        
        velocity_error = np.array(desired_velocity) - np.array(current_velocity)
        acceleration_term = self.k_d * (np.array(current_velocity) - self.prev_velocity)
        
        force = self.k_p * velocity_error - acceleration_term
        force = np.clip(force, -self.max_force, self.max_force)
        force[2] += self.mass * self.gravity
        
        p.applyExternalForce(self.drone, -1, force.tolist(), [0, 0, 0], p.WORLD_FRAME)
        p.stepSimulation()
        
        # Update state
        self.prev_velocity = current_velocity
        pos, _ = p.getBasePositionAndOrientation(self.drone)
        self.path_history.append(np.array(pos))
        
        # Calculate distances and progress
        distance_to_goal = np.linalg.norm(np.array(pos) - self.goal_pos)
        progress = self.prev_distance - distance_to_goal
        self.prev_distance = distance_to_goal
        
        # Update best distance
        if distance_to_goal < self.best_distance:
            self.best_distance = distance_to_goal
            
        # Get minimum obstacle distance
        min_obstacle_distance = float('inf')
        for center, half_extents in self.obstacle_params:
            dist = np.linalg.norm(np.array(pos) - np.array(center)) - max(half_extents)
            min_obstacle_distance = min(min_obstacle_distance, dist)
            
        # Check termination conditions
        collision = self._check_collision()
        success = distance_to_goal < self.goal_threshold
        timeout = self.current_step >= self.episode_length
        
        # Calculate reward using adaptive system
        state_info = {
            'progress': progress,
            'collision': collision,
            'success': success,
            'min_obstacle_distance': min_obstacle_distance,
            'velocity_magnitude': np.linalg.norm(current_velocity)
        }
        
        reward = self.reward_system.calculate_reward(state_info)
        
        done = success or collision or timeout
        self.current_step += 1
        
        info = {
            'distance_to_goal': distance_to_goal,
            'progress_ratio': 1 - (distance_to_goal / self.initial_distance),
            'collision': collision,
            'success': success,
            'min_obstacle_distance': min_obstacle_distance,
            'episode_length': self.current_step
        }
        
        return self._get_observation(), reward, done, False, info
        
    def close(self):
        p.disconnect()

class CurriculumManager:
    """Main curriculum management system"""
    
    def __init__(self, save_dir: str = "./curriculum_checkpoints/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.current_stage_idx = 0
        self.stage_names = CurriculumConfig.STAGE_ORDER
        self.current_stage = self.stage_names[0]
        
        self.performance_tracker = PerformanceTracker()
        self.transition_progress = 0.0
        self.episodes_in_stage = 0
        self.stage_start_time = time.time()
        
        # Environment and model references
        self.env = None
        self.model = None
        
        # Logging
        self.curriculum_log = []
        
    def initialize_environment(self):
        """Initialize environment with first stage configuration"""
        stage_config = CurriculumConfig.STAGES[self.current_stage]
        self.env = AdaptiveDroneEnv(stage_config)
        return DummyVecEnv([lambda: self.env])
        
    def initialize_model(self, vec_env):
        """Initialize SAC model with stage-appropriate hyperparameters"""
        stage_config = CurriculumConfig.STAGES[self.current_stage]
        hyperparams = stage_config['hyperparams']
        
        self.model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=hyperparams['learning_rate'],
            buffer_size=hyperparams['buffer_size'],
            learning_starts=2000,
            batch_size=hyperparams['batch_size'],
            tau=hyperparams['tau'],
            gamma=0.95,
            ent_coef='auto',
            target_entropy='auto',
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn=nn.ReLU
            ),
            device="auto",
            tensorboard_log="./curriculum_logs/"
        )
        
        return self.model
        
    def update_episode_result(self, episode_info: Dict[str, Any]):
        """Update performance tracking with episode results"""
        success = episode_info['success']
        # Calculate episode reward (you'd get this from the training loop)
        reward = episode_info.get('total_reward', 0)
        episode_length = episode_info['episode_length']
        
        self.performance_tracker.update(success, reward, episode_length)
        self.episodes_in_stage += 1
        
        # Check if we should advance to next stage
        stage_config = CurriculumConfig.STAGES[self.current_stage]
        criteria = stage_config['success_criteria']
        
        if self.performance_tracker.should_advance(criteria):
            if self.current_stage_idx < len(self.stage_names) - 1:
                self._advance_to_next_stage()
            else:
                print(f"Curriculum completed! Mastered final stage: {self.current_stage}")
                
        # Update transition progress for smooth transitions
        metrics = self.performance_tracker.get_metrics()
        if metrics['total_episodes'] > criteria['min_episodes'] * 0.7:
            # Start preparing for transition in last 30% of stage
            progress_in_stage = (metrics['total_episodes'] - criteria['min_episodes'] * 0.7) / (criteria['min_episodes'] * 0.3)
            self.transition_progress = min(progress_in_stage, 1.0)
            
            # Update environment with transition settings
            if self.transition_progress > 0 and self.current_stage_idx < len(self.stage_names) - 1:
                next_stage_config = CurriculumConfig.STAGES[self.stage_names[self.current_stage_idx + 1]]
                self.env.reward_system.set_transition(self.transition_progress, next_stage_config)
                
    def _advance_to_next_stage(self):
        """Advance to the next curriculum stage"""
        # Save current model
        model_path = os.path.join(self.save_dir, f"model_stage_{self.current_stage}.zip")
        self.model.save(model_path)
        
        # Log stage completion
        metrics = self.performance_tracker.get_metrics()
        stage_duration = time.time() - self.stage_start_time
        
        stage_log = {
            'stage': self.current_stage,
            'episodes': self.episodes_in_stage,
            'duration_minutes': stage_duration / 60,
            'final_success_rate': metrics['success_rate'],
            'final_stability': metrics['stability']
        }
        self.curriculum_log.append(stage_log)
        
        print(f"\n{'='*60}")
        print(f"STAGE COMPLETED: {self.current_stage}")
        print(f"Episodes: {self.episodes_in_stage}")
        print(f"Duration: {stage_duration/60:.1f} minutes")
        print(f"Final Success Rate: {metrics['success_rate']:.3f}")
        print(f"Final Stability: {metrics['stability']:.3f}")
        print(f"{'='*60}\n")
        
        # Advance to next stage
        self.current_stage_idx += 1
        self.current_stage = self.stage_names[self.current_stage_idx]
        
        # Reset tracking for new stage
        self.performance_tracker = PerformanceTracker()
        self.episodes_in_stage = 0
        self.transition_progress = 0.0
        self.stage_start_time = time.time()
        
        # Update environment configuration
        new_stage_config = CurriculumConfig.STAGES[self.current_stage]
        self.env.update_stage_config(new_stage_config)
        
        # Update model hyperparameters
        self._update_model_hyperparameters(new_stage_config['hyperparams'])
        
        print(f"ADVANCED TO STAGE: {self.current_stage}")
        print(f"New Configuration:")
        print(f"- Goal Threshold: {new_stage_config['environment']['goal_threshold']}")
        print(f"- Episode Length: {new_stage_config['environment']['episode_length']}")
        print(f"- Obstacles: {len(new_stage_config['environment']['obstacles'])}")
        print(f"- Learning Rate: {new_stage_config['hyperparams']['learning_rate']}")
        print()
        
    def _update_model_hyperparameters(self, new_hyperparams: Dict[str, Any]):
        """Update model hyperparameters for new stage"""
        # Update learning rate
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_hyperparams['learning_rate']
            
        # Update other hyperparameters
        self.model.batch_size = new_hyperparams['batch_size']
        self.model.tau = new_hyperparams['tau']
        
    def get_current_stage_info(self) -> Dict[str, Any]:
        """Get information about current curriculum stage"""
        metrics = self.performance_tracker.get_metrics()
        stage_config = CurriculumConfig.STAGES[self.current_stage]
        
        return {
            'stage': self.current_stage,
            'stage_index': self.current_stage_idx,
            'total_stages': len(self.stage_names),
            'episodes_in_stage': self.episodes_in_stage,
            'current_metrics': metrics,
            'success_criteria': stage_config['success_criteria'],
            'transition_progress': self.transition_progress
        }
        
    def save_curriculum_log(self):
        """Save curriculum learning progress log"""
        log_path = os.path.join(self.save_dir, "curriculum_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.curriculum_log, f, indent=2)

class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning integration with Stable Baselines3"""
    
    def __init__(self, curriculum_manager: CurriculumManager, update_frequency: int = 100):
        super(CurriculumCallback, self).__init__()
        self.curriculum_manager = curriculum_manager
        self.update_frequency = update_frequency
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
    def _on_step(self) -> bool:
        # Collect episode information when episode ends
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                info = self.locals['infos'][i]
                
                # Calculate total episode reward (approximation)
                episode_reward = sum(self.episode_rewards[-info.get('episode_length', 100):])
                
                episode_info = {
                    'success': info.get('success', False),
                    'total_reward': episode_reward,
                    'episode_length': info.get('episode_length', 0),
                    'collision': info.get('collision', False),
                    'progress_ratio': info.get('progress_ratio', 0)
                }
                
                self.curriculum_manager.update_episode_result(episode_info)
                
                # Print progress every update_frequency episodes
                if self.curriculum_manager.episodes_in_stage % self.update_frequency == 0:
                    stage_info = self.curriculum_manager.get_current_stage_info()
                    metrics = stage_info['current_metrics']
                    
                    print(f"\nStage: {stage_info['stage']} | "
                          f"Episode: {stage_info['episodes_in_stage']} | "
                          f"Success Rate: {metrics['success_rate']:.3f} | "
                          f"Stability: {metrics['stability']:.3f}")
                    
        return True

def train_curriculum_drone():
    """Main training function with comprehensive curriculum learning"""
    
    print("="*80)
    print("COMPREHENSIVE CURRICULUM LEARNING DRONE TRAINING")
    print("="*80)
    print()
    
    # Initialize curriculum manager
    curriculum_manager = CurriculumManager()
    
    # Initialize environment and model
    vec_env = curriculum_manager.initialize_environment()
    model = curriculum_manager.initialize_model(vec_env)
    
    # Create curriculum callback
    curriculum_callback = CurriculumCallback(curriculum_manager, update_frequency=50)
    
    print("Curriculum Learning Configuration:")
    print(f"- Total Stages: {len(CurriculumConfig.STAGE_ORDER)}")
    print(f"- Stages: {', '.join(CurriculumConfig.STAGE_ORDER)}")
    print()
    
    for stage_name in CurriculumConfig.STAGE_ORDER:
        stage_config = CurriculumConfig.STAGES[stage_name]
        print(f"{stage_name.upper()}:")
        print(f"  - Distance Range: {stage_config['environment']['max_distance_range']}")
        print(f"  - Obstacles: {len(stage_config['environment']['obstacles'])}")
        print(f"  - Goal Threshold: {stage_config['environment']['goal_threshold']}")
        print(f"  - Success Criteria: {stage_config['success_criteria']['success_rate']:.2f}")
        print(f"  - Min Episodes: {stage_config['success_criteria']['min_episodes']}")
        print()
    
    print("Starting curriculum training...")
    
    # Train with curriculum learning
    start_time = time.time()
    total_timesteps = 500000  # Increased for comprehensive curriculum
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=curriculum_callback,
        tb_log_name="CurriculumDrone"
    )
    
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    # Save final model and curriculum log
    model.save("curriculum_drone_final")
    curriculum_manager.save_curriculum_log()
    
    # Print curriculum summary
    print("\nCURRICULUM LEARNING SUMMARY:")
    print("="*50)
    
    for log_entry in curriculum_manager.curriculum_log:
        print(f"Stage: {log_entry['stage']}")
        print(f"  Episodes: {log_entry['episodes']}")
        print(f"  Duration: {log_entry['duration_minutes']:.1f} min")
        print(f"  Success Rate: {log_entry['final_success_rate']:.3f}")
        print(f"  Stability: {log_entry['final_stability']:.3f}")
        print()
    
    return model, curriculum_manager

def evaluate_curriculum_model(model_path: str = "curriculum_drone_final", num_episodes: int = 20):
    """Comprehensive evaluation across all curriculum stages"""
    
    model = SAC.load(model_path)
    
    print("CURRICULUM MODEL EVALUATION")
    print("="*50)
    
    overall_results = {}
    
    # Test on each curriculum stage
    for stage_name in CurriculumConfig.STAGE_ORDER:
        print(f"\nTesting on {stage_name.upper()} stage...")
        
        stage_config = CurriculumConfig.STAGES[stage_name]
        test_env = AdaptiveDroneEnv(stage_config)
        
        stage_results = {
            'successes': 0,
            'collisions': 0,
            'timeouts': 0,
            'total_rewards': [],
            'episode_lengths': [],
            'progress_ratios': []
        }
        
        for episode in range(num_episodes):
            obs, _ = test_env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = test_env.step(action)
                total_reward += reward
                
            # Record results
            stage_results['total_rewards'].append(total_reward)
            stage_results['episode_lengths'].append(info['episode_length'])
            stage_results['progress_ratios'].append(info['progress_ratio'])
            
            if info['success']:
                stage_results['successes'] += 1
            elif info['collision']:
                stage_results['collisions'] += 1
            else:
                stage_results['timeouts'] += 1
                
        # Calculate statistics
        success_rate = stage_results['successes'] / num_episodes
        avg_reward = np.mean(stage_results['total_rewards'])
        avg_length = np.mean(stage_results['episode_lengths'])
        avg_progress = np.mean(stage_results['progress_ratios'])
        
        print(f"Results for {stage_name}:")
        print(f"  Success Rate: {success_rate:.3f}")
        print(f"  Collision Rate: {stage_results['collisions']/num_episodes:.3f}")
        print(f"  Timeout Rate: {stage_results['timeouts']/num_episodes:.3f}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Episode Length: {avg_length:.0f}")
        print(f"  Avg Progress: {avg_progress:.3f}")
        
        overall_results[stage_name] = {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_progress': avg_progress
        }
        
        test_env.close()
    
    # Overall curriculum evaluation
    print(f"\nOVERALL CURRICULUM PERFORMANCE:")
    print("="*40)
    
    total_success_rate = np.mean([r['success_rate'] for r in overall_results.values()])
    print(f"Average Success Rate Across All Stages: {total_success_rate:.3f}")
    
    # Check for catastrophic forgetting
    basic_success = overall_results['basic_navigation']['success_rate']
    expert_success = overall_results['expert_level']['success_rate']
    
    if basic_success > 0.7:  # Should maintain basic skills
        print(f"✓ No catastrophic forgetting detected (basic: {basic_success:.3f})")
    else:
        print(f"⚠ Potential catastrophic forgetting (basic: {basic_success:.3f})")
        
    print(f"Expert Level Performance: {expert_success:.3f}")
    
    return overall_results

if __name__ == "__main__":
    # Train with comprehensive curriculum learning
    model, curriculum_manager = train_curriculum_drone()
    
    # Evaluate the trained model
    print("\n" + "="*80)
    evaluation_results = evaluate_curriculum_model()
    
    print(f"\nCurriculum learning completed successfully!")
    print(f"Check './curriculum_checkpoints/' for stage-wise model saves")
    print(f"Check './curriculum_logs/' for detailed training logs")