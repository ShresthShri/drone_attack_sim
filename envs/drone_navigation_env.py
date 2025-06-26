import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p


class DroneNavigationEnv(VelocityAviary):
    """
    Custom drone navigation environment that extends VelocityAviary.
    Task: Navigate from start position to target position.
    """
    
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,  # Fixed frequency to avoid issues
                 ctrl_freq: int = 30,   # Lower control frequency
                 gui=False,
                 record=False):
        
        # Set target position bounds
        self.target_bounds = {
            'x': (-2.0, 2.0),
            'y': (-2.0, 2.0), 
            'z': (0.5, 2.0)
        }
        
        # Navigation parameters
        self.target_position = None
        self.start_position = None
        self.max_distance = 5.0  # Maximum distance for normalization
        self.position_tolerance = 0.1  # Success threshold
        self.max_steps = 1000
        self.current_step = 0
        
        # Reward parameters
        self.prev_distance = None
        self.success_reward = 100.0
        self.distance_reward_scale = 10.0
        self.step_penalty = -0.01
        self.collision_penalty = -50.0
        self.out_of_bounds_penalty = -20.0
        
        # Initialize parent first
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record
        )
        
        # Now setup our custom observation space
        self._setup_custom_observation_space()
        
    def _setup_custom_observation_space(self):
        """Setup custom observation space including drone state and target position."""
        # Get the original observation space from parent
        original_obs_space = super()._observationSpace()
        
        # Debug: print original observation space info
        print(f"Original obs space shape: {original_obs_space.shape}")
        print(f"Original obs space low shape: {original_obs_space.low.shape}")
        print(f"Original obs space high shape: {original_obs_space.high.shape}")
        
        # Flatten the original observation space if it's multidimensional
        if len(original_obs_space.shape) > 1:
            original_low = original_obs_space.low.flatten()
            original_high = original_obs_space.high.flatten()
        else:
            original_low = original_obs_space.low
            original_high = original_obs_space.high
        
        # Add target position (3D) and distance to target (1D)
        target_low = np.array([
            self.target_bounds['x'][0], 
            self.target_bounds['y'][0], 
            self.target_bounds['z'][0],
            0.0  # distance is always positive
        ], dtype=np.float32)
        
        target_high = np.array([
            self.target_bounds['x'][1], 
            self.target_bounds['y'][1], 
            self.target_bounds['z'][1],
            self.max_distance
        ], dtype=np.float32)
        
        # Combine original observation with target information
        combined_low = np.concatenate([original_low, target_low])
        combined_high = np.concatenate([original_high, target_high])
        
        self.observation_space = spaces.Box(
            low=combined_low,
            high=combined_high,
            dtype=np.float32
        )
        
        print(f"Custom obs space shape: {self.observation_space.shape}")
        print(f"Custom obs space low shape: {self.observation_space.low.shape}")
        print(f"Custom obs space high shape: {self.observation_space.high.shape}")
    
    def _observationSpace(self):
        """Return the observation space."""
        # If we haven't set up our custom observation space yet, use parent's
        if hasattr(self, 'observation_space'):
            return self.observation_space
        else:
            return super()._observationSpace()
    
    def reset(self, seed=None, options=None):
        """Reset the environment and generate new target position."""
        # Reset parent environment
        obs, info = super().reset(seed=seed, options=options)
        
        # Generate random target position
        self.target_position = np.array([
            np.random.uniform(*self.target_bounds['x']),
            np.random.uniform(*self.target_bounds['y']),
            np.random.uniform(*self.target_bounds['z'])
        ])
        
        # Store start position
        self.start_position = self.pos[0].copy()
        
        # Calculate initial distance
        self.prev_distance = np.linalg.norm(self.pos[0] - self.target_position)
        
        # Reset step counter
        self.current_step = 0
        
        # Add target information to observation
        enhanced_obs = self._enhance_observation(obs)
        
        return enhanced_obs, info
    
    def step(self, action):
        """Execute one step in the environment."""
        # Execute action in parent environment
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Calculate custom reward
        custom_reward = self._calculate_reward()
        
        # Check for termination conditions
        custom_terminated = self._check_termination()
        
        # Check for truncation (max steps)
        self.current_step += 1
        custom_truncated = self.current_step >= self.max_steps
        
        # Enhance observation with target information
        enhanced_obs = self._enhance_observation(obs)
        
        # Update info with navigation-specific information
        info.update({
            'distance_to_target': np.linalg.norm(self.pos[0] - self.target_position),
            'success': self._is_success(),
            'target_position': self.target_position.copy(),
            'current_position': self.pos[0].copy()
        })
        
        return enhanced_obs, custom_reward, custom_terminated, custom_truncated, info
    
    def _enhance_observation(self, obs):
        """Add target position and distance to the observation."""
        current_pos = self.pos[0]
        distance_to_target = np.linalg.norm(current_pos - self.target_position)
        
        # Normalize distance
        normalized_distance = min(distance_to_target / self.max_distance, 1.0)
        
        # Add target position and distance to observation
        target_info = np.array([
            self.target_position[0],
            self.target_position[1], 
            self.target_position[2],
            normalized_distance
        ], dtype=np.float32)
        
        # Flatten original observation if needed
        if len(obs.shape) > 1:
            obs_flat = obs.flatten()
        else:
            obs_flat = obs
        
        enhanced_obs = np.concatenate([obs_flat, target_info])
        return enhanced_obs.astype(np.float32)
    
    def _calculate_reward(self):
        """Calculate reward based on navigation progress."""
        current_pos = self.pos[0]
        current_distance = np.linalg.norm(current_pos - self.target_position)
        
        reward = 0.0
        
        # Success reward
        if self._is_success():
            reward += self.success_reward
        
        # Distance-based reward (reward for getting closer)
        if self.prev_distance is not None:
            distance_improvement = self.prev_distance - current_distance
            reward += distance_improvement * self.distance_reward_scale
        
        # Step penalty (encourage efficiency)
        reward += self.step_penalty
        
        # Collision penalty
        if self._check_collision():
            reward += self.collision_penalty
        
        # Out of bounds penalty
        if self._check_out_of_bounds():
            reward += self.out_of_bounds_penalty
        
        # Update previous distance
        self.prev_distance = current_distance
        
        return reward
    
    def _is_success(self):
        """Check if drone reached the target."""
        distance = np.linalg.norm(self.pos[0] - self.target_position)
        return distance < self.position_tolerance
    
    def _check_termination(self):
        """Check if episode should terminate."""
        return (self._is_success() or 
                self._check_collision() or 
                self._check_out_of_bounds())
    
    def _check_collision(self):
        """Check if drone collided with ground or obstacles."""
        # Check if drone is too close to ground
        if self.pos[0][2] < 0.1:
            return True
        
        # Check for high velocity (potential crash indicator)
        velocity_magnitude = np.linalg.norm(self.vel[0])
        if velocity_magnitude > 10.0:  # Adjust threshold as needed
            return True
        
        return False
    
    def _check_out_of_bounds(self):
        """Check if drone is out of bounds."""
        pos = self.pos[0]
        bounds = 3.0  # Boundary limit
        
        return (abs(pos[0]) > bounds or 
                abs(pos[1]) > bounds or 
                pos[2] > bounds or 
                pos[2] < 0)
    
    def render(self, mode='human'):
        """Render the environment with target visualization."""
        if self.GUI and self.target_position is not None:
            # Draw target position as a red sphere
            p.addUserDebugLine(
                lineFromXYZ=self.target_position - [0.1, 0, 0],
                lineToXYZ=self.target_position + [0.1, 0, 0],
                lineColorRGB=[1, 0, 0],
                lineWidth=3,
                physicsClientId=self.CLIENT
            )
            p.addUserDebugLine(
                lineFromXYZ=self.target_position - [0, 0.1, 0],
                lineToXYZ=self.target_position + [0, 0.1, 0],
                lineColorRGB=[1, 0, 0],
                lineWidth=3,
                physicsClientId=self.CLIENT
            )
            p.addUserDebugLine(
                lineFromXYZ=self.target_position - [0, 0, 0.1],
                lineToXYZ=self.target_position + [0, 0, 0.1],
                lineColorRGB=[1, 0, 0],
                lineWidth=3,
                physicsClientId=self.CLIENT
            )
        
        return super().render(mode)