import os
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
from datetime import datetime

class DroneNavEnv(VelocityAviary):
	def __init__(self):
		self.difficulty = 0
		self.max_difficulty = 7
		self.step_count = 0
		self.max_steps = 1200

		# Waypoints for progressive difficulty
		self.start_pos = np.array([-1.0,-3.0,1.0])
		self.goal_positions = [
			np.array([0.0, 1.5, 1.2]),   # Level 0
            np.array([0.0, 2.0, 1.3]),   # Level 1
            np.array([0.5, 2.5, 1.4]),   # Level 2
            np.array([1.0, 3.0, 1.5]),   # Level 3
            np.array([1.5, 3.5, 1.7]),   # Level 4
            np.array([2.0, 4.0, 1.8]),   # Level 5
            np.array([2.5, 4.2, 2.0]),   # Level 6
            np.array([3.0, 4.5, 2.2])    # Level 7
		]

		# Obstacle configuration
		self.obstacles = []
		self.obstacle_ids = []
		# Training up to 10 obstacles for simplicity
		self.obstacle_counts = [0,1,2,3,5,6,8,10]
		# Obstacles sizes defined manually
		self.obstacle_sizes = [
			[0.1, 0.15], [0.15, 0.20], [0.15, 0.20], [0.12, 0.18],
            [0.12, 0.18], [0.10, 0.16], [0.10, 0.15], [0.08, 0.14]
		]

		# Collision Parameters 
		# Drone is modelled as a sphere 
		self.drone_radius = 0.15
		# When checking for collision I am adding an extra 5 cm buffer
		# Therefore total collision distance = 0.15 + 0.20 = 0.20 
		self.collision_threshold = 0.05

		# Reward Parameters
		# Progress reward is the dominant signal and success reward is large enough to be worth pursuing
		# Penalties are smaller to avoid discouraging exploration
		# So a balance need to be made between the exploration and accuracy
		self.success_reward  = 500.0
		self.collision_penalty = -100.0
		self.progress_reward_scale = 200.0
		self.step_penalty = -0.0001
		self.velocity_reward_scale = 5.0
		self.efficiency_bonus_scale = 50.0

		# State tracking 
		# This tracks the state of the Markov Decision Process.
		# The prev stagnation counter penalises hovering in plae without progress 
		self.prev_distance_to_goal = None
		self.prev_pos = None
		self.initial_distance = None
		self.stagnation_counter = 0
		
		# Parent Environment Setup
		super().__init__(
			drone_model=DroneModel.CF2X,     # Use Crazyflie 2.X drone model
			num_drones=1,                    # Single drone
			initial_xyzs=np.array([self.start_pos]),  # Start at [-1, -3, 1]
			physics=Physics.PYB,             # Use PyBullet physics
			pyb_freq=240,                    # Physics runs at 240Hz
			ctrl_freq=30,                    # Control commands at 30Hz
			gui=False                        # No visual interface (faster training)
		)
		
		# The observatio space defines the infoprmation the AI agent can see each time step
		# The observation space include the drone position, orientation,velocity, angular velocity
		# Other features are the goal, direction, goal distance, progress ratio, velocity alignment and efficiency metric
		# In total 19 numbers
		def _observationSpace(self):
			    return spaces.Box(
				low=-np.inf,        # Minimum possible values
				high=np.inf,        # Maximum possible values  
				shape=(19,),        # 19 numbers total
				dtype=np.float32    # 32-bit floating point numbers
			)
		
		# Reset function prepares a fresh environment for a new training epsisode
		def reset(self, seed = None, options = None):
			# This resets the drone position, and physics similation allowing for reproducible random episodes
			obs, info = super().reset(seed=seed, options=options)
        
			# Clear old obstacles and deletes them from the physics simulation starting with fresh obstacles
			self._clear_obstacles()
        
			# Generate new obstacles with advanced path clearance
			self._generate_obstacles_with_path_clearance()
			
			# Reset tracking
			self.step_count = 0											# Reset Step counter
			goal = self.goal_positions[self.difficulty]					# Get the target position for current difficulty level
			self.initial_distance = np.linalg.norm(self.pos[0] - goal)	# Initial Distance from the start to goal
			self.prev_distance_to_goal = self.initial_distance			# Initialise with startign distance
			self.prev_pos = self.pos[0].copy()							# Store starting position to detect movement
			self.stagnation_counter = 0									# Reset counter for detecting if drone gets stuck
			
			return self._get_obs(),info
		
		# Executes one time step of the simulation
		def step(self, action):
			# Takes the action, runs the physic simulation and updates the drone position, velocity and orientation
			obs, _, done, truncated, info = super().step(action)
        
			# Advanced collision detection to check for Obstacle and Ground collison
			# If collision is detected a large negative reward of -100 is given 
			if self._check_collision_3d_box_sphere() or self.pos[0][2] < 0.05:
				reward = self.collision_penalty
				done = True
			
			# If no collision then calculate sophisticated reward based on progress
			else:
				reward = self._calculate_advanced_reward()
				done = self._reached_goal()
			
			# Update the counters and check time limit
			self.step_count += 1
			
			if self.step_count >= self.max_steps:
				truncated = True
        
			info['success'] = self._reached_goal()
			info['difficulty'] = self.difficulty
        
			return self._get_obs(), reward, done, truncated, info
		
		
		def _get_obs(self):
			# Get the base drone state of 12 numbers
			base_obs = super()._computeObs().flatten()
			current_pos = self.pos[0]
			current_vel = self.vel[0] if hasattr(self, 'vel') else np.zeros(3)
			goal = self.goal_positions[self.difficulty]

			# Calculate the Goal information 
			goal_vector = goal - current_pos
			goal_distance = np.linalg.norm(goal_vector)
			goal_direction = goal_vector / (goal_distance + 1e-8)

			# Calculate the enhanced features to get the progress ratio
			# How much of the journey is complete 
			progress_ratio = 1.0 - (goal_distance / self.initial_distance) if self.initial_distance else 0.0
			progress_ratio = np.clip(progress_ratio, 0.0, 1.0)
			
			# Is the drone moving toward or away from goal
			velocity_toward_goal = np.dot(current_vel, goal_direction)
			# How efficiently is drone moving 
			efficiency = self._get_efficiency_metric()
		
			# Build the final 19 number array 
			extra_obs = np.array([
				goal_direction[0], goal_direction[1], goal_direction[2],
				goal_distance / self.initial_distance,
				progress_ratio,
				np.clip(velocity_toward_goal, -2.0, 2.0),
				efficiency
			])

			return np.concatenate([base_obs, extra_obs])
		
	############################ REWARD CALCULATION ################################
	# Calculate the advanced rewards
	# The reward system creates a strong gradient pointing toward the goal with the biggest rewards for making progress
	# Setup variables 
	def _calculate_advanced_reward(self):
		current_pos = self.pos[0]
        current_vel = self.vel[0] if hasattr(self, 'vel') else np.zeros(3)
        goal = self.goal_positions[self.difficulty]
        current_distance = np.linalg.norm(current_pos - goal)

        # Success reward with difficulty bonus
		# Higher difficulties give much bigger rewards 
        if self._reached_goal():
            difficulty_bonus = self.difficulty * 120.0
            efficiency_bonus = self.efficiency_bonus_scale * self._get_efficiency_metric()
            return self.success_reward + difficulty_bonus + efficiency_bonus

        total_reward = 0.0

		# Progress reward is the main learning signal as every tiny step toward the goal is rewarded immediately
        if self.prev_distance_to_goal is not None:
            progress = self.prev_distance_to_goal - current_distance
            if progress > 0:
                total_reward += progress * self.progress_reward_scale
            elif progress < -0.01:
                total_reward += progress * 50.0  # retreat penalty

        self.prev_distance_to_goal = current_distance
		
		# Velocity alignment reward
		# So rewards assigned based on flying toward goal and flying away from the goal
        goal_direction = (goal - current_pos)
        goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
        velocity_alignment = np.dot(current_vel, goal_direction)
        total_reward += max(0, velocity_alignment) * self.velocity_reward_scale

		# Proximity reward
		# Ensures drone always has motivation to stay close to goal 
        max_distance = np.linalg.norm(goal - self.start_pos)
        proximity_reward = (1.0 - current_distance / max_distance) * 2.0
        total_reward += proximity_reward

		# Anti-stagnation penalty
		# This forces the drone to explore and move instead of getting stuck osciallating 
        if self.prev_pos is not None:
            movement = np.linalg.norm(current_pos - self.prev_pos)
            if movement < 0.003:
                self.stagnation_counter += 1
                if self.stagnation_counter > 20:
                    total_reward -= 0.1
            else:
                self.stagnation_counter = 0

        self.prev_pos = current_pos.copy()

        # Step penalty
		# Efficiency is encouraged by small pressure to complete the epsiodes quickly
        total_reward += self.step_penalty

        return total_reward

	################### GENERATE OBSTACLES ######################

	def _generate_obstacles_with_path_clearance(self):
        # Setup and early exit
        num_obstacles = self.obstacle_counts[self.difficulty]
        if num_obstacles == 0:
            return

        goal = self.goal_positions[self.difficulty]
        obstacle_size_range = self.obstacle_sizes[self.difficulty]

        # Create protected path points 
		# 20 evenly spaced points along the direct line from start to goal 
		# Like placing checkpoints every 5% of the journey 
        path_vector = goal - self.start_pos
        protected_points = []
        for i in range(20):
            t = i / 19.0
            point = self.start_pos + t * path_vector
            protected_points.append(point)

        print(f"Generating {num_obstacles} obstacles for difficulty {self.difficulty}")

        for i in range(num_obstacles):
            obstacle_placed = False

            for attempt in range(200):  # Try up to 200 times
                # Generate position based on difficulty
                if self.difficulty <= 1:
                    # Easy: use polar coordinates away from path and from center
                    angle = np.random.uniform(0, 2*np.pi)
                    radius = np.random.uniform(1.5, 2.5)
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    z = np.random.uniform(0.8, 2.0)
                else:
                    # Harder: random but with clearance checks
                    x = np.random.uniform(-2.5, 2.5)
                    y = np.random.uniform(-2.5, 2.5)
                    z = np.random.uniform(0.8, 2.2)

                pos = np.array([x, y, z])

				# Safety Checks that need to be passed
                # Check waypoint distances
                if (np.linalg.norm(pos - self.start_pos) < 1.0 or
                    np.linalg.norm(pos - goal) < 1.0):
                    continue

                # Check path clearance
				# Guarantees there is always a tube of clear space 
                min_path_clearance = min(np.linalg.norm(pos - p) for p in protected_points)
                required_clearance = max(0.4, 1.2 - (self.difficulty * 0.1))
                if min_path_clearance < required_clearance:
                    continue

                # Check distance to other obstacles
                too_close = any(np.linalg.norm(pos - obs['position']) < 1.0
                              for obs in self.obstacles)
                if too_close:
                    continue

                # Create obstacle
                size = np.random.uniform(*obstacle_size_range)
                size = min(size, 0.25)  # Cap size

                obstacle_id = self._create_obstacle(pos, size)
                if obstacle_id is not None:
                    self.obstacle_ids.append(obstacle_id)
                    self.obstacles.append({
                        'position': pos,
                        'size': size,
                        'id': obstacle_id
                    })
                    obstacle_placed = True
                    break

            if not obstacle_placed:
                print(f"Could not place obstacle {i}")
		

		def _create_obstacle(self, pos, size):
			# Creates a physical 3D cube in the PyBullet Physics simulation that the drone can collide with 
			try:
				collision_shape = p.createCollisionShape(
					p.GEOM_BOX, halfExtents=[size, size, size]
				)

				if self.GUI:
					visual_shape = p.createVisualShape(
							p.GEOM_BOX, halfExtents=[size, size, size],
							rgbaColor=[0.8, 0.2, 0.2, 0.8]
							)
					obstacle_id = p.createMultiBody(
							baseMass=0,
							baseCollisionShapeIndex=collision_shape,
							baseVisualShapeIndex=visual_shape,
							basePosition=pos
							)
				else:
					obstacle_id = p.createMultiBody(
							baseMass=0,
							baseCollisionShapeIndex=collision_shape,
							basePosition=pos
							)

				return obstacle_id
			except:
				return None
