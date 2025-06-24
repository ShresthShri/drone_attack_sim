# envs/drone_navigation_env.py (Improved Version)

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel, Physics
import gymnasium as gym
import numpy as np
import os
import pybullet as p
from gymnasium import spaces

class DroneNavigationEnv(VelocityAviary):
    def __init__(self, gui=False, obstacles=True, initial_xyzs=None, goal_xyz=None):
        # Define the drone model and physics
        drone_model = DroneModel.CF2X
        num_drones = 1
        
        # Store initial position for deviation calculation
        if initial_xyzs is None:
            self.initial_xyzs = np.array([[0., 0., 1.]])
        else:
            self.initial_xyzs = initial_xyzs.copy()
        
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=self.initial_xyzs,
            physics=Physics.PYB,
            gui=gui,
            record=False
        )
        
        # Define the goal position
        if goal_xyz is None:
            self.goal = np.array([2.0, 2.0, 1.0])
        else:
            self.goal = goal_xyz.copy()

        self._reached_goal = False
        self._collision_occurred = False
        self._last_distance = None
        self._start_pos = self.initial_xyzs[0].copy()  # Store for deviation calc

        # Reward function weighting factors
        # W_Progress for moving towards the goal
        self.W_PROGRESS = 10.0
        # Continous Proximity
        self.W_GOAL_PROXIMITY = 5.0
        # W_Deviation for deviating from the straight path
        self.W_DEVIATION = 0.5
        # W_Collison for colliding
        self.W_COLLISION = 200.0 #Penalty increased from 100 to 200
        # W_Time for time spent in the episode 
        self.W_TIME = 0.1
        # Penalty for large actions 
        self.W_ACTION_MAGNITUDE = 0.01 

        # Obstacle setup
        self.obstacles_enabled = obstacles
        self.obstacle_ids = []
        self.OBSTACLE_RADIUS = 0.5
        self.OBSTACLE_HEIGHT = 2.0

        # Don't define observation space here - let parent class handle it
        # VelocityAviary returns 20-element observations, not 12
        # The parent class will set the correct observation space

    def _add_obstacles(self):
        """Adds simple static obstacles to the environment."""
        # Clear previous obstacles if any
        for obs_id in self.obstacle_ids:
            if obs_id in [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]:
                try:
                    p.removeBody(obs_id)
                except:
                    pass

        self.obstacle_ids = []

        # Example obstacle positions
        obstacle_positions = [
            [2.0, 1.0, 1.0],
            [3.5, -1.5, 1.0],
            [1.5, 3.0, 1.0],
            [4.0, 2.5, 1.0],
            [2.5, -0.5, 1.0]
        ]

        for i, pos in enumerate(obstacle_positions):
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=self.OBSTACLE_RADIUS,
                length=self.OBSTACLE_HEIGHT,
                rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=self.OBSTACLE_RADIUS,
                height=self.OBSTACLE_HEIGHT
            )
            
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=pos
            )
            self.obstacle_ids.append(obstacle_id)

    def _check_obstacle_collision(self):
        """Check if drone has collided with any obstacles."""
        drone_id = self.DRONE_IDS[0]  # First (and only) drone
        
        for obstacle_id in self.obstacle_ids:
            contact_points = p.getContactPoints(bodyA=drone_id, bodyB=obstacle_id)
            if len(contact_points) > 0:
                return True
        return False

    def compute_reward(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        
        # 1. Progress Reward
        distance_to_goal = np.linalg.norm(drone_pos - self.goal)
        if self._last_distance is None:
            self._last_distance = distance_to_goal
        
        progress_reward = (self._last_distance - distance_to_goal) * self.W_PROGRESS
        
        # 2. Goal Reward
        goal_bonus = 0
        if distance_to_goal < 0.2:
            goal_bonus = 50.0
            self._reached_goal = True
        
        # 3. Collision Penalty
        collision_penalty = 0
        if self._collision_occurred:
            collision_penalty = -self.W_COLLISION

        # 4. Deviation Penalty (using actual start position)
        deviation_penalty = 0
        if not self._reached_goal:
            vec_to_goal = self.goal - self._start_pos
            vec_norm = np.linalg.norm(vec_to_goal)
            
            if vec_norm > 1e-6:  # Avoid division by zero
                # Project drone's current position onto the ideal path
                vec_from_start = drone_pos - self._start_pos
                projection_factor = np.dot(vec_from_start, vec_to_goal) / (vec_norm ** 2)
                projected_point = self._start_pos + projection_factor * vec_to_goal
                
                # Distance from drone to the projected point
                lateral_deviation = np.linalg.norm(drone_pos - projected_point)
                deviation_penalty = -lateral_deviation * self.W_DEVIATION
                deviation_penalty = np.clip(deviation_penalty, -50.0, 0)

        # 5. Time Penalty
        time_penalty = -self.W_TIME

        # Add continous proximity reward
        proximity_reward = self.W_GOAL_PROXIMITY / (distance_to_goal + 0.1) # 0.1 to avoid div by zero

        # Add Action magnitude penalty (Assuming action is a numpy array)
        # THis helps prevent jerky movements and encourages smoother control
        action_magnitude_penalty = -np.sum(np.square(self._last_action)) * self.W_ACTION_MAGNITUDE if hasattr(self, '_last_action') else 0

        # Combine all rewards
        reward = progress_reward + goal_bonus + collision_penalty + deviation_penalty + time_penalty + proximity_reward + action_magnitude_penalty
        
        # Clip the total reward
        #reward = np.clip(reward, -self.W_COLLISION - 50.0, 50.0 + self.W_PROGRESS * 10)

        self._last_distance = distance_to_goal
        return reward
    
    def check_termination(self):
        pos = self._getDroneStateVector(0)[0:3]
        
        # Collision with ground
        if pos[2] < 0.1:
            self._collision_occurred = True
            return True
        
        # Collision with obstacles
        if self.obstacles_enabled and self._check_obstacle_collision():
            self._collision_occurred = True
            return True
        
        # Out of bounds
        if np.linalg.norm(pos) > 10:
            return True
        
        # Reached goal
        if self._reached_goal:
            return True
        
        return False

    def step(self, action):
        # Call parent step - VelocityAviary returns 20-element observations
        obs, _, terminated, truncated, info = super().step(action)
        
        # Get current drone state for reward calculation (12 elements)
        drone_state = self._getDroneStateVector(0)
        if drone_state is None:
            drone_state = np.zeros(12)
        
        # Compute reward based on drone state
        reward = self.compute_reward()
        
        # Check termination
        terminated = self.check_termination()

        # Prepare info dictionary
        drone_pos = drone_state[0:3]
        info.update({
            "drone_position": drone_pos.tolist(),
            "goal_position": self.goal.tolist(),
            "true_reward": float(reward),
            "collision_occurred": self._collision_occurred,
            "reached_goal": self._reached_goal
        })
        
        # Return the full observation from parent (20 elements)
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, dict(info)

    def reset(self, seed=None, options=None):
        # Reset internal flags
        self._reached_goal = False
        self._collision_occurred = False
        self._last_distance = None

        # Randomize goal position
        self.goal = np.random.uniform(low=[3.0, -3.0, 1.0], high=[7.0, 3.0, 2.0])
        
        # Update start position for deviation calculation
        self._start_pos = self.initial_xyzs[0].copy()
        
        # Call parent reset - returns 20-element observation
        obs, info = super().reset(seed=seed, options=options)
        
        # Add obstacles if enabled
        if self.obstacles_enabled:
            self._add_obstacles()

        # Get drone state for info (first 3 elements are position)
        drone_state = self._getDroneStateVector(0)
        if drone_state is not None:
            drone_pos = drone_state[0:3]
        else:
            drone_pos = [0.0, 0.0, 1.0]  # Default position
        
        # Update info
        info.update({
            "drone_position": drone_pos,
            "goal_position": self.goal.tolist(),
            "true_reward": 0.0,
            "collision_occurred": False,
            "reached_goal": False
        })

        return np.array(obs, dtype=np.float32), info