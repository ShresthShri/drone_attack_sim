# envs/drone_navigation_env.py

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel, Physics
import gym
import numpy as np

class DroneNavigationEnv(VelocityAviary):
    def __init__(self):
        super().__init__(
            drone_model=DroneModel.CF2X,                     # Default drone model
            num_drones=1,
            initial_xyzs=np.array([[0., 0., 1.]]),
            physics=Physics.PYB,
            gui=False,
            record=False
        )
        self.goal = np.array([5.0, 5.0, 1.0])  # Example goal position
        self._reached_goal = False

    def compute_reward(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        goal = self.goal  # Example goal location
        distance = np.linalg.norm(drone_pos - goal)

        if not hasattr(self, "_last_distance"):
            self._last_distance = distance
        
        progress = self._last_distance - distance

        reward = progress * 10 #Encourage movement toward the goal 
        reward = np.clip(reward, -10.0, 10)  

        if distance < 0.2:
            reward += 50  # smaller bonus
            self._reached_goal = True
        
        self._last_distance = distance
        return reward
    
    def check_termination(self):
        pos = self._getDroneStateVector(0)[0:3]
        
        if pos[2] < 0.1:  # Crashed into ground
            return True
        if np.linalg.norm(pos) > 20:  # Out of bounds
            return True
        if self._reached_goal:  # Reached goal
            return True
        
        return False


    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        
        reward = self.compute_reward()
        terminated = self.check_termination()

        obs = self._getDroneStateVector(0)
        if obs is None:
            obs = np.zeros(12)  # fallback in case sim is unstable
        
        return np.array(obs), float(reward), terminated, truncated, dict(info)



    def reset(self, seed=None, options=None):
        self._reached_goal = False
        self.goal = np.random.uniform(low=[3.0, 3.0, 1.0], high=[6.0, 6.0, 1.0])
        result = super().reset(seed=seed, options=options)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        return np.array(obs), info
