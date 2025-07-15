import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
import time

class AdvancedDroneEnv(gym.Env):
    def __init__(self):
        super(AdvancedDroneEnv, self).__init__()
        
        # Environment parameters - More reasonable for learning
        self.max_speed = 4.0  # Reduced for better control
        self.max_force = 30.0  # Reduced
        self.gravity = 9.8
        self.mass = 1.0
        self.k_p = 10.0  # Reduced for stability
        self.k_d = 2.0   # Reduced
        self.episode_length = 800  # Shorter episodes
        self.goal_threshold = 1.5  # Slightly larger for easier success
        self.min_altitude = 0.3
        self.max_altitude = 8.0
        
        # Action and observation spaces - Fixed bounds
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Proper observation bounds
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(9,), dtype=np.float32)
        
        # PyBullet setup
        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(fixedTimeStep=1/120, numSubSteps=2)  # Simpler physics
        
        # Environment state
        self.drone = None
        self.obstacles = []
        self.start_pos = np.array([0, 0, 2], dtype=np.float32)
        self.goal_pos = np.array([15, 0, 2], dtype=np.float32)  # Closer goal
        self.current_step = 0
        self.prev_distance = None
        self.prev_velocity = np.zeros(3)
        self.obstacle_params = []
        self.best_distance = None
        self.initial_distance = None

    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            
        p.resetSimulation()
        p.setGravity(0, 0, -self.gravity)
        
        # Create ground plane
        ground = p.loadURDF("plane.urdf")
        
        # Simpler, less random positions for initial learning
        self.start_pos = np.array([
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1), 
            np.random.uniform(1.5, 2.5)
        ])
        
        self.goal_pos = np.array([
            np.random.uniform(12, 18),  # Closer and less variable
            np.random.uniform(-2, 2),
            np.random.uniform(1.5, 2.5)
        ])
        
        # Create drone (sphere for better collision detection)
        drone_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2)  # Slightly larger
        drone_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0, 0, 1])
        self.drone = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=drone_col,
            baseVisualShapeIndex=drone_vis,
            basePosition=self.start_pos
        )
        
        # Create goal marker
        goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 0, 0.5])
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_vis,
            basePosition=self.goal_pos
        )
        
        # Create obstacle course
        self._create_obstacle_course()
        
        # Reset state variables
        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.start_pos - self.goal_pos)
        self.initial_distance = self.prev_distance
        self.best_distance = self.prev_distance
        self.prev_velocity = np.zeros(3)
        
        return self._get_observation(), {}

    def _create_obstacle_course(self):
        """Create a simpler, learnable obstacle course"""
        self.obstacles = []
        self.obstacle_params = []
        
        # Much simpler obstacle layout - curriculum learning approach
        obstacles = [
            # Simple corridor walls
            (7, -3, 2, 0.5, 0.5, 2),   # Left wall segment
            (7, 3, 2, 0.5, 0.5, 2),    # Right wall segment
            
            # Mid-point challenge
            (10, -1, 2, 0.5, 2, 2),    # Narrow passage
            (10, 1, 2, 0.5, 2, 2),
            
            # Final approach
            (13, 0, 2, 2, 0.5, 2),     # Single barrier to navigate around
            
            # Boundary walls (wider spacing)
            (7.5, -5, 2, 15, 0.3, 3),   # Bottom boundary
            (7.5, 5, 2, 15, 0.3, 3),    # Top boundary
        ]
        
        for x, y, z, w, d, h in obstacles:
            half_extents = [w/2, d/2, h/2]
            obstacle_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            obstacle_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, 
                                             rgbaColor=[0.7, 0.7, 0.7, 1])
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=obstacle_col,
                baseVisualShapeIndex=obstacle_vis,
                basePosition=[x, y, z]
            )
            self.obstacles.append(obstacle_id)
            self.obstacle_params.append(([x, y, z], half_extents))

    def _get_observation(self):
        """Enhanced observation space with velocity and relative positioning"""
        pos, _ = p.getBasePositionAndOrientation(self.drone)
        vel, ang_vel = p.getBaseVelocity(self.drone)
        
        # Relative position to goal
        relative_goal = self.goal_pos - np.array(pos)
        
        obs = np.concatenate([
            pos,                    # Current position (3)
            vel,                    # Current velocity (3) 
            relative_goal           # Relative goal position (3)
        ]).astype(np.float32)
        
        return obs

    def _check_collision(self):
        """Improved collision detection"""
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        
        # Check altitude bounds
        if drone_pos[2] < self.min_altitude or drone_pos[2] > self.max_altitude:
            return True
        
        # Check obstacle collisions
        for obstacle in self.obstacles:
            contact_points = p.getContactPoints(self.drone, obstacle)
            if contact_points:
                return True
        return False

    def _calculate_proximity_reward(self, drone_pos):
        """Calculate gentler proximity rewards/penalties"""
        min_distance = float('inf')
        
        for (center, half_extents) in self.obstacle_params:
            # Calculate distance to obstacle surface
            closest_point = np.clip(drone_pos, 
                                  np.array(center) - np.array(half_extents),
                                  np.array(center) + np.array(half_extents))
            distance = np.linalg.norm(drone_pos - closest_point)
            min_distance = min(min_distance, distance)
        
        # Much gentler proximity rewards
        if min_distance < 0.5:
            return -5 * (0.5 - min_distance)  # Mild penalty for very close
        elif min_distance < 1.0:
            return -2 * (1.0 - min_distance)  # Light penalty for close
        elif min_distance > 2.0:
            return 1.0  # Small bonus for safe distance
        else:
            return 0  # Neutral zone
    
    def step(self, action):
        # Enhanced PD controller with better stability
        desired_velocity = action * self.max_speed
        current_velocity, _ = p.getBaseVelocity(self.drone)
        
        # PD control with derivative term
        velocity_error = np.array(desired_velocity) - np.array(current_velocity)
        acceleration_term = self.k_d * (np.array(current_velocity) - self.prev_velocity)
        
        force = self.k_p * velocity_error - acceleration_term
        force = np.clip(force, -self.max_force, self.max_force)
        
        # Gravity compensation
        force[2] += self.mass * self.gravity
        
        # Apply force
        p.applyExternalForce(self.drone, -1, force.tolist(), [0, 0, 0], p.WORLD_FRAME)
        p.stepSimulation()
        
        # Update state
        self.prev_velocity = current_velocity
        
        # Get new observation
        obs = self._get_observation()
        drone_pos = obs[:3]
        
        # Calculate distances
        distance_to_goal = np.linalg.norm(drone_pos - self.goal_pos)
        progress = self.prev_distance - distance_to_goal
        self.prev_distance = distance_to_goal
        
        # Track best distance for bonus rewards
        new_best = distance_to_goal < self.best_distance
        if new_best:
            self.best_distance = distance_to_goal
        
        # IMPROVED REWARD SYSTEM
        reward = 0.0
        
        # 1. Strong progress reward - main learning signal
        if progress > 0:
            reward += progress * 50  # Increased reward for progress
        
        # 2. Distance-based reward - always positive baseline
        max_distance = 30.0  # Reasonable max distance
        proximity_bonus = (max_distance - distance_to_goal) / max_distance * 10
        reward += proximity_bonus
        
        # 3. Best distance achievement bonus
        if new_best:
            improvement = self.initial_distance - distance_to_goal
            reward += improvement * 5  # Bonus for new best distance
        
        # 4. Gentle proximity guidance
        proximity_reward = self._calculate_proximity_reward(drone_pos)
        reward += proximity_reward
        
        # 5. Movement encouragement
        velocity_magnitude = np.linalg.norm(current_velocity)
        if velocity_magnitude > 0.2:
            reward += 0.1  # Small bonus for moving
        
        # 6. Very light time penalty
        reward -= 0.02
        
        # Check termination conditions
        done = False
        collision = self._check_collision()
        
        if distance_to_goal < self.goal_threshold:
            # Success with significant reward
            efficiency_bonus = max(0, 50 - self.current_step / 10)
            reward += 100 + efficiency_bonus
            done = True
        elif collision:
            reward -= 50  # Moderate collision penalty
            done = True
        elif self.current_step >= self.episode_length:
            # Mild timeout penalty based on progress
            progress_ratio = 1 - (distance_to_goal / self.initial_distance)
            reward += -10 + (progress_ratio * 20)
            done = True
        
        self.current_step += 1
        
        info = {
            'distance_to_goal': distance_to_goal,
            'progress_ratio': 1 - (distance_to_goal / self.initial_distance),
            'collision': collision,
            'success': distance_to_goal < self.goal_threshold
        }
        
        return obs, reward, done, False, info

    def close(self):
        p.disconnect()

def train_advanced_drone():
    """Train the advanced drone navigation system"""
    
    # Create environment
    env = AdvancedDroneEnv()
    env = DummyVecEnv([lambda: env])
    
    # Create SAC agent with more conservative hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,  # Smaller buffer
        learning_starts=5000,  # Earlier learning start
        batch_size=128,      # Smaller batch size
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        target_entropy='auto',
        policy_kwargs=dict(
            net_arch=[256, 256],  # Smaller network to prevent overfitting
            activation_fn=nn.ReLU
        ),
        device="auto",
        tensorboard_log="./advanced_drone_logs/"
    )
    
    print("Starting advanced drone navigation training...")
    print("Fixed Environment Issues:")
    print("- Proper observation space bounds (-100 to 100)")
    print("- Simpler obstacle course for initial learning")
    print("- Shorter navigation distance (12-18 units vs 22-28)")
    print("- Gentler proximity penalties (-2 to -5 vs -20)")
    print("- Always positive baseline reward (proximity bonus)")
    print("- Strong progress rewards (+50 per unit progress)")
    print("- Conservative network size (256x256 vs 512x512x256)")
    print("- Earlier learning start (5k vs 10k steps)")
    print("- Larger goal threshold (1.5 vs 1.0)")
    print("- More stable physics (120Hz vs 240Hz)")
    
    # Train the agent
    start_time = time.time()
    model.learn(total_timesteps=300000, tb_log_name="FixedAdvancedSAC")  # Reduced timesteps
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Save the model
    model.save("advanced_drone_navigation_sac")
    
    return model

def test_model(model_path="advanced_drone_navigation_sac", num_episodes=5):
    """Test the trained model"""
    
    # Load model
    model = SAC.load(model_path)
    
    # Create test environment
    env = AdvancedDroneEnv()
    
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        print(f"Start: {env.start_pos}")
        print(f"Goal: {env.goal_pos}")
        print(f"Distance: {np.linalg.norm(env.goal_pos - env.start_pos):.2f}")
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        print(f"Steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final distance: {info['distance_to_goal']:.2f}")
        print(f"Progress: {info['progress_ratio']*100:.1f}%")
        
        if info['success']:
            print("SUCCESS!")
            success_count += 1
        elif info['collision']:
            print("COLLISION")
        else:
            print("TIMEOUT")
    
    print(f"\nOverall success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    
    env.close()

if __name__ == "__main__":
    # Train the model
    model = train_advanced_drone()
    
    # Test the trained model
    test_model()
