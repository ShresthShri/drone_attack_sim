import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque
import random

class RewardAgent:
    """Assistant Reward Agent based on ReLara framework"""
    
    def __init__(self, state_dim, action_dim, reward_range=(-1.0, 1.0)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_range = reward_range
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ReLara dual network architecture
        self.policy_net = self._build_policy_network()
        self.q_net = self._build_q_network()
        self.target_q_net = self._build_q_network()
        
        # Copy weights to target network
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        
        # Experience buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # ReLara specific parameters
        self.burn_in_episodes = 10  # Episode-based burn-in
        self.burn_in_steps = 5000   # Step-based burn-in
        self.episodes_completed = 0
        self.total_steps = 0
        self.tau = 0.005  # Target network update rate
        
    def _build_policy_network(self):
        """Build policy network for reward generation"""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        ).to(self.device)
    
    def _build_q_network(self):
        """Build Q-network for reward agent"""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim + 1, 256),  # +1 for suggested reward
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
    
    def get_suggested_reward(self, state, action):
        """Generate suggested reward based on ReLara methodology"""
        self.total_steps += 1
        
        # Exploration phase: random rewards during burn-in
        if (self.episodes_completed < self.burn_in_episodes or 
            self.total_steps < self.burn_in_steps):
            suggested_reward = np.random.uniform(*self.reward_range)
            return suggested_reward
        
        # Exploitation phase: use learned policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            input_tensor = torch.cat([state_tensor, action_tensor], dim=1)
            suggested_reward = self.policy_net(input_tensor).item()
            # Scale from [-1, 1] to reward range
            suggested_reward = suggested_reward * (self.reward_range[1] - self.reward_range[0]) / 2
        
        return suggested_reward
    
    def store_transition(self, state, action, next_state, next_action, 
                        suggested_reward, env_reward):
        """Store transition in replay buffer"""
        self.replay_buffer.append({
            'state': state.copy(),
            'action': action.copy(),
            'next_state': next_state.copy(),
            'next_action': next_action.copy(),
            'suggested_reward': suggested_reward,
            'env_reward': env_reward
        })
    
    def episode_ended(self):
        """Call when episode ends"""
        self.episodes_completed += 1
    
    def update_networks(self):
        """Update reward agent networks using ReLara algorithm"""
        if (len(self.replay_buffer) < 256 or 
            self.episodes_completed < self.burn_in_episodes):
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, min(256, len(self.replay_buffer)))
        
        # Prepare tensors
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        actions = torch.FloatTensor([exp['action'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
        next_actions = torch.FloatTensor([exp['next_action'] for exp in batch]).to(self.device)
        suggested_rewards = torch.FloatTensor([exp['suggested_reward'] for exp in batch]).to(self.device)
        env_rewards = torch.FloatTensor([exp['env_reward'] for exp in batch]).to(self.device)
        
        # Update Q-network
        current_q_input = torch.cat([states, actions, suggested_rewards.unsqueeze(1)], dim=1)
        current_q_values = self.q_net(current_q_input)
        
        with torch.no_grad():
            next_suggested_rewards = self.policy_net(torch.cat([next_states, next_actions], dim=1))
            next_q_input = torch.cat([next_states, next_actions, next_suggested_rewards], dim=1)
            next_q_values = self.target_q_net(next_q_input)
            target_q_values = env_rewards.unsqueeze(1) + 0.99 * next_q_values
        
        q_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Update Policy network
        current_suggested_rewards = self.policy_net(torch.cat([states, actions], dim=1))
        policy_q_input = torch.cat([states, actions, current_suggested_rewards], dim=1)
        policy_loss = -self.q_net(policy_q_input).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update target network
        self._soft_update_target_network()
    
    def _soft_update_target_network(self):
        """Soft update target network"""
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def is_exploration_phase(self):
        """Check if in exploration phase"""
        return (self.episodes_completed < self.burn_in_episodes or 
                self.total_steps < self.burn_in_steps)

class AdaptiveDroneEnv(gym.Env):
    """Enhanced drone environment with ReLara reward shaping"""
    
    def __init__(self):
        super(AdaptiveDroneEnv, self).__init__()
        
        # Environment parameters
        self.max_speed = 4.0
        self.max_force = 15.0
        self.gravity = 9.8
        self.mass = 1.0
        self.k_p = 5.0
        self.k_d = 1.0
        self.episode_length = 1000
        self.goal_threshold = 0.5
        self.min_altitude = 0.3
        self.max_altitude = 8.0
        
        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-30, -30, 0, -15, -15, -15, -50, -50, -15]),
            high=np.array([30, 30, 15, 15, 15, 15, 50, 50, 15]),
            dtype=np.float32
        )
        
        # PyBullet setup
        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(fixedTimeStep=1/60, numSubSteps=1)
        
        # Initialize ReLara reward agent
        self.reward_agent = RewardAgent(
            state_dim=self.observation_space.shape[0],
            action_dim=self.action_space.shape[0],
            reward_range=(-1.0, 1.0)
        )
        
        # ReLara parameters
        self.beta = 0.3  # Suggested reward weight
        self.success_count = 0
        self.total_episodes = 0
        
        # Environment state
        self.drone = None
        self.obstacles = []
        self.start_pos = np.array([0, 0, 2], dtype=np.float32)
        self.goal_pos = np.array([15, 0, 2], dtype=np.float32)
        self.current_step = 0
        self.prev_distance = None
        self.prev_velocity = np.zeros(3)
        self.obstacle_params = []
        self.initial_distance = None
        
        # Store previous state-action for reward agent
        self.prev_state = None
        self.prev_action = None
        self.prev_suggested_reward = None

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            
        p.resetSimulation()
        p.setGravity(0, 0, -self.gravity)
        
        # Create ground plane
        ground = p.loadURDF("plane.urdf")
        
        # Generate start and goal positions
        self.start_pos = np.array([
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1), 
            np.random.uniform(1.5, 2.5)
        ])
        
        self.goal_pos = np.array([
            np.random.uniform(12, 18),
            np.random.uniform(-2, 2),
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
        goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 0, 0.5])
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_vis,
            basePosition=self.goal_pos
        )
        
        # Create obstacle course with tighter boundaries
        self._create_obstacle_course()
        
        # Reset state variables
        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.start_pos - self.goal_pos)
        self.initial_distance = self.prev_distance
        self.prev_velocity = np.zeros(3)
        self.prev_state = None
        self.prev_action = None
        self.prev_suggested_reward = None
        
        # Stabilize
        for _ in range(10):
            p.stepSimulation()
        
        return self._get_observation(), {}

    def _create_obstacle_course(self):
        """Create obstacle course with tighter boundaries to prevent cheating"""
        self.obstacles = []
        self.obstacle_params = []
        
        # Fixed obstacle layout with tighter boundaries
        obstacles = [
            # Corridor walls
            (7, -3, 2, 0.5, 0.5, 2),   # Left wall
            (7, 3, 2, 0.5, 0.5, 2),    # Right wall
            
            # Narrow passage
            (10, -1, 2, 0.5, 2, 2),    
            (10, 1, 2, 0.5, 2, 2),
            
            # Final barrier
            (13, 0, 2, 2, 0.5, 2),
            
            # TIGHTER boundary walls to prevent cheating
            (7.5, -3.8, 2, 15, 0.3, 3),   # Bottom boundary (closer)
            (7.5, 3.8, 2, 15, 0.3, 3),    # Top boundary (closer)
        ]
        
        # Create obstacles
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
        """Get normalized observation"""
        pos, _ = p.getBasePositionAndOrientation(self.drone)
        vel, ang_vel = p.getBaseVelocity(self.drone)
        
        relative_goal = self.goal_pos - np.array(pos)
        
        obs = np.concatenate([
            pos,
            vel,
            relative_goal
        ]).astype(np.float32)
        
        # Normalize
        obs[:3] = obs[:3] / 20.0
        obs[3:6] = obs[3:6] / 10.0
        obs[6:9] = obs[6:9] / 20.0
        
        return obs

    def _check_collision(self):
        """Enhanced collision detection"""
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        
        # Altitude collision
        if drone_pos[2] < self.min_altitude:
            return True
        elif drone_pos[2] > self.max_altitude:
            return True
        
        # Obstacle collision
        for obstacle in self.obstacles:
            contact_points = p.getContactPoints(self.drone, obstacle)
            if contact_points:
                return True
        return False

    def _calculate_environmental_reward(self, drone_pos, current_velocity):
        """Calculate environmental reward with enhanced obstacle penalties"""
        distance_to_goal = np.linalg.norm(drone_pos - self.goal_pos)
        progress = self.prev_distance - distance_to_goal
        self.prev_distance = distance_to_goal
        
        env_reward = 0.0
        
        # Base proximity reward
        max_distance = 30.0
        proximity_reward = (max_distance - distance_to_goal) / max_distance
        env_reward += proximity_reward
        
        # Progress reward
        if progress > 0:
            env_reward += progress * 10.0
        
        # Y-constraint penalty to prevent going around obstacles
        y_pos = drone_pos[1]
        if abs(y_pos) > 3.5:  # Penalize going too far in Y direction
            env_reward -= 20.0 * (abs(y_pos) - 3.5)
        
        # Obstacle avoidance reward
        min_distance = float('inf')
        for (center, half_extents) in self.obstacle_params:
            closest_point = np.clip(drone_pos, 
                                  np.array(center) - np.array(half_extents),
                                  np.array(center) + np.array(half_extents))
            distance = np.linalg.norm(drone_pos - closest_point)
            min_distance = min(min_distance, distance)
        
        if min_distance < 0.5:
            env_reward -= 30.0 * (0.5 - min_distance)
        elif min_distance < 1.0:
            env_reward -= 15.0 * (1.0 - min_distance)
        
        # Altitude safety reward
        altitude = drone_pos[2]
        if altitude < 0.8:
            env_reward -= 20.0 * (0.8 - altitude)
        elif altitude > 6.0:
            env_reward -= 10.0 * (altitude - 6.0)
        elif 1.5 <= altitude <= 4.0:
            env_reward += 1.0
        
        return env_reward

    def step(self, action):
        # Apply action with enhanced altitude safety
        desired_velocity = action * self.max_speed
        current_velocity, _ = p.getBaseVelocity(self.drone)
        
        # Get current position
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        current_altitude = drone_pos[2]
        
        # Enhanced altitude safety
        if current_altitude < 1.0 and desired_velocity[2] < 0:
            desired_velocity[2] = max(desired_velocity[2], 0.5)
        elif current_altitude > 6.0 and desired_velocity[2] > 0:
            desired_velocity[2] = min(desired_velocity[2], -0.2)
        
        # PD controller
        velocity_error = np.array(desired_velocity) - np.array(current_velocity)
        acceleration_term = self.k_d * (np.array(current_velocity) - self.prev_velocity)
        
        force = self.k_p * velocity_error - acceleration_term
        force = np.clip(force, -self.max_force, self.max_force)
        
        # Enhanced gravity compensation
        gravity_compensation = self.mass * self.gravity
        if current_altitude < 1.5:
            gravity_compensation *= 1.3
        
        force[2] += gravity_compensation
        
        p.applyExternalForce(self.drone, -1, force.tolist(), [0, 0, 0], p.WORLD_FRAME)
        p.stepSimulation()
        
        self.prev_velocity = current_velocity
        
        # Get observation
        obs = self._get_observation()
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        
        # Calculate environmental reward
        env_reward = self._calculate_environmental_reward(drone_pos, current_velocity)
        
        # Get suggested reward from ReLara agent
        suggested_reward = self.reward_agent.get_suggested_reward(obs, action)
        
        # Store transition for reward agent training
        if self.prev_state is not None:
            self.reward_agent.store_transition(
                self.prev_state, self.prev_action, obs, action,
                self.prev_suggested_reward, env_reward
            )
        
        # ReLara augmented reward
        total_reward = env_reward + self.beta * suggested_reward
        
        # Update reward agent
        if self.current_step % 10 == 0:
            self.reward_agent.update_networks()
        
        # Time penalty
        total_reward -= 0.01
        
        # Check termination
        done = False
        collision = self._check_collision()
        distance_to_goal = np.linalg.norm(drone_pos - self.goal_pos)
        
        if distance_to_goal < self.goal_threshold:
            total_reward += 100.0
            done = True
            self.success_count += 1
        elif collision:
            total_reward -= 50.0
            done = True
        elif self.current_step >= self.episode_length:
            progress_ratio = 1 - (distance_to_goal / self.initial_distance)
            total_reward += progress_ratio * 20.0 - 10.0
            done = True
        
        if done:
            self.total_episodes += 1
            self.reward_agent.episode_ended()
        
        # Store current state-action for next iteration
        self.prev_state = obs.copy()
        self.prev_action = action.copy()
        self.prev_suggested_reward = suggested_reward
        
        self.current_step += 1
        
        info = {
            'distance_to_goal': distance_to_goal,
            'progress_ratio': 1 - (distance_to_goal / self.initial_distance),
            'collision': collision,
            'success': distance_to_goal < self.goal_threshold,
            'env_reward': env_reward,
            'aux_reward': suggested_reward,
            'exploration_phase': self.reward_agent.is_exploration_phase()
        }
        
        return obs, total_reward, done, False, info

    def close(self):
        p.disconnect()

def train_relara_drone():
    """Train drone with proper ReLara implementation"""
    
    env = AdaptiveDroneEnv()
    env = DummyVecEnv([lambda: env])
    
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef=0.1,
        policy_kwargs=dict(
            net_arch=[512, 512, 256],
            activation_fn=nn.ReLU
        ),
        device="auto",
        tensorboard_log="./relara_drone_logs/"
    )
    
    print("Training Drone with ReLara Framework...")
    print("Key Features:")
    print("- Proper dual-agent architecture (Policy + Reward Agent)")
    print("- Episode-based exploration/exploitation balance")
    print("- Enhanced obstacle avoidance with Y-constraints")
    print("- Tighter boundary walls to prevent cheating")
    print("- Proper reward agent learning with Q-networks")
    
    start_time = time.time()
    model.learn(total_timesteps=500000, tb_log_name="ReLara_Drone")
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    model.save("relara_drone_navigation")
    return model

def test_relara_model(model_path="relara_drone_navigation", num_episodes=5):
    """Test the ReLara trained model"""
    model = SAC.load(model_path)
    env = AdaptiveDroneEnv()
    
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        env_reward_sum = 0
        aux_reward_sum = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        print(f"Start: {env.start_pos}")
        print(f"Goal: {env.goal_pos}")
        print(f"Distance: {np.linalg.norm(env.goal_pos - env.start_pos):.2f}")
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            env_reward_sum += info['env_reward']
            aux_reward_sum += info['aux_reward']
            steps += 1
            
            if done or truncated:
                break
        
        print(f"Steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Env reward: {env_reward_sum:.2f}")
        print(f"Aux reward: {aux_reward_sum:.2f}")
        print(f"Final distance: {info['distance_to_goal']:.2f}")
        print(f"Progress: {info['progress_ratio']*100:.1f}%")
        print(f"Exploration phase: {info['exploration_phase']}")
        
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
    model = train_relara_drone()
    test_relara_model()