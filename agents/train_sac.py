#!/usr/bin/env python3
 
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
    # This class will ensure that the environment difficulty is increased automatically during training. 
    
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.last_difficulty_update = 0
	
	# So every 150,000 training steps the training difficulty is increased
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_difficulty_update >= 150000:
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
    FIXED: Proper 3D box-sphere collision detection with path clearance guarantee
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
        self.workspace_bounds = 5.0
        self.obstacles = []
        self.obstacle_ids = []
        
        # Progressive difficulty parameters
        self.difficulty_level = 0  # Start at level 0 (easiest)
        self.max_difficulty = 7
        
        # Difficulty-dependent parameters
        self.num_obstacles_by_level = [0, 1, 2, 3, 5, 6, 8, 10]  # Start with NO obstacles
        self.obstacle_sizes_by_level = [
            [0.1, 0.15],   # Level 0
            [0.15, 0.20],  # Level 1
            [0.15, 0.20],  # Level 2
            [0.12, 0.18],  # Level 3
            [0.12, 0.18],  # Level 4
            [0.10, 0.16],  # Level 5
            [0.10, 0.15],  # Level 6
            [0.08, 0.14]   # Level 7 (10 obstacles)
        ]
        
        # Waypoints - start closer, get further
        self.start_waypoint = np.array([-1.0, -3.0, 1.0])
        self.end_waypoints_by_level = [
            np.array([0.0, 1.5, 1.2]),   # Level 0
            np.array([0.0, 2.0, 1.3]),   # Level 1
            np.array([0.5, 2.5, 1.4]),   # Level 2
            np.array([1.0, 3.0, 1.5]),   # Level 3
            np.array([1.5, 3.5, 1.7]),   # Level 4
            np.array([2.0, 4.0, 1.8]),   # Level 5
            np.array([2.5, 4.2, 2.0]),   # Level 6
            np.array([3.0, 4.5, 2.2])    # Level 7
        ]
        
        self.waypoint_threshold = 0.3
        
        # FIXED: Safety parameters - more conservative
        self.min_obstacle_distance = 0.8
        self.collision_threshold = 0.05  # REDUCED from 0.3
        self.drone_radius = 0.15         # REDUCED from 0.2
        self.max_episode_steps = 1200
        self.current_step = 0
        
        # OPTIMIZED REWARD PARAMETERS FOR EARLY LEARNING
        # Terminal rewards
        self.success_reward = 500.0
        self.collision_penalty = -100.0
        self.oob_penalty = -15.0
        
        # Progress rewards (dominant signal)
        self.progress_reward_scale = 200.0
        self.retreat_penalty_scale = 50.0
        
        # Minimal penalties
        self.step_penalty = -0.0001
        self.distance_penalty_scale = 0.0
        
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
        self.success_count = 0
        
        # NEW: Debug visualization
        self.debug_lines = []
        
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
        
        additional_dim = 7
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
        
        # Clear obstacles and debug lines safely
        self._clear_obstacles_safe()
        self._clear_debug_lines()
        
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
        """Execute step with FIXED collision detection."""
        obs, _, terminated, truncated, info = super().step(action)
        
        # Get current position for debugging
        current_pos = self.pos[0]
        
        # FIXED: Check collision ONCE with proper 3D detection
        collision_detected = self._check_collision_3d_box_sphere()
        
        if collision_detected:
            print(f"💥 COLLISION DETECTED at step {self.current_step}")
            print(f"   Position: {current_pos}")
            reward = self.collision_penalty
            terminated = True
        else:
            # Calculate reward without collision checking
            reward = self._calculate_optimized_reward()
            # Check other termination conditions (success, out of bounds)
            terminated = self._check_termination_no_collision()
        
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
            'collision': collision_detected,
            'distance_to_goal': np.linalg.norm(self.pos[0] - current_end),
            'episode_length': self.current_step,
            'difficulty_level': self.difficulty_level,
            'success_count': self.success_count
        })
        
        return enhanced_obs, reward, terminated, truncated, info
    
    def _create_enhanced_observation(self, original_obs):
        """Create destination-only observation - NO obstacle information."""
        obs_flat = original_obs.flatten() if hasattr(original_obs, 'flatten') else np.array(original_obs).flatten()
        
        current_pos = self.pos[0]
        current_vel = self.vel[0] if hasattr(self, 'vel') else np.zeros(3)
        current_end = self.get_current_end_waypoint()
        
        # Goal information ONLY
        goal_vector = current_end - current_pos
        goal_distance = np.linalg.norm(goal_vector)
        goal_direction = goal_vector / (goal_distance + 1e-8)
        
        # Progress metrics
        progress_ratio = self._get_progress_ratio()
        
        # Velocity toward goal
        velocity_toward_goal = np.dot(current_vel, goal_direction)
        
        # Efficiency metric
        efficiency = self._get_efficiency_metric()
        
        # DESTINATION-ONLY FEATURES (no obstacle info)
        additional_features = np.array([
            # Goal direction (normalized)
            goal_direction[0], goal_direction[1], goal_direction[2],
            # Goal distance (normalized)
            goal_distance / self.initial_distance,
            # Progress ratio
            progress_ratio,
            # Velocity alignment with goal
            np.clip(velocity_toward_goal, -2.0, 2.0),
            # Efficiency metric
            efficiency
        ], dtype=np.float32)
        
        return np.concatenate([obs_flat, additional_features])
    
    def _calculate_optimized_reward(self):
        """FIXED: Optimized reward function WITHOUT collision checking."""
        current_pos = self.pos[0]
        current_vel = self.vel[0] if hasattr(self, 'vel') else np.zeros(3)
        current_end = self.get_current_end_waypoint()
        total_reward = 0.0
        
        current_distance = np.linalg.norm(current_pos - current_end)
        
        # 1. SUCCESS REWARD (collision already handled in step())
        if self._is_success():
            # Massive success reward with difficulty bonus
            difficulty_bonus = self.difficulty_level * 120.0
            efficiency_bonus = self.efficiency_bonus_scale * self._get_efficiency_metric()
            total_reward = self.success_reward + difficulty_bonus + efficiency_bonus
            print(f"SUCCESS! Level {self.difficulty_level}, Base: {self.success_reward}, "
                  f"Difficulty: +{difficulty_bonus}, Efficiency: +{efficiency_bonus:.1f}, "
                  f"Total: {total_reward:.1f}")
            return total_reward
        
        # 2. OUT OF BOUNDS (only non-collision termination checked here)
        if self._check_out_of_bounds():
            return self.oob_penalty
        
        # 3. MASSIVE PROGRESS REWARD (main learning signal)
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
        
        # 4. VELOCITY ALIGNMENT REWARD (encourage right direction)
        goal_direction = (current_end - current_pos)
        goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
        velocity_alignment = np.dot(current_vel, goal_direction)
        velocity_reward = max(0, velocity_alignment) * self.velocity_reward_scale  # Only positive alignment
        total_reward += velocity_reward
        
        # 5. DISTANCE-BASED MOTIVATION (closer to goal = higher baseline reward)
        # Give higher baseline reward for being closer to goal
        max_distance = np.linalg.norm(self.get_current_end_waypoint() - self.start_waypoint)
        proximity_reward = (1.0 - current_distance / max_distance) * 2.0
        total_reward += proximity_reward
        
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
        """COMPLETELY REWRITTEN: Generate obstacles with guaranteed path clearance."""
        self.obstacles = []
        self.obstacle_ids = []
        
        num_obstacles = self.num_obstacles_by_level[self.difficulty_level]
        if num_obstacles == 0:
            return  # No obstacles at difficulty 0
        
        current_end = self.get_current_end_waypoint()
        obstacle_size_range = self.obstacle_sizes_by_level[self.difficulty_level]
        
        # CRITICAL: Calculate direct path and ensure clearance
        path_vector = current_end - self.start_waypoint
        path_length = np.linalg.norm(path_vector)
        path_direction = path_vector / path_length
        
        # Create path points to protect
        protected_points = []
        num_path_points = 20
        for i in range(num_path_points):
            t = i / (num_path_points - 1)
            point = self.start_waypoint + t * path_vector
            protected_points.append(point)
        
        print(f"Generating {num_obstacles} obstacles for difficulty level {self.difficulty_level}")
        
        for i in range(num_obstacles):
            max_attempts = 200  # Increased attempts
            obstacle_placed = False
            
            for attempt in range(max_attempts):
                # Generate position with better distribution
                if self.difficulty_level <= 1:
                    # Easy: well away from path
                    angle = np.random.uniform(0, 2*np.pi)
                    radius = np.random.uniform(1.5, 2.5)
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    z = np.random.uniform(0.8, 2.0)
                else:
                    # Harder: can be closer but still maintain clearance
                    x = np.random.uniform(-2.5, 2.5)
                    y = np.random.uniform(-2.5, 2.5)
                    z = np.random.uniform(0.8, 2.2)
                
                pos = np.array([x, y, z])
                
                # Check waypoint distances
                start_dist = np.linalg.norm(pos - self.start_waypoint)
                end_dist = np.linalg.norm(pos - current_end)
                
                if start_dist < 1.0 or end_dist < 1.0:
                    continue
                
                # CRITICAL: Check clearance from ALL protected path points
                min_path_clearance = float('inf')
                for path_point in protected_points:
                    clearance = np.linalg.norm(pos - path_point)
                    min_path_clearance = min(min_path_clearance, clearance)
                
                # Ensure sufficient path clearance based on difficulty
                base_clearance = max(0.4, 1.2 - (self.difficulty_level * 0.1))
                required_clearance = base_clearance
                if min_path_clearance < required_clearance:
                    continue
                
                # Check overlap with existing obstacles
                too_close_to_obstacles = False
                min_obstacle_spacing = 1.0
                for existing in self.obstacles:
                    distance = np.linalg.norm(pos - existing['position'])
                    if distance < min_obstacle_spacing:
                        too_close_to_obstacles = True
                        break
                
                if not too_close_to_obstacles:
                    # Valid position found!
                    size = np.random.uniform(*obstacle_size_range)
                    size = min(size, 0.25)  # Cap maximum size
                    
                    obstacle_id = self._create_obstacle_with_debug(pos, size)
                    if obstacle_id is not None:
                        self.obstacle_ids.append(obstacle_id)
                        self.obstacles.append({
                            'position': pos,
                            'size': size,
                            'id': obstacle_id
                        })
                        print(f"  ✅ Obstacle {i} placed at {pos[:2]} (z={pos[2]:.1f}), size={size:.2f}, path_clearance={min_path_clearance:.2f}")
                        obstacle_placed = True
                        break
            
            if not obstacle_placed:
                print(f"  ⚠️ Could not place obstacle {i} after {max_attempts} attempts")
    
    def _create_obstacle_with_debug(self, pos, size):
        """ENHANCED: Create obstacle with visual debug information."""
        try:
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size, size, size],
                physicsClientId=self.CLIENT
            )
            
            if self.GUI:
                # Main obstacle (red)
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
                
                # ADD VISUAL DEBUG: Show collision boundary
                danger_zone = size + self.drone_radius + 0.1  # Show actual collision zone
                
                # Create wireframe box around danger zone
                corners = [
                    [pos[0]-danger_zone, pos[1]-danger_zone, pos[2]-danger_zone],
                    [pos[0]+danger_zone, pos[1]-danger_zone, pos[2]-danger_zone],
                    [pos[0]+danger_zone, pos[1]+danger_zone, pos[2]-danger_zone],
                    [pos[0]-danger_zone, pos[1]+danger_zone, pos[2]-danger_zone],
                    [pos[0]-danger_zone, pos[1]-danger_zone, pos[2]+danger_zone],
                    [pos[0]+danger_zone, pos[1]-danger_zone, pos[2]+danger_zone],
                    [pos[0]+danger_zone, pos[1]+danger_zone, pos[2]+danger_zone],
                    [pos[0]-danger_zone, pos[1]+danger_zone, pos[2]+danger_zone],
                ]
                
                # Draw wireframe (yellow lines)
                edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                for edge in edges:
                    line_id = p.addUserDebugLine(
                        corners[edge[0]], corners[edge[1]], 
                        [1, 1, 0], 1, 0, physicsClientId=self.CLIENT
                    )
                    self.debug_lines.append(line_id)
                
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
    
    def _clear_debug_lines(self):
        """NEW: Clear debug visualization lines."""
        for line_id in self.debug_lines:
            try:
                p.removeUserDebugItem(line_id, physicsClientId=self.CLIENT)
            except:
                pass
        self.debug_lines = []
    
    def _check_collision_3d_box_sphere(self):
        """COMPLETELY REWRITTEN: Proper 3D box-sphere collision detection."""
        current_pos = self.pos[0]
        
        # Ground collision
        if current_pos[2] < 0.05:
            print(f"🚨 GROUND COLLISION at z={current_pos[2]:.3f}")
            return True
        
        # 3D Box-Sphere collision detection
        if self.obstacles:
            for i, obstacle in enumerate(self.obstacles):
                obs_pos = obstacle['position']
                obs_size = obstacle['size']  # half-extent of box
                
                # Calculate closest point on box to sphere center
                closest_point = np.array([
                    max(obs_pos[0] - obs_size, min(current_pos[0], obs_pos[0] + obs_size)),
                    max(obs_pos[1] - obs_size, min(current_pos[1], obs_pos[1] + obs_size)),
                    max(obs_pos[2] - obs_size, min(current_pos[2], obs_pos[2] + obs_size))
                ])
                
                # Distance from drone center to closest point on box
                distance_to_box = np.linalg.norm(current_pos - closest_point)
                
                # Debug output when getting close
                if distance_to_box < self.drone_radius + 0.2:
                    print(f"🔍 APPROACHING OBSTACLE {i}:")
                    print(f"    Drone pos: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]")
                    print(f"    Obstacle pos: [{obs_pos[0]:.2f}, {obs_pos[1]:.2f}, {obs_pos[2]:.2f}], size: {obs_size:.2f}")
                    print(f"    Closest point: [{closest_point[0]:.2f}, {closest_point[1]:.2f}, {closest_point[2]:.2f}]")
                    print(f"    Distance to box: {distance_to_box:.3f}m")
                    print(f"    Drone radius: {self.drone_radius:.3f}m")
                    print(f"    Collision threshold: {self.collision_threshold:.3f}m")
                
                # Check collision: drone sphere intersects box
                if distance_to_box <= (self.drone_radius + self.collision_threshold):
                    print(f"🚨 BOX-SPHERE COLLISION DETECTED!")
                    print(f"    Obstacle {i}: distance {distance_to_box:.3f} <= threshold {self.drone_radius + self.collision_threshold:.3f}")
                    return True
        
        return False
    
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
    
    def _check_termination_no_collision(self):
        """Check termination without collision (collision handled in step())."""
        return self._is_success() or self._check_out_of_bounds()


def make_progressive_env(gui=False):
    """Create progressive environment."""
    def _init():
        env = ProgressiveObstacleNavigationEnv(gui=gui)
        return Monitor(env)
    return _init


def train_progressive_drone_navigation():
    """Training with progressive difficulty."""
    
    config = {
        'total_timesteps': 2500000,  # Longer training for progression
        'learning_rate': 5e-4,  # Higher learning rate for faster convergence
        'buffer_size': 500000,  # Smaller buffer for faster turnover
        'batch_size': 128,  # Smaller batches for more frequent updates
        'learning_starts': 3000,  # Start learning very early
        'eval_freq': 10000,  # More frequent evaluation
        'n_eval_episodes': 5,  # Fewer eval episodes for speed
        'reward_threshold': 10000.0,  # Higher threshold for better performance
        'n_envs': 8
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/progressive_drone_nav_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("🚀 Starting Progressive Difficulty Drone Training - COLLISION DETECTION COMPLETELY FIXED")
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
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        deterministic=True,
        verbose=1,
    )
    
    # Callbacks
    progressive_callback = ProgressiveDifficultyCallback(train_env, verbose=1)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        deterministic=True,
        verbose=1,
    )
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callback_list = CallbackList([progressive_callback, eval_callback])
   
    # Train
    print("📊 Level 0: No obstacles, close goal")
    print("📊 Level 1: 1 obstacle, simple navigation") 
    print("📊 Level 2: 2 obstacles, basic avoidance")
    print("📊 Level 3: 3 obstacles, path planning")
    print("📊 Level 4: 5 obstacles, complex navigation")
    print("📊 Level 5: 6 obstacles, precision flying")
    print("📊 Level 6: 8 obstacles, advanced scenarios")
    print("📊 Level 7: 10 obstacles, ultimate challenge")
    
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
