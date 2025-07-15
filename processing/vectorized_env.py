# vectorized_oliver_env.py
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from preprocessing import FrameStackWrapper
from action_space import get_action_from_index, get_num_actions
import cv2

class OliverEnhancedWrapper(gym.Wrapper):
    """Wrapper that adds Oliver's features: road visibility, consecutive negative tracking"""
    
    def __init__(self, env, max_consecutive_neg=20, max_penalty=-5):
        super().__init__(env)
        self.max_consecutive_neg = max_consecutive_neg
        self.max_penalty = max_penalty
        self.consecutive_neg_count = 0
        self.episode_reward = 0
        
    def reset(self, **kwargs):
        self.consecutive_neg_count = 0
        self.episode_reward = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track episode reward
        self.episode_reward += reward
        
        # Oliver's consecutive negative reward tracking
        if reward < 0:
            self.consecutive_neg_count += 1
        else:
            self.consecutive_neg_count = 0
        
        # Oliver's termination conditions
        if self.consecutive_neg_count >= self.max_consecutive_neg:
            terminated = True
            info['termination_reason'] = 'consecutive_negative'
        
        if self.episode_reward <= self.max_penalty:
            terminated = True
            info['termination_reason'] = 'max_penalty'
        
        # Check road visibility
        can_see_road = self.check_road_visibility(obs)
        if not can_see_road:
            terminated = True
            info['termination_reason'] = 'no_road_visible'
        
        return obs, reward, terminated, truncated, info
    
    def check_road_visibility(self, observation):
        """Oliver's road visibility check"""
        try:
            # Take the last frame from the stack
            if len(observation.shape) == 3 and observation.shape[0] == 4:
                frame = observation[-1]  # Last frame in stack
            else:
                frame = observation
            
            # Convert to RGB if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            
            # Convert to 3-channel if grayscale
            if len(frame.shape) == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            
            # Create mask to detect road (gray pixels)
            mask = cv2.inRange(frame, np.array([100, 100, 100]), np.array([150, 150, 150]))
            return np.any(mask == 255)
        except:
            return True  # Default to True if check fails

def make_oliver_env():
    """Create a single Oliver-enhanced environment"""
    def _init():
        env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=False)
        env = FrameStackWrapper(env)
        env = OliverEnhancedWrapper(env)
        return env
    return _init

def create_vectorized_oliver_env(num_envs=10):
    """Create 10 parallel Oliver-enhanced environments"""
    envs = AsyncVectorEnv([make_oliver_env() for _ in range(num_envs)])
    return envs

def convert_continuous_to_discrete_batch(action_tuples):
    """Convert batch of Oliver's continuous actions to CarRacing-v3 discrete actions"""
    discrete_actions = []
    
    for action_tuple in action_tuples:
        steering, gas, brake = action_tuple
        
        # Oliver's action mapping to discrete
        if brake > 0:
            discrete_action = 4  # Brake
        elif gas > 0:
            if steering < -0.5:
                discrete_action = 1  # Left
            elif steering > 0.5:
                discrete_action = 2  # Right
            else:
                discrete_action = 3  # Gas
        else:
            if steering < -0.5:
                discrete_action = 1  # Left
            elif steering > 0.5:
                discrete_action = 2  # Right
            else:
                discrete_action = 0  # Do nothing
        
        discrete_actions.append(discrete_action)
    
    return np.array(discrete_actions)