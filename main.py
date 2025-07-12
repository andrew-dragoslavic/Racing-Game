# Task 1: Environment Setup and Understanding

# Required installations (run in terminal):
# pip install gymnasium[box2d] torch torchvision numpy matplotlib opencv-python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def explore_environment():
    """Explore the CarRacing environment to understand its properties"""
    
    # Create the environment (v3 has both continuous and discrete action modes)
    env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=False)
    
    print("=== CarRacing Environment Analysis ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    if hasattr(env.action_space, 'n'):
        print(f"Number of discrete actions: {env.action_space.n}")
    else:
        print(f"Action space low: {env.action_space.low}")
        print(f"Action space high: {env.action_space.high}")
    
    # Reset environment and get initial observation
    obs, info = env.reset()
    print(f"\nObservation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min()}, {obs.max()}]")
    
    # Take a few random actions to see the environment
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Total reward after random actions: {total_reward}")
    
    # Display a sample observation
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(obs)
    plt.title('Raw Observation (96x96x3)')
    plt.axis('off')
    
    # Show what a grayscale version looks like
    gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    plt.subplot(1, 2, 2)
    plt.imshow(gray_obs, cmap='gray')
    plt.title('Grayscale Version')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('environment_exploration.png', dpi=150, bbox_inches='tight')
    print("Saved environment visualization to 'environment_exploration.png'")
    
    # Print some basic info about what we see
    print(f"\nVisual analysis:")
    print(f"- Image has {np.sum(obs > 200)} bright pixels (likely sky/track)")
    print(f"- Image has {np.sum(obs < 50)} dark pixels (likely grass/borders)")
    print(f"- Average pixel intensity: {np.mean(obs):.1f}")
    
    # Check if car is visible (usually red pixels in center)
    center_region = obs[40:56, 40:56]  # Center 16x16 region
    red_pixels = np.sum(center_region[:,:,0] > center_region[:,:,1])
    print(f"- Red pixels in center (car indicator): {red_pixels}")
    
    env.close()
    
    return obs.shape, env.action_space.n

def analyze_action_space():
    """Analyze the action space for CarRacing-v3"""
    
    print("\n=== Action Space Analysis ===")
    print("CarRacing-v3 can use discrete actions when continuous=False")
    print("Discrete actions: 0=do nothing, 1=steer left, 2=steer right, 3=gas, 4=brake")
    
    # Let's verify this
    env = gym.make('CarRacing-v3', continuous=False)
    print(f"Action space: {env.action_space}")
    print(f"Number of discrete actions: {env.action_space.n}")
    env.close()
    
    # These are the 5 discrete actions available
    action_meanings = [
        "Do nothing",
        "Steer left", 
        "Steer right",
        "Gas",
        "Brake"
    ]
    
    print(f"\nDiscrete action meanings:")
    for i, meaning in enumerate(action_meanings):
        print(f"Action {i}: {meaning}")
    
    return action_meanings

if __name__ == "__main__":
    # Run the exploration
    obs_shape, action_shape = explore_environment()
    action_meanings = analyze_action_space()
    
    print(f"\n=== Summary ===")
    print(f"Observation shape: {obs_shape}")
    print(f"Number of discrete actions: {action_shape}")
    print(f"Actions: {action_meanings}")