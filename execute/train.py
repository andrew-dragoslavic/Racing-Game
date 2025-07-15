# train.py - Oliver's approach with 10 parallel environments
import gymnasium as gym
import numpy as np
import torch
from collections import deque
import random

from processing.vectorized_env import create_vectorized_oliver_env, convert_continuous_to_discrete_batch
from action_space import get_num_actions, get_action_from_index
from replay_buffer import ReplayBuffer
from ddqn_agent import DDQNAgent

# Oliver's hyperparameters
EPISODES = 5000
START_TRAINING = 1000
TARGET_UPDATE_FREQ = 500
BATCH_SIZE = 10                # Oliver's small batch size
SKIP_FRAMES = 2               # Oliver's frame skipping
NUM_PARALLEL_ENVS = 10        # Run 10 episodes in parallel

# Network parameters  
INPUT_SHAPE = (4, 84, 84)
NUM_ACTIONS = get_num_actions()  # 12 actions from Oliver's action space
BUFFER_SIZE = 50000
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.9999

def train_parallel_oliver():
    """Train using Oliver's approach with 10 parallel environments"""
    
    # Create 10 parallel environments
    envs = create_vectorized_oliver_env(num_envs=NUM_PARALLEL_ENVS)
    
    agent = DDQNAgent(INPUT_SHAPE, NUM_ACTIONS, LEARNING_RATE, GAMMA, 
                      EPSILON_START, EPSILON_END, EPSILON_DECAY)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    # Track rewards for each environment
    episode_rewards = [[] for _ in range(NUM_PARALLEL_ENVS)]
    step_count = 0
    total_episodes = 0
    
    # Reset all environments
    states, infos = envs.reset()
    episode_rewards_current = np.zeros(NUM_PARALLEL_ENVS)
    
    print(f"Starting parallel training with {NUM_PARALLEL_ENVS} environments...")
    print(f"Using Oliver's enhanced action space with {NUM_ACTIONS} actions")
    
    while total_episodes < EPISODES:
        # Get actions for all 10 environments
        action_tuples = []
        action_indices = []
        
        for env_idx in range(NUM_PARALLEL_ENVS):
            action_tuple, action_index = agent.select_action(states[env_idx], training=True)
            action_tuples.append(action_tuple)
            action_indices.append(action_index)
        
        # Convert Oliver's continuous actions to discrete actions for the environment
        discrete_actions = convert_continuous_to_discrete_batch(action_tuples)
        
        # Apply frame skipping (Oliver's approach)
        total_rewards = np.zeros(NUM_PARALLEL_ENVS)
        final_next_states = None
        final_terminated = None
        final_truncated = None
        final_infos = None
        
        for skip_step in range(SKIP_FRAMES + 1):
            next_states, rewards, terminated, truncated, infos = envs.step(discrete_actions)
            total_rewards += rewards
            
            # Store final results
            final_next_states = next_states
            final_terminated = terminated
            final_truncated = truncated  
            final_infos = infos
            
            # Break if any environment is done
            if np.any(terminated | truncated):
                break
        
        dones = final_terminated | final_truncated
        
        # Store experiences and track rewards for each environment
        for env_idx in range(NUM_PARALLEL_ENVS):
            # Store experience in replay buffer
            replay_buffer.push(
                states[env_idx], 
                action_indices[env_idx], 
                total_rewards[env_idx], 
                final_next_states[env_idx], 
                dones[env_idx]
            )
            
            episode_rewards_current[env_idx] += total_rewards[env_idx]
            step_count += 1
            
            # Episode finished for this environment
            if dones[env_idx]:
                episode_rewards[env_idx].append(episode_rewards_current[env_idx])
                episode_rewards_current[env_idx] = 0
                total_episodes += 1
                
                # Print progress for this environment
                if len(episode_rewards[env_idx]) % 10 == 0:
                    recent_avg = np.mean(episode_rewards[env_idx][-10:])
                    termination_reason = final_infos[env_idx].get('termination_reason', 'normal')
                    print(f"Env {env_idx}: Episode {len(episode_rewards[env_idx])}, "
                          f"Reward: {episode_rewards[env_idx][-1]:.2f}, "
                          f"Avg: {recent_avg:.2f}, "
                          f"Reason: {termination_reason}")
        
        # Training with Oliver's small batch sizes
        if len(replay_buffer) >= START_TRAINING and len(replay_buffer) >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            if batch is not None:
                agent.learn(batch)
        
        # Update target network
        if step_count % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
            print(f"Target network updated at step {step_count}")
        
        # Progress monitoring
        if step_count % 5000 == 0:
            all_recent_rewards = []
            for env_rewards in episode_rewards:
                if len(env_rewards) > 0:
                    all_recent_rewards.extend(env_rewards[-20:])  # Last 20 episodes per env
            
            if all_recent_rewards:
                avg_reward = np.mean(all_recent_rewards)
                print(f"\n=== Step {step_count} Summary ===")
                print(f"Total Episodes Completed: {total_episodes}")
                print(f"Average Reward (recent): {avg_reward:.2f}")
                print(f"Epsilon: {agent.epsilon:.4f}")
                print(f"Replay Buffer Size: {len(replay_buffer)}")
                print("="*40 + "\n")
        
        # Save model periodically
        if total_episodes > 0 and total_episodes % 1000 == 0:
            torch.save(agent.main_network.state_dict(), f'parallel_oliver_ddqn_episodes_{total_episodes}.pth')
            print(f"Model saved at {total_episodes} episodes")
        
        # Update states for next iteration
        states = final_next_states
    
    # Final save
    envs.close()
    torch.save(agent.main_network.state_dict(), 'parallel_oliver_ddqn_final.pth')
    
    # Print final statistics
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    
    for env_idx in range(NUM_PARALLEL_ENVS):
        if episode_rewards[env_idx]:
            avg_reward = np.mean(episode_rewards[env_idx])
            max_reward = max(episode_rewards[env_idx])
            episodes_completed = len(episode_rewards[env_idx])
            print(f"Env {env_idx}: {episodes_completed} episodes, Avg: {avg_reward:.2f}, Max: {max_reward:.2f}")

if __name__ == "__main__":
    train_parallel_oliver()