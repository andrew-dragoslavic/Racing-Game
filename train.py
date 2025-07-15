import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch

from preprocessing import FrameStackWrapper
from replay_buffer import ReplayBuffer
from ddqn_agent import DDQNAgent

EPISODES = 2000
START_TRAINING = 1000
TARGET_UPDATE_FREQ = 200
BATCH_SIZE = 32

LEARNING_RATE = 0.0001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

BUFFER_SIZE = 50000
INPUT_SHAPE = (4, 84, 84)
NUM_ACTIONS = 5

def train():
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    env = FrameStackWrapper(env)
    agent = DDQNAgent(INPUT_SHAPE, NUM_ACTIONS, LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    episode_rewards = []
    step_count = 0
    for episode in range(EPISODES):
        state, info = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            replay_buffer.push(state, action, reward, next_state, done)
            step_count += 1

            if len(replay_buffer) >= START_TRAINING:
                batch = replay_buffer.sample(BATCH_SIZE)
                if batch is not None:
                    agent.learn(batch)
            
            if step_count % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            state = next_state
        episode_rewards.append(episode_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # Save model periodically (add this)
        if episode % 500 == 0 and episode > 0:
            torch.save(agent.main_network.state_dict(), f'ddqn_model_episode_{episode}.pth')

    torch.save(agent.main_network.state_dict(), 'ddqn_model_final.pth')
    env.close()

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_progress.png')
    print("Training completed!")

if __name__ == "__main__":
    train()