import gymnasium as gym
import torch
import numpy as np
from processing.preprocessing import FrameStackWrapper
from models.ddqn_agent import DDQNAgent

# Same parameters as training
INPUT_SHAPE = (4, 84, 84)
NUM_ACTIONS = 5
LEARNING_RATE = 0.00005
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.9995

def test_agent(model_path, num_episodes=5, render=True):
    # Create environment with rendering
    render_mode = "human" if render else "rgb_array"
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=False)
    env = FrameStackWrapper(env)
    
    # Create agent and load trained model
    agent = DDQNAgent(INPUT_SHAPE, NUM_ACTIONS, LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY)
    agent.main_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    agent.main_network.eval()  # Set to evaluation mode
    print("Model loaded successfully!")

    test_state = np.random.random((4, 84, 84)).astype(np.float32)
    test_action = agent.select_action(test_state, training=False)
    print(f"Test action prediction: {test_action}")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        actions_taken = []
        
        while not done:
            # Use trained agent (no exploration)
            action = agent.select_action(state, training=False)
            actions_taken.append(action)

            if step < 10:
                print(f"Step {step}: Action = {action}")

            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step}")

        action_counts = {}
        for action in actions_taken:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step}")
        print(f"Action distribution: {action_counts}")
        print("---")
    
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(f"Best episode: {max(episode_rewards):.2f}")
    print(f"Worst episode: {min(episode_rewards):.2f}")

if __name__ == "__main__":
    # Test with the final saved model
    test_agent("ddqn_model_episode_1500.pth", num_episodes=5, render=True)