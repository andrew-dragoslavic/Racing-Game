import torch
import numpy as np
import os
from tqdm import tqdm
import time
from ..agents.ddqn_agent import DDQNAgent
from ..environment.env_wrapper import CarRacingWrapper
from ..environment.parallel_env import make_parallel_env

class DDQNTrainer:
    def __init__(self, config):
        self.config = config
        self.agent = DDQNAgent(config)
        if hasattr(config, 'num_envs') and config.num_envs > 1:
            self.env = make_parallel_env(
                num_envs=config.num_envs,
                render_mode="rgb_array" if config.render else None
            )
            self.is_parallel = True
        else:
            self.env = CarRacingWrapper(
                render_mode="rgb_array" if config.render else None
            )
            self.is_parallel = False
        
        # Setup directories for saving
        self.model_dir = f"./models/{config.username if hasattr(config, 'username') else 'default'}/"
        self.reward_dir = f"./rewards/{config.username if hasattr(config, 'username') else 'default'}/"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.reward_dir, exist_ok=True)
        
        # Training tracking
        self.episode_rewards = []
        self.episode_count = 0

    def train(self, num_episodes):
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in tqdm(range(num_episodes)):
            print(f"[INFO]: Starting Episode {episode}")
            
            # Run single episode
            episode_reward = self._run_episode()
            
            # Store episode results
            self.episode_rewards.append([episode_reward, self.agent.get_exploration_rate()])
            
            # Update target network periodically
            if episode % self.config.target_update_steps == 0:
                self.agent.update_target_network()
            
            # Save model periodically
            if episode % self.config.save_training_freq == 0:
                self._save_checkpoint(episode)
            
            print(f"[INFO]: Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {self.agent.epsilon:.4f}")
        
        print("Training completed!")

    def _run_episode(self):
    # Reset environment
        state = self.env.reset()
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Choose action
            action_idx = self.agent.choose_action(state)
            
            # Execute action with frame skipping (like DDQN3)
            total_reward = 0
            for _ in range(self.config.skip_frames + 1):
                next_state, reward, done, info = self.env.step(action_idx)
                total_reward += reward
                if done:
                    break
            
            # Store experience
            self.agent.store_transition(state, action_idx, total_reward, next_state, done)
            
            # Train agent if buffer is ready
            self.agent.experience_replay()
            
            # Update for next iteration
            state = next_state
            episode_reward += total_reward
            step += 1
        
        return episode_reward
    
    def _save_checkpoint(self, episode):
        # Save agent model
        model_filename = f"episode_{episode}.pth"
        model_path = os.path.join(self.model_dir, model_filename)
        self.agent.save_agent(model_path)
        
        # Save episode rewards data (matching DDQN3 CSV format)
        reward_filename = f"episode_{episode}.csv"
        reward_path = os.path.join(self.reward_dir, reward_filename)
        np.savetxt(reward_path, self.episode_rewards, delimiter=",")
        
        print(f"[INFO]: Saved checkpoint at episode {episode}")

    def evaluate_agent(self, num_episodes=10):
        print(f"[INFO]: Evaluating agent for {num_episodes} episodes...")
        
        # Set agent to evaluation mode
        self.agent.eval_mode()
        
        test_rewards = []
        test_times = []
        
        for test_episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            done = False
            start_time = time.time()
            
            while not done:
                # Choose best action (no exploration)
                action_idx = self.agent.choose_action(state, best=True)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action_idx)
                
                state = next_state
                episode_reward += reward
            
            episode_time = time.time() - start_time
            test_rewards.append(episode_reward)
            test_times.append(episode_time)
            
            print(f"[INFO]: Test Episode {test_episode} | Reward: {episode_reward:.2f} | Time: {episode_time:.2f}s")
        
        # Calculate statistics (matching DDQN3 format)
        avg_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        min_reward = np.min(test_rewards)
        max_reward = np.max(test_rewards)
        avg_time = np.mean(test_times)
        
        print(f"[INFO]: Evaluation Results | Avg: {avg_reward:.2f} | Std: {std_reward:.2f} | Min: {min_reward:.2f} | Max: {max_reward:.2f}")
        
        # Set agent back to training mode
        self.agent.train_mode()
        
        return avg_reward, std_reward, min_reward, max_reward, avg_time
    
    def save_training_state(self, episode, filepath):
        """Save complete training state for resuming"""
        training_state = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_count': self.episode_count,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        }
        torch.save(training_state, f"{filepath}_training_state.pth")
        
        # Save agent separately
        self.agent.save_agent(f"{filepath}_agent.pth")
        
        print(f"[INFO]: Saved complete training state at episode {episode}")

    def load_training_state(self, filepath):
        """Load training state to resume training"""
        training_state = torch.load(f"{filepath}_training_state.pth")
        
        self.episode_count = training_state['episode']
        self.episode_rewards = training_state['episode_rewards']
        
        # Load agent
        self.agent.load_agent(f"{filepath}_agent.pth")
        
        print(f"[INFO]: Resumed training from episode {self.episode_count}")
        return training_state['episode']

    def get_training_stats(self):
        """Return current training statistics"""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = [reward[0] for reward in self.episode_rewards[-100:]]  # Last 100 episodes
        
        return {
            'total_episodes': len(self.episode_rewards),
            'current_epsilon': self.agent.get_exploration_rate(),
            'avg_reward_last_100': np.mean(recent_rewards) if recent_rewards else 0,
            'buffer_size': len(self.agent.buffer),
            'recent_rewards': recent_rewards
        }
    
    def _run_parallel_episode(self):
        """Run episode with parallel environments for faster data collection"""
        # Reset all environments
        states = self.env.reset()  # Returns array of states from all envs
        
        episode_rewards = np.zeros(self.config.num_envs)
        dones = np.zeros(self.config.num_envs, dtype=bool)
        
        while not np.all(dones):
            # Choose actions for all environments
            actions = []
            for i, state in enumerate(states):
                if not dones[i]:
                    action_idx = self.agent.choose_action(state)
                    actions.append(action_idx)
                else:
                    actions.append(0)  # Dummy action for finished environments
            
            # Execute actions in all environments
            next_states, rewards, new_dones, infos = self.env.step(actions)
            
            # Store experiences for active environments
            for i in range(self.config.num_envs):
                if not dones[i]:
                    self.agent.store_transition(
                        states[i], actions[i], rewards[i], next_states[i], new_dones[i]
                    )
                    episode_rewards[i] += rewards[i]
            
            # Train agent multiple times with parallel data
            for _ in range(self.config.num_envs):
                self.agent.experience_replay()
            
            # Update states and done flags
            states = next_states
            dones = np.logical_or(dones, new_dones)
        
        return np.mean(episode_rewards)  # Return average reward across environments
    
    def _run_episode(self):
        """Run episode - automatically handles single or parallel environments"""
        if self.is_parallel:
            return self._run_parallel_episode()
        else:
            # Your existing single environment code
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and episode_reward > self.config.max_penalty:
                action_idx = self.agent.choose_action(state)
                
                total_reward = 0
                for _ in range(self.config.skip_frames + 1):
                    next_state, reward, done, info = self.env.step(action_idx)
                    total_reward += reward
                    if done:
                        break
                
                self.agent.store_transition(state, action_idx, total_reward, next_state, done)
                self.agent.experience_replay()
                
                state = next_state
                episode_reward += total_reward
                step += 1
            
            return episode_reward