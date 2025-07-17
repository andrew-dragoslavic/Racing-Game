import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from ..models.ddqn_network import DDQNNetwork
from ..replay_buffer.prioritized_buffer import PrioritizedReplayBuffer

class DDQNAgent:
    """
    A Deep Double Q-Network (DDQN) agent for training and evaluation.

    Attributes:
        config: Configuration object containing hyperparameters.
        device: Device to run computations on (CPU or GPU).
        main_network: Main DDQN network for action-value estimation.
        target_network: Target DDQN network for stable training.
        buffer: Replay buffer for storing and sampling experiences.
        optimizer: Optimizer for training the main network.
        epsilon: Exploration rate for epsilon-greedy policy.
    """

    def __init__(self, config):
        """
        Initialize the DDQNAgent with the given configuration.

        Args:
            config: Configuration object containing hyperparameters.
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.main_network = DDQNNetwork(
            input_channels=1, 
            n_actions=12,  # From action space size
            hidden_dim=216
        ).to(self.device)
        
        self.target_network = DDQNNetwork(
            input_channels=1,
            n_actions=12,
            hidden_dim=216
        ).to(self.device)
        
        # Initialize target network with main network weights
        self.target_network.copy_weights_from(self.main_network)
        
        # Initialize replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=config.replay_buffer
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.main_network.parameters(), 
            lr=config.learning_rate
        )
        
        # Store current epsilon for exploration
        self.epsilon = config.epsilon

    def choose_action(self, state, best=False):
        """
        Choose an action based on the current state using epsilon-greedy policy.

        Args:
            state (np.ndarray or torch.Tensor): Current state of the environment.
            best (bool): If True, always choose the best action (default: False).

        Returns:
            int: Index of the chosen action.
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.float().to(self.device)
        
        with torch.no_grad():
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            q_values = self.main_network(state)
            action_idx = torch.argmax(q_values).item()
        
        if best:
            return action_idx
        
        if random.random() < self.epsilon:
            return  random.randint(0,11)
        else:
            return action_idx
        
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state (np.ndarray or torch.Tensor): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray or torch.Tensor): Next state.
            done (bool): Whether the episode is done.
        """
        # Convert tensors to numpy for storage (saves memory)
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Store in replay buffer
        self.buffer.add(state, action, reward, next_state, done)

    def experience_replay(self):
        """
        Perform experience replay to train the main network using sampled experiences.

        Returns:
            None
        """
        if not self.buffer.is_ready(self.config.batch_size):
            return

        batch_data = self.buffer.sample(self.config.batch_size)
        if batch_data is None:
            return
        
        experiences, weights, indices = batch_data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for experience in experiences:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        current_q_values = self.main_network(states).gather(1, actions.unsqueeze(1))
        next_q_values_main = self.main_network(next_states)
        next_actions = torch.argmax(next_q_values_main, dim=1)

        next_q_values_target = self.target_network(next_states)
        next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target_q_values = rewards + (self.config.gamma * next_q_values * (~dones))

        td_errors = (current_q_values.squeeze(1) - target_q_values).detach()
        loss = (weights * (current_q_values.squeeze(1) - target_q_values) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors.cpu().numpy())
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay

    def update_target_network(self):
        """
        Update the target network by copying weights from the main network.

        Returns:
            None
        """
        self.target_network.copy_weights_from(self.main_network)

    def get_exploration_rate(self):
        """
        Get the current exploration rate (epsilon).

        Returns:
            float: Current epsilon value.
        """
        """Return current epsilon value"""
        return self.epsilon
    
    def eval_mode(self):
        """
        Set the agent to evaluation mode.

        Returns:
            None
        """
        """Set agent to evaluation mode"""
        self.main_network.eval()
        self.target_network.eval()

    def train_mode(self):
        """
        Set the agent to training mode.

        Returns:
            None
        """
        """Set agent to training mode"""
        self.main_network.train()
        self.target_network.train()

    def save_agent(self, filepath):
        """
        Save the agent's state to a file.

        Args:
            filepath (str): Path to save the agent's state.
        """
        """Save complete agent state"""
        checkpoint = {
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config
        }
        torch.save(checkpoint, filepath)

    def load_agent(self, filepath):
        """
        Load the agent's state from a file.

        Args:
            filepath (str): Path to the saved agent's state.
        """
        """Load saved agent state"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
