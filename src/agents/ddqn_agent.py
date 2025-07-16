import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from ..models.ddqn_network import DDQNNetwork
from ..replay_buffer.prioritized_buffer import PrioritizedReplayBuffer

class DDQNAgent:
    def __init__(self, config):
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
            capacity=config.replay_buffer_max_size
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.main_network.parameters(), 
            lr=config.learning_rate
        )
        
        # Store current epsilon for exploration
        self.epsilon = config.epsilon

    def choose_action(self, state, best=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        
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
        # Convert tensors to numpy for storage (saves memory)
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Store in replay buffer
        self.buffer.add(state, action, reward, next_state, done)

    def experience_replay(self):
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


