import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.dqn import DQNNetwork
import torch.nn.functional as F
import random
from processing.action_space import get_action_from_index, get_num_actions


class DDQNAgent:
    def __init__(self, input_shape, num_actions):
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.learning_rate = 0.001      # Higher than your 0.0001
        self.gamma = 0.95               # Lower than your 0.99
        self.epsilon = 1.0              # Same start
        self.epsilon_end = 0.1          # Same end
        self.epsilon_decay = 0.9999

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.main_network = DQNNetwork(input_shape, num_actions).to(self.device)
        self.target_network = DQNNetwork(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        

    def select_action(self, state, training=True):
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        
        if not training:
            # Always use best action during testing
            q_values = self.main_network(state)
            action_index = torch.argmax(q_values).item()
        else:
            # Epsilon-greedy during training
            if random.random() < self.epsilon:
                action_index = random.choice(range(self.num_actions))
            else:
                q_values = self.main_network(state)
                action_index = torch.argmax(q_values).item()
            
            # Oliver's epsilon decay
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
        
        # Convert action index to (steering, gas, brake) tuple
        return get_action_from_index(action_index), action_index
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)      
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        all_q_values = self.main_network.forward(states)
        current_q_values = all_q_values[range(len(actions)), actions]
        next_q_values_main = self.main_network.forward(next_states)
        best_next_actions = torch.argmax(next_q_values_main, dim=1)
        next_q_values_target = self.target_network.forward(next_states)
        target_next_q_values = next_q_values_target[range(len(best_next_actions)), best_next_actions]
        targets = rewards + self.gamma * target_next_q_values * (1 - dones)
        loss = F.mse_loss(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


