import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQNNetwork
import torch.nn.functional as F
import random


class DDQNAgent:
    def __init__(self, input_shape, num_actions, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.main_network = DQNNetwork(input_shape, num_actions)
        self.target_network = DQNNetwork(input_shape, num_actions)
        self.optimizer = optim.Adam(params=self.main_network.parameters(), lr=learning_rate)

    def select_action(self, state, training=True):
        state = torch.Tensor(state).unsqueeze(0)
        actions = self.main_network.forward(state)
        if not training:
            action = torch.argmax(actions).item()
        else:
            if random.random() < self.epsilon:
                action = random.choice(range(self.num_actions))
            else:
                action = torch.argmax(actions).item()
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return action
    
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


