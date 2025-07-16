from collections import deque
import random
import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity=150000, alpha = 0.6, beta = 0.4, beta_increment= 0.001):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        priorities = np.array(self.priorities)[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return experiences, weights, indices
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority

            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self, min_size):
        return len(self.buffer) >= min_size
    
    def save_buffer(self, filepath='checkpoint'):
        """Save buffer with metadata"""
        checkpoint = {
            'buffer': list(self.buffer),
            'priorities': list(self.priorities),
            'capacity': self.capacity,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment,
            'max_priority': self.max_priority
        }
        torch.save(checkpoint, f'{filepath}_replay_buffer.pth')
    
    def load_buffer(self, filepath='checkpoint'):
        """Load buffer with metadata"""
        checkpoint = torch.load(f'{filepath}_replay_buffer.pth')
        
        self.buffer = deque(checkpoint['buffer'], maxlen=self.capacity)
        self.priorities = deque(checkpoint['priorities'], maxlen=self.capacity)
        self.alpha = checkpoint['alpha']
        self.beta = checkpoint['beta']
        self.beta_increment = checkpoint['beta_increment']
        self.max_priority = checkpoint['max_priority']