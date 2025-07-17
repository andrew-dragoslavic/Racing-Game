from collections import deque
import random
import numpy as np
import torch

class PrioritizedReplayBuffer:
    """
    A prioritized replay buffer for storing and sampling experiences with importance sampling.

    Attributes:
        buffer: Deque to store experiences.
        capacity: Maximum number of experiences the buffer can hold.
        alpha: Degree of prioritization (0 = no prioritization, 1 = full prioritization).
        beta: Degree of importance sampling correction (0 = no correction, 1 = full correction).
        beta_increment: Increment value for beta per sampling step.
        priorities: Deque to store priorities of experiences.
        max_priority: Maximum priority value in the buffer.
    """

    def __init__(self, capacity=150000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize the prioritized replay buffer.

        Args:
            capacity (int): Maximum number of experiences to store.
            alpha (float): Degree of prioritization.
            beta (float): Initial value of importance sampling correction.
            beta_increment (float): Increment value for beta per sampling step.
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode is done.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            tuple: A tuple containing sampled experiences, importance sampling weights, and indices.
        """
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
        """
        Update the priorities of sampled experiences.

        Args:
            indices (list): Indices of the sampled experiences.
            td_errors (list): Temporal difference errors for the sampled experiences.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority

            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            int: Number of experiences in the buffer.
        """
        return len(self.buffer)
    
    def is_ready(self, min_size):
        """
        Check if the buffer has enough experiences for sampling.

        Args:
            min_size (int): Minimum number of experiences required.

        Returns:
            bool: True if the buffer has enough experiences, False otherwise.
        """
        return len(self.buffer) >= min_size
    
    def save_buffer(self, filepath='checkpoint'):
        """
        Save the buffer and its metadata to a file.

        Args:
            filepath (str): Path to save the buffer.
        """
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
        """
        Load the buffer and its metadata from a file.

        Args:
            filepath (str): Path to the saved buffer.
        """
        """Load buffer with metadata"""
        checkpoint = torch.load(f'{filepath}_replay_buffer.pth')
        
        self.buffer = deque(checkpoint['buffer'], maxlen=self.capacity)
        self.priorities = deque(checkpoint['priorities'], maxlen=self.capacity)
        self.alpha = checkpoint['alpha']
        self.beta = checkpoint['beta']
        self.beta_increment = checkpoint['beta_increment']
        self.max_priority = checkpoint['max_priority']