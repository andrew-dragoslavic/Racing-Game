import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        states, actions, rewards, next_states, dones = [np.array(x) for x in [states, actions, rewards, next_states, dones]]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

