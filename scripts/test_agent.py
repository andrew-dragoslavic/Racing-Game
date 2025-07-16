import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import DDQNConfig
from src.agents.ddqn_agent import DDQNAgent
import torch

print("Testing agent initialization...")
config = DDQNConfig()
agent = DDQNAgent(config)
print("Agent works!")

print("Testing agent methods...")
# Test action selection with dummy state
dummy_state = torch.randn(1, 96, 96)  # Random state tensor
action = agent.choose_action(dummy_state)
print(f"Action selection works! Chosen action: {action}")

print("Testing network forward pass...")
q_values = agent.main_network(dummy_state.unsqueeze(0))  # Add batch dimension
print(f"Network forward pass works! Q-values shape: {q_values.shape}")

print("Testing replay buffer...")
print(f"Buffer size: {len(agent.buffer)}")
print(f"Buffer ready for training: {agent.buffer.is_ready(config.batch_size)}")

print("All agent tests passed!")