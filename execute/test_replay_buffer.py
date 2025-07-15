import numpy as np
from processing.replay_buffer import ReplayBuffer

def test_basic_functionality():
    print("=== Testing Basic ReplayBuffer Functionality ===")
    
    # Create a small buffer for testing
    buffer = ReplayBuffer(capacity=5)
    print(f"Initial buffer length: {len(buffer)}")
    
    # Test that empty buffer returns None when sampling
    result = buffer.sample(2)
    print(f"Sampling from empty buffer: {result}")

    for i in range(3):
        state = np.random.random((4, 84, 84)).astype(np.float32)
        action = i % 5  # Actions 0-4
        reward = float(i)
        next_state = np.random.random((4, 84, 84)).astype(np.float32)
        done = (i == 2)  # Last one is done
        
        buffer.push(state, action, reward, next_state, done)
        print(f"Added experience {i+1}, buffer length: {len(buffer)}")

    batch = buffer.sample(2)
    if batch is not None:
        states, actions, rewards, next_states, dones = batch
        print(f"Sampled batch shapes:")
        print(f"  States: {states.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Rewards: {rewards.shape}")
        print(f"  Next states: {next_states.shape}")
        print(f"  Dones: {dones.shape}")

    for i in range(5):  # Add 5 more (total would be 8, but capacity is 5)
        state = np.random.random((4, 84, 84)).astype(np.float32)
        buffer.push(state, 0, 0.0, state, False)
    
    print(f"Buffer length after overflow: {len(buffer)} (should be 5)")


if __name__ == "__main__":
    test_basic_functionality()