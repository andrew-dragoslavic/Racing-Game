import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.env_wrapper import CarRacingWrapper
from src.utils.preprocessing import convert_greyscale_pytorch

def test_environment():
    """
    Test the CarRacing environment wrapper.

    Prints the state shape after resetting the environment.
    """
    print("Testing environment...")
    env = CarRacingWrapper()
    state = env.reset()
    print(f"Environment works! State shape: {state.shape}")

def test_preprocessing():
    """
    Test the preprocessing functions for the CarRacing environment.

    Creates a dummy state, processes it, and checks road visibility and grass detection.
    """
    print("Testing preprocessing...")
    import numpy as np
    dummy_state = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    processed_state, road_visible, on_grass = convert_greyscale_pytorch(dummy_state)
    print(f"Preprocessing works! Processed shape: {processed_state.shape}")
    print(f"Road visible: {road_visible}, On grass: {on_grass}")

# Run tests
test_environment()
test_preprocessing()