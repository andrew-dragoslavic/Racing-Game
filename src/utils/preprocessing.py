import torch
import cv2
import numpy as np
from torchvision import transforms

class CarRacingPreprocessor:
    """
    A class for preprocessing CarRacing environment states.

    Attributes:
        input_size (tuple): The target size for resizing input states.
    """

    def __init__(self, input_size=(96, 96)):
        """
        Initialize the CarRacingPreprocessor.

        Args:
            input_size (tuple): The target size for resizing input states.
        """
        self.input_size = input_size

    def process_state(self, state):
        """
        Convert a state to grayscale, normalize it, and convert it to a PyTorch tensor.

        Args:
            state (numpy.ndarray): The input state as a color image.

        Returns:
            torch.Tensor: The processed state as a grayscale tensor.
        """
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = state.astype(float) / 255.0
        state = torch.tensor(state).float().unsqueeze(0)
        return state

    def check_road_visible(self, state):
        """
        Check if the road is visible in the given state.

        Args:
            state (numpy.ndarray): The input state as a color image.

        Returns:
            bool: True if the road is visible, False otherwise.
        """
        x, y, _ = state.shape
        cropped = state[0: int(0.85 * y), 0:x]
        mask = cv2.inRange(cropped, np.array([100, 100, 100]), np.array([150, 150, 150]))
        return np.any(mask == 255)

    def check_grass_detection(self, state):
        """
        Check if the car is on the grass in the given state.

        Args:
            state (numpy.ndarray): The input state as a color image.

        Returns:
            bool: True if the car is on the grass, False otherwise.
        """
        x, y, _ = state.shape
        xc = int(x / 2)
        car = state[67:76, xc - 2:xc + 2]
        mask = cv2.inRange(car, np.array([50, 180, 0]), np.array([150, 255, 255]))
        return np.any(mask == 255)

def convert_greyscale_pytorch(state):
    """
    Convert a state to grayscale and check road visibility and grass detection.

    Args:
        state (numpy.ndarray): The input state as a color image.

    Returns:
        tuple: A tuple containing the processed state (torch.Tensor),
               a boolean indicating road visibility, and a boolean indicating grass detection.
    """
    preprocessor = CarRacingPreprocessor()
    processed_image = preprocessor.process_state(state)
    road_visible = preprocessor.check_road_visible(state)
    on_grass = preprocessor.check_grass_detection(state)
    return processed_image, road_visible, on_grass