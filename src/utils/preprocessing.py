import torch
import cv2
import numpy as np
from torchvision import transforms

class CarRacingPreprocessor:
    def __init__(self, input_size = (96,96)):
        self.input_size = input_size
    
    def process_state(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = state.astype(float) / 255.0
        state = torch.tensor(state).unsqueeze(0)
        return state
    
    def check_road_visible(self, state):
        x, y, _ = state.shape
        cropped = state[0: int(0.85*y), 0:x]
        mask = cv2.inRange(cropped, np.array([100,100,100]), np.array([150,150,150]))
        return np.any(mask == 255)
    
    def check_grass_detection(self, state):
        x,y,_ = state.shape
        xc = int(x / 2)
        car = state[67:76, xc-2:xc+2]
        mask = cv2.inRange(car, np.array([50,180,0]), np.array([150,255,255]))
        return np.any(mask == 255)
    
def convert_greyscale_pytorch(state):
    preprocessor = CarRacingPreprocessor()
    processed_image = preprocessor.process_state(state)
    road_visible = preprocessor.check_road_visible(state)
    on_grass = preprocessor.check_grass_detection(state)
    return processed_image, road_visible, on_grass