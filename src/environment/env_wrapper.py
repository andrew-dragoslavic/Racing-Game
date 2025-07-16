import gymnasium as gym
import numpy as np
from ..utils.preprocessing import CarRacingPreprocessor

class CarRacingWrapper:
    def __init__(self, env_name = "CarRacing-v3", render_mode = None):
        self.env = gym.make(env_name, render_mode=render_mode, continuous = False)
        self.preprocessor = CarRacingPreprocessor()
        self.negative_reward_counter = 0
        self.grass_counter = 0
        self.action_space = [(-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #            Action Space Structure
                            (-1, 1,   0), (0, 1,   0), (1, 1,   0), #           (Steering, Gas, Break)
                            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range       -1~1     0~`1`   0~1
                            (-1, 0,   0), (0, 0,   0), (1, 0,   0)]
        
    def reset(self):
        raw_state, info = self.env.reset()
        self.negative_reward_counter = 0
        self.grass_counter = 0
        processed_state = self.preprocessor.process_state(raw_state)
        return processed_state
    
    def step(self, action_index):
        action = self.action_space[action_index]
        raw_next_state, reward, terminated, truncated, info = self.env.step(action)
        reward = np.clip(reward, a_min = -10, a_max = 1)
        if reward < 0:
            self.negative_reward_counter += 1
        else:
            self.negative_reward_counter = 0

        if self.preprocessor.check_grass_detection(raw_next_state):
            self.grass_counter += 1
        else:
            self.grass_counter = 0

        road_visible = self.preprocessor.check_road_visible(raw_next_state)
        processed_state = self.preprocessor.process_state(raw_next_state)
        done = terminated or truncated or self.negative_reward_counter >= 25 or self.grass_counter >= 20 or not road_visible

        return processed_state, reward, done, info