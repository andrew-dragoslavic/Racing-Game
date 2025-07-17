import gymnasium as gym
import numpy as np
from ..utils.preprocessing import CarRacingPreprocessor

class CarRacingWrapper:
    """
    A wrapper for the CarRacing-v3 environment that preprocesses states and manages custom termination conditions.

    Attributes:
        env (gym.Env): The CarRacing environment instance.
        preprocessor (CarRacingPreprocessor): Preprocessor for state processing and checks.
        negative_reward_counter (int): Counter for consecutive negative rewards.
        grass_counter (int): Counter for consecutive frames on grass.
        action_space (list): List of predefined action tuples.
    """

    def __init__(self, env_name="CarRacing-v3", render_mode=None):
        """
        Initialize the CarRacingWrapper.

        Args:
            env_name (str): Name of the environment to load. Defaults to "CarRacing-v3".
            render_mode (str, optional): Render mode for the environment. Defaults to None.
        """
        self.env = gym.make(env_name, render_mode="human", continuous = True)
        self.preprocessor = CarRacingPreprocessor()
        self.negative_reward_counter = 0
        self.grass_counter = 0
        self.action_space = [(-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #            Action Space Structure
                            (-1, 1,   0), (0, 1,   0), (1, 1,   0), #           (Steering, Gas, Break)
                            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range       -1~1     0~`1`   0~1
                            (-1, 0,   0), (0, 0,   0), (1, 0,   0)]
        
    def reset(self):
        """
        Reset the environment and preprocess the initial state.

        Returns:
            np.ndarray: The processed initial state.
        """
        raw_state, info = self.env.reset()
        self.negative_reward_counter = 0
        self.grass_counter = 0
        processed_state = self.preprocessor.process_state(raw_state)
        return processed_state
    
    def step(self, action_index):
        """
        Take a step in the environment using the specified action index.

        Args:
            action_index (int): Index of the action to take from the predefined action space.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The processed next state.
                - float: The clipped reward.
                - bool: Whether the episode is done.
                - dict: Additional information from the environment.
        """
        action_tuple = self.action_space[action_index]
        action = np.array(action_tuple, dtype=np.float32)
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