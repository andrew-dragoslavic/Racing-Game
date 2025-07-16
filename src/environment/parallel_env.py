import gymnasium as gym
from gymnasium.vector import make_vec_env
from .env_wrapper import CarRacingWrapper

def make_parallel_env(num_envs = 4, **kwargs):

    def make_env():
        return CarRacingWrapper(**kwargs)
    
    vec_env = make_vec_env(make_env, num_envs)
    return vec_env