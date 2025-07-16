import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from .env_wrapper import CarRacingWrapper

def make_parallel_env(num_envs=4, **kwargs):
    """Create parallel environments using SyncVectorEnv"""
    
    def make_env():
        return CarRacingWrapper(**kwargs)
    
    if num_envs == 1:
        return CarRacingWrapper(**kwargs)
    
    # Create list of environment functions
    env_fns = [make_env for _ in range(num_envs)]
    
    # Use SyncVectorEnv instead of make_vec_env
    vec_env = SyncVectorEnv(env_fns)
    return vec_env