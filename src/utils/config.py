import yaml

class DDQNConfig:
    def __init__(self, episodes = 2000, batch_size = 20, target_update_steps = 5, save_training_freq = 100, learning_rate = 0.001, gamma = 0.95, epsilon = 1.0, epsilon_min = 0.1, epsilon_decay = 0.9999, max_penalty = -30, consecutive_neg_reward = 25, skip_frames = 2, steps_on_grass = 20, replay_buffer = 150000, render = True, device = 'cpu'):
        self.episodes = episodes
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.save_training_freq = save_training_freq
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_penalty = max_penalty
        self.consecutive_neg_reward = consecutive_neg_reward
        self.skip_frames = skip_frames
        self.steps_on_grass = steps_on_grass
        self.replay_buffer = replay_buffer
        self.render = render
        self.device = device

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        episodes = yaml_data.get('episodes', 2000)
        batch_size = yaml_data.get('batch_size', 20)
        target_update_steps = yaml_data.get('target_update_steps', 5)
        save_training_freq = yaml_data.get('save_training_freq', 100)
        learning_rate = yaml_data.get('learning_rate', 0.001)
        gamma = yaml_data.get('gamma', 0.95)
        epsilon = yaml_data.get('epsilon', 1.0)
        epsilon_min = yaml_data.get('epsilon_min', 0.1)
        epsilon_decay = yaml_data.get('epsilon_decay', 0.9999)
        max_penalty = yaml_data.get('max_penalty', -30)
        consecutive_neg_reward = yaml_data.get('consecutive_neg_reward', 25)
        skip_frames = yaml_data.get('skip_frames', 2)
        steps_on_grass = yaml_data.get('steps_on_grass', 20)
        replay_buffer = yaml_data.get('replay_buffer', 150000)
        render = yaml_data.get('render', True)
        device = yaml_data.get('device', 'cpu')

        return cls(episodes, batch_size, target_update_steps, save_training_freq, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay, max_penalty, consecutive_neg_reward, skip_frames, steps_on_grass, replay_buffer, render, device)
    
    def to_dict(self):
        return {
            'episodes': self.episodes,
            'batch_size': self.batch_size,
            'target_update_steps': self.target_update_steps,
            'save_training_freq': self.save_training_freq,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'max_penalty': self.max_penalty,
            'consecutive_neg_reward': self.consecutive_neg_reward,
            'skip_frames': self.skip_frames,
            'steps_on_grass': self.steps_on_grass,
            'replay_buffer': self.replay_buffer,
            'render': self.render,
            'device': self.device
        }
    
    def validate(self):
        if not isinstance(self.episodes, int) or self.episodes <= 0:
            raise ValueError("Episodes must be a positive integer.")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if not isinstance(self.target_update_steps, int) or self.target_update_steps <= 0:
            raise ValueError("Target update steps must be a positive integer.")
        if not isinstance(self.save_training_freq, int) or self.save_training_freq <= 0:
            raise ValueError("Save training frequency must be a positive integer.")
        if not (0 < self.learning_rate <= 1):
            raise ValueError("Learning rate must be between 0 and 1.")
        if not (0 < self.gamma < 1):
            raise ValueError("Gamma must be between 0 and 1.")
        if not (0 < self.epsilon <= 1):
            raise ValueError("Epsilon must be between 0 and 1.")
        if not (0 < self.epsilon_min <= 1):
            raise ValueError("Epsilon min must be between 0 and 1.")
        if not (0 < self.epsilon_decay < 1):
            raise ValueError("Epsilon decay must be between 0 and 1.")
        if not isinstance(self.max_penalty, (int, float)):
            raise ValueError("Max penalty must be a number.")
        if not isinstance(self.consecutive_neg_reward, int) or self.consecutive_neg_reward < 0:
            raise ValueError("Consecutive negative reward must be a non-negative integer.")
        if not isinstance(self.skip_frames, int) or self.skip_frames <= 0:
            raise ValueError("Skip frames must be a positive integer.")
        if not isinstance(self.steps_on_grass, int) or self.steps_on_grass <= 0:
            raise ValueError("Steps on grass must be a positive integer.")
        if not isinstance(self.replay_buffer, int) or self.replay_buffer <= 0:
            raise ValueError("Replay buffer size must be a positive integer.")
        if not isinstance(self.render, bool):
            raise ValueError("Render must be a boolean value.")
        if not isinstance(self.device, str):
            raise ValueError("Device must be a string.")
        if self.device not in ['cpu', 'cuda']:
            raise ValueError("Device must be either 'cpu' or 'cuda'.")
        return True
