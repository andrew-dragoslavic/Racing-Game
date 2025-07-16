import yaml

class DDQNConfig:
    def __init__(self, episodes=2000, batch_size=20, target_update_steps=5, save_training_freq=100, 
                 learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9999, 
                 max_penalty=-30, consecutive_neg_reward=25, skip_frames=2, steps_on_grass=20, 
                 replay_buffer=150000, render=True, device='cpu'):
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
        
        # Extract nested values
        training = yaml_data.get('training', {})
        environment = yaml_data.get('environment', {})
        replay_buffer = yaml_data.get('replay_buffer', {})
        parallel = yaml_data.get('parallel', {})
        system = yaml_data.get('system', {})
        
        return cls(
            episodes=training.get('episodes', 2000),
            batch_size=training.get('batch_size', 20),
            target_update_steps=training.get('target_update_steps', 5),
            save_training_freq=training.get('save_training_freq', 100),
            learning_rate=training.get('learning_rate', 0.001),
            gamma=training.get('gamma', 0.95),
            epsilon=training.get('epsilon', 1.0),
            epsilon_min=training.get('epsilon_min', 0.1),
            epsilon_decay=training.get('epsilon_decay', 0.9999),
            max_penalty=environment.get('max_penalty', -30),
            consecutive_neg_reward=environment.get('consecutive_neg_reward', 25),
            skip_frames=environment.get('skip_frames', 2),
            steps_on_grass=environment.get('steps_on_grass', 20),
            replay_buffer=replay_buffer.get('replay_buffer', 150000),
            render=environment.get('render', True),
            device=system.get('device', 'cpu')
        )

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
        # Your existing validation code
        if not isinstance(self.episodes, int) or self.episodes <= 0:
            raise ValueError("Episodes must be a positive integer.")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        # ... rest of your validation
        return True