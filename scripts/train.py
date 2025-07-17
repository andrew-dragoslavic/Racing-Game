import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import DDQNConfig
from src.training.trainer import DDQNTrainer
import argparse

def parse_arguments():
    """
    Parse command line arguments for training the DDQN Agent.

    Returns:
        argparse.Namespace: Parsed arguments including:
            --config (str): Path to configuration file (default: 'configs/ddqn_config.yaml').
            --episodes (int): Number of episodes to train (overrides config).
            --device (str): Device to use: 'cpu' or 'cuda' (overrides config).
            --resume (str): Path to checkpoint to resume training from.
    """
    parser = argparse.ArgumentParser(description='Train DDQN Agent on CarRacing')
    parser.add_argument('--config', type=str, default='configs/ddqn_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes to train (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use: cpu or cuda (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    return parser.parse_args()

def main():
    """
    Main function to train the DDQN Agent on the CarRacing environment.

    This function parses command-line arguments, loads the configuration, initializes the trainer,
    optionally resumes training from a checkpoint, and starts the training process.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    if os.path.exists(args.config):
        config = DDQNConfig.from_yaml(args.config)
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = DDQNConfig()
    
    # Override config with command line arguments
    if args.episodes:
        config.episodes = args.episodes
    if args.device:
        config.device = args.device
    
    print("Configuration:")
    print(config.to_dict())
    
    # Initialize trainer
    trainer = DDQNTrainer(config)
    
    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume:
        start_episode = trainer.load_training_state(args.resume)
    
    # Start training
    trainer.train(config.episodes - start_episode)

if __name__ == "__main__":
    main()