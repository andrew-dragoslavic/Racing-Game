import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import DDQNConfig
from src.training.trainer import DDQNTrainer
import argparse
import numpy as np

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test trained DDQN Agent on CarRacing')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--config', type=str, default='configs/ddqn_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during testing')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use: cpu or cuda')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    if os.path.exists(args.config):
        config = DDQNConfig.from_yaml(args.config)
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = DDQNConfig()
    
    # Override config for testing
    config.device = args.device
    config.render = args.render
    
    print("Testing Configuration:")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {args.render}")
    print(f"Device: {args.device}")
    
    # Initialize trainer
    trainer = DDQNTrainer(config)
    
    # Load trained model
    try:
        trainer.agent.load_agent(args.model)
        print(f"Successfully loaded model from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run evaluation
    print(f"\nStarting evaluation for {args.episodes} episodes...")
    avg_reward, std_reward, min_reward, max_reward, avg_time = trainer.evaluate_agent(args.episodes)
    
    # Print results in DDQN3 format
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes: {args.episodes}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Standard Deviation: {std_reward:.2f}")
    print(f"Minimum Reward: {min_reward:.2f}")
    print(f"Maximum Reward: {max_reward:.2f}")
    print(f"Average Time: {avg_time:.2f}s")
    print("="*60)
    
    # Save results to CSV (matching DDQN3 format)
    results_file = f"test_results_{os.path.basename(args.model).replace('.pth', '.csv')}"
    test_data = [[avg_reward, 0, avg_time, max_reward, min_reward, std_reward]]
    np.savetxt(results_file, test_data, delimiter=",")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()