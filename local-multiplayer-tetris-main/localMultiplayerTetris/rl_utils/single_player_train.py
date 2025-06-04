import argparse
from .train import train_single_player

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--eval-interval', type=int, default=50, help='Evaluate agent every N episodes')
    parser.add_argument('--visualize', action='store_true', help='Enable GUI visualization')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to load')
    parser.add_argument('--no-eval', action='store_true', help='Disable evaluation during training')
    parser.add_argument('--verbose', action='store_true', help='Enable per-step logging')
    args = parser.parse_args()
    
    train_single_player(
        num_episodes=args.episodes,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        visualize=args.visualize,
        checkpoint=args.checkpoint,
        no_eval=args.no_eval,
        verbose=args.verbose
    )