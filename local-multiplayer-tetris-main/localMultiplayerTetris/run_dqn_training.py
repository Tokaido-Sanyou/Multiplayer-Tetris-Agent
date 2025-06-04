#!/usr/bin/env python3
"""
Wrapper script to run DQN training from the correct directory level to avoid import issues.

Usage:
    python run_dqn_training.py --mode vectorized --num_episodes 10000 --num_envs 8

This script sets up the environment correctly and then calls the DQN training with all arguments.
"""

import os
import sys
import subprocess
import argparse

def main():
    # Parse arguments to pass through
    parser = argparse.ArgumentParser(description='Train DQN agent on Tetris')
    parser.add_argument('--mode', choices=['single', 'vectorized'], default='vectorized',
                       help='Training mode: single environment or vectorized')
    parser.add_argument('--num_episodes', type=int, default=10000,
                       help='Number of episodes to train')
    parser.add_argument('--num_envs', type=int, default=4,
                       help='Number of parallel environments (vectorized mode only)')
    parser.add_argument('--eval_interval', type=int, default=1000,
                       help='Episodes between evaluations')
    parser.add_argument('--update_frequency', type=int, default=4,
                       help='Environment steps between agent updates (vectorized mode)')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='Initial epsilon for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                       help='Final epsilon for exploration')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                       help='Epsilon decay rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--buffer_size', type=int, default=100000,
                       help='Replay buffer size')
    parser.add_argument('--target_update', type=int, default=10,
                       help='Steps between target network updates')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Episodes between checkpoints')
    
    args = parser.parse_args()
    
    # Get the current directory (should be localMultiplayerTetris)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Running DQN training from: {current_dir}")
    
    # Build command arguments
    cmd_args = [
        sys.executable, '-m', 'rl_utils.dqn_train',
        '--mode', args.mode,
        '--num_episodes', str(args.num_episodes),
        '--num_envs', str(args.num_envs),
        '--eval_interval', str(args.eval_interval),
        '--update_frequency', str(args.update_frequency),
        '--learning_rate', str(args.learning_rate),
        '--gamma', str(args.gamma),
        '--epsilon_start', str(args.epsilon_start),
        '--epsilon_end', str(args.epsilon_end),
        '--epsilon_decay', str(args.epsilon_decay),
        '--batch_size', str(args.batch_size),
        '--buffer_size', str(args.buffer_size),
        '--target_update', str(args.target_update),
        '--save_interval', str(args.save_interval)
    ]
    
    # Add checkpoint if provided
    if args.checkpoint:
        cmd_args.extend(['--checkpoint', args.checkpoint])
    
    print(f"Starting DQN training with command: {' '.join(cmd_args)}")
    
    try:
        # Run the training script as a module from the current directory
        subprocess.run(cmd_args, cwd=current_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 