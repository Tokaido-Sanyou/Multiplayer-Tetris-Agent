#!/usr/bin/env python3
"""
Play Trained Model Script

This script loads a trained Tetris RL agent and displays its gameplay.
Run this script from the localMultiplayerTetris directory.

Example usage:
    python play_model.py --checkpoint checkpoints/unified/checkpoint_batch_49.pt
    python play_model.py --checkpoint checkpoints/unified/checkpoint_batch_49.pt --episodes 5 --step_delay 0.2
"""

import os
import sys
import argparse
import logging

# Add current directory to path for imports  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_utils.play_model import main as play_main

def main():
    parser = argparse.ArgumentParser(description="Play trained Tetris model")
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=3, 
                       help='Number of episodes to play (default: 3)')
    parser.add_argument('--max_steps', type=int, default=2000, 
                       help='Maximum steps per episode (default: 2000)')
    parser.add_argument('--step_delay', type=float, default=0.1, 
                       help='Delay between steps in seconds (default: 0.1)')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto/cuda/mps/cpu)')
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return
    
    print(f"\n=== Playing Trained Model ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Step delay: {args.step_delay}s")
    print(f"Device: {args.device}")
    print(f"=============================\n")
    
    # Set up sys.argv for the play_model script
    sys.argv = [
        'play_model.py',
        '--checkpoint', args.checkpoint,
        '--episodes', str(args.episodes),
        '--max_steps', str(args.max_steps),
        '--step_delay', str(args.step_delay),
        '--device', args.device
    ]
    
    try:
        play_main()
    except KeyboardInterrupt:
        print("\n⚠️  Playback interrupted by user")
    except Exception as e:
        print(f"\n❌ Playback failed: {e}")
        raise

if __name__ == '__main__':
    main() 