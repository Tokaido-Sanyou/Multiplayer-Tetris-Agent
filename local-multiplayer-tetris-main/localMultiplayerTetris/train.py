#!/usr/bin/env python3
"""
Main Training Script for Tetris RL Agent

This script provides a simple interface to train the Tetris RL agent with the 6-phase algorithm.
Run this script from the localMultiplayerTetris directory.

Example usage:
    python train.py --visualize --num_batches 50
    python train.py --num_batches 100 --log_dir custom_logs
"""

import os
import sys
import argparse
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_utils.unified_trainer import UnifiedTrainer, TrainingConfig

def main():
    parser = argparse.ArgumentParser(description="Train Tetris RL Agent")
    parser.add_argument('--num_batches', type=int, default=50, 
                       help='Number of training batches (default: 50)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Enable visualization for last episode of each batch')
    parser.add_argument('--log_dir', type=str, default='logs/unified_training', 
                       help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/unified', 
                       help='Checkpoint save directory')
    parser.add_argument('--save_interval', type=int, default=10, 
                       help='Save checkpoint every N batches')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto/cuda/mps/cpu)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create config
    config = TrainingConfig()
    config.num_batches = args.num_batches
    config.visualize = args.visualize
    config.log_dir = args.log_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.save_interval = args.save_interval
    
    # Override device if specified
    if args.device != 'auto':
        config.device = args.device
        print(f"Using specified device: {args.device}")
    
    print(f"\n=== Training Configuration ===")
    print(f"Batches: {config.num_batches}")
    print(f"Device: {config.device}")
    print(f"Visualization: {config.visualize}")
    print(f"Log directory: {config.log_dir}")
    print(f"Checkpoint directory: {config.checkpoint_dir}")
    print(f"Save interval: {config.save_interval}")
    print(f"===============================\n")
    
    # Initialize and run trainer
    try:
        trainer = UnifiedTrainer(config)
        trainer.run_training()
        print("\n🎉 Training completed successfully!")
        print(f"📁 Logs saved to: {config.log_dir}")
        print(f"💾 Checkpoints saved to: {config.checkpoint_dir}")
        print(f"📊 View training progress: tensorboard --logdir {config.log_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        logging.info("Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    main() 