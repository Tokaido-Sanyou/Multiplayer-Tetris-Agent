#!/usr/bin/env python3
"""
Command-line interface for training the Standard Dreamer agent on Tetris,
with customizable episodes and world-model pretraining.
Outputs TensorBoard log directory for monitoring.
"""
import argparse
from train_dreamer_standard import train_dreamer

def main():
    parser = argparse.ArgumentParser(
        description="Train Standard Dreamer Agent on Tetris"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=1000,
        help="Number of training episodes (phase 2)"
    )
    parser.add_argument(
        "--pretrain", "-p",
        type=int,
        default=100,
        help="Number of world-model pretrain episodes (phase 1)"
    )
    parser.add_argument(
        "--reward_mode", "-r",
        type=str,
        choices=["lines_only", "standard"],
        default="lines_only",
        help="Reward mode: lines_only or standard"
    )
    parser.add_argument(
        "--visualize_interval", "-v",
        type=int,
        default=0,
        help="Interval (in episodes) at which to render gameplay (0=off)"
    )
    parser.add_argument(
        "--step_mode", "-s",
        type=str,
        choices=["action", "block_placed"],
        default="block_placed",
        help="Step mode: 'action' for tick-by-tick, 'block_placed' to step per piece placement"
    )
    args = parser.parse_args()

    print(f"🚀 Starting Dreamer training: pretrain={args.pretrain}, episodes={args.episodes}, reward_mode={args.reward_mode}, visualize_interval={args.visualize_interval}, step_mode={args.step_mode}")

    # Run training (train_dreamer prints the log directory)
    agent = train_dreamer(
        episodes=args.episodes,
        world_model_pretrain=args.pretrain,
        reward_mode=args.reward_mode,
        visualize_interval=args.visualize_interval,
        step_mode=args.step_mode
    )

    print("✅ Training completed. Review TensorBoard logs above for details.")

if __name__ == "__main__":
    main() 