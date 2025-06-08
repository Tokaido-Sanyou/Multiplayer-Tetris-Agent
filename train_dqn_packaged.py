#!/usr/bin/env python3
"""
Packaged DQN Locked State Training Script
Streamlined, optimized training with tensor performance fixes
"""

import sys
import os
import argparse
import time
from typing import Optional

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def main():
    """Main packaged DQN training function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DQN Locked State Training - Optimized Package')
    parser.add_argument('--episodes', type=int, default=100, help='Total episodes to train (default: 100)')
    parser.add_argument('--episodes-per-batch', type=int, default=10, help='Episodes per batch (default: 10)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], 
                       help='Device to use (default: auto)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--epsilon-decay', type=int, default=800, help='Episodes to decay epsilon over (default: 800)')
    parser.add_argument('--memory-size', type=int, default=100000, help='Experience replay buffer size (default: 100000)')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size (default: 32)')
    parser.add_argument('--save-freq', type=int, default=1, help='Save checkpoint every N batches (default: 1)')
    parser.add_argument('--no-demo', action='store_true', help='Disable agent demonstrations')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Import after parsing args to speed up help display
    import torch
    from training.train_dqn_locked import ComprehensiveDQNTrainer
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Validate GPU availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    print("🚀 PACKAGED DQN LOCKED STATE TRAINING")
    print("=" * 50)
    print(f"📊 Configuration:")
    print(f"   Total Episodes: {args.episodes}")
    print(f"   Episodes per Batch: {args.episodes_per_batch}")
    print(f"   Device: {device.upper()}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Epsilon Decay: {args.epsilon_decay} episodes")
    print(f"   Memory Size: {args.memory_size:,}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Action Space: 1600 (200 coords × 4 rotations × 2 lock states)")
    print(f"   State Space: 585 (425 observation + 160 selection)")
    print("=" * 50)
    
    # Initialize trainer with optimized settings
    try:
        trainer = ComprehensiveDQNTrainer(
            device=device,
            episodes_per_batch=args.episodes_per_batch,
            learning_rate=args.learning_rate,
            epsilon_decay_episodes=args.epsilon_decay,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            save_freq=args.save_freq
        )
        
        if not args.quiet:
            print(f"✅ Trainer initialized successfully")
            print(f"📱 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if device == 'cuda' else "💻 CPU Training")
            
    except Exception as e:
        print(f"❌ Failed to initialize trainer: {e}")
        return 1
    
    # Run training with error handling
    start_time = time.time()
    try:
        print(f"\n🎯 Starting Training...")
        if args.no_demo:
            print(f"   (Agent demonstrations disabled)")
            
        trainer.train(total_episodes=args.episodes)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training Completed Successfully!")
        print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"📈 Final Performance:")
        print(f"   Episodes Trained: {len(trainer.episode_rewards)}")
        print(f"   Final Epsilon: {trainer.agent.epsilon:.4f}")
        print(f"   Memory Size: {len(trainer.agent.memory):,}")
        if trainer.batch_rewards:
            print(f"   Best Batch Reward: {max(trainer.batch_rewards):.2f}")
            print(f"   Latest Batch Reward: {trainer.batch_rewards[-1]:.2f}")
        
        # Display results location
        print(f"\n📁 Results saved to: results/dqn_locked/")
        print(f"   Training logs, plots, and checkpoints available")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        elapsed = time.time() - start_time
        print(f"   Trained for {elapsed:.1f}s")
        if hasattr(trainer, 'episode_rewards') and trainer.episode_rewards:
            print(f"   Completed {len(trainer.episode_rewards)} episodes")
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        if not args.quiet:
            traceback.print_exc()
        return 1
    
    finally:
        # Clean up environments
        try:
            if 'trainer' in locals():
                trainer.env.close()
                trainer.demo_env.close()
        except:
            pass

def quick_train(episodes: int = 100, device: str = 'auto', quiet: bool = False) -> bool:
    """
    Quick training function for programmatic use
    
    Args:
        episodes: Number of episodes to train
        device: Device to use ('auto', 'cuda', 'cpu')
        quiet: Reduce output verbosity
        
    Returns:
        True if training succeeded, False otherwise
    """
    import torch
    from training.train_dqn_locked import ComprehensiveDQNTrainer
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not quiet:
        print(f"🚀 Quick DQN Training: {episodes} episodes on {device.upper()}")
    
    try:
        trainer = ComprehensiveDQNTrainer(
            device=device,
            episodes_per_batch=min(10, episodes//5),  # Adaptive batch size
            epsilon_decay_episodes=episodes
        )
        
        trainer.train(total_episodes=episodes)
        
        if not quiet:
            print(f"✅ Quick training completed!")
            if trainer.batch_rewards:
                print(f"   Final performance: {trainer.batch_rewards[-1]:.2f}")
        
        # Clean up
        trainer.env.close() 
        trainer.demo_env.close()
        
        return True
        
    except Exception as e:
        if not quiet:
            print(f"❌ Quick training failed: {e}")
        return False

def evaluate_model(checkpoint_path: str, episodes: int = 5, device: str = 'auto') -> dict:
    """
    Evaluate a trained DQN model
    
    Args:
        checkpoint_path: Path to model checkpoint
        episodes: Number of evaluation episodes
        device: Device to use
        
    Returns:
        Dictionary with evaluation results
    """
    import torch
    from training.train_dqn_locked import ComprehensiveDQNTrainer
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🧪 Evaluating model: {checkpoint_path}")
    print(f"   Episodes: {episodes}, Device: {device.upper()}")
    
    try:
        trainer = ComprehensiveDQNTrainer(device=device)
        trainer.agent.load_checkpoint(checkpoint_path)
        trainer.agent.set_training_mode(False)
        
        rewards = []
        lengths = []
        
        for i in range(episodes):
            reward, length, _ = trainer.run_episode(training=False)
            rewards.append(reward)
            lengths.append(length)
            print(f"   Episode {i+1}: {reward:.2f} reward, {length} steps")
        
        results = {
            'avg_reward': sum(rewards) / len(rewards),
            'avg_length': sum(lengths) / len(lengths),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'all_rewards': rewards,
            'all_lengths': lengths
        }
        
        print(f"📊 Evaluation Results:")
        print(f"   Average Reward: {results['avg_reward']:.2f}")
        print(f"   Average Length: {results['avg_length']:.1f}")
        print(f"   Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        
        # Clean up
        trainer.env.close()
        trainer.demo_env.close()
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return {}

if __name__ == "__main__":
    exit(main()) 