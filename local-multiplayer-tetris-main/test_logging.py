#!/usr/bin/env python3

from localMultiplayerTetris.rl_utils.unified_trainer import UnifiedTrainer, TrainingConfig
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_initialization():
    """Test basic trainer initialization"""
    print("Testing UnifiedTrainer initialization...")
    
    config = TrainingConfig()
    config.num_batches = 1
    config.exploration_episodes = 5
    config.visualize = False
    config.device = 'cpu'
    
    try:
        trainer = UnifiedTrainer(config)
        print("✓ Trainer initialized successfully")
        print(f"✓ Writer: {trainer.writer}")
        print(f"✓ ActorCritic has writer: {hasattr(trainer.actor_critic, 'writer')}")
        print(f"✓ Exploration actor: {trainer.exploration_actor}")
        return trainer
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_phase_1(trainer):
    """Test Phase 1: Exploration"""
    if trainer is None:
        return
        
    print("\nTesting Phase 1: Exploration...")
    try:
        trainer.phase_1_exploration(batch=0)
        print("✓ Phase 1 completed successfully")
        print(f"✓ Exploration data collected: {len(trainer.exploration_data)} items")
    except Exception as e:
        print(f"✗ Phase 1 Error: {e}")
        import traceback
        traceback.print_exc()

def test_tensorboard_logging(trainer):
    """Test if TensorBoard logging is working"""
    if trainer is None:
        return
        
    print("\nTesting TensorBoard logging...")
    try:
        # Test manual logging
        trainer.writer.add_scalar('Test/Value', 42.0, 0)
        trainer.writer.flush()
        print("✓ Manual TensorBoard logging works")
        
        # Check log directory
        import os
        log_files = os.listdir(trainer.config.log_dir) if os.path.exists(trainer.config.log_dir) else []
        print(f"✓ Log directory: {trainer.config.log_dir}")
        print(f"✓ Log files: {log_files}")
        
    except Exception as e:
        print(f"✗ TensorBoard logging error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    trainer = test_initialization()
    test_tensorboard_logging(trainer)
    test_phase_1(trainer)
    
    if trainer:
        trainer.writer.close()
    print("\nTest completed.")
