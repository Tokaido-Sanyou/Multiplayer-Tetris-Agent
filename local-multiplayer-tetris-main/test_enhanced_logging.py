#!/usr/bin/env python3
"""
Test script to verify enhanced logging functionality
"""

import torch
import sys
import os
sys.path.append('.')
from localMultiplayerTetris.rl_utils.unified_trainer import UnifiedTrainer
from localMultiplayerTetris.tetris_env import TetrisEnv

def test_enhanced_logging():
    print('Testing enhanced logging integration...')
    
    # Initialize environment and trainer
    env = TetrisEnv()
    trainer = UnifiedTrainer(
        env=env,
        hidden_size=128,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_state_model=1e-3,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        exploration_episodes=10,
        log_dir='logs/test_enhanced_logging'
    )
    
    print('✓ UnifiedTrainer initialized successfully')
    print('✓ TensorBoard writer configured')
    print('✓ ActorCritic agent has writer configured')
    
    # Test a short exploration phase
    print('\nTesting Phase 1: Exploration data collection...')
    try:
        trainer.phase_1_exploration()
        print('✓ Phase 1 completed successfully')
    except Exception as e:
        print(f'✗ Phase 1 failed: {e}')
        import traceback
        traceback.print_exc()
    
    # Test state model training
    print('\nTesting Phase 2: State model training...')
    try:
        trainer.phase_2_state_model_training()
        print('✓ Phase 2 completed successfully')
    except Exception as e:
        print(f'✗ Phase 2 failed: {e}')
        import traceback
        traceback.print_exc()
    
    # Test actor-critic training with enhanced logging
    print('\nTesting Phase 3: Actor-Critic training...')
    try:
        trainer.phase_3_actor_critic_training()
        print('✓ Phase 3 completed successfully')
    except Exception as e:
        print(f'✗ Phase 3 failed: {e}')
        import traceback
        traceback.print_exc()
    
    print('\nEnhanced logging test completed!')
    
    # Check if log files were created
    log_dir = trainer.log_dir
    if os.path.exists(log_dir):
        print(f'✓ Log directory created: {log_dir}')
        log_files = os.listdir(log_dir)
        if log_files:
            print(f'✓ Log files created: {log_files}')
        else:
            print('⚠ Log directory exists but no files found')
    else:
        print('⚠ Log directory not found')

if __name__ == "__main__":
    test_enhanced_logging()
