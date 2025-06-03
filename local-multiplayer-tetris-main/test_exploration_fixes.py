#!/usr/bin/env python3

"""
Test script for exploration actor functionality
"""

import os
import sys
import torch
import numpy as np
import logging

# Add the project path to sys.path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_exploration_actor():
    """Test ExplorationActor functionality"""
    try:
        from localMultiplayerTetris.tetris_env import TetrisEnv
        from localMultiplayerTetris.rl_utils.exploration_actor import ExplorationActor
        
        print("Creating environment...")
        env = TetrisEnv(single_player=True, headless=True)
        
        print("Creating exploration actor...")
        exploration_actor = ExplorationActor(env)
        
        print("Testing collect_placement_data with small num_episodes...")
        placement_data = exploration_actor.collect_placement_data(num_episodes=2)
        
        print(f"✓ ExplorationActor test completed successfully!")
        print(f"✓ Collected {len(placement_data)} placement data points")
        
        if placement_data:
            sample_data = placement_data[0]
            print(f"✓ Sample data keys: {list(sample_data.keys())}")
            print(f"✓ Sample terminal reward: {sample_data['terminal_reward']:.3f}")
            print(f"✓ Sample placement: {sample_data['placement']}")
        
        return True
        
    except Exception as e:
        print(f"✗ ExplorationActor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unified_trainer_phase1():
    """Test Phase 1 of UnifiedTrainer"""
    try:
        from localMultiplayerTetris.rl_utils.unified_trainer import UnifiedTrainer, TrainingConfig
        
        print("Creating training config...")
        config = TrainingConfig()
        config.exploration_episodes = 2  # Small number for testing
        config.visualize = False
        config.log_dir = 'logs/test'
        config.checkpoint_dir = 'checkpoints/test'
        
        # Create directories
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        print("Creating unified trainer...")
        trainer = UnifiedTrainer(config)
        
        print("Testing Phase 1: Exploration...")
        trainer.phase_1_exploration(batch=0)
        
        print(f"✓ UnifiedTrainer Phase 1 test completed successfully!")
        print(f"✓ Exploration data collected: {len(trainer.exploration_data)} items")
        
        return True
        
    except Exception as e:
        print(f"✗ UnifiedTrainer Phase 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Exploration Logging Fixes ===\n")
    
    print("1. Testing ExplorationActor...")
    exploration_success = test_exploration_actor()
    
    print("\n2. Testing UnifiedTrainer Phase 1...")
    phase1_success = test_unified_trainer_phase1()
    
    print("\n=== Test Summary ===")
    if exploration_success and phase1_success:
        print("✓ All tests passed! Exploration logging is working properly.")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        sys.exit(1)
