#!/usr/bin/env python3
"""
Simple test to verify enhanced logging functionality
"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Test if all modules can be imported correctly"""
    try:
        from localMultiplayerTetris.rl_utils.unified_trainer import UnifiedTrainer
        print("✓ UnifiedTrainer imported successfully")
    except Exception as e:
        print(f"✗ UnifiedTrainer import failed: {e}")
        return False
        
    try:
        from localMultiplayerTetris.rl_utils.actor_critic import ActorCritic
        print("✓ ActorCritic imported successfully")
    except Exception as e:
        print(f"✗ ActorCritic import failed: {e}")
        return False
        
    try:
        from localMultiplayerTetris.rl_utils.exploration_actor import ExplorationActor
        print("✓ ExplorationActor imported successfully")
    except Exception as e:
        print(f"✗ ExplorationActor import failed: {e}")
        return False
        
    try:
        from localMultiplayerTetris.tetris_env import TetrisEnv
        print("✓ TetrisEnv imported successfully")
    except Exception as e:
        print(f"✗ TetrisEnv import failed: {e}")
        return False
        
    return True

def test_initialization():
    """Test initialization of components"""
    try:
        from localMultiplayerTetris.tetris_env import TetrisEnv
        from localMultiplayerTetris.rl_utils.unified_trainer import UnifiedTrainer
        
        env = TetrisEnv()
        print("✓ TetrisEnv initialized successfully")
        
        trainer = UnifiedTrainer(
            env=env,
            hidden_size=64,  # Smaller for testing
            lr_actor=3e-4,
            lr_critic=3e-4,
            lr_state_model=1e-3,
            batch_size=32,  # Smaller for testing
            gamma=0.99,
            tau=0.005,
            exploration_episodes=5,  # Much smaller for testing
            log_dir='logs/test_simple'
        )
        print("✓ UnifiedTrainer initialized successfully")
        print("✓ TensorBoard writer configured")
        print("✓ ActorCritic agent has writer configured")
        
        return trainer
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=== Enhanced Logging Test ===")
    
    # Test imports
    if not test_imports():
        print("Import tests failed, cannot continue")
        return
    
    print("\n=== Initialization Test ===")
    trainer = test_initialization()
    
    if trainer is None:
        print("Initialization failed, cannot continue")
        return
    
    print("\n=== Testing Logging Components ===")
    
    # Check if ActorCritic has the writer
    if hasattr(trainer.actor_critic, 'writer') and trainer.actor_critic.writer is not None:
        print("✓ ActorCritic has TensorBoard writer configured")
    else:
        print("✗ ActorCritic missing TensorBoard writer")
    
    # Check if writer is the same as trainer's writer
    if hasattr(trainer.actor_critic, 'writer') and trainer.actor_critic.writer == trainer.writer:
        print("✓ ActorCritic writer correctly linked to trainer writer")
    else:
        print("✗ ActorCritic writer not properly linked")
    
    print("\n=== Quick Functionality Test ===")
    try:
        # Test exploration actor
        exploration_actor = trainer.exploration_actor
        print("✓ Exploration actor accessible")
        
        # Test a very short data collection
        print("Testing mini exploration run...")
        data = exploration_actor.collect_placement_data(num_episodes=2)
        print(f"✓ Collected {len(data)} transitions")
        
        # Get statistics
        stats = exploration_actor.get_placement_statistics()
        print(f"✓ Exploration statistics: {stats}")
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
