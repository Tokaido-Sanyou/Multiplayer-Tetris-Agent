#!/usr/bin/env python3
"""
Comprehensive Integration Test for Streamlined Enhanced Tetris RL System
Tests RND exploration, future state prediction, model persistence, and phase reporting
"""

import torch
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localMultiplayerTetris.config import TetrisConfig
from localMultiplayerTetris.tetris_env import TetrisEnv
from localMultiplayerTetris.rl_utils.rnd_exploration import RNDExplorationActor
from localMultiplayerTetris.rl_utils.actor_critic import ActorCriticAgent
from localMultiplayerTetris.rl_utils.state_model import StateModel
from localMultiplayerTetris.rl_utils.future_reward_predictor import FutureRewardPredictor
from localMultiplayerTetris.rl_utils.unified_trainer import UnifiedTrainer, TrainingConfig

def test_enhanced_system():
    """Test all enhanced system components"""
    print("🧪 Testing Enhanced Streamlined Tetris RL System")
    print("="*60)
    
    # Test 1: Configuration and dimensions
    print("\n1️⃣ Testing Centralized Configuration...")
    config = TetrisConfig()
    assert config.STATE_DIM == 410, f"Expected STATE_DIM=410, got {config.STATE_DIM}"
    assert config.ACTION_DIM == 8, f"Expected ACTION_DIM=8, got {config.ACTION_DIM}"
    assert config.GOAL_DIM == 36, f"Expected GOAL_DIM=36, got {config.GOAL_DIM}"
    print("✅ Configuration dimensions correct")
    
    # Test 2: Enhanced RND Exploration
    print("\n2️⃣ Testing Enhanced RND Exploration...")
    env = TetrisEnv(single_player=True, headless=True)
    rnd_actor = RNDExplorationActor(env)
    
    # Test RND components
    assert hasattr(rnd_actor, 'rnd_exploration'), "RND exploration component missing"
    assert hasattr(rnd_actor, 'visited_terminal_states'), "Terminal state tracking missing"
    assert hasattr(rnd_actor, 'novelty_bonus_scale'), "Novelty bonus system missing"
    
    # Test RND collection (mini batch)
    placement_data = rnd_actor.collect_placement_data(num_episodes=2)
    assert len(placement_data) > 0, "No placement data collected"
    
    # Verify enhanced data structure
    sample_data = placement_data[0]
    required_fields = ['state', 'placement', 'terminal_reward', 'resulting_state', 
                      'intrinsic_reward', 'terminal_value', 'novelty_score']
    for field in required_fields:
        assert field in sample_data, f"Missing field: {field}"
    
    # Test novelty stats
    novelty_stats = rnd_actor._get_novelty_stats()
    assert 'unique_terminals' in novelty_stats, "Novelty stats missing unique_terminals"
    print("✅ Enhanced RND exploration working")
    
    # Test 3: Future State Prediction in Actor-Critic
    print("\n3️⃣ Testing Future State Prediction...")
    actor_critic = ActorCriticAgent()
    
    # Test network has future state predictor
    assert hasattr(actor_critic.network, 'future_state_predictor'), "Future state predictor missing"
    assert hasattr(actor_critic, 'future_state_optimizer'), "Future state optimizer missing"
    
    # Test forward pass with future prediction
    dummy_state = torch.randn(1, config.STATE_DIM)
    dummy_goal = torch.randn(1, config.GOAL_DIM)
    
    # Without future prediction
    action_probs, state_value = actor_critic.network(dummy_state, dummy_goal, predict_future=False)
    assert action_probs.shape == (1, config.ACTION_DIM), f"Wrong action shape: {action_probs.shape}"
    assert state_value.shape == (1, 1), f"Wrong value shape: {state_value.shape}"
    
    # With future prediction
    action_probs, state_value, future_state = actor_critic.network(dummy_state, dummy_goal, predict_future=True)
    assert future_state.shape == (1, config.STATE_DIM), f"Wrong future state shape: {future_state.shape}"
    print("✅ Future state prediction working")
    
    # Test 4: Enhanced Model Persistence
    print("\n4️⃣ Testing Enhanced Model Persistence...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test actor-critic save/load
        save_path = os.path.join(temp_dir, 'actor_critic.pt')
        actor_critic.save(save_path)
        
        # Modify agent state
        original_epsilon = actor_critic.epsilon
        actor_critic.epsilon = 0.5
        
        # Load and verify
        actor_critic.load(save_path)
        assert actor_critic.epsilon == original_epsilon, "Epsilon not restored correctly"
        print("✅ Actor-critic persistence working")
        
        # Test unified trainer checkpoint
        print("\n5️⃣ Testing Unified Trainer Checkpoint System...")
        training_config = TrainingConfig()
        training_config.num_batches = 1
        training_config.exploration_episodes = 2
        training_config.exploitation_episodes = 2
        training_config.log_dir = temp_dir
        training_config.checkpoint_dir = temp_dir
        
        trainer = UnifiedTrainer(training_config)
        
        # Save checkpoint
        trainer.save_checkpoint(0)
        
        # Verify files exist
        assert os.path.exists(os.path.join(temp_dir, 'checkpoint_batch_0.pt')), "Batch checkpoint missing"
        assert os.path.exists(os.path.join(temp_dir, 'latest_checkpoint.pt')), "Latest checkpoint missing"
        
        # Test loading
        checkpoint = trainer.load_checkpoint(os.path.join(temp_dir, 'latest_checkpoint.pt'))
        assert 'config' in checkpoint, "Config missing from checkpoint"
        assert 'rnd_exploration_state' in checkpoint, "RND state missing from checkpoint"
        print("✅ Unified trainer checkpoints working")
    
    # Test 5: Phase Integration and Statistics
    print("\n6️⃣ Testing Phase Integration...")
    
    # Test batch statistics tracking
    trainer.update_batch_stats('exploration', {'avg_terminal': 12.5, 'success_rate': 0.68})
    trainer.update_batch_stats('state_model', {'total_loss': 0.0457, 'loss_improvement': 0.123})
    
    assert 'exploration' in trainer.batch_stats, "Exploration stats not tracked"
    assert 'state_model' in trainer.batch_stats, "State model stats not tracked"
    
    # Test batch summary (captures output)
    from io import StringIO
    import contextlib
    
    captured_output = StringIO()
    with contextlib.redirect_stdout(captured_output):
        trainer.print_batch_summary(0)
    
    output = captured_output.getvalue()
    assert "📊 BATCH 1 SUMMARY" in output, "Batch summary header missing"
    assert "🔍 EXPLORATION" in output, "Exploration summary missing"
    assert "🎯 STATE MODEL" in output, "State model summary missing"
    print("✅ Phase integration working")
    
    # Test 6: Integration Test - Mini Training Run
    print("\n7️⃣ Testing Mini Training Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Very minimal training config for integration test
        mini_config = TrainingConfig()
        mini_config.num_batches = 1
        mini_config.exploration_episodes = 1
        mini_config.exploitation_episodes = 1
        mini_config.eval_episodes = 1
        mini_config.state_epochs = 1
        mini_config.ppo_iterations = 1
        mini_config.ppo_epochs = 1
        mini_config.log_dir = temp_dir
        mini_config.checkpoint_dir = temp_dir
        
        mini_trainer = UnifiedTrainer(mini_config)
        
        # Test each phase works in sequence
        try:
            mini_trainer.phase_1_exploration(0)
            print("   • Phase 1 ✅")
            
            mini_trainer.phase_2_state_learning(0)
            print("   • Phase 2 ✅")
            
            mini_trainer.phase_3_reward_prediction(0)
            print("   • Phase 3 ✅")
            
            mini_trainer.phase_4_exploitation(0)
            print("   • Phase 4 ✅")
            
            mini_trainer.phase_5_ppo_training(0)
            print("   • Phase 5 ✅")
            
            mini_trainer.phase_6_evaluation(0)
            print("   • Phase 6 ✅")
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False
    
    print("✅ Mini training integration working")
    
    # Final verification
    print("\n🎉 All Enhanced System Tests Passed!")
    print("="*60)
    print("✅ Enhanced RND exploration with terminal value focus")
    print("✅ Future state prediction in actor-critic")
    print("✅ Comprehensive model persistence")
    print("✅ Streamlined phase reporting")
    print("✅ Console batch summaries")
    print("✅ Complete integration working")
    
    return True

if __name__ == "__main__":
    try:
        success = test_enhanced_system()
        if success:
            print("\n🚀 System ready for enhanced training!")
            print("Run: python -m localMultiplayerTetris.rl_utils.unified_trainer --visualize")
        else:
            print("\n❌ Tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 