#!/usr/bin/env python3
"""
Test script for the enhanced Multi-Attempt Goal-Focused Training with HER
Tests the new multi-attempt mechanism with Hindsight Experience Replay (HER) using randomized future goals
"""

import sys
import os
import torch
import numpy as np

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from localMultiplayerTetris.rl_utils.unified_trainer import UnifiedTrainer, TrainingConfig
from localMultiplayerTetris.tetris_env import TetrisEnv

def test_multi_attempt_training():
    print("🧪 Testing Enhanced Multi-Attempt Goal-Focused Training with HER")
    print("="*70)
    
    # Create minimal config for testing
    config = TrainingConfig()
    config.num_batches = 1  # Just test one batch
    config.exploration_episodes = 2  # Minimal exploration
    config.exploitation_episodes = 2  # Test multi-attempt exploitation
    config.eval_episodes = 1  # Minimal evaluation
    config.exploration_mode = 'deterministic'  # Use deterministic for predictable results
    config.visualize = False  # No visualization for testing
    
    print(f"📋 Test Configuration:")
    print(f"   • Batches: {config.num_batches}")
    print(f"   • Exploration episodes: {config.exploration_episodes}")
    print(f"   • Exploitation episodes: {config.exploitation_episodes}")
    print(f"   • Exploration mode: {config.exploration_mode}")
    print(f"   • Device: {config.device}")
    
    try:
        # Initialize trainer
        print(f"\n🔧 Initializing Enhanced Trainer...")
        trainer = UnifiedTrainer(config)
        
        # Test Phase 1: Exploration
        print(f"\n🔍 Testing Phase 1: Exploration...")
        trainer.phase_1_exploration(0)
        exploration_data_count = len(trainer.exploration_data)
        print(f"   ✅ Generated {exploration_data_count} exploration data points")
        
        if exploration_data_count == 0:
            print(f"   ⚠️  No exploration data generated - skipping remaining phases")
            return False
        
        # Test Phase 2: State Learning
        print(f"\n🎯 Testing Phase 2: State Learning...")
        trainer.phase_2_state_learning(0)
        print(f"   ✅ State model training completed")
        
        # Test Phase 3: Reward Prediction
        print(f"\n🔮 Testing Phase 3: Reward Prediction...")
        trainer.phase_3_reward_prediction(0)
        print(f"   ✅ Reward predictor training completed")
        
        # Test Phase 4: Multi-Attempt Exploitation
        print(f"\n🎮 Testing Phase 4: Multi-Attempt Goal-Focused Exploitation...")
        old_buffer_size = len(trainer.experience_buffer)
        trainer.phase_4_exploitation(0)
        new_buffer_size = len(trainer.experience_buffer)
        
        # Check multi-attempt statistics
        exploitation_stats = trainer.batch_stats.get('exploitation', {})
        multi_attempt_enabled = exploitation_stats.get('multi_attempt_enabled', False)
        avg_attempts = exploitation_stats.get('avg_attempts_per_episode', 0)
        step_goal_success_rate = exploitation_stats.get('step_goal_success_rate', 0)
        episode_goal_success_rate = exploitation_stats.get('episode_goal_success_rate', 0)
        hindsight_trajectories = exploitation_stats.get('hindsight_trajectories', 0)
        total_hindsight_experiences = exploitation_stats.get('total_hindsight_experiences', 0)
        
        print(f"   ✅ Multi-attempt exploitation completed")
        print(f"   📊 Multi-attempt enabled: {multi_attempt_enabled}")
        print(f"   📊 Average attempts per episode: {avg_attempts:.1f}")
        print(f"   📊 Step-level goal success rate: {step_goal_success_rate:.3f}")
        print(f"   📊 Episode-level goal success rate: {episode_goal_success_rate:.3f}")
        print(f"   📊 Hindsight trajectories created: {hindsight_trajectories}")
        print(f"   📊 Total hindsight experiences: {total_hindsight_experiences}")
        print(f"   📊 Experience buffer growth: {old_buffer_size} → {new_buffer_size}")
        
        # Test Phase 5: PPO Training
        print(f"\n🏋️ Testing Phase 5: PPO with Hindsight...")
        if len(trainer.experience_buffer) >= trainer.config.min_buffer_size:
            trainer.phase_5_ppo_training(0)
            print(f"   ✅ PPO training with hindsight completed")
        else:
            print(f"   ⚠️  Insufficient experience for PPO training ({len(trainer.experience_buffer)} < {trainer.config.min_buffer_size})")
        
        # Test Phase 6: Evaluation
        print(f"\n📊 Testing Phase 6: Evaluation...")
        trainer.phase_6_evaluation(0)
        print(f"   ✅ Evaluation completed")
        
        # Print comprehensive test summary
        print(f"\n🎯 Enhanced Training Test Summary:")
        trainer.print_batch_summary(0)
        
        # Verify enhanced multi-attempt + HER features
        success_indicators = []
        
        if multi_attempt_enabled:
            success_indicators.append("✅ Multi-attempt mechanism enabled")
        else:
            success_indicators.append("❌ Multi-attempt mechanism not enabled")
        
        if avg_attempts >= 2.5:  # Should be close to 3 attempts per episode
            success_indicators.append("✅ Multiple attempts per episode detected")
        else:
            success_indicators.append("❌ Insufficient attempts per episode")
        
        if total_hindsight_experiences > avg_attempts * 0.5:  # Should have hindsight for significant portion of attempts
            success_indicators.append("✅ ALL-ATTEMPT HER (Hindsight Experience Replay) working")
        else:
            success_indicators.append("❌ ALL-ATTEMPT HER (Hindsight Experience Replay) not working")
        
        if new_buffer_size > old_buffer_size:
            success_indicators.append("✅ Experience buffer populated with hindsight")
        else:
            success_indicators.append("❌ Experience buffer not growing")
        
        # Additional HER-specific checks
        if step_goal_success_rate > 0 or episode_goal_success_rate > 0:
            success_indicators.append("✅ Goal achievement detected (step or episode level)")
        else:
            success_indicators.append("❌ No goal achievement detected")
        
        print(f"\n🔍 Enhanced Test Results:")
        for indicator in success_indicators:
            print(f"   {indicator}")
        
        # Overall success
        success_count = sum(1 for ind in success_indicators if ind.startswith("✅"))
        total_checks = len(success_indicators)
        
        print(f"\n🏆 Overall Test Result: {success_count}/{total_checks} checks passed")
        print(f"📊 Detailed Metrics:")
        print(f"   • Multi-attempt: {multi_attempt_enabled}")
        print(f"   • Attempts/episode: {avg_attempts:.1f}")
        print(f"   • Step success rate: {step_goal_success_rate:.3f} ({step_goal_success_rate*100:.1f}%)")
        print(f"   • Episode success rate: {episode_goal_success_rate:.3f} ({episode_goal_success_rate*100:.1f}%)")
        print(f"   • HER trajectories: {hindsight_trajectories}")
        print(f"   • Total hindsight experiences: {total_hindsight_experiences}")
        print(f"   • Buffer growth: {old_buffer_size} → {new_buffer_size}")
        print(f"   • Hindsight coverage: {total_hindsight_experiences / max(1, avg_attempts * config.exploitation_episodes):.1%}")
        
        if success_count >= 4:  # At least 4/5 checks should pass
            print(f"🎉 Multi-Attempt + HER Enhanced Training: WORKING")
            return True
        else:
            print(f"⚠️  Multi-Attempt + HER Enhanced Training: NEEDS ATTENTION")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_attempt_training()
    exit(0 if success else 1) 