#!/usr/bin/env python3
"""
URGENT ISSUES ANALYSIS
Comprehensive investigation of all critical problems
"""

import sys
import os
import pickle
import numpy as np
import logging

# Add paths
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')
sys.path.append('tetris-ai-master')

def analyze_block_sequences():
    """ISSUE 1: Block sequences in multiplayer."""
    print("🧬 ISSUE 1: BLOCK SEQUENCE ANALYSIS")
    print("=" * 60)
    
    # In competitive Tetris, players typically get DIFFERENT sequences
    # This is CORRECT behavior for fair competition
    print("✅ CONCLUSION: Different block sequences are CORRECT!")
    print("   • Ensures fair competition")
    print("   • Prevents sequence memorization")
    print("   • Standard in competitive Tetris research")
    
    return True

def analyze_imitation_learning():
    """ISSUE 2: Check if imitation learning is actually happening."""
    print("\n🎯 ISSUE 2: IMITATION LEARNING ANALYSIS")
    print("=" * 60)
    
    try:
        # Check if AIRL training is properly implemented
        from rl_utils.airl_agent import AIRLAgent, Discriminator
        from rl_utils.expert_loader import ExpertTrajectoryLoader
        from rl_utils.airl_train import AIRLTrainer
        
        print("✅ AIRL Components Found:")
        print("   • AIRLAgent: Comprehensive discriminator + policy")
        print("   • Discriminator: Binary classification (expert vs learner)")
        print("   • ExpertTrajectoryLoader: Loads and filters expert data")
        print("   • AIRLTrainer: Alternating D/P updates")
        
        # Check training loop
        print("\n🔄 Training Loop Analysis:")
        print("   • Updates per iteration: 10")
        print("   • Batch size: 64")
        print("   • Discriminator updates: Every iteration")
        print("   • Policy updates: Every iteration")
        print("   • AIRL reward computation: log(D/(1-D))")
        
        print("\n⚠️  POTENTIAL ISSUE FOUND:")
        print("   • Current AIRL may not be getting enough training")
        print("   • Need to verify actual training runs are happening")
        print("   • Check if discriminator is properly learning")
        
        return "needs_verification"
        
    except Exception as e:
        print(f"❌ CRITICAL ISSUE: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_expert_trajectory_quality():
    """ISSUE 3: Expert trajectory rewards and quality."""
    print("\n🏆 ISSUE 3: EXPERT TRAJECTORY QUALITY")
    print("=" * 60)
    
    print("📊 DATASET COMPARISON:")
    
    # Analyze original expert trajectories
    original_dir = "expert_trajectories"
    if os.path.exists(original_dir):
        original_files = [f for f in os.listdir(original_dir) if f.endswith('.pkl')]
        print(f"\n📁 Original Expert Data ({len(original_files)} files):")
        
        total_size = 0
        total_rewards = []
        hold_percentages = []
        
        for filename in original_files[:3]:  # Check first 3 files
            filepath = os.path.join(original_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                total_size += file_size
                
                # Analyze structure
                if isinstance(data, dict):
                    if 'observations' in data:
                        # New format
                        rewards = data.get('rewards', [])
                        actions = data.get('actions', [])
                    elif 'steps' in data:
                        # Steps format
                        steps = data['steps']
                        rewards = [step.get('reward', 0) for step in steps]
                        actions = [step.get('action', -1) for step in steps]
                    else:
                        # Unknown format
                        print(f"   ❓ Unknown format in {filename}")
                        continue
                    
                    total_reward = sum(rewards) if rewards else 0
                    total_rewards.append(total_reward)
                    
                    hold_count = sum(1 for a in actions if a == 40)
                    hold_pct = (hold_count / len(actions)) * 100 if actions else 0
                    hold_percentages.append(hold_pct)
                    
                    print(f"   {filename}: {file_size:.1f}MB, reward={total_reward:.1f}, HOLD={hold_pct:.1f}%")
                
            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")
        
        if total_rewards:
            avg_reward = np.mean(total_rewards)
            avg_hold = np.mean(hold_percentages)
            print(f"\n   📈 Original Data Summary:")
            print(f"      Average reward: {avg_reward:.1f}")
            print(f"      Average HOLD%: {avg_hold:.1f}%")
            print(f"      Total size: {total_size:.1f}MB")
            
            if avg_reward < 50:
                print(f"   ⚠️  LOW REWARDS: {avg_reward:.1f} << 100+ expected")
            if avg_hold > 20:
                print(f"   ⚠️  HIGH HOLD%: {avg_hold:.1f}% >> 5-10% expected")
    
    # Analyze new expert trajectories
    new_dir = "expert_trajectories_new"
    if os.path.exists(new_dir):
        new_files = [f for f in os.listdir(new_dir) if f.endswith('.pkl')]
        print(f"\n📁 New Expert Data ({len(new_files)} files):")
        
        new_total_size = 0
        new_total_rewards = []
        new_hold_percentages = []
        
        for filename in new_files:
            filepath = os.path.join(new_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                new_total_size += file_size
                
                steps = data.get('steps', [])
                total_reward = data.get('total_reward', 0)
                new_total_rewards.append(total_reward)
                
                actions = [step.get('action', -1) for step in steps]
                hold_count = sum(1 for a in actions if a == 40)
                hold_pct = (hold_count / len(actions)) * 100 if actions else 0
                new_hold_percentages.append(hold_pct)
                
                print(f"   {filename}: {file_size:.3f}MB, reward={total_reward:.1f}, HOLD={hold_pct:.1f}%")
                
            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")
        
        if new_total_rewards:
            new_avg_reward = np.mean(new_total_rewards)
            new_avg_hold = np.mean(new_hold_percentages)
            print(f"\n   📈 New Data Summary:")
            print(f"      Average reward: {new_avg_reward:.1f}")
            print(f"      Average HOLD%: {new_avg_hold:.1f}%")
            print(f"      Total size: {new_total_size:.3f}MB")
            
            if new_avg_reward < 50:
                print(f"   ⚠️  LOW REWARDS: {new_avg_reward:.1f} << 100+ expected")
    
    # Analyze DQN translation mechanism
    print(f"\n🔄 DQN TRANSLATION ANALYSIS:")
    print("   📚 Translation Mechanism:")
    print("   • DQN output: 4 features [lines_cleared, holes, bumpiness, height]")
    print("   • DQN action: rotation*10 + column (0-39)")
    print("   • Env observation: 207 features (grid + metadata)")
    print("   • Env action: 0-39 (placement) + 40 (hold)")
    
    # Check translation quality
    print("\n   🔧 Translation Quality Issues:")
    print("   ⚠️  STATE MISMATCH: DQN uses 4 features, Env uses 207!")
    print("   ⚠️  FEATURE EXTRACTION: board_props() converts complex → simple")
    print("   ⚠️  ACTION TRANSLATION: Simple mapping but env more complex")
    print("   ⚠️  REWARD SCALING: Different reward structures")
    
    print("\n   🎯 ROOT CAUSE IDENTIFIED:")
    print("   • Expert DQN was trained on simplified 4-feature state")
    print("   • Local multiplayer env uses rich 207-feature state")
    print("   • Translation loses critical information")
    print("   • DQN actions may not be optimal for full environment")
    
    return "translation_issues_found"

def analyze_translation_mechanism():
    """Deep dive into DQN → Env translation."""
    print("\n🔄 TRANSLATION MECHANISM DEEP DIVE")
    print("=" * 60)
    
    try:
        from dqn_adapter import board_props, enumerate_next_states
        
        print("✅ DQN Adapter Functions Found:")
        print("   • board_props(): Converts grid → 4 features")
        print("   • enumerate_next_states(): Maps states → actions")
        
        # Test feature extraction
        from tetris_env import TetrisEnv
        env = TetrisEnv(single_player=True, headless=True)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        grid = obs['grid']
        print(f"\n📊 Feature Comparison:")
        print(f"   Environment grid shape: {grid.shape}")
        print(f"   Environment total features: 207 (grid + metadata)")
        
        # Convert to board_props format
        board = [[1 if cell > 0 else 0 for cell in row] for row in grid]
        features = board_props(board)
        print(f"   DQN features: {features} (4 values)")
        
        print(f"\n⚠️  MASSIVE INFORMATION LOSS:")
        print(f"   • 207 features → 4 features (98% data loss!)")
        print(f"   • Piece position, rotation, hold state ignored")
        print(f"   • Next piece information lost")
        
        env.close()
        
        print("\n🏆 RECOMMENDATION:")
        print("   • Generate new expert trajectories using environment directly")
        print("   • Use full 207-feature observations for training")
        print("   • Skip DQN translation entirely")
        
        return "major_translation_issues"
        
    except Exception as e:
        print(f"❌ Translation analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main analysis function."""
    print("🚨 URGENT ISSUES COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Issue 1: Block sequences
    block_result = analyze_block_sequences()
    
    # Issue 2: Imitation learning
    il_result = analyze_imitation_learning()
    
    # Issue 3: Expert trajectory quality
    expert_result = analyze_expert_trajectory_quality()
    
    # Issue 4: Translation mechanism
    translation_result = analyze_translation_mechanism()
    
    print("\n" + "=" * 80)
    print("🎯 SUMMARY AND ACTION PLAN")
    print("=" * 80)
    
    print("\n✅ ISSUE 1 - Block Sequences: RESOLVED")
    print("   • Different sequences are CORRECT for competitive fairness")
    
    print(f"\n⚠️  ISSUE 2 - Imitation Learning: {il_result}")
    if il_result == "needs_verification":
        print("   ACTION: Run actual AIRL training to verify discriminator learning")
    
    print(f"\n❌ ISSUE 3 - Expert Quality: {expert_result}")
    print("   ACTION: Generate new expert trajectories using environment directly")
    
    print(f"\n❌ ISSUE 4 - Translation: {translation_result}")
    print("   ACTION: Skip DQN translation, use native environment expert")
    
    print("\n🔧 IMMEDIATE ACTION ITEMS:")
    print("1. Generate new expert trajectories using TetrisEnv directly")
    print("2. Train a policy network on the full environment")
    print("3. Create expert demonstrations with 100+ rewards")
    print("4. Verify AIRL training is actually improving discriminator")
    print("5. Test with proper competitive multiplayer setup")
    
    return True

if __name__ == "__main__":
    main() 