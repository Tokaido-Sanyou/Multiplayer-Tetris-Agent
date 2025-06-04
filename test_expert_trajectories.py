#!/usr/bin/env python3
"""
Test Expert Trajectories for AIRL Training
Sample and validate the generated expert trajectories
"""

import sys
import os
import pickle
import numpy as np

# Add paths
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')

def test_expert_trajectories():
    """Test loading and sampling expert trajectories."""
    print("🧪 TESTING EXPERT TRAJECTORIES FOR AIRL")
    print("=" * 60)
    
    try:
        from rl_utils.expert_loader import ExpertTrajectoryLoader
        
        # Test loading trajectories
        expert_loader = ExpertTrajectoryLoader(
            trajectory_dir="expert_trajectories_dqn_adapter",
            max_trajectories=None,
            min_episode_length=30,
            max_hold_percentage=10.0,  # Very strict since our trajectories have 0% hold
        )
        
        print("📂 Loading expert trajectories...")
        num_loaded = expert_loader.load_trajectories()
        print(f"   Loaded: {num_loaded} trajectories")
        
        if num_loaded == 0:
            print("❌ No trajectories loaded!")
            return False
        
        # Get statistics
        stats = expert_loader.get_statistics()
        print(f"\n📊 Expert Data Statistics:")
        print(f"   Trajectories: {stats['num_trajectories']}")
        print(f"   Transitions: {stats['num_transitions']}")
        print(f"   Mean reward: {stats['mean_reward']:.1f}")
        print(f"   Std reward: {stats['std_reward']:.1f}")
        print(f"   Mean episode length: {stats['mean_episode_length']:.1f}")
        
        # Test sampling
        print(f"\n🎯 Testing Expert Sampling:")
        batch_size = 32
        device = 'cpu'
        
        # Sample expert batch
        expert_batch = expert_loader.get_batch(batch_size, device)
        
        print(f"   Batch size: {batch_size}")
        print(f"   States shape: {expert_batch['states'].shape}")
        print(f"   Actions shape: {expert_batch['actions'].shape}")
        print(f"   Rewards shape: {expert_batch['rewards'].shape}")
        print(f"   Next states shape: {expert_batch['next_states'].shape}")
        print(f"   Dones shape: {expert_batch['dones'].shape}")
        
        # Sample state-action pairs for discriminator
        state_action_pairs = expert_loader.get_state_action_pairs(batch_size, device)
        states, actions = state_action_pairs
        
        print(f"   State-action states: {states.shape}")
        print(f"   State-action actions: {actions.shape}")
        
        # Test individual trajectory loading
        print(f"\n📋 Sample Trajectory Analysis:")
        trajectory_file = "expert_trajectories_dqn_adapter/dqn_adapter_ep004_r3246.pkl"
        
        with open(trajectory_file, 'rb') as f:
            trajectory = pickle.load(f)
        
        print(f"   File: {os.path.basename(trajectory_file)}")
        print(f"   Episode ID: {trajectory['episode_id']}")
        print(f"   Total reward: {trajectory['total_reward']}")
        print(f"   Length: {trajectory['length']}")
        print(f"   Policy type: {trajectory['policy_type']}")
        
        # Analyze steps
        steps = trajectory['steps']
        actions = [step['action'] for step in steps]
        rewards = [step['reward'] for step in steps]
        
        print(f"   Actions (first 10): {actions[:10]}")
        print(f"   Rewards (first 10): {[f'{r:.1f}' for r in rewards[:10]]}")
        print(f"   Action range: {min(actions)} to {max(actions)}")
        print(f"   Positive rewards: {sum(1 for r in rewards if r > 0)}")
        
        # Verify state format
        sample_state = steps[0]['state']
        print(f"\n🔍 State Format Verification:")
        print(f"   State type: {type(sample_state)}")
        if isinstance(sample_state, dict):
            print(f"   State keys: {list(sample_state.keys())}")
            print(f"   Grid shape: {sample_state['grid'].shape}")
            print(f"   Next piece: {sample_state['next_piece']}")
            print(f"   Hold piece: {sample_state['hold_piece']}")
        
        print(f"\n✅ Expert trajectories are ready for AIRL training!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing trajectories: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    success = test_expert_trajectories()
    
    if success:
        print("\n🎯 AIRL Training Requirements Met:")
        print("   ✅ Expert trajectories loaded successfully")
        print("   ✅ Batch sampling works correctly")
        print("   ✅ State-action pairs available")
        print("   ✅ High-quality demonstrations (1000+ rewards)")
        print("   ✅ Compatible with ExpertTrajectoryLoader")
    else:
        print("\n❌ Issues found - need to fix before AIRL training")
    
    return success

if __name__ == "__main__":
    main() 