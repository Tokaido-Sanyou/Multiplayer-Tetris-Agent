#!/usr/bin/env python3
"""
Simple test to verify checkpoint compatibility and trajectory generation
"""

import torch
import numpy as np
import sys
import os

# Add local paths for imports
sys.path.append('local-multiplayer-tetris-main')
from localMultiplayerTetris.rl_utils.checkpoint_compatible_actor_critic import CheckpointCompatibleAgent

def test_checkpoint_agent():
    """Test checkpoint loading and basic agent functionality"""
    print("🧪 Testing Checkpoint Agent")
    print("=" * 50)
    
    checkpoint_path = 'local-multiplayer-tetris-main/localMultiplayerTetris/checkpoints/80k_bad.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    # Create and load agent
    agent = CheckpointCompatibleAgent()
    agent.load(checkpoint_path)
    
    print(f"✓ Checkpoint loaded successfully!")
    print(f"✓ Network has {sum(p.numel() for p in agent.network.parameters())} parameters")
    print(f"✓ Running on device: {agent.device}")
    
    # Test action selection with various inputs
    print("\n🎯 Testing Action Selection:")
    
    test_cases = [
        np.random.randn(207),           # Random state
        np.zeros(207),                  # Zero state
        np.ones(207),                   # Ones state
        np.random.randn(207) * 10,      # Large values
    ]
    
    for i, test_state in enumerate(test_cases):
        action = agent.select_action(test_state, deterministic=True)
        action_val = agent.select_action_with_value(test_state)
        
        print(f"  Test {i+1}: Action={action}, Value={action_val[1]:.2f}")
        
        # Verify action is valid
        assert 0 <= action <= 40, f"Invalid action: {action}"
        assert 0 <= action_val[0] <= 40, f"Invalid action: {action_val[0]}"
    
    print("✓ All action selections valid!")
    
    return True

def test_trajectory_generation():
    """Test trajectory generation for AIRL training"""
    print("\n📊 Testing Trajectory Generation")
    print("=" * 50)
    
    checkpoint_path = 'local-multiplayer-tetris-main/localMultiplayerTetris/checkpoints/80k_bad.pt'
    agent = CheckpointCompatibleAgent()
    agent.load(checkpoint_path)
    
    # Simulate expert trajectory collection
    trajectories = []
    total_steps = 0
    
    print("Generating sample trajectories...")
    
    for episode in range(5):
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        # Simulate an episode
        state = np.random.randn(207)
        episode_length = np.random.randint(50, 200)  # Variable episode lengths
        
        for step in range(episode_length):
            # Get action from agent
            action = agent.select_action(state, deterministic=False)  # Use stochastic for variety
            
            # Simulate next state and reward
            next_state = np.random.randn(207)
            reward = np.random.randn()  # Random reward
            done = (step == episode_length - 1)
            
            # Store transition
            episode_data['states'].append(state.copy())
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['next_states'].append(next_state.copy())
            episode_data['dones'].append(done)
            
            state = next_state
            if done:
                break
        
        # Convert to numpy arrays
        for key in episode_data:
            episode_data[key] = np.array(episode_data[key])
        
        trajectories.append(episode_data)
        total_steps += len(episode_data['states'])
        
        print(f"  Episode {episode+1}: {len(episode_data['states'])} steps, "
              f"actions in [{min(episode_data['actions'])}, {max(episode_data['actions'])}]")
    
    print(f"\n✓ Generated {len(trajectories)} trajectories with {total_steps} total steps")
    print(f"✓ Average episode length: {total_steps / len(trajectories):.1f}")
    
    # Verify trajectory format for AIRL
    sample_traj = trajectories[0]
    print(f"\n📋 Trajectory Format Verification:")
    print(f"  States shape: {sample_traj['states'].shape}")
    print(f"  Actions shape: {sample_traj['actions'].shape}")
    print(f"  Rewards shape: {sample_traj['rewards'].shape}")
    print(f"  Next states shape: {sample_traj['next_states'].shape}")
    print(f"  Dones shape: {sample_traj['dones'].shape}")
    
    # Verify all actions are valid
    all_actions = np.concatenate([traj['actions'] for traj in trajectories])
    print(f"  Action range: [{all_actions.min()}, {all_actions.max()}]")
    print(f"  Valid actions: {np.all((all_actions >= 0) & (all_actions <= 40))}")
    
    return trajectories

def test_data_format_for_airl():
    """Test that our data format is suitable for AIRL training"""
    print("\n🤖 Testing AIRL Data Compatibility")
    print("=" * 50)
    
    # Generate sample data
    trajectories = test_trajectory_generation()
    
    # Convert to format expected by AIRL
    print("\nConverting to AIRL format...")
    
    all_states = []
    all_actions = []
    all_next_states = []
    all_dones = []
    
    for traj in trajectories:
        all_states.extend(traj['states'])
        all_actions.extend(traj['actions'])
        all_next_states.extend(traj['next_states'])
        all_dones.extend(traj['dones'])
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(np.array(all_states))
    actions_tensor = torch.LongTensor(np.array(all_actions))
    next_states_tensor = torch.FloatTensor(np.array(all_next_states))
    dones_tensor = torch.BoolTensor(np.array(all_dones))
    
    print(f"✓ AIRL Training Data Ready:")
    print(f"  States: {states_tensor.shape}")
    print(f"  Actions: {actions_tensor.shape}")
    print(f"  Next States: {next_states_tensor.shape}")
    print(f"  Dones: {dones_tensor.shape}")
    print(f"  Total transitions: {len(all_states)}")
    
    return {
        'states': states_tensor,
        'actions': actions_tensor,
        'next_states': next_states_tensor,
        'dones': dones_tensor
    }

def main():
    """Run all tests"""
    print("🚀 Checkpoint Compatibility Test")
    print("=" * 60)
    
    # Test 1: Basic checkpoint loading
    if not test_checkpoint_agent():
        print("❌ Checkpoint loading failed!")
        return
    
    # Test 2: Trajectory generation  
    trajectories = test_trajectory_generation()
    
    # Test 3: AIRL data format
    airl_data = test_data_format_for_airl()
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("\n📋 Summary:")
    print(f"✓ Checkpoint loaded with 41-action architecture")
    print(f"✓ Expert trajectories can be generated")
    print(f"✓ Data format compatible with AIRL training")
    print(f"✓ Ready for full AIRL implementation")
    
    print("\n🔄 Next Steps:")
    print("1. Run replay_agent_with_trajectories.py to collect real expert data")
    print("2. Train AIRL discriminator and policy with expert trajectories")
    print("3. Test dual-agent environment with AIRL-trained agents")

if __name__ == "__main__":
    main() 