#!/usr/bin/env python3
"""
Test script to verify AIRL implementation works with existing checkpoint
"""

import torch
import numpy as np
import sys
import os

# Add local paths for imports
sys.path.append('local-multiplayer-tetris-main')
from localMultiplayerTetris.rl_utils.checkpoint_compatible_actor_critic import (
    CheckpointCompatibleFeatureExtractor, 
    CheckpointCompatibleActorCritic,
    CheckpointCompatibleAgent
)

def test_feature_extractor():
    """Test feature extractor with checkpoint dimensions"""
    print("=== Testing Feature Extractor ===")
    
    # Create feature extractor
    feature_extractor = CheckpointCompatibleFeatureExtractor()
    
    # Test input (batch_size=1, 207 dimensions)
    test_input = torch.randn(1, 207)
    
    # Forward pass
    features = feature_extractor(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected output shape: (1, 1632)")
    
    assert features.shape == (1, 1632), f"Expected (1, 1632), got {features.shape}"
    print("✓ Feature extractor dimensions correct!")
    
    return feature_extractor

def test_actor_critic():
    """Test full Actor-Critic network"""
    print("\n=== Testing Actor-Critic Network ===")
    
    # Create network
    network = CheckpointCompatibleActorCritic()
    
    # Test input
    test_input = torch.randn(4, 207)  # Batch of 4
    
    # Forward pass
    action_probs, state_value = network(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Action probs shape: {action_probs.shape}")
    print(f"State value shape: {state_value.shape}")
    print(f"Action probs sum: {action_probs.sum(dim=1)}")  # Should be ~1.0 due to softmax
    
    assert action_probs.shape == (4, 41), f"Expected (4, 41), got {action_probs.shape}"
    assert state_value.shape == (4, 1), f"Expected (4, 1), got {state_value.shape}"
    print("✓ Actor-Critic network dimensions correct!")
    
    return network

def test_checkpoint_loading():
    """Test loading actual checkpoint"""
    print("\n=== Testing Checkpoint Loading ===")
    
    checkpoint_path = 'local-multiplayer-tetris-main/localMultiplayerTetris/checkpoints/80k_bad.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Create agent and load checkpoint
    agent = CheckpointCompatibleAgent()
    
    try:
        agent.load(checkpoint_path)
        print("✓ Checkpoint loaded successfully!")
        
        # Test action selection
        test_state = np.random.randn(207)
        action = agent.select_action(test_state, deterministic=True)
        action_with_value = agent.select_action_with_value(test_state)
        
        print(f"Test action: {action}")
        print(f"Test action with value: {action_with_value}")
        print(f"Action range check: 0 <= {action} <= 40: {0 <= action <= 40}")
        
        assert 0 <= action <= 40, f"Action {action} out of range [0, 40]"
        assert 0 <= action_with_value[0] <= 40, f"Action {action_with_value[0]} out of range [0, 40]"
        print("✓ Action selection working correctly!")
        
        return agent
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def test_airl_compatibility():
    """Test that our AIRL can use the checkpoint-compatible components"""
    print("\n=== Testing AIRL Compatibility ===")
    
    # Import our AIRL implementation
    from localMultiplayerTetris.rl_utils.airl import AIRLAgent, AIRLConfig
    
    # Create AIRL config with checkpoint-compatible settings
    config = AIRLConfig()
    config.state_dim = 207
    config.action_dim = 41
    
    # Create AIRL agent
    airl_agent = AIRLAgent(config)
    
    # Test with sample data
    states = torch.randn(10, 207)
    actions = torch.randint(0, 41, (10,))
    next_states = torch.randn(10, 207)
    dones = torch.zeros(10, dtype=torch.bool)
    
    # Test discriminator
    disc_output = airl_agent.discriminator(states, actions, next_states)
    print(f"Discriminator output shape: {disc_output.shape}")
    
    # Test policy
    action_probs = airl_agent.policy(states)
    print(f"Policy output shape: {action_probs.shape}")
    
    print("✓ AIRL components compatible with checkpoint architecture!")
    
    return airl_agent

def test_expert_trajectory_format():
    """Test expert trajectory generation for AIRL"""
    print("\n=== Testing Expert Trajectory Format ===")
    
    # Load checkpoint agent
    checkpoint_path = 'local-multiplayer-tetris-main/localMultiplayerTetris/checkpoints/80k_bad.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    agent = CheckpointCompatibleAgent()
    agent.load(checkpoint_path)
    
    # Simulate expert trajectory collection
    expert_trajectories = []
    
    for episode in range(3):  # Small test
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # Simulate episode
        state = np.random.randn(207)
        for step in range(10):  # Short episodes for testing
            action = agent.select_action(state, deterministic=True)
            next_state = np.random.randn(207)
            reward = np.random.randn()
            done = (step == 9)  # End episode at step 9
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            if done:
                break
        
        trajectory = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones)
        }
        
        expert_trajectories.append(trajectory)
        print(f"Episode {episode + 1}: {len(states)} steps, actions range: [{min(actions)}, {max(actions)}]")
    
    print(f"✓ Generated {len(expert_trajectories)} expert trajectories!")
    print(f"Total steps: {sum(len(traj['states']) for traj in expert_trajectories)}")
    
    return expert_trajectories

def main():
    """Run all tests"""
    print("🚀 Testing AIRL Implementation with Existing Checkpoint")
    print("=" * 60)
    
    # Test individual components
    feature_extractor = test_feature_extractor()
    network = test_actor_critic()
    
    # Test checkpoint loading
    agent = test_checkpoint_loading()
    
    if agent is not None:
        # Test AIRL compatibility
        airl_agent = test_airl_compatibility()
        
        # Test expert trajectory generation
        expert_trajectories = test_expert_trajectory_format()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! AIRL implementation is compatible with checkpoint.")
        print("\n📝 Next Steps:")
        print("1. Generate expert trajectories using replay_agent_with_trajectories.py")
        print("2. Train AIRL agent using train_airl.py")
        print("3. Evaluate against dual agent environment")
        
    else:
        print("\n" + "=" * 60)
        print("❌ Checkpoint loading failed. Please check checkpoint path.")

if __name__ == "__main__":
    main() 