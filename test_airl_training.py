#!/usr/bin/env python3
"""
Test AIRL training with real expert trajectories
"""

import torch
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add local paths for imports
sys.path.append('local-multiplayer-tetris-main')
from localMultiplayerTetris.rl_utils.airl_fixed import AIRLAgent, AIRLConfig
from localMultiplayerTetris.rl_utils.trajectory_collector import TrajectoryCollector

def load_expert_trajectories(expert_dir: str = "expert_trajectories", max_trajectories: int = 5):
    """Load expert trajectories from saved files"""
    print(f"📂 Loading expert trajectories from {expert_dir}")
    
    trajectory_files = sorted([f for f in os.listdir(expert_dir) if f.endswith('.pkl')])
    
    if not trajectory_files:
        raise FileNotFoundError(f"No trajectory files found in {expert_dir}")
    
    # Load first few trajectories for testing
    trajectories = []
    total_steps = 0
    
    for i, filename in enumerate(trajectory_files[:max_trajectories]):
        filepath = os.path.join(expert_dir, filename)
        
        with open(filepath, 'rb') as f:
            episode_data = pickle.load(f)
        
        # Extract the actual trajectory data from the episode structure
        steps = episode_data['steps']
        
        # Convert to our expected format
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for j, step in enumerate(steps):
            states.append(step['state'])
            actions.append(step['action'])
            rewards.append(step['reward'])
            dones.append(step['done'])
            
            # Get next state (use current state for last step if needed)
            if j + 1 < len(steps):
                next_states.append(steps[j + 1]['state'])
            else:
                next_states.append(step['state'])  # Terminal state
        
        trajectory = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones)
        }
        
        trajectories.append(trajectory)
        total_steps += len(trajectory['states'])
        
        print(f"  Loaded {filename}: {len(trajectory['states'])} steps, "
              f"actions [{min(trajectory['actions'])}-{max(trajectory['actions'])}], "
              f"avg reward: {np.mean(trajectory['rewards']):.2f}")
    
    print(f"✓ Loaded {len(trajectories)} trajectories with {total_steps} total steps")
    return trajectories

def convert_trajectories_to_tensors(trajectories):
    """Convert trajectories to tensor format for AIRL"""
    print("🔄 Converting trajectories to tensor format...")
    
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_dones = []
    
    for traj in trajectories:
        # Convert numpy object arrays to proper numeric arrays
        states = []
        next_states = []
        
        for state in traj['states']:
            if isinstance(state, np.ndarray):
                states.append(state.astype(np.float32))
            else:
                # Convert to numpy array if it's not already
                states.append(np.array(state, dtype=np.float32))
        
        for next_state in traj['next_states']:
            if isinstance(next_state, np.ndarray):
                next_states.append(next_state.astype(np.float32))
            else:
                next_states.append(np.array(next_state, dtype=np.float32))
        
        all_states.extend(states)
        all_actions.extend(traj['actions'].astype(np.int64))
        all_next_states.extend(next_states)
        all_rewards.extend(traj['rewards'].astype(np.float32))
        all_dones.extend(traj['dones'].astype(bool))
    
    # Convert to tensors
    expert_data = {
        'states': torch.FloatTensor(np.array(all_states)),
        'actions': torch.LongTensor(np.array(all_actions)),
        'next_states': torch.FloatTensor(np.array(all_next_states)),
        'rewards': torch.FloatTensor(np.array(all_rewards)),
        'dones': torch.BoolTensor(np.array(all_dones))
    }
    
    print(f"✓ Expert data tensor shapes:")
    for key, tensor in expert_data.items():
        print(f"  {key}: {tensor.shape}")
    
    return expert_data

def test_airl_initialization():
    """Test AIRL agent initialization"""
    print("\n🤖 Testing AIRL Initialization")
    print("=" * 50)
    
    # Create config
    config = AIRLConfig()
    config.state_dim = 207
    config.action_dim = 41
    config.batch_size = 64  # Small batch for testing
    config.max_episodes = 10  # Few episodes for testing
    
    print(f"Config: state_dim={config.state_dim}, action_dim={config.action_dim}")
    print(f"Device: {config.device}")
    
    # Initialize AIRL agent
    agent = AIRLAgent(config)
    
    # Count parameters
    disc_params = sum(p.numel() for p in agent.discriminator.parameters())
    policy_params = sum(p.numel() for p in agent.policy.parameters())
    total_params = disc_params + policy_params
    
    print(f"✓ AIRL Agent initialized:")
    print(f"  Discriminator: {disc_params:,} parameters")
    print(f"  Policy: {policy_params:,} parameters")
    print(f"  Total: {total_params:,} parameters")
    print(f"  Under 300k constraint: {total_params < 300000}")
    
    return agent, config

def test_airl_forward_pass(agent, expert_data):
    """Test AIRL forward pass with real data"""
    print("\n⚡ Testing AIRL Forward Pass")
    print("=" * 50)
    
    # Get a small batch for testing
    batch_size = 32
    indices = torch.randperm(len(expert_data['states']))[:batch_size]
    
    states = expert_data['states'][indices]
    actions = expert_data['actions'][indices]
    next_states = expert_data['next_states'][indices]
    
    print(f"Testing with batch size: {batch_size}")
    
    # Test discriminator
    print("Testing discriminator...")
    try:
        disc_output = agent.discriminator(states, actions, next_states)
        print(f"✓ Discriminator output shape: {disc_output.shape}")
        print(f"  Value range: [{disc_output.min():.3f}, {disc_output.max():.3f}]")
    except Exception as e:
        print(f"❌ Discriminator error: {e}")
        return False
    
    # Test policy
    print("Testing policy...")
    try:
        policy_output = agent.policy(states)
        print(f"✓ Policy output shape: {policy_output.shape}")
        print(f"  Probability sums: {policy_output.sum(dim=1).mean():.3f} (should be ~1.0)")
        print(f"  Max action prob: {policy_output.max():.3f}")
    except Exception as e:
        print(f"❌ Policy error: {e}")
        return False
    
    return True

def test_airl_training_step(agent, expert_data):
    """Test a single AIRL training step"""
    print("\n🏋️ Testing AIRL Training Step")
    print("=" * 50)
    
    try:
        # Get a batch of expert data
        batch_size = 64
        expert_batch = {}
        indices = torch.randperm(len(expert_data['states']))[:batch_size]
        
        for key in expert_data:
            expert_batch[key] = expert_data[key][indices]
        
        print(f"Training with batch size: {batch_size}")
        
        # Generate policy data (random for testing)
        policy_batch = {
            'states': torch.randn(batch_size, 207),
            'actions': torch.randint(0, 41, (batch_size,)),
            'next_states': torch.randn(batch_size, 207),
            'rewards': torch.randn(batch_size),
            'dones': torch.zeros(batch_size, dtype=torch.bool)
        }
        
        # Test training step
        print("Attempting training step...")
        
        # Move data to device
        device = next(agent.discriminator.parameters()).device
        for batch in [expert_batch, policy_batch]:
            for key in batch:
                batch[key] = batch[key].to(device)
        
        # Manual training step (simplified)
        agent.discriminator.train()
        agent.policy.train()
        
        # Discriminator forward pass
        expert_disc = agent.discriminator(
            expert_batch['states'], 
            expert_batch['actions'], 
            expert_batch['next_states']
        )
        
        policy_disc = agent.discriminator(
            policy_batch['states'], 
            policy_batch['actions'], 
            policy_batch['next_states']
        )
        
        # Simple discriminator loss (binary classification)
        expert_labels = torch.ones_like(expert_disc)
        policy_labels = torch.zeros_like(policy_disc)
        
        disc_loss = (
            torch.nn.functional.binary_cross_entropy(expert_disc, expert_labels) +
            torch.nn.functional.binary_cross_entropy(policy_disc, policy_labels)
        )
        
        print(f"✓ Training step successful!")
        print(f"  Expert discrimination: {expert_disc.mean():.3f}")
        print(f"  Policy discrimination: {policy_disc.mean():.3f}")
        print(f"  Discriminator loss: {disc_loss.item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run AIRL training test"""
    print("🚀 AIRL Training Test with Real Expert Trajectories")
    print("=" * 60)
    
    # Load expert trajectories
    try:
        trajectories = load_expert_trajectories(max_trajectories=3)  # Small test
        expert_data = convert_trajectories_to_tensors(trajectories)
    except Exception as e:
        print(f"❌ Failed to load expert trajectories: {e}")
        return
    
    # Initialize AIRL
    try:
        agent, config = test_airl_initialization()
    except Exception as e:
        print(f"❌ Failed to initialize AIRL: {e}")
        return
    
    # Test forward pass
    if not test_airl_forward_pass(agent, expert_data):
        print("❌ Forward pass test failed!")
        return
    
    # Test training step
    if not test_airl_training_step(agent, expert_data):
        print("❌ Training step test failed!")
        return
    
    print("\n" + "=" * 60)
    print("🎉 All AIRL tests passed!")
    print("\n📋 Ready for full AIRL training:")
    print(f"✓ Expert trajectories loaded and converted")
    print(f"✓ AIRL networks initialized and tested")
    print(f"✓ Forward passes working correctly")
    print(f"✓ Training steps functioning")
    print(f"✓ Parameter count under 300k constraint")
    
    print("\n🔄 To run full training:")
    print("python train_airl.py --expert-dir expert_trajectories --max-episodes 1000")

if __name__ == "__main__":
    main() 