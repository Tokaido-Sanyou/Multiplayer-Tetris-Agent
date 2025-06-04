"""Test script to verify model loading compatibility.

This script attempts to load the saved weights for both the reward network
and PPO model to verify our implementations are compatible.
"""
import os
from pathlib import Path
import torch
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO

from localMultiplayerTetris.reward_model import RewardModel
from localMultiplayerTetris.ppo_agent import PPOAgent

def test_reward_model_loading():
    """Test loading the reward network weights."""
    print("\nTesting Reward Model Loading...")
    
    # Create model instance
    reward_model = RewardModel()
    reward_model.to('cpu')  # Ensure model is on CPU
    
    # Load weights - use relative path
    weights_path = Path("weights/100k/reward_net_100000.pth")
    try:
        state_dict = torch.load(weights_path, map_location='cpu')
        reward_model.load_state_dict(state_dict)
        print("✓ Successfully loaded reward model weights")
        
        # Test forward pass
        dummy_state = torch.randn(1, 207)  # Batch size 1, state dim 207
        dummy_action = torch.tensor([0])  # Single action
        reward = reward_model(dummy_state, dummy_action)
        print("✓ Successfully ran forward pass")
        print(f"  Output shape: {reward.shape}")
        print(f"  Output value: {reward.item():.3f}")
        
    except Exception as e:
        print(f"✗ Failed to load reward model: {str(e)}")
        return False
    
    return True

def test_ppo_loading():
    """Test loading the PPO model weights."""
    print("\nTesting PPO Model Loading...")
    
    # Create our PPO model instance
    ppo_agent = PPOAgent()
    ppo_agent.to('cpu')  # Ensure model is on CPU
    
    # Load the Stable Baselines 3 model - use relative path
    sb3_path = Path("weights/100k/ppo_generator_100000/policy.pth")
    try:
        # Load policy weights
        state_dict = torch.load(sb3_path, map_location='cpu')
        print("✓ Successfully loaded SB3 policy weights")
        
        # Create mapping between SB3 and our model's keys
        key_mapping = {
            # Feature extractor
            'features_extractor.grid_encoder.0.weight': 'feature_extractor.grid_encoder.0.weight',
            'features_extractor.grid_encoder.0.bias': 'feature_extractor.grid_encoder.0.bias',
            'features_extractor.grid_encoder.2.weight': 'feature_extractor.grid_encoder.2.weight',
            'features_extractor.grid_encoder.2.bias': 'feature_extractor.grid_encoder.2.bias',
            
            # Policy
            'pi.0.weight': 'policy.0.weight',
            'pi.0.bias': 'policy.0.bias',
            'pi.2.weight': 'policy.2.weight',
            'pi.2.bias': 'policy.2.bias',
            
            # Value
            'vf.0.weight': 'value.0.weight',
            'vf.0.bias': 'value.0.bias',
            'vf.2.weight': 'value.2.weight',
            'vf.2.bias': 'value.2.bias',
        }
        
        # Create new state dict with mapped keys
        new_state_dict = {}
        for sb3_key, our_key in key_mapping.items():
            if sb3_key in state_dict:
                new_state_dict[our_key] = state_dict[sb3_key]
        
        # Load weights into our model
        ppo_agent.load_state_dict(new_state_dict, strict=False)
        print("✓ Successfully transferred weights to our model")
        
        # Test forward pass
        dummy_state = torch.randn(1, 207)  # Batch size 1, state dim 207
        action_probs, value = ppo_agent(dummy_state)
        print("✓ Successfully ran forward pass")
        print(f"  Action probs shape: {action_probs.shape}")
        print(f"  Value shape: {value.shape}")
        
        # Test action selection
        action = ppo_agent.act(dummy_state[0], deterministic=True)
        print(f"  Selected action: {action}")
        
    except Exception as e:
        print(f"✗ Failed to load PPO model: {str(e)}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Starting model loading tests...")
    
    # Set random seeds for reproducibility
    set_random_seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Test reward model
    reward_success = test_reward_model_loading()
    
    # Test PPO model
    ppo_success = test_ppo_loading()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Reward Model: {'✓ PASS' if reward_success else '✗ FAIL'}")
    print(f"PPO Model: {'✓ PASS' if ppo_success else '✗ FAIL'}")

if __name__ == "__main__":
    main() 