#!/usr/bin/env python3
"""
🎯 FLAWLESS DREAM COMPONENT TEST

Tests all DREAM components to ensure they work perfectly together.
"""

import torch
import numpy as np
import sys
import traceback
from pathlib import Path

def test_dream_components():
    """Test all DREAM components"""
    print("🔍 TESTING DREAM COMPONENTS...")
    
    try:
        # Import components
        from dream.configs.dream_config import DREAMConfig
        from dream.models.world_model import WorldModel
        from dream.models.actor_critic import ActorCritic
        from envs.tetris_env import TetrisEnv
        
        print("✅ All imports successful")
        
        # Test environment
        env = TetrisEnv(num_agents=1, headless=True)
        obs = env.reset()
        print(f"✅ Environment: obs shape {obs.shape}")
        
        # Test padding (206 -> 212)
        if obs.shape[0] == 206:
            padded_obs = np.concatenate([obs, np.zeros(6)], axis=0)
            print(f"✅ Padding: {obs.shape} -> {padded_obs.shape}")
        else:
            padded_obs = obs
            print(f"✅ No padding needed: {obs.shape}")
        
        # Test config
        config = DREAMConfig.get_default_config(action_mode='direct')
        print(f"✅ Config loaded")
        print(f"   World model obs_dim: {config.world_model['observation_dim']}")
        print(f"   Actor-critic state_dim: {config.actor_critic['state_dim']}")
        
        # Test models
        world_model = WorldModel(**config.world_model)
        actor_critic = ActorCritic(**config.actor_critic)
        print(f"✅ Models initialized")
        
        # Test world model forward pass
        batch_size, seq_len = 2, 5
        test_obs = torch.randn(batch_size, seq_len, 212)
        test_actions = torch.randint(0, 8, (batch_size, seq_len))
        
        wm_output = world_model(test_obs, test_actions)
        print(f"✅ World Model forward: {test_obs.shape} -> rewards {wm_output['predicted_rewards'].shape}")
        
        # Test actor-critic forward pass
        test_state = torch.randn(batch_size, 212)
        dist, value = actor_critic(test_state)
        print(f"✅ Actor-Critic forward: {test_state.shape} -> value {value.shape}")
        
        # Test action sampling
        action, log_prob, _ = actor_critic.get_action_and_value(test_state)
        print(f"✅ Action sampling: action {action.shape}, log_prob {log_prob.shape}")
        
        # Test environment step with model
        obs_tensor = torch.tensor(padded_obs, dtype=torch.float32).unsqueeze(0)
        action, _, _ = actor_critic.get_action_and_value(obs_tensor)
        action_scalar = action.squeeze(0).item()
        
        next_obs, reward, done, info = env.step(action_scalar)
        print(f"✅ Environment step: action={action_scalar}, reward={reward:.2f}")
        
        env.close()
        
        print("🎉 ALL DREAM COMPONENTS WORKING FLAWLESSLY!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dream_components()
    if success:
        print("\n🚀 DREAM is ready for training!")
    else:
        print("\n❌ Issues detected. Check output above.")
    
    sys.exit(0 if success else 1) 