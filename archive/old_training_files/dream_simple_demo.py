#!/usr/bin/env python3
"""
🎯 SIMPLIFIED DREAM DEMONSTRATION

Shows all DREAM components working together in a simple, clear way.
"""

import torch
import numpy as np
from dream.configs.dream_config import DREAMConfig
from dream.models.world_model import WorldModel
from dream.models.actor_critic import ActorCritic
from envs.tetris_env import TetrisEnv

def pad_observation(obs):
    """Pad 206→212 dimensions"""
    if isinstance(obs, np.ndarray) and obs.shape[0] == 206:
        return np.concatenate([obs, np.zeros(6)], axis=0)
    return obs

def main():
    print("🎯 SIMPLIFIED DREAM DEMONSTRATION")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Test Configuration
    print("\n1️⃣ Testing Configuration...")
    config = DREAMConfig.get_default_config(action_mode='direct')
    print(f"✅ Config loaded: obs_dim={config.world_model['observation_dim']}, state_dim={config.actor_critic['state_dim']}")
    
    # 2. Test Models
    print("\n2️⃣ Testing Models...")
    world_model = WorldModel(**config.world_model).to(device)
    actor_critic = ActorCritic(**config.actor_critic).to(device)
    
    world_params = sum(p.numel() for p in world_model.parameters())
    actor_params = sum(p.numel() for p in actor_critic.parameters())
    print(f"✅ World Model: {world_params:,} parameters")
    print(f"✅ Actor-Critic: {actor_params:,} parameters")
    
    # 3. Test Environment
    print("\n3️⃣ Testing Environment...")
    env = TetrisEnv(num_agents=1, headless=True, action_mode='direct')
    obs = env.reset()
    padded_obs = pad_observation(obs)
    print(f"✅ Environment: {obs.shape} → {padded_obs.shape} (padded)")
    
    # 4. Test Forward Passes
    print("\n4️⃣ Testing Forward Passes...")
    
    # World model test
    batch_size, seq_len = 2, 5
    test_obs = torch.randn(batch_size, seq_len, 212).to(device)
    test_actions = torch.randint(0, 8, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        wm_output = world_model(test_obs, test_actions)
    print(f"✅ World Model: {test_obs.shape} → rewards {wm_output['predicted_rewards'].shape}")
    
    # Actor-critic test
    test_state = torch.randn(batch_size, 212).to(device)
    with torch.no_grad():
        dist, value = actor_critic(test_state)
    print(f"✅ Actor-Critic: {test_state.shape} → value {value.shape}")
    
    # 5. Test Action Sampling
    print("\n5️⃣ Testing Action Sampling...")
    obs_tensor = torch.tensor(padded_obs, dtype=torch.float32).to(device).unsqueeze(0)
    
    with torch.no_grad():
        action, log_prob, value = actor_critic.get_action_and_value(obs_tensor)
        # For direct mode, convert 8-element binary vector to single action
        action_scalar = torch.argmax(action.squeeze(0)).cpu().item()
    
    print(f"✅ Action sampling: {action.shape} → scalar {action_scalar}")
    print(f"   Log prob: {log_prob.item():.3f}, Value: {value.item():.3f}")
    
    # 6. Test Environment Step
    print("\n6️⃣ Testing Environment Step...")
    next_obs, reward, done, info = env.step(action_scalar)
    next_padded_obs = pad_observation(next_obs)
    print(f"✅ Environment step: action={action_scalar}, reward={reward:.2f}, done={done}")
    print(f"   Next obs: {next_obs.shape} → {next_padded_obs.shape} (padded)")
    
    # 7. Test Imagination
    print("\n7️⃣ Testing Imagination...")
    initial_state = world_model.get_initial_state(1, device)
    imagination_actions = torch.randint(0, 8, (1, 10)).to(device)
    
    with torch.no_grad():
        imagination_output = world_model.imagine(initial_state, imagination_actions)
    
    print(f"✅ Imagination: {imagination_actions.shape} → rewards {imagination_output['predicted_rewards'].shape}")
    
    env.close()
    
    print("\n" + "=" * 50)
    print("🎉 ALL DREAM COMPONENTS WORKING PERFECTLY!")
    print("✅ Configuration: Correct dimensions")
    print("✅ Models: Forward passes successful")
    print("✅ Environment: Padding working")
    print("✅ Actions: Sampling and conversion working")
    print("✅ Imagination: World model imagination working")
    print("✅ Ready for full training!")
    print("=" * 50)

if __name__ == "__main__":
    main() 