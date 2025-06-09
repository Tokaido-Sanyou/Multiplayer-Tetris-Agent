#!/usr/bin/env python3
"""Test all DREAM components for flawless operation"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))

try:
    from dream.configs.dream_config import DREAMConfig
    from dream.models.world_model import WorldModel
    from dream.models.actor_critic import ActorCritic
    from envs.tetris_env import TetrisEnv
    
    print('🔍 TESTING DREAM COMPONENTS...')
    
    # Test environment
    env = TetrisEnv(num_agents=1, headless=True)
    obs = env.reset()
    print(f'✅ Environment: obs shape {obs.shape}')
    
    # Test padding
    padded_obs = np.concatenate([obs, np.zeros(6)], axis=0)
    print(f'✅ Padding: {obs.shape} -> {padded_obs.shape}')
    
    # Test config
    config = DREAMConfig.get_default_config(action_mode='direct')
    print(f'✅ Config: world_model obs_dim = {config.world_model["observation_dim"]}')
    print(f'✅ Config: actor_critic state_dim = {config.actor_critic["state_dim"]}')
    
    # Test models
    world_model = WorldModel(**config.world_model)
    actor_critic = ActorCritic(**config.actor_critic)
    print(f'✅ Models initialized successfully')
    
    # Test forward passes
    test_obs = torch.randn(1, 10, 212)
    test_actions = torch.randint(0, 8, (1, 10))
    wm_output = world_model(test_obs, test_actions)
    print(f'✅ World Model: {test_obs.shape} -> rewards {wm_output["predicted_rewards"].shape}')
    
    test_state = torch.randn(1, 212)
    dist, value = actor_critic(test_state)
    print(f'✅ Actor-Critic: {test_state.shape} -> value {value.shape}')
    
    # Test action sampling
    action, log_prob = dist.sample(), dist.log_prob(dist.sample())
    print(f'✅ Action sampling: action shape {action.shape}, log_prob shape {log_prob.shape}')
    
    env.close()
    print('🎉 ALL TESTS PASSED!')
    
except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc() 