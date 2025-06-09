import torch
import numpy as np
from dream.configs.dream_config import DREAMConfig
from dream.models.world_model import WorldModel
from dream.models.actor_critic import ActorCritic
from envs.tetris_env import TetrisEnv

print("Testing DREAM components...")

# Test config
config = DREAMConfig.get_default_config(action_mode='direct')
print(f"Config loaded: obs_dim={config.world_model['observation_dim']}, state_dim={config.actor_critic['state_dim']}")

# Test models
world_model = WorldModel(**config.world_model)
actor_critic = ActorCritic(**config.actor_critic)
print("Models created successfully")

# Test environment
env = TetrisEnv(num_agents=1, headless=True)
obs = env.reset()
print(f"Environment: obs shape {obs.shape}")

# Test padding
if obs.shape[0] == 206:
    padded_obs = np.concatenate([obs, np.zeros(6)], axis=0)
    print(f"Padded: {obs.shape} -> {padded_obs.shape}")
else:
    padded_obs = obs

# Test forward pass
test_obs = torch.randn(1, 5, 212)
test_actions = torch.randint(0, 8, (1, 5))
wm_output = world_model(test_obs, test_actions)
print(f"World model works: {wm_output['predicted_rewards'].shape}")

test_state = torch.randn(1, 212)
dist, value = actor_critic(test_state)
print(f"Actor-critic works: value shape {value.shape}")

env.close()
print("All tests passed! DREAM is ready.") 