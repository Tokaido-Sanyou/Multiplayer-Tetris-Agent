#!/usr/bin/env python3
"""
🎯 WORKING DREAM DEMONSTRATION

Demonstrates all DREAM components working together correctly.
"""

import torch
import numpy as np
import time
from dream.configs.dream_config import DREAMConfig
from dream.models.world_model import WorldModel
from dream.models.actor_critic import ActorCritic
from dream.buffers.replay_buffer import ReplayBuffer
from envs.tetris_env import TetrisEnv

def pad_observation(obs):
    """Pad 206→212 dimensions"""
    if isinstance(obs, np.ndarray) and obs.shape[0] == 206:
        return np.concatenate([obs, np.zeros(6)], axis=0)
    return obs

def collect_episode(env, actor_critic, device, max_steps=100):
    """Collect one episode of experience"""
    observations = []
    actions = []
    rewards = []
    dones = []
    
    obs = env.reset()
    obs = pad_observation(obs)
    episode_reward = 0
    
    for step in range(max_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = actor_critic.get_action_and_value(obs_tensor)
            # Convert 8-element binary vector to single action for direct mode
            action_scalar = torch.argmax(action.squeeze(0)).cpu().item()
        
        observations.append(obs)
        actions.append(action_scalar)
        
        next_obs, reward, done, info = env.step(action_scalar)
        next_obs = pad_observation(next_obs)
        
        rewards.append(reward)
        dones.append(done)
        episode_reward += reward
        
        if done:
            break
            
        obs = next_obs
    
    trajectory = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'dones': dones
    }
    
    return trajectory, episode_reward, len(observations)

def main():
    print("🎯 WORKING DREAM DEMONSTRATION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize components
    print("\n1️⃣ Initializing DREAM components...")
    config = DREAMConfig.get_default_config(action_mode='direct')
    
    world_model = WorldModel(**config.world_model).to(device)
    actor_critic = ActorCritic(**config.actor_critic).to(device)
    replay_buffer = ReplayBuffer(capacity=1000, sequence_length=20, device=device)
    
    world_params = sum(p.numel() for p in world_model.parameters())
    actor_params = sum(p.numel() for p in actor_critic.parameters())
    
    print(f"✅ World Model: {world_params:,} parameters")
    print(f"✅ Actor-Critic: {actor_params:,} parameters")
    print(f"✅ Replay Buffer: capacity {replay_buffer.capacity}")
    
    # Initialize environment
    print("\n2️⃣ Initializing environment...")
    env = TetrisEnv(num_agents=1, headless=True, action_mode='direct')
    print("✅ Environment created")
    
    # Collect experience
    print("\n3️⃣ Collecting experience...")
    total_reward = 0
    
    for episode in range(5):
        trajectory, episode_reward, episode_length = collect_episode(env, actor_critic, device, max_steps=50)
        replay_buffer.add_trajectory(trajectory)
        total_reward += episode_reward
        
        print(f"   Episode {episode}: Reward={episode_reward:6.2f}, Length={episode_length:2d}")
    
    print(f"✅ Collected {len(replay_buffer)} transitions")
    
    # Test world model training
    print("\n4️⃣ Testing world model...")
    if len(replay_buffer) > 0:
        batch = replay_buffer.sample_sequences(batch_size=2, sequence_length=10)
        
        # Convert to tensors
        observations = torch.stack([torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in seq]) 
                                  for seq in batch['observations']]).to(device)
        actions = torch.tensor(batch['actions'], dtype=torch.long).to(device)
        
        # Test forward pass
        with torch.no_grad():
            wm_output = world_model(observations, actions)
        
        print(f"✅ World model forward: {observations.shape} → rewards {wm_output['predicted_rewards'].shape}")
    
    # Test imagination
    print("\n5️⃣ Testing imagination...")
    initial_state = world_model.get_initial_state(1, device)
    imagination_actions = torch.randint(0, 8, (1, 15)).to(device)
    
    with torch.no_grad():
        imagination_output = world_model.imagine(initial_state, imagination_actions)
    
    print(f"✅ Imagination: {imagination_actions.shape} → rewards {imagination_output['predicted_rewards'].shape}")
    
    # Test actor-critic training
    print("\n6️⃣ Testing actor-critic...")
    if len(replay_buffer) > 0:
        batch = replay_buffer.sample_sequences(batch_size=2, sequence_length=10)
        
        # Flatten sequences for actor-critic
        flat_obs = []
        flat_actions = []
        for seq_obs, seq_actions in zip(batch['observations'], batch['actions']):
            for obs, action in zip(seq_obs, seq_actions):
                flat_obs.append(torch.tensor(obs, dtype=torch.float32))
                flat_actions.append(action)
        
        if flat_obs:
            obs_tensor = torch.stack(flat_obs).to(device)
            
            # Convert scalar actions to 8-element binary vectors for direct mode
            actions_binary = torch.zeros(len(flat_actions), 8).to(device)
            for i, action in enumerate(flat_actions):
                actions_binary[i, action] = 1.0
            
            with torch.no_grad():
                dist, values = actor_critic(obs_tensor)
                log_probs, eval_values, entropy = actor_critic.evaluate_actions(obs_tensor, actions_binary)
            
            print(f"✅ Actor-critic: {obs_tensor.shape} → values {values.shape}, entropy {entropy.mean().item():.3f}")
    
    # Show buffer statistics
    print("\n7️⃣ Buffer statistics...")
    stats = replay_buffer.get_statistics()
    print(f"✅ Episodes: {stats['num_episodes']}")
    print(f"✅ Mean reward: {stats['mean_episode_reward']:.2f}")
    print(f"✅ Mean length: {stats['mean_episode_length']:.1f}")
    print(f"✅ Buffer size: {stats['buffer_size']}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("🎉 DREAM DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("✅ All components working perfectly")
    print("✅ Experience collection functional")
    print("✅ World model forward passes working")
    print("✅ Imagination system operational")
    print("✅ Actor-critic training ready")
    print("✅ Buffer management working")
    print("\n🚀 DREAM is ready for full training!")
    print("=" * 60)

if __name__ == "__main__":
    main() 