#!/usr/bin/env python3
"""
🎯 DREAM TRAINER - COMPREHENSIVE FINAL FIX

Fixed Issues:
1. ✅ World model config dimension mismatch (212 vs 206)
2. ✅ Hard drop action correction (5 not 6) 
3. ✅ Testing until line clearing actually happens
4. ✅ Action strategy for line clearing
5. ✅ Proper reward mode configuration
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from pathlib import Path
import logging

# Environment and DREAM imports
from envs.tetris_env import TetrisEnv
from dream.models.replay_buffer import ReplayBuffer
from dream.configs.dream_config import DREAMConfig
from dream.models.world_model import WorldModel

class FixedActorCritic(torch.nn.Module):
    """Fixed actor-critic with correct categorical distribution"""
    
    def __init__(self, state_dim=206, action_dim=8, hidden_dim=400):
        super().__init__()
        self.action_dim = action_dim
        
        # Shared feature extraction
        self.features = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        
        # Separate heads
        self.actor = torch.nn.Linear(hidden_dim, action_dim)  # Logits for categorical
        self.critic = torch.nn.Linear(hidden_dim, 1)        # Value estimate
    
    def forward(self, state):
        """Forward pass returning distribution and value"""
        features = self.features(state)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.critic(features)
        return dist, value
    
    def get_action_and_value(self, state, deterministic=False):
        """Get action and value for given state"""
        dist, value = self(state)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)
    
    def evaluate_actions(self, state, action):
        """Evaluate actions for training"""
        dist, value = self(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy

class CompleteDREAMTrainer:
    """Final comprehensive DREAM trainer with all fixes"""
    
    def __init__(self, reward_mode='lines_only', episodes=1000, max_buffer_size=50000, 
                 show_dashboard=False, use_tensorboard=True):
        """Initialize with all critical fixes"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 INITIALIZING COMPREHENSIVE DREAM TRAINER")
        print(f"   Device: {self.device}")
        print(f"   Reward mode: {reward_mode}")
        print(f"   Episodes: {episodes}")
        print(f"   Max buffer size: {max_buffer_size:,}")
        
        # Logging setup
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir = f'logs/dream_comprehensive_fix'
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir + '/tensorboard')
        else:
            self.writer = None
        
        # Environment with native dimensions
        self.env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='direct',
            reward_mode=reward_mode
        )
        
        # Get actual observation dimension
        obs = self.env.reset()
        self.state_dim = obs.shape[0]  # Should be 206
        print(f"   ✅ Environment state dimension: {self.state_dim}")
        
        # Fixed Actor-Critic
        self.actor_critic = FixedActorCritic(
            state_dim=self.state_dim, 
            action_dim=8, 
            hidden_dim=400
        ).to(self.device)
        
        # World model with CORRECTED configuration
        try:
            self.config = DREAMConfig.get_default_config(action_mode='direct')
            world_config = self.config.world_model.copy()
            
            # CRITICAL FIX: Use correct observation dimension
            if 'observation_dim' in world_config:
                world_config['observation_dim'] = self.state_dim
            if 'obs_dim' in world_config:
                world_config['obs_dim'] = self.state_dim
            
            self.world_model = WorldModel(**world_config).to(self.device)
            self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
            print(f"   ✅ World model initialized with corrected obs_dim={self.state_dim}")
        except Exception as e:
            print(f"   ⚠️  World model failed: {e}")
            self.world_model = None
            self.world_optimizer = None
        
        # Actor-critic optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=max_buffer_size,
            sequence_length=20,
            device=self.device
        )
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.lines_cleared = []
        self.pieces_placed = []
        self.world_losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.buffer_sizes = []
        
        print(f"   ✅ Actor-Critic: {sum(p.numel() for p in self.actor_critic.parameters()):,} params")
        if self.world_model:
            print(f"   ✅ World Model: {sum(p.numel() for p in self.world_model.parameters()):,} params")
        print("   🎯 ALL CRITICAL FIXES APPLIED!")
    
    def collect_episode_with_smart_strategy(self):
        """Collect episode with line-clearing strategy"""
        obs = self.env.reset()
        episode_reward = 0
        episode_lines = 0
        episode_pieces = 0
        trajectory = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
        
        for step in range(500):  # Max steps
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                if step < 100:  # First 100 steps: mostly exploration
                    if random.random() < 0.7:
                        # Smart strategy: mix of actions favoring piece placement
                        action_probs = [0.1, 0.1, 0.15, 0.1, 0.1, 0.4, 0.05, 0.0]  # Favor hard drop
                        action = np.random.choice(8, p=action_probs)
                    else:
                        action, _, _ = self.actor_critic.get_action_and_value(obs_tensor)
                        action = action.item()
                else:  # Later: mostly use learned policy
                    action, _, _ = self.actor_critic.get_action_and_value(obs_tensor)
                    action = action.item()
            
            # Execute action
            next_obs, reward, done, info = self.env.step(action)
            
            # Track metrics
            if 'lines_cleared' in info and info['lines_cleared'] > 0:
                episode_lines += info['lines_cleared']
                print(f"   🎉 Step {step}: Cleared {info['lines_cleared']} lines! Total: {episode_lines}")
            
            if 'pieces_placed' in info:
                episode_pieces = info['pieces_placed']
            
            # Store experience
            trajectory['observations'].append(obs.copy())
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            
            episode_reward += reward
            
            if done:
                print(f"   Game ended at step {step}")
                break
                
            obs = next_obs
        
        return trajectory, episode_reward, len(trajectory['observations']), episode_lines, episode_pieces
    
    def train_world_model(self):
        """Train world model with fixed dimensions"""
        if len(self.replay_buffer) < 32 or self.world_model is None:
            return {'world_loss': 0.0}
        
        try:
            batch = self.replay_buffer.sample_sequences(batch_size=16, sequence_length=10)
            
            # Convert to tensors with STRICT dimension checking
            observations = []
            for seq in batch['observations']:
                seq_tensors = []
                for obs in seq:
                    if isinstance(obs, np.ndarray) and obs.shape[0] == self.state_dim:
                        seq_tensors.append(torch.tensor(obs, dtype=torch.float32))
                    else:
                        print(f"   ⚠️  Dimension mismatch: expected {self.state_dim}, got {obs.shape if hasattr(obs, 'shape') else type(obs)}")
                        continue
                if len(seq_tensors) == len(seq):
                    observations.append(torch.stack(seq_tensors))
            
            if not observations:
                return {'world_loss': 0.0}
            
            observations = torch.stack(observations).to(self.device)
            actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)
            rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
            
            # Forward pass
            world_output = self.world_model(observations, actions)
            
            # Compute losses
            reward_loss = torch.nn.functional.mse_loss(
                world_output['predicted_rewards'], 
                rewards
            )
            
            if 'predicted_observations' in world_output:
                obs_loss = torch.nn.functional.mse_loss(
                    world_output['predicted_observations'], 
                    observations
                )
                total_loss = reward_loss + 0.1 * obs_loss
            else:
                total_loss = reward_loss
            
            # Backward pass
            self.world_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.world_optimizer.step()
            
            return {'world_loss': total_loss.item()}
            
        except Exception as e:
            print(f"   ⚠️  World model training error: {e}")
            return {'world_loss': 0.0}
    
    def train_actor_critic(self):
        """Train actor-critic with fixed categorical actions"""
        if len(self.replay_buffer) < 32:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
        
        try:
            batch = self.replay_buffer.sample_sequences(batch_size=16, sequence_length=10)
            
            # Flatten sequences with dimension checking
            flat_obs = []
            flat_actions = []
            flat_rewards = []
            
            for seq_obs, seq_actions, seq_rewards in zip(batch['observations'], batch['actions'], batch['rewards']):
                for obs, action, reward in zip(seq_obs, seq_actions, seq_rewards):
                    if isinstance(obs, np.ndarray) and obs.shape[0] == self.state_dim:
                        flat_obs.append(torch.tensor(obs, dtype=torch.float32))
                        flat_actions.append(action)
                        flat_rewards.append(reward)
            
            if not flat_obs:
                return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
            
            obs_tensor = torch.stack(flat_obs).to(self.device)
            actions_tensor = torch.tensor(flat_actions, dtype=torch.long).to(self.device)
            rewards_tensor = torch.tensor(flat_rewards, dtype=torch.float32).to(self.device)
            
            # Forward pass
            log_probs, values, entropy = self.actor_critic.evaluate_actions(obs_tensor, actions_tensor)
            
            # Simple advantage estimation
            advantages = rewards_tensor - values.detach()
            
            # Losses
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, rewards_tensor)
            entropy_loss = -entropy.mean()
            
            total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            # Backward pass
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
            self.actor_optimizer.step()
            
            return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'entropy': entropy.mean().item()
            }
            
        except Exception as e:
            print(f"   ⚠️  Actor-critic training error: {e}")
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
    
    def log_metrics(self, episode, metrics):
        """Log metrics to tensorboard"""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, episode)
            self.writer.flush()
    
    def train(self):
        """Main training loop with comprehensive fixes"""
        print("\n🚀 STARTING COMPREHENSIVE TRAINING")
        print("=" * 60)
        
        total_lines_cleared = 0
        best_episode_lines = 0
        
        for episode in range(1000):
            # Collect episode with smart strategy
            trajectory, episode_reward, episode_length, episode_lines, episode_pieces = self.collect_episode_with_smart_strategy()
            total_lines_cleared += episode_lines
            
            if episode_lines > best_episode_lines:
                best_episode_lines = episode_lines
                print(f"   🎉 NEW BEST! Episode {episode}: {episode_lines} lines cleared!")
            
            # Add to replay buffer
            self.replay_buffer.add_episode(trajectory)
            
            # Train models
            world_metrics = self.train_world_model()
            actor_metrics = self.train_actor_critic()
            
            # Combine metrics
            metrics = {
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'episode_lines': episode_lines,
                'episode_pieces': episode_pieces,
                'total_lines': total_lines_cleared,
                'world_loss': world_metrics['world_loss'],
                'actor_loss': actor_metrics['actor_loss'],
                'critic_loss': actor_metrics['critic_loss'],
                'entropy': actor_metrics['entropy'],
                'buffer_size': len(self.replay_buffer),
                'best_lines': best_episode_lines
            }
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.lines_cleared.append(episode_lines)
            self.pieces_placed.append(episode_pieces)
            self.world_losses.append(world_metrics['world_loss'])
            self.actor_losses.append(actor_metrics['actor_loss'])
            self.critic_losses.append(actor_metrics['critic_loss'])
            self.entropies.append(actor_metrics['entropy'])
            self.buffer_sizes.append(len(self.replay_buffer))
            
            # Log to tensorboard
            self.log_metrics(episode, metrics)
            
            # Progress report
            if episode % 20 == 0:
                avg_reward = np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0
                avg_lines = np.mean(self.lines_cleared[-20:]) if self.lines_cleared else 0
                recent_world_loss = np.mean(self.world_losses[-20:]) if self.world_losses else 0
                
                print(f"\n📊 Episode {episode:4d}")
                print(f"   Avg Reward (20): {avg_reward:8.2f}")
                print(f"   Avg Lines (20):  {avg_lines:8.2f}")
                print(f"   Total Lines:     {total_lines_cleared:8d}")
                print(f"   Best Episode:    {best_episode_lines:8d} lines")
                print(f"   World Loss:      {recent_world_loss:8.4f}")
                print(f"   Buffer Size:     {len(self.replay_buffer):8,}")
                
                # Stop if we've achieved good line clearing
                if total_lines_cleared >= 50:
                    print(f"\n🎯 SUCCESS! Cleared {total_lines_cleared} total lines!")
                    break
        
        self.cleanup()
        print("\n✅ TRAINING COMPLETE!")
        print(f"   Total lines cleared: {total_lines_cleared}")
        print(f"   Best episode: {best_episode_lines} lines")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.writer:
            self.writer.close()
        self.env.close()

def main():
    """Run comprehensive training"""
    trainer = CompleteDREAMTrainer(
        reward_mode='lines_only',  # Focus on line clearing
        episodes=1000,
        max_buffer_size=50000,
        show_dashboard=False,
        use_tensorboard=True
    )
    trainer.train()

if __name__ == "__main__":
    main() 