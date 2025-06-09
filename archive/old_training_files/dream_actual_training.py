#!/usr/bin/env python3
"""
🚀 ACTUAL DREAM TRAINING SESSION

Runs real DREAM training episodes with comprehensive stats and logging.
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os

from dream.configs.dream_config import DREAMConfig
from dream.models.world_model import WorldModel
from dream.models.actor_critic import ActorCritic
from dream.buffers.replay_buffer import ReplayBuffer
from envs.tetris_env import TetrisEnv

class DREAMTrainingSession:
    """Complete DREAM training session with stats and logging"""
    
    def __init__(self, num_episodes=100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_episodes = num_episodes
        
        # Initialize components
        self.config = DREAMConfig.get_default_config(action_mode='direct')
        self.config.max_episode_length = 500  # Reasonable episode length
        self.config.min_buffer_size = 10      # Start training early
        
        # Create padded environment
        self.env = self._create_padded_environment()
        
        # Initialize models
        self.world_model = WorldModel(**self.config.world_model).to(self.device)
        self.actor_critic = ActorCritic(**self.config.actor_critic).to(self.device)
        
        # Initialize optimizers
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.config.world_model_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.config.actor_lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            sequence_length=self.config.sequence_length,
            device=self.device
        )
        
        # Stats tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.pieces_placed = []
        self.lines_cleared = []
        self.world_model_losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        
        # Create logging directory
        self.log_dir = Path("logs/dream_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 DREAM Training Session Initialized:")
        print(f"   Device: {self.device}")
        print(f"   Episodes: {num_episodes}")
        print(f"   World Model params: {sum(p.numel() for p in self.world_model.parameters()):,}")
        print(f"   Actor-Critic params: {sum(p.numel() for p in self.actor_critic.parameters()):,}")
        print(f"   Log directory: {self.log_dir}")
    
    def _create_padded_environment(self):
        """Create environment with automatic dimension padding"""
        class PaddedTetrisEnv:
            def __init__(self, base_env):
                self.base_env = base_env
                self.observation_space = base_env.observation_space
                self.action_space = base_env.action_space
                
            def reset(self):
                obs = self.base_env.reset()
                return self._pad_observation(obs)
                
            def step(self, action):
                next_obs, reward, done, info = self.base_env.step(action)
                return self._pad_observation(next_obs), reward, done, info
                
            def _pad_observation(self, obs):
                """Pad 206→212 dimensions"""
                if isinstance(obs, np.ndarray) and obs.shape[0] == 206:
                    return np.concatenate([obs, np.zeros(6)], axis=0)
                return obs
                
            def close(self):
                return self.base_env.close()
        
        base_env = TetrisEnv(num_agents=1, headless=True, action_mode='direct')
        return PaddedTetrisEnv(base_env)
    
    def collect_episode(self):
        """Collect one episode of experience"""
        observations = []
        actions = []
        rewards = []
        dones = []
        
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        pieces_placed = 0
        lines_cleared = 0
        
        for step in range(self.config.max_episode_length):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = self.actor_critic.get_action_and_value(obs_tensor)
                action_scalar = torch.argmax(action.squeeze(0)).cpu().item()
            
            observations.append(obs)
            actions.append(action_scalar)
            
            next_obs, reward, done, info = self.env.step(action_scalar)
            
            rewards.append(reward)
            dones.append(done)
            episode_reward += reward
            episode_length += 1
            
            # Track Tetris-specific stats
            if 'pieces_placed' in info:
                pieces_placed = info['pieces_placed']
            if 'lines_cleared' in info:
                lines_cleared = info['lines_cleared']
            
            if done:
                break
                
            obs = next_obs
        
        trajectory = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }
        
        return trajectory, episode_reward, episode_length, pieces_placed, lines_cleared
    
    def train_world_model(self):
        """Train the world model"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {'world_loss': 0.0}
        
        batch = self.replay_buffer.sample_sequences(batch_size=self.config.batch_size, sequence_length=20)
        
        # Convert to tensors
        observations = torch.stack([torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in seq]) 
                                  for seq in batch['observations']]).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
        
        # Forward pass
        world_output = self.world_model(observations, actions)
        
        # Compute losses
        reward_loss = torch.nn.functional.mse_loss(world_output['predicted_rewards'], rewards)
        obs_loss = torch.nn.functional.mse_loss(world_output['predicted_observations'], observations)
        
        total_loss = reward_loss + 0.1 * obs_loss
        
        # Backward pass
        self.world_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip_norm)
        self.world_optimizer.step()
        
        return {
            'world_loss': total_loss.item(),
            'reward_loss': reward_loss.item(),
            'obs_loss': obs_loss.item()
        }
    
    def train_actor_critic(self):
        """Train the actor-critic"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
        
        batch = self.replay_buffer.sample_sequences(batch_size=self.config.batch_size, sequence_length=20)
        
        # Flatten sequences
        flat_obs = []
        flat_actions = []
        flat_rewards = []
        
        for seq_obs, seq_actions, seq_rewards in zip(batch['observations'], batch['actions'], batch['rewards']):
            for obs, action, reward in zip(seq_obs, seq_actions, seq_rewards):
                flat_obs.append(torch.tensor(obs, dtype=torch.float32))
                flat_actions.append(action)
                flat_rewards.append(reward)
        
        if not flat_obs:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
        
        obs_tensor = torch.stack(flat_obs).to(self.device)
        rewards_tensor = torch.tensor(flat_rewards, dtype=torch.float32).to(self.device)
        
        # Convert scalar actions to binary vectors for direct mode
        actions_binary = torch.zeros(len(flat_actions), 8).to(self.device)
        for i, action in enumerate(flat_actions):
            actions_binary[i, action] = 1.0
        
        # Forward pass
        dist, values = self.actor_critic(obs_tensor)
        log_probs, eval_values, entropy = self.actor_critic.evaluate_actions(obs_tensor, actions_binary)
        
        # Compute losses
        advantages = rewards_tensor - values.detach()
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = torch.nn.functional.mse_loss(values, rewards_tensor)
        entropy_loss = -entropy.mean()
        
        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        
        # Backward pass
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.grad_clip_norm)
        self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.mean().item()
        }
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 STARTING DREAM TRAINING ({self.num_episodes} episodes)")
        print("=" * 80)
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            episode_start = time.time()
            
            # Collect experience
            trajectory, episode_reward, episode_length, pieces, lines = self.collect_episode()
            self.replay_buffer.add_trajectory(trajectory)
            
            # Train components
            world_losses = self.train_world_model()
            policy_losses = self.train_actor_critic()
            
            # Store stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.pieces_placed.append(pieces)
            self.lines_cleared.append(lines)
            self.world_model_losses.append(world_losses.get('world_loss', 0))
            self.actor_losses.append(policy_losses.get('actor_loss', 0))
            self.critic_losses.append(policy_losses.get('critic_loss', 0))
            self.entropies.append(policy_losses.get('entropy', 0))
            
            episode_time = time.time() - episode_start
            
            # Logging
            if episode % 10 == 0 or episode < 5:
                print(f"Episode {episode:3d}: "
                      f"Reward={episode_reward:7.2f}, "
                      f"Length={episode_length:3d}, "
                      f"Pieces={pieces:2d}, "
                      f"Lines={lines:1d}, "
                      f"Buffer={len(self.replay_buffer):4d}, "
                      f"WLoss={world_losses.get('world_loss', 0):.4f}, "
                      f"ALoss={policy_losses.get('actor_loss', 0):.4f}, "
                      f"Time={episode_time:.2f}s")
        
        total_time = time.time() - start_time
        
        print("=" * 80)
        print(f"🎉 TRAINING COMPLETE!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Episodes: {self.num_episodes}")
        print(f"   Buffer size: {len(self.replay_buffer)}")
        print(f"   Avg time/episode: {total_time/self.num_episodes:.2f}s")
        
        # Generate comprehensive stats
        self.generate_stats_dashboard()
    
    def generate_stats_dashboard(self):
        """Generate comprehensive training statistics dashboard"""
        print(f"\n📊 GENERATING TRAINING DASHBOARD...")
        
        # Calculate statistics
        recent_rewards = self.episode_rewards[-20:] if len(self.episode_rewards) >= 20 else self.episode_rewards
        recent_pieces = self.pieces_placed[-20:] if len(self.pieces_placed) >= 20 else self.pieces_placed
        recent_lines = self.lines_cleared[-20:] if len(self.lines_cleared) >= 20 else self.lines_cleared
        
        stats = {
            'training_summary': {
                'total_episodes': int(len(self.episode_rewards)),
                'total_time': float(time.time()),
                'buffer_size': int(len(self.replay_buffer))
            },
            'performance_metrics': {
                'mean_reward': float(np.mean(self.episode_rewards)),
                'std_reward': float(np.std(self.episode_rewards)),
                'best_reward': float(np.max(self.episode_rewards)),
                'worst_reward': float(np.min(self.episode_rewards)),
                'recent_mean_reward': float(np.mean(recent_rewards)),
                'mean_episode_length': float(np.mean(self.episode_lengths)),
                'mean_pieces_placed': float(np.mean(self.pieces_placed)),
                'total_lines_cleared': int(np.sum(self.lines_cleared)),
                'recent_mean_pieces': float(np.mean(recent_pieces)),
                'recent_mean_lines': float(np.mean(recent_lines))
            },
            'learning_metrics': {
                'final_world_loss': float(self.world_model_losses[-1] if self.world_model_losses else 0),
                'final_actor_loss': float(self.actor_losses[-1] if self.actor_losses else 0),
                'final_critic_loss': float(self.critic_losses[-1] if self.critic_losses else 0),
                'final_entropy': float(self.entropies[-1] if self.entropies else 0),
                'mean_world_loss': float(np.mean(self.world_model_losses) if self.world_model_losses else 0),
                'mean_actor_loss': float(np.mean(self.actor_losses) if self.actor_losses else 0)
            }
        }
        
        # Save stats to JSON
        with open(self.log_dir / 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print dashboard
        print("\n" + "=" * 80)
        print("📊 DREAM TRAINING DASHBOARD")
        print("=" * 80)
        print(f"🎯 TRAINING SUMMARY:")
        print(f"   Episodes completed: {stats['training_summary']['total_episodes']}")
        print(f"   Buffer size: {stats['training_summary']['buffer_size']}")
        
        print(f"\n🏆 PERFORMANCE METRICS:")
        print(f"   Mean reward: {stats['performance_metrics']['mean_reward']:.2f} ± {stats['performance_metrics']['std_reward']:.2f}")
        print(f"   Best reward: {stats['performance_metrics']['best_reward']:.2f}")
        print(f"   Recent reward: {stats['performance_metrics']['recent_mean_reward']:.2f}")
        print(f"   Mean pieces placed: {stats['performance_metrics']['mean_pieces_placed']:.1f}")
        print(f"   Total lines cleared: {stats['performance_metrics']['total_lines_cleared']}")
        print(f"   Recent pieces: {stats['performance_metrics']['recent_mean_pieces']:.1f}")
        print(f"   Recent lines: {stats['performance_metrics']['recent_mean_lines']:.2f}")
        
        print(f"\n🧠 LEARNING METRICS:")
        print(f"   World model loss: {stats['learning_metrics']['final_world_loss']:.4f}")
        print(f"   Actor loss: {stats['learning_metrics']['final_actor_loss']:.4f}")
        print(f"   Critic loss: {stats['learning_metrics']['final_critic_loss']:.4f}")
        print(f"   Entropy: {stats['learning_metrics']['final_entropy']:.3f}")
        
        # Create plots if possible
        try:
            self.create_training_plots()
            print(f"\n📈 Training plots saved to: {self.log_dir}")
        except Exception as e:
            print(f"\n⚠️  Could not create plots: {e}")
        
        print("=" * 80)
        
        return stats
    
    def create_training_plots(self):
        """Create training visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('DREAM Training Dashboard', fontsize=16)
        
        # Reward progression
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Pieces placed
        axes[0, 1].plot(self.pieces_placed)
        axes[0, 1].set_title('Pieces Placed per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Pieces')
        axes[0, 1].grid(True)
        
        # Lines cleared
        axes[0, 2].plot(self.lines_cleared)
        axes[0, 2].set_title('Lines Cleared per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Lines')
        axes[0, 2].grid(True)
        
        # World model loss
        if self.world_model_losses:
            axes[1, 0].plot(self.world_model_losses)
            axes[1, 0].set_title('World Model Loss')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # Actor loss
        if self.actor_losses:
            axes[1, 1].plot(self.actor_losses)
            axes[1, 1].set_title('Actor Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        # Entropy
        if self.entropies:
            axes[1, 2].plot(self.entropies)
            axes[1, 2].set_title('Policy Entropy')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Entropy')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.env.close()
        except:
            pass

def main():
    """Run actual DREAM training session"""
    print("🚀 STARTING ACTUAL DREAM TRAINING SESSION")
    print("=" * 80)
    
    session = DREAMTrainingSession(num_episodes=50)  # Start with 50 episodes
    
    try:
        session.train()
        print("\n✅ Training session completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        session.generate_stats_dashboard()
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        session.cleanup()

if __name__ == "__main__":
    main() 