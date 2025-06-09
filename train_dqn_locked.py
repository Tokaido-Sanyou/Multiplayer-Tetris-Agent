#!/usr/bin/env python3
"""
🤖 FINAL DQN LOCKED STATE TRAINING

Definitive DQN training for locked state actions with both reward modes.
Uses 212-dimensional input and 800 action outputs.
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path

from agents.dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent
from envs.tetris_env import TetrisEnv

class DQNLockedTrainer:
    """DQN trainer for locked state actions"""
    
    def __init__(self, reward_mode='standard', episodes=1000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reward_mode = reward_mode
        self.episodes = episodes
        
        # Create environment
        self.env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='locked_position',
            reward_mode=reward_mode
        )
        
        # Initialize agent with reward mode configuration
        if reward_mode == 'lines_only':
            # Optimized for sparse rewards
            self.agent = RedesignedLockedStateDQNAgent(
                input_dim=206,  # Environment provides 206 dimensions
                num_actions=800,
                device=str(self.device),
                learning_rate=0.0001,
                epsilon_start=0.95,      # High exploration for sparse rewards
                epsilon_end=0.05,        # Maintain exploration
                epsilon_decay=episodes * 20,  # Slower decay
                buffer_size=200000,      # Large buffer for rare positive experiences
                reward_mode=reward_mode
            )
        else:
            # Standard configuration
            self.agent = RedesignedLockedStateDQNAgent(
                input_dim=206,
                num_actions=800,
                device=str(self.device),
                learning_rate=0.0001,
                epsilon_start=0.9,
                epsilon_end=0.01,
                epsilon_decay=episodes * 10,
                reward_mode=reward_mode
            )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.lines_cleared = []
        self.losses = []
        
        print(f"🤖 DQN Locked Trainer Initialized:")
        print(f"   Device: {self.device}")
        print(f"   Reward mode: {reward_mode}")
        print(f"   Episodes: {episodes}")
        print(f"   Agent parameters: {self.agent.get_parameter_count():,}")
        print(f"   Action space: 800 (10×20×4 locked positions)")
    
    def train(self):
        """Main training loop"""
        print(f"\n🤖 STARTING DQN LOCKED TRAINING ({self.episodes} episodes)")
        print("=" * 70)
        
        start_time = time.time()
        total_lines = 0
        
        for episode in range(self.episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_lines = 0
            
            for step in range(500):  # Max steps per episode
                # Select action
                action = self.agent.select_action(obs, training=True, env=self.env)
                
                # Environment step
                next_obs, reward, done, info = self.env.step(action)
                
                # Track episode stats
                if 'lines_cleared' in info:
                    episode_lines += info['lines_cleared']
                
                # Store experience and update
                self.agent.store_experience(obs, action, reward, next_obs, done)
                loss_dict = self.agent.update(obs, action, reward, next_obs, done)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                    
                obs = next_obs
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.lines_cleared.append(episode_lines)
            if loss_dict and 'loss' in loss_dict:
                self.losses.append(loss_dict['loss'])
            
            total_lines += episode_lines
            
            # Logging
            if episode % 50 == 0 or episode < 5:
                avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
                recent_lines = sum(self.lines_cleared[-50:]) if len(self.lines_cleared) >= 50 else sum(self.lines_cleared)
                
                print(f"Episode {episode:4d}: "
                      f"Reward={episode_reward:7.2f}, "
                      f"Length={episode_length:3d}, "
                      f"Lines={episode_lines:1d}, "
                      f"TotalLines={total_lines:3d}, "
                      f"ε={self.agent.epsilon:.3f}, "
                      f"Recent50Lines={recent_lines:2d}")
        
        training_time = time.time() - start_time
        
        print("=" * 70)
        print(f"🎉 DQN LOCKED TRAINING COMPLETE!")
        print(f"   Total time: {training_time:.1f}s")
        print(f"   Episodes: {self.episodes}")
        print(f"   Total lines cleared: {total_lines}")
        print(f"   Mean reward: {np.mean(self.episode_rewards):.2f}")
        print(f"   Final epsilon: {self.agent.epsilon:.3f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'lines_cleared': self.lines_cleared,
            'training_time': training_time,
            'total_lines': total_lines
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.env.close()
        except:
            pass

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='DQN Locked State Training')
    parser.add_argument('--reward_mode', choices=['standard', 'lines_only'], 
                       default='standard', help='Reward mode')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    args = parser.parse_args()
    
    print("🤖 DQN LOCKED STATE TRAINING")
    print("=" * 80)
    
    trainer = DQNLockedTrainer(reward_mode=args.reward_mode, episodes=args.episodes)
    
    try:
        results = trainer.train()
        print(f"\n✅ Training completed successfully!")
        print(f"   Total lines cleared: {results['total_lines']}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 