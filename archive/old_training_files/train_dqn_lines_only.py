#!/usr/bin/env python3
"""
🎯 DQN TRAINING WITH LINES-ONLY REWARDS

Trains a DQN agent using the sparse lines-only reward function.
Demonstrates how to handle sparse rewards effectively.
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import json

from envs.tetris_env import TetrisEnv
from agents.dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent

class LinesClearingDQNTrainer:
    """DQN trainer specialized for lines-only rewards"""
    
    def __init__(self, episodes=1000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.episodes = episodes
        
        # Create environment with lines-only rewards
        self.env = self._create_padded_env()
        
        # Create DQN agent with modified parameters for sparse rewards
        self.agent = RedesignedLockedStateDQNAgent(
            input_dim=212,  # Padded observation size
            num_actions=8,  # Direct action mode
            hidden_dim=800,
            device=self.device,
            learning_rate=0.0001,    # Lower LR for sparse rewards
            gamma=0.99,              # High discount for long-term rewards
            epsilon_start=0.9,       # High exploration
            epsilon_end=0.05,        # Maintain some exploration
            epsilon_decay=20000,     # Slower decay for sparse rewards
            buffer_size=100000,      # Large buffer for rare positive experiences
            batch_size=64,           # Larger batches for stability
            target_update=1000       # Frequent target updates
        )
        
        # Stats tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.lines_cleared_history = []
        self.exploration_rates = []
        self.losses = []
        
        # Line clearing tracking
        self.total_lines_cleared = 0
        self.episodes_with_lines = 0
        
        print(f"🎯 Lines-Only DQN Trainer Initialized:")
        print(f"   Device: {self.device}")
        print(f"   Episodes: {episodes}")
        print(f"   Reward mode: lines_only")
        print(f"   Agent params: {sum(p.numel() for p in self.agent.q_network.parameters()):,}")
    
    def _create_padded_env(self):
        """Create environment with padding wrapper"""
        class PaddedTetrisEnv:
            def __init__(self, base_env):
                self.base_env = base_env
                
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
        
        base_env = TetrisEnv(
            num_agents=1, 
            headless=True, 
            action_mode='direct',
            reward_mode='lines_only'  # Key: sparse rewards only for line clearing
        )
        return PaddedTetrisEnv(base_env)
    
    def train(self):
        """Main training loop with sparse reward handling"""
        print(f"\n🚀 STARTING LINES-ONLY DQN TRAINING")
        print("=" * 70)
        
        start_time = time.time()
        best_performance = 0
        
        for episode in range(self.episodes):
            episode_start = time.time()
            
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            lines_this_episode = 0
            
            for step in range(1000):  # Longer episodes for line clearing opportunities
                # Get action from agent
                action = self.agent.select_action(obs)
                
                # Take environment step
                next_obs, reward, done, info = self.env.step(action)
                
                # Track lines cleared
                if 'lines_cleared' in info and info['lines_cleared'] > 0:
                    lines_this_episode += info['lines_cleared']
                    self.total_lines_cleared += info['lines_cleared']
                
                # Store experience (important for sparse rewards)
                self.agent.store_experience(obs, action, reward, next_obs, done)
                
                # Update agent
                loss = self.agent.update()
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                    
                obs = next_obs
            
            # Track episode statistics
            if lines_this_episode > 0:
                self.episodes_with_lines += 1
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.lines_cleared_history.append(lines_this_episode)
            self.exploration_rates.append(self.agent.epsilon)
            if loss is not None:
                self.losses.append(loss)
            
            episode_time = time.time() - episode_start
            
            # Performance tracking
            recent_lines = sum(self.lines_cleared_history[-100:]) if len(self.lines_cleared_history) >= 100 else sum(self.lines_cleared_history)
            if recent_lines > best_performance:
                best_performance = recent_lines
            
            # Logging
            if episode % 50 == 0 or episode < 10:
                avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
                total_recent_lines = sum(self.lines_cleared_history[-50:]) if len(self.lines_cleared_history) >= 50 else sum(self.lines_cleared_history)
                
                print(f"Episode {episode:4d}: "
                      f"Reward={episode_reward:5.1f}, "
                      f"Length={episode_length:3d}, "
                      f"Lines={lines_this_episode:1d}, "
                      f"TotalLines={self.total_lines_cleared:3d}, "
                      f"ε={self.agent.epsilon:.3f}, "
                      f"Recent50Lines={total_recent_lines:2d}, "
                      f"Time={episode_time:.2f}s")
        
        total_time = time.time() - start_time
        
        print("=" * 70)
        print(f"🎉 LINES-ONLY DQN TRAINING COMPLETE!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Episodes: {self.episodes}")
        print(f"   Total lines cleared: {self.total_lines_cleared}")
        print(f"   Episodes with lines: {self.episodes_with_lines}/{self.episodes}")
        print(f"   Line clearing rate: {self.episodes_with_lines/self.episodes*100:.1f}%")
        
        # Generate analysis
        self.generate_analysis()
        
        return {
            'total_lines': self.total_lines_cleared,
            'episodes_with_lines': self.episodes_with_lines,
            'final_epsilon': self.agent.epsilon,
            'training_time': total_time
        }
    
    def generate_analysis(self):
        """Generate comprehensive analysis of sparse reward training"""
        print(f"\n📊 LINES-ONLY TRAINING ANALYSIS")
        print("=" * 70)
        
        # Learning progress analysis
        if len(self.episode_rewards) > 100:
            early_rewards = self.episode_rewards[:100]
            late_rewards = self.episode_rewards[-100:]
            early_lines = sum(self.lines_cleared_history[:100])
            late_lines = sum(self.lines_cleared_history[-100:])
            
            print(f"📈 LEARNING PROGRESS:")
            print(f"   Early episodes (0-99): {early_lines} lines cleared")
            print(f"   Late episodes ({self.episodes-100}-{self.episodes-1}): {late_lines} lines cleared")
            print(f"   Improvement: {late_lines - early_lines:+d} lines")
            print(f"   Line clearing frequency: {late_lines/100:.2f} lines per episode (recent)")
        
        # Exploration analysis
        if self.exploration_rates:
            print(f"\n🔍 EXPLORATION ANALYSIS:")
            print(f"   Initial epsilon: {self.exploration_rates[0]:.3f}")
            print(f"   Final epsilon: {self.exploration_rates[-1]:.3f}")
            print(f"   Epsilon decay: {'Appropriate' if self.exploration_rates[-1] > 0.01 else 'Too aggressive'}")
        
        # Sparse reward challenges
        zero_reward_episodes = sum(1 for r in self.episode_rewards if r == 0)
        print(f"\n⚠️  SPARSE REWARD CHALLENGES:")
        print(f"   Zero-reward episodes: {zero_reward_episodes}/{len(self.episode_rewards)} ({zero_reward_episodes/len(self.episode_rewards)*100:.1f}%)")
        print(f"   Line clearing episodes: {self.episodes_with_lines}/{len(self.episode_rewards)} ({self.episodes_with_lines/len(self.episode_rewards)*100:.1f}%)")
        
        # Success metrics
        if self.total_lines_cleared > 0:
            print(f"\n✅ SUCCESS METRICS:")
            print(f"   Total lines cleared: {self.total_lines_cleared}")
            print(f"   Average lines per successful episode: {self.total_lines_cleared/max(1, self.episodes_with_lines):.2f}")
            print(f"   Best 100-episode window: {max([sum(self.lines_cleared_history[i:i+100]) for i in range(len(self.lines_cleared_history)-99)]) if len(self.lines_cleared_history) >= 100 else sum(self.lines_cleared_history)}")
        
        # Training recommendations
        print(f"\n💡 TRAINING RECOMMENDATIONS:")
        if self.total_lines_cleared == 0:
            print(f"   🎯 Increase exploration (higher epsilon_end)")
            print(f"   🎯 Longer episodes (more line clearing opportunities)")
            print(f"   🎯 Reward shaping (small rewards for getting close to lines)")
        elif self.episodes_with_lines < self.episodes * 0.1:
            print(f"   🎯 Good start! Continue training for more consistency")
            print(f"   🎯 Consider curriculum learning (easier scenarios first)")
        else:
            print(f"   🎯 Excellent progress! Agent is learning to clear lines")
            print(f"   🎯 Fine-tune hyperparameters for optimization")
        
        print("=" * 70)
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.env.close()
        except:
            pass

def main():
    """Main training function"""
    print("🎯 DQN TRAINING WITH LINES-ONLY REWARDS")
    print("=" * 80)
    
    trainer = LinesClearingDQNTrainer(episodes=500)  # Start with 500 episodes
    
    try:
        results = trainer.train()
        
        print(f"\n🎉 TRAINING SESSION COMPLETED!")
        print(f"✅ Agent learned to clear {results['total_lines']} lines")
        print(f"✅ Line clearing in {results['episodes_with_lines']} episodes")
        print(f"✅ Final exploration rate: {results['final_epsilon']:.3f}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        trainer.generate_analysis()
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 