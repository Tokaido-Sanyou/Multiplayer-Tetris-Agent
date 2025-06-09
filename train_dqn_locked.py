#!/usr/bin/env python3
"""
DQN LOCKED STATE TRAINING - COMPLETE WITH TENSORBOARD LOGGING
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from agents.dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent
from envs.tetris_env import TetrisEnv

class DQNLockedTrainer:
    """DQN trainer for locked state actions with tensorboard logging"""
    
    def __init__(self, reward_mode='standard', episodes=1000, use_tensorboard=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reward_mode = reward_mode
        self.episodes = episodes
        
        # Tensorboard logging setup
        if use_tensorboard:
            self.log_dir = f'logs/dqn_locked_{reward_mode}'
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir + '/tensorboard')
            print(f"TensorBoard: tensorboard --logdir={self.log_dir}/tensorboard --port=6007")
        else:
            self.writer = None
        
        # Create environment
        self.env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='locked_position',
            reward_mode=reward_mode
        )
        
        # Initialize agent
        if reward_mode == 'lines_only':
            self.agent = RedesignedLockedStateDQNAgent(
                input_dim=206,
                num_actions=800,
                device=str(self.device),
                learning_rate=0.0001,
                epsilon_start=0.95,
                epsilon_end=0.05,
                epsilon_decay=episodes * 20,
                buffer_size=200000,
                reward_mode=reward_mode
            )
        else:
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
        
        print(f"DQN Locked Trainer Initialized:")
        print(f"   Device: {self.device}")
        print(f"   Reward mode: {reward_mode}")
        print(f"   Episodes: {episodes}")
        print(f"   Agent parameters: {self.agent.get_parameter_count():,}")
        print(f"   Action space: 800 (10×20×4 locked positions)")
    
    def log_metrics(self, episode, episode_reward, episode_length, episode_lines, loss, epsilon):
        """Log metrics to tensorboard"""
        if self.writer:
            self.writer.add_scalar('Episode/Reward', episode_reward, episode)
            self.writer.add_scalar('Episode/Length', episode_length, episode)
            self.writer.add_scalar('Episode/Lines_Cleared', episode_lines, episode)
            self.writer.add_scalar('Training/Loss', loss, episode)
            self.writer.add_scalar('Training/Epsilon', epsilon, episode)
            self.writer.add_scalar('Cumulative/Total_Lines', sum(self.lines_cleared), episode)
            self.writer.flush()
    
    def train(self):
        """Main training loop with tensorboard logging"""
        print(f"\nSTARTING DQN LOCKED TRAINING ({self.episodes} episodes)")
        print("=" * 70)
        
        start_time = time.time()
        total_lines = 0
        
        for episode in range(self.episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_lines = 0
            episode_loss = 0
            
            for step in range(500):
                action = self.agent.select_action(obs, training=True, env=self.env)
                next_obs, reward, done, info = self.env.step(action)
                
                if 'lines_cleared' in info:
                    episode_lines += info['lines_cleared']
                
                self.agent.store_experience(obs, action, reward, next_obs, done)
                loss_dict = self.agent.update(obs, action, reward, next_obs, done)
                
                if loss_dict and 'loss' in loss_dict:
                    episode_loss = loss_dict['loss']
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                    
                obs = next_obs
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.lines_cleared.append(episode_lines)
            if episode_loss > 0:
                self.losses.append(episode_loss)
            
            total_lines += episode_lines
            
            # Log to tensorboard
            self.log_metrics(episode, episode_reward, episode_length, episode_lines, 
                           episode_loss, self.agent.epsilon)
            
            # Console logging
            if episode % 50 == 0 or episode < 5:
                avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
                recent_lines = sum(self.lines_cleared[-50:]) if len(self.lines_cleared) >= 50 else sum(self.lines_cleared)
                
                print(f"Episode {episode:4d}: "
                      f"Reward={episode_reward:7.2f}, "
                      f"Length={episode_length:3d}, "
                      f"Lines={episode_lines:1d}, "
                      f"Loss={episode_loss:.4f}, "
                      f"ε={self.agent.epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        print("=" * 70)
        print(f"DQN LOCKED TRAINING COMPLETE!")
        print(f"   Total time: {training_time:.1f}s")
        print(f"   Total lines cleared: {total_lines}")
        print(f"   Mean reward: {np.mean(self.episode_rewards):.2f}")
        
        return {'total_lines': total_lines, 'training_time': training_time}
    
    def cleanup(self):
        """Clean up resources"""
        if self.writer:
            self.writer.close()
        try:
            self.env.close()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description='DQN Locked State Training')
    parser.add_argument('--reward_mode', choices=['standard', 'lines_only'], 
                       default='standard', help='Reward mode')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    args = parser.parse_args()
    
    trainer = DQNLockedTrainer(reward_mode=args.reward_mode, episodes=args.episodes)
    
    try:
        results = trainer.train()
        print(f"\nTraining completed! Lines cleared: {results['total_lines']}")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted")
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 