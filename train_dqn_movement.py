#!/usr/bin/env python3
"""
DQN MOVEMENT AGENT TRAINING - COMPLETE WITH PROPER INPUT DIMENSIONS
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from agents.dqn_movement_agent_redesigned import RedesignedMovementAgent
from agents.dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent
from envs.tetris_env import TetrisEnv

class DQNMovementTrainer:
    """DQN trainer for hierarchical movement agent with tensorboard logging"""
    
    def __init__(self, reward_mode='standard', episodes=1000, use_tensorboard=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reward_mode = reward_mode
        self.episodes = episodes
        
        # Tensorboard logging setup
        if use_tensorboard:
            self.log_dir = f'logs/dqn_movement_{reward_mode}'
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir + '/tensorboard')
            print(f"TensorBoard: tensorboard --logdir={self.log_dir}/tensorboard --port=6009")
        else:
            self.writer = None
        
        # Create environment
        self.env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='direct',
            reward_mode=reward_mode
        )
        
        # Initialize locked agent (for Q-values)
        self.locked_agent = RedesignedLockedStateDQNAgent(
            input_dim=206,
            num_actions=800,
            device=str(self.device),
            learning_rate=0.0001
        )
        
        # Initialize movement agent with proper input dimensions
        # Input: 206 (env state) + 800 (locked Q-values) = 1006
        # But we want board + current + next + locked = 200 + 6 + 6 + 800 = 1012
        self.movement_agent = RedesignedMovementAgent(
            input_dim=1012,  # FIXED: proper dimensions
            num_actions=8,
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
        
        print(f"DQN Movement Trainer Initialized:")
        print(f"   Device: {self.device}")
        print(f"   Reward mode: {reward_mode}")
        print(f"   Episodes: {episodes}")
        print(f"   Locked agent parameters: {self.locked_agent.get_parameter_count():,}")
        print(f"   Movement agent parameters: {self.movement_agent.get_parameter_count():,}")
        print(f"   Movement input dimensions: 1012 (200 board + 6 current + 6 next + 800 locked)")
    
    def extract_state_components(self, obs):
        """Extract components from 206-dimensional observation"""
        # obs is 206-dimensional: [board_flat(200), current_piece(6)]
        # For some environments, it might be different, so let's be robust
        if len(obs) == 206:
            board_flat = obs[:200]  # 20x10 board flattened
            current_piece = obs[200:206]  # current piece info (6 dims)
            next_piece = np.zeros(6)  # placeholder for next piece (not in obs)
        else:
            # Fallback - pad or truncate as needed
            board_flat = obs[:200] if len(obs) >= 200 else np.pad(obs, (0, max(0, 200-len(obs))))
            current_piece = obs[200:206] if len(obs) >= 206 else np.zeros(6)
            next_piece = np.zeros(6)
        
        return {
            'board': board_flat,
            'current': current_piece,
            'next': next_piece
        }
    
    def create_movement_state(self, obs):
        """Create proper input state for movement agent"""
        # Extract components
        components = self.extract_state_components(obs)
        
        # Get locked Q-values
        locked_q_values = self.locked_agent.get_q_values(obs)
        
        # Combine all components: board(200) + current(6) + next(6) + locked_q(800) = 1012
        movement_state = np.concatenate([
            components['board'],     # 200 dims
            components['current'],   # 6 dims  
            components['next'],      # 6 dims
            locked_q_values         # 800 dims
        ])
        
        return movement_state
    
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
        print(f"\nSTARTING DQN MOVEMENT TRAINING ({self.episodes} episodes)")
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
                # Create proper movement state
                movement_state = self.create_movement_state(obs)
                
                # Select movement action
                action = self.movement_agent.select_action(movement_state, training=True)
                next_obs, reward, done, info = self.env.step(action)
                
                if 'lines_cleared' in info:
                    episode_lines += info['lines_cleared']
                
                # Create next movement state
                next_movement_state = self.create_movement_state(next_obs)
                
                # Store experience and update
                self.movement_agent.store_experience(movement_state, action, reward, next_movement_state, done)
                loss_dict = self.movement_agent.update(movement_state, action, reward, next_movement_state, done)
                
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
                           episode_loss, self.movement_agent.epsilon)
            
            # Console logging
            if episode % 50 == 0 or episode < 5:
                print(f"Episode {episode:4d}: "
                      f"Reward={episode_reward:7.2f}, "
                      f"Length={episode_length:3d}, "
                      f"Lines={episode_lines:1d}, "
                      f"Loss={episode_loss:.4f}, "
                      f"ε={self.movement_agent.epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        print("=" * 70)
        print(f"DQN MOVEMENT TRAINING COMPLETE!")
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
    parser = argparse.ArgumentParser(description='DQN Movement Agent Training')
    parser.add_argument('--reward_mode', choices=['standard', 'lines_only'], 
                       default='standard', help='Reward mode')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    args = parser.parse_args()
    
    trainer = DQNMovementTrainer(reward_mode=args.reward_mode, episodes=args.episodes)
    
    try:
        results = trainer.train()
        print(f"\nTraining completed! Lines cleared: {results['total_lines']}")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted")
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 