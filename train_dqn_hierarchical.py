#!/usr/bin/env python3
"""
HIERARCHICAL DQN TRAINING - COMPLETE WITH DUAL LOSS LOGGING
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from agents.dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent
from agents.dqn_movement_agent_redesigned import RedesignedMovementAgent
from envs.tetris_env import TetrisEnv

class HierarchicalDQNTrainer:
    """Hierarchical DQN trainer with dual agent training and tensorboard logging"""
    
    def __init__(self, reward_mode='standard', episodes=1000, use_tensorboard=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reward_mode = reward_mode
        self.episodes = episodes
        
        # Tensorboard logging setup
        if use_tensorboard:
            self.log_dir = f'logs/dqn_hierarchical_{reward_mode}'
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir + '/tensorboard')
            print(f"TensorBoard: tensorboard --logdir={self.log_dir}/tensorboard --port=6008")
        else:
            self.writer = None
        
        # Create environment for movement actions (hierarchical uses movement)
        self.env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='direct',
            reward_mode=reward_mode
        )
        
        # Initialize locked agent (first level of hierarchy)
        self.locked_agent = RedesignedLockedStateDQNAgent(
            input_dim=206,
            num_actions=800,
            device=str(self.device),
            learning_rate=0.0001,
            epsilon_start=0.9,
            epsilon_end=0.01,
            epsilon_decay=episodes * 10,
            reward_mode=reward_mode
        )
        
        # Initialize movement agent (second level of hierarchy)
        # Input: 1012 (200 board + 6 current + 6 next + 800 locked_q)
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
        self.locked_losses = []
        self.movement_losses = []
        
        # Calculate total parameters
        locked_params = self.locked_agent.get_parameter_count()
        movement_params = self.movement_agent.get_parameter_count()
        total_params = locked_params + movement_params
        
        print(f"Hierarchical DQN Trainer Initialized:")
        print(f"   Device: {self.device}")
        print(f"   Reward mode: {reward_mode}")
        print(f"   Episodes: {episodes}")
        print(f"   Locked agent: 206 → 800 ({locked_params:,} params)")
        print(f"   Movement agent: 1012 → 8 ({movement_params:,} params)")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Movement input: 200 board + 6 current + 6 next + 800 locked")
    
    def extract_state_components(self, obs):
        """Extract components from 206-dimensional observation"""
        if len(obs) == 206:
            board_flat = obs[:200]
            current_piece = obs[200:206]
            next_piece = np.zeros(6)  # placeholder
        else:
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
        
        # Get locked Q-values from locked agent
        locked_q_values = self.locked_agent.get_q_values(obs)
        
        # Combine: board(200) + current(6) + next(6) + locked_q(800) = 1012
        movement_state = np.concatenate([
            components['board'],
            components['current'],
            components['next'],
            locked_q_values
        ])
        
        return movement_state
    
    def log_metrics(self, episode, episode_reward, episode_length, episode_lines, 
                   locked_loss, movement_loss, locked_epsilon, movement_epsilon):
        """Log metrics to tensorboard with dual losses"""
        if self.writer:
            # Episode metrics
            self.writer.add_scalar('Episode/Reward', episode_reward, episode)
            self.writer.add_scalar('Episode/Length', episode_length, episode)
            self.writer.add_scalar('Episode/Lines_Cleared', episode_lines, episode)
            
            # Training losses
            self.writer.add_scalar('Training/Locked_Loss', locked_loss, episode)
            self.writer.add_scalar('Training/Movement_Loss', movement_loss, episode)
            
            # Exploration
            self.writer.add_scalar('Training/Locked_Epsilon', locked_epsilon, episode)
            self.writer.add_scalar('Training/Movement_Epsilon', movement_epsilon, episode)
            
            # Cumulative
            self.writer.add_scalar('Cumulative/Total_Lines', sum(self.lines_cleared), episode)
            
            self.writer.flush()
    
    def train(self):
        """Main hierarchical training loop with dual loss logging"""
        print(f"\nSTARTING HIERARCHICAL DQN TRAINING ({self.episodes} episodes)")
        print("=" * 70)
        
        start_time = time.time()
        total_lines = 0
        
        for episode in range(self.episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_lines = 0
            locked_loss = 0
            movement_loss = 0
            
            for step in range(500):
                # Level 1: Locked agent analyzes state
                locked_q_values = self.locked_agent.get_q_values(obs)
                
                # Level 2: Movement agent uses locked analysis + full state
                movement_state = self.create_movement_state(obs)
                action = self.movement_agent.select_action(movement_state, training=True)
                
                # Environment step
                next_obs, reward, done, info = self.env.step(action)
                
                if 'lines_cleared' in info:
                    episode_lines += info['lines_cleared']
                
                # Prepare next states
                next_locked_q_values = self.locked_agent.get_q_values(next_obs)
                next_movement_state = self.create_movement_state(next_obs)
                
                # Train both agents
                # Locked agent: learns to predict best locked positions
                # Use a dummy locked action for training (the agent still learns state evaluation)
                dummy_locked_action = np.random.randint(0, 800)
                locked_loss_dict = self.locked_agent.update(obs, dummy_locked_action, reward, next_obs, done)
                
                # Movement agent: learns to convert locked analysis to actions
                movement_loss_dict = self.movement_agent.update(movement_state, action, reward, next_movement_state, done)
                
                if locked_loss_dict and 'loss' in locked_loss_dict:
                    locked_loss = locked_loss_dict['loss']
                
                if movement_loss_dict and 'loss' in movement_loss_dict:
                    movement_loss = movement_loss_dict['loss']
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                    
                obs = next_obs
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.lines_cleared.append(episode_lines)
            if locked_loss > 0:
                self.locked_losses.append(locked_loss)
            if movement_loss > 0:
                self.movement_losses.append(movement_loss)
            
            total_lines += episode_lines
            
            # Log to tensorboard with dual losses
            self.log_metrics(episode, episode_reward, episode_length, episode_lines,
                           locked_loss, movement_loss, 
                           self.locked_agent.epsilon, self.movement_agent.epsilon)
            
            # Console logging
            if episode % 50 == 0 or episode < 5:
                print(f"Episode {episode:4d}: "
                      f"Reward={episode_reward:7.2f}, "
                      f"Length={episode_length:3d}, "
                      f"Lines={episode_lines:1d}")
                print(f"   Locked Loss={locked_loss:.4f}, ε={self.locked_agent.epsilon:.3f}")
                print(f"   Move Loss={movement_loss:.4f}, ε={self.movement_agent.epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        print("=" * 70)
        print(f"HIERARCHICAL DQN TRAINING COMPLETE!")
        print(f"   Total time: {training_time:.1f}s")
        print(f"   Total lines cleared: {total_lines}")
        print(f"   Mean reward: {np.mean(self.episode_rewards):.2f}")
        print(f"   Locked agent final ε: {self.locked_agent.epsilon:.3f}")
        print(f"   Movement agent final ε: {self.movement_agent.epsilon:.3f}")
        
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
    parser = argparse.ArgumentParser(description='Hierarchical DQN Training')
    parser.add_argument('--reward_mode', choices=['standard', 'lines_only'], 
                       default='standard', help='Reward mode')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    args = parser.parse_args()
    
    trainer = HierarchicalDQNTrainer(reward_mode=args.reward_mode, episodes=args.episodes)
    
    try:
        results = trainer.train()
        print(f"\nTraining completed! Lines cleared: {results['total_lines']}")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted")
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 