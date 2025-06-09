#!/usr/bin/env python3
"""
Fixed Hierarchical DQN Training Pipeline for Redesigned System
Works with 800-action space and CNN architecture
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import argparse
from typing import Dict, Any, Tuple, Optional, List
from collections import deque

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent
from envs.tetris_env import TetrisEnv


class FixedHierarchicalTrainer:
    """
    Fixed Hierarchical Training Pipeline for Redesigned DQN System
    
    Features:
    - 800 action space (10×20×4)
    - CNN architecture with 206-dimensional observations
    - Progressive penalty system for invalid actions
    - No max_steps termination
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 batch_size: int = 32,
                 debug_mode: bool = True):
        """Initialize fixed trainer"""
        self.device = device
        self.batch_size = batch_size
        self.debug_mode = debug_mode
        
        # Initialize environment with correct action mode
        self.env = TetrisEnv(num_agents=1, headless=True, action_mode='locked_position')
        
        # Initialize redesigned agent
        self.agent = RedesignedLockedStateDQNAgent(
            device=device,
            learning_rate=0.001,
            epsilon_decay_steps=20000,
            batch_size=batch_size,
            memory_size=50000
        )
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'pieces_placed': [],
            'lines_cleared': [],
            'losses': [],
            'invalid_actions': []
        }
        
        print(f"Fixed Hierarchical Trainer initialized:")
        print(f"   Device: {self.device}")
        print(f"   Environment: {self.env.action_space} action space")
        print(f"   Agent: {self.agent.get_parameter_count():,} parameters")
        print(f"   Batch Size: {self.batch_size}")
    
    def train_episodes(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """Train the redesigned agent for specified episodes"""
        print(f"\n=== TRAINING REDESIGNED DQN AGENT ===")
        print(f"Episodes: {num_episodes}")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_reward, episode_length, pieces_placed, lines_cleared, invalid_count = self._run_episode(episode)
            
            # Store metrics
            self.training_metrics['episode_rewards'].append(episode_reward)
            self.training_metrics['episode_lengths'].append(episode_length)
            self.training_metrics['pieces_placed'].append(pieces_placed)
            self.training_metrics['lines_cleared'].append(lines_cleared)
            self.training_metrics['invalid_actions'].append(invalid_count)
            
            # Logging
            if episode % 100 == 0 or episode < 10:
                avg_reward = np.mean(self.training_metrics['episode_rewards'][-100:])
                avg_pieces = np.mean(self.training_metrics['pieces_placed'][-100:])
                print(f"Episode {episode:4d}: Reward={episode_reward:6.1f}, "
                      f"Pieces={pieces_placed:2d}, Lines={lines_cleared:2d}, "
                      f"Avg100={avg_reward:6.1f}, Epsilon={self.agent.epsilon:.3f}")
        
        total_time = time.time() - start_time
        
        # Save checkpoint
        try:
            os.makedirs("checkpoints", exist_ok=True)
            self.agent.save_checkpoint("checkpoints/redesigned_agent_checkpoint.pt")
            print(f"✅ Checkpoint saved")
        except Exception as e:
            print(f"⚠️ Could not save checkpoint: {e}")
        
        # Final summary
        final_metrics = self._generate_summary()
        final_metrics['total_time'] = total_time
        
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Average Reward: {final_metrics['avg_reward']:.1f}")
        print(f"Total Pieces Placed: {final_metrics['total_pieces']}")
        print(f"Total Lines Cleared: {final_metrics['total_lines']}")
        print(f"Success Rate: {final_metrics['success_rate']:.1%}")
        
        return final_metrics
    
    def _run_episode(self, episode_num: int) -> Tuple[float, int, int, int, int]:
        """Run a single training episode"""
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        pieces_placed = 0
        lines_cleared = 0
        done = False
        
        # Track invalid actions at start of episode
        initial_invalid_count = self.agent.invalid_action_count
        
        while not done:
            # Agent selects action
            action = self.agent.select_action(obs, training=True, env=self.env)
            
            # Environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # Track metrics
            episode_reward += reward
            episode_length += 1
            
            if info.get('piece_placed', False):
                pieces_placed += 1
            
            lines_this_step = info.get('lines_cleared', 0)
            if lines_this_step > 0:
                lines_cleared += lines_this_step
            
            # Agent update (store experience and train)
            update_metrics = self.agent.update(obs, action, reward, next_obs, done)
            
            # Store loss if training occurred
            if update_metrics['loss'] > 0:
                self.training_metrics['losses'].append(update_metrics['loss'])
            
            obs = next_obs
        
        # Calculate invalid actions for this episode
        invalid_count = self.agent.invalid_action_count - initial_invalid_count
        
        return episode_reward, episode_length, pieces_placed, lines_cleared, invalid_count
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate training summary"""
        rewards = self.training_metrics['episode_rewards']
        pieces = self.training_metrics['pieces_placed']
        lines = self.training_metrics['lines_cleared']
        
        return {
            'total_episodes': len(rewards),
            'avg_reward': np.mean(rewards) if rewards else 0.0,
            'total_pieces': sum(pieces),
            'total_lines': sum(lines),
            'success_rate': sum(1 for r in rewards if r > -50) / len(rewards) if rewards else 0.0,
            'avg_episode_length': np.mean(self.training_metrics['episode_lengths']) if self.training_metrics['episode_lengths'] else 0.0,
            'total_invalid_actions': sum(self.training_metrics['invalid_actions']),
            'final_epsilon': self.agent.epsilon
        }
    
    def evaluate_agent(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        print(f"\n=== EVALUATING AGENT ===")
        
        eval_rewards = []
        eval_pieces = []
        eval_lines = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            pieces_placed = 0
            lines_cleared = 0
            done = False
            
            while not done:
                # Agent selects action (no training)
                action = self.agent.select_action(obs, training=False, env=self.env)
                
                # Environment step
                next_obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                if info.get('piece_placed', False):
                    pieces_placed += 1
                if info.get('lines_cleared', 0) > 0:
                    lines_cleared += info['lines_cleared']
                
                obs = next_obs
            
            eval_rewards.append(episode_reward)
            eval_pieces.append(pieces_placed)
            eval_lines.append(lines_cleared)
            
            print(f"Eval Episode {episode+1}: Reward={episode_reward:.1f}, Pieces={pieces_placed}, Lines={lines_cleared}")
        
        eval_summary = {
            'avg_reward': np.mean(eval_rewards),
            'avg_pieces': np.mean(eval_pieces),
            'avg_lines': np.mean(eval_lines),
            'total_episodes': num_episodes
        }
        
        print(f"Evaluation Summary: Avg Reward={eval_summary['avg_reward']:.1f}, "
              f"Avg Pieces={eval_summary['avg_pieces']:.1f}, "
              f"Avg Lines={eval_summary['avg_lines']:.1f}")
        
        return eval_summary


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Fixed Hierarchical DQN Training')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    print("Fixed Hierarchical DQN Training")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Initialize trainer
    trainer = FixedHierarchicalTrainer(
        device=args.device,
        batch_size=args.batch_size,
        debug_mode=True
    )
    
    # Train agent
    training_results = trainer.train_episodes(args.episodes)
    
    # Evaluate agent
    eval_results = trainer.evaluate_agent(args.eval_episodes)
    
    print("\n=== FINAL RESULTS ===")
    print(f"Training Success: {'✅' if training_results['success_rate'] > 0.1 else '❌'}")
    print(f"Pieces Placed: {training_results['total_pieces']}")
    print(f"Lines Cleared: {training_results['total_lines']}")
    print(f"Final Performance: {eval_results['avg_reward']:.1f} avg reward")


if __name__ == "__main__":
    main() 