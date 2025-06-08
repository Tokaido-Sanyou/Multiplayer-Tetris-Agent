#!/usr/bin/env python3
"""
Training script for Actor-Locked Hierarchical System

Features:
- Hierarchical training: Locked model + Actor model
- Hindsight Experience Replay (HER)
- Configurable actor trials
- Visual demonstrations
- Checkpoint resuming
"""

import torch
import numpy as np
import time
import os
import argparse
import json
import glob
from typing import Dict, Any, Optional

from envs.tetris_env import TetrisEnv
from agents.actor_locked_system import ActorLockedSystem

class ActorLockedTrainer:
    """Trainer for Actor-Locked hierarchical system"""
    
    def __init__(self,
                 device: str = 'cuda',
                 actor_trials: int = 10,
                 locked_model_path: Optional[str] = None,
                 actor_learning_rate: float = 0.0001):
        
        self.device = device
        self.actor_trials = actor_trials
        
        # Training state for resuming
        self.start_episode = 0
        self.total_training_time = 0.0
        self.training_history = {
            'episode_rewards': [],
            'episode_pieces': [],
            'episode_lines': [],
            'locked_losses': [],
            'actor_losses': [],
            'actor_success_rates': []
        }
        
        # Initialize environment and agent
        self.env = TetrisEnv(action_mode='locked_position', headless=True)
        self.agent = ActorLockedSystem(
            device=device,
            max_movement_steps=actor_trials,
            locked_model_path=locked_model_path,
            actor_learning_rate=actor_learning_rate
        )
        
        print(f"Actor-Locked Trainer initialized:")
        print(f"   Device: {self.device}")
        print(f"   Actor trials: {self.actor_trials}")
        print(f"   Environment action space: {self.env.action_space}")
        print(f"   Environment observation space: {self.env.observation_space}")
    
    def train(self, 
              num_episodes: int = 1000,
              save_interval: int = 100,
              resume: bool = True,
              show_visualization: bool = False,
              visualization_interval: int = 100) -> Dict[str, Any]:
        """Train the Actor-Locked system"""
        
        # Try to resume from checkpoint if requested
        if resume:
            resumed = self.resume_from_checkpoint()
            if resumed:
                remaining_episodes = max(0, num_episodes - self.start_episode)
                if remaining_episodes == 0:
                    print(f"Training already completed {num_episodes} episodes!")
                    return self.training_history
                print(f"Continuing training for {remaining_episodes} more episodes")
                num_episodes = remaining_episodes
        
        print(f"\n=== TRAINING ACTOR-LOCKED SYSTEM ===")
        print(f"Episodes: {num_episodes} (starting from episode {self.start_episode})")
        
        episode_rewards = self.training_history['episode_rewards'].copy()
        episode_pieces = self.training_history['episode_pieces'].copy()
        episode_lines = self.training_history['episode_lines'].copy()
        locked_losses = self.training_history['locked_losses'].copy()
        actor_losses = self.training_history['actor_losses'].copy()
        actor_success_rates = self.training_history['actor_success_rates'].copy()
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            actual_episode = self.start_episode + episode
            
            # Run episode
            episode_result = self._run_episode(actual_episode, show_visualization and (episode % visualization_interval == 0))
            
            # Store results
            episode_rewards.append(episode_result['reward'])
            episode_pieces.append(episode_result['pieces'])
            episode_lines.append(episode_result['lines'])
            locked_losses.append(episode_result['locked_loss'])
            actor_losses.append(episode_result['actor_loss'])
            actor_success_rates.append(episode_result['actor_success_rate'])
            
            # Logging
            if episode % 10 == 0 or episode < 10:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_pieces = np.mean(episode_pieces[-10:])
                avg_lines = np.mean(episode_lines[-10:])
                avg_locked_loss = np.mean(locked_losses[-10:])
                avg_actor_loss = np.mean(actor_losses[-10:])
                avg_success_rate = np.mean(actor_success_rates[-10:])
                
                print(f"Episode {actual_episode:4d}: "
                      f"Reward={episode_result['reward']:6.1f} (avg={avg_reward:6.1f}), "
                      f"Pieces={episode_result['pieces']:2d} (avg={avg_pieces:4.1f}), "
                      f"Lines={episode_result['lines']:2d} (avg={avg_lines:4.1f}), "
                      f"LockedLoss={avg_locked_loss:.4f}, "
                      f"ActorLoss={avg_actor_loss:.4f}, "
                      f"ActorSuccess={avg_success_rate:.3f}")
            
            # Save checkpoint
            if episode > 0 and episode % save_interval == 0:
                self.save_checkpoint(f"checkpoints/actor_locked_episode_{actual_episode}.pt", 
                                   episode_rewards, episode_pieces, episode_lines, 
                                   locked_losses, actor_losses, actor_success_rates)
        
        total_time = time.time() - start_time
        self.total_training_time += total_time
        
        # Final statistics
        final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        final_avg_pieces = np.mean(episode_pieces[-100:]) if len(episode_pieces) >= 100 else np.mean(episode_pieces)
        final_avg_lines = np.mean(episode_lines[-100:]) if len(episode_lines) >= 100 else np.mean(episode_lines)
        final_avg_success = np.mean(actor_success_rates[-100:]) if len(actor_success_rates) >= 100 else np.mean(actor_success_rates)
        
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Total time: {total_time:.1f}s")
        print(f"Final average reward (last 100): {final_avg_reward:.1f}")
        print(f"Final average pieces (last 100): {final_avg_pieces:.1f}")
        print(f"Final average lines (last 100): {final_avg_lines:.1f}")
        print(f"Final actor success rate: {final_avg_success:.3f}")
        print(f"Total episodes: {num_episodes}")
        
        # Save final checkpoint
        self.save_checkpoint("checkpoints/actor_locked_final.pt",
                           episode_rewards, episode_pieces, episode_lines,
                           locked_losses, actor_losses, actor_success_rates)
        
        return {
            'episode_rewards': episode_rewards,
            'episode_pieces': episode_pieces,
            'episode_lines': episode_lines,
            'locked_losses': locked_losses,
            'actor_losses': actor_losses,
            'actor_success_rates': actor_success_rates,
            'final_avg_reward': final_avg_reward,
            'final_avg_pieces': final_avg_pieces,
            'final_avg_lines': final_avg_lines,
            'final_actor_success': final_avg_success,
            'total_time': total_time
        }
    
    def _run_episode(self, episode_num: int, show_visualization: bool = False) -> Dict[str, Any]:
        """Run a single episode"""
        observation = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        pieces_placed = 0
        lines_cleared = 0
        locked_losses = []
        actor_losses = []
        actor_success_rates = []
        
        if show_visualization:
            print(f"\n--- EPISODE {episode_num} VISUALIZATION ---")
            self._visualize_state("INITIAL", observation, None, None)
        
        while not done and episode_length < 200:  # Max 200 steps per episode
            # Select action using Actor-Locked system
            action = self.agent.select_action(observation, training=True, env=self.env)
            
            # Execute action
            next_observation, reward, done, info = self.env.step(action)
            
            if show_visualization:
                locked_action = self.agent.locked_model.select_action(observation, training=False, env=self.env)
                self._visualize_action_comparison(episode_length, observation, locked_action, action, reward, info)
            
            # Track statistics
            if info.get('piece_placed', False):
                pieces_placed += 1
            
            lines_this_step = info.get('lines_cleared', 0)
            if lines_this_step > 0:
                lines_cleared += lines_this_step
            
            # Update agent (both locked and actor models)
            train_result = self.agent.update(observation, action, reward, next_observation, done)
            
            if 'loss' in train_result:
                locked_losses.append(train_result['loss'])
            if 'actor_loss' in train_result:
                actor_losses.append(train_result['actor_loss'])
            if 'actor_success_rate' in train_result:
                actor_success_rates.append(train_result['actor_success_rate'])
            
            observation = next_observation
            episode_reward += reward
            episode_length += 1
        
        if show_visualization:
            print(f"--- EPISODE {episode_num} COMPLETE ---")
            print(f"Total reward: {episode_reward:.1f}, Pieces: {pieces_placed}, Lines: {lines_cleared}")
        
        return {
            'reward': episode_reward,
            'pieces': pieces_placed,
            'lines': lines_cleared,
            'locked_loss': np.mean(locked_losses) if locked_losses else 0.0,
            'actor_loss': np.mean(actor_losses) if actor_losses else 0.0,
            'actor_success_rate': np.mean(actor_success_rates) if actor_success_rates else 0.0
        }
    
    def _visualize_state(self, label: str, observation: np.ndarray, action: Optional[int], info: Optional[Dict]):
        """Visualize current state (text-based)"""
        board = observation[:200].reshape(20, 10)
        piece_info = observation[200:206]
        
        print(f"{label} STATE:")
        print("Board (top 5 rows):")
        for row in board[:5]:
            print("".join("█" if cell > 0 else "·" for cell in row))
        print(f"Piece info: {piece_info}")
        if action is not None:
            coords = self.agent.locked_model.map_action_to_board(action)
            print(f"Action: {action} -> coords {coords}")
        if info:
            print(f"Info: {info}")
        print()
    
    def _visualize_action_comparison(self, step: int, observation: np.ndarray, locked_action: int, actor_action: int, reward: float, info: Dict):
        """Compare locked vs actor action choices"""
        locked_coords = self.agent.locked_model.map_action_to_board(locked_action)
        actor_coords = self.agent.locked_model.map_action_to_board(actor_action)
        
        print(f"Step {step}: Locked={locked_action}{locked_coords} vs Actor={actor_action}{actor_coords}, Reward={reward:.1f}")
        if locked_action != actor_action:
            print("  → Actor chose different action!")
    
    def resume_from_checkpoint(self, checkpoint_pattern: str = "checkpoints/actor_locked_episode_*.pt") -> bool:
        """Resume training from the latest checkpoint"""
        try:
            # Find latest checkpoint
            checkpoint_files = glob.glob(checkpoint_pattern)
            if not checkpoint_files:
                print("No Actor-Locked checkpoints found, starting fresh training")
                return False
            
            # Get latest checkpoint by episode number
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # Load agent state
            self.agent.load_checkpoint(latest_checkpoint)
            
            # Extract episode number from filename
            episode_num = int(latest_checkpoint.split('_')[-1].split('.')[0])
            self.start_episode = episode_num
            
            # Load training history if exists
            history_file = latest_checkpoint.replace('.pt', '_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    saved_history = json.load(f)
                    self.training_history.update(saved_history)
                    self.total_training_time = saved_history.get('total_training_time', 0.0)
            
            print(f"✅ Resumed Actor-Locked training from: {latest_checkpoint}")
            print(f"   Starting from episode: {self.start_episode}")
            print(f"   Previous training time: {self.total_training_time:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to resume Actor-Locked training: {e}")
            return False
    
    def save_checkpoint(self, filepath: str, episode_rewards=None, episode_pieces=None, episode_lines=None, 
                       locked_losses=None, actor_losses=None, actor_success_rates=None):
        """Save checkpoint with training history"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save agent state
            self.agent.save_checkpoint(filepath)
            
            # Save training history
            if episode_rewards is not None:
                history_data = {
                    'episode_rewards': episode_rewards,
                    'episode_pieces': episode_pieces or [],
                    'episode_lines': episode_lines or [],
                    'locked_losses': locked_losses or [],
                    'actor_losses': actor_losses or [],
                    'actor_success_rates': actor_success_rates or [],
                    'total_training_time': self.total_training_time,
                    'start_episode': self.start_episode
                }
                
                history_file = filepath.replace('.pt', '_history.json')
                with open(history_file, 'w') as f:
                    json.dump(history_data, f, indent=2)
            
            print(f"Actor-Locked checkpoint saved: {filepath}")
        except Exception as e:
            print(f"Warning: Could not save Actor-Locked checkpoint {filepath}: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Actor-Locked Hierarchical System')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train (default: 1000)')
    parser.add_argument('--save-interval', type=int, default=100, help='Save checkpoint every N episodes (default: 100)')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh training (ignore checkpoints)')
    
    # Actor-Locked specific parameters
    parser.add_argument('--actor-trials', type=int, default=10, help='Number of actor trials per state (default: 10)')
    parser.add_argument('--locked-model-path', type=str, default=None, help='Path to pre-trained locked model')
    parser.add_argument('--actor-learning-rate', type=float, default=0.0001, help='Actor learning rate (default: 0.0001)')
    
    # Visualization parameters
    parser.add_argument('--show-visualization', action='store_true', help='Show text-based visualization during training')
    parser.add_argument('--visualization-interval', type=int, default=100, help='Show visualization every N episodes')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use (default: auto)')
    
    return parser.parse_args()

def main():
    """Main training function with command line arguments"""
    args = parse_arguments()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Actor-Locked training configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg.replace('_', '-')}: {value}")
    
    # Initialize trainer
    trainer = ActorLockedTrainer(
        device=device,
        actor_trials=args.actor_trials,
        locked_model_path=args.locked_model_path,
        actor_learning_rate=args.actor_learning_rate
    )
    
    # Train the system
    results = trainer.train(
        num_episodes=args.episodes,
        save_interval=args.save_interval,
        resume=not args.no_resume,
        show_visualization=args.show_visualization,
        visualization_interval=args.visualization_interval
    )
    
    print("\nActor-Locked training completed successfully!")
    if 'final_avg_reward' in results:
        print(f"Final performance: {results['final_avg_reward']:.1f} reward, "
              f"{results['final_avg_pieces']:.1f} pieces, "
              f"{results['final_avg_lines']:.1f} lines, "
              f"{results['final_actor_success']:.3f} actor success rate")

if __name__ == "__main__":
    main() 