#!/usr/bin/env python3
"""
Hierarchical DQN Training Pipeline for Tetris
Combines Locked State DQN (upper level) with Action DQN (lower level)
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
from typing import Dict, Any, Tuple, Optional

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_locked_agent_optimized import OptimizedLockedStateDQNAgent
from agents.dqn_action_agent import ActionDQNAgent
from envs.tetris_env import TetrisEnv


class HierarchicalDQNTrainer:
    """
    Hierarchical DQN Trainer for Tetris
    
    Two-level control:
    1. Upper Level (Locked State DQN): Selects target positions
    2. Lower Level (Action DQN): Executes 8 basic actions to reach targets
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 max_steps_per_target: int = 50,
                 reward_shaping: bool = True):
        """
        Initialize hierarchical trainer
        
        Args:
            device: Device to run on
            max_steps_per_target: Max steps to reach each target
            reward_shaping: Whether to use reward shaping
        """
        self.device = device
        self.max_steps_per_target = max_steps_per_target
        self.reward_shaping = reward_shaping
        
        # Initialize environment
        self.env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='direct'  # Use direct action mode for basic actions
        )
        
        # Initialize agents
        self.locked_agent = OptimizedLockedStateDQNAgent(
            device=device,
            use_valid_action_selection=False,  # Use full action space
            learning_rate=0.001,
            epsilon_decay_steps=2000,
            batch_size=16
        )
        
        self.action_agent = ActionDQNAgent(
            device=device,
            learning_rate=0.003,  # Higher learning rate for faster action learning
            epsilon_decay_steps=1000,
            batch_size=16
        )
        
        print(f"Hierarchical DQN Trainer initialized:")
        print(f"   Device: {device}")
        print(f"   Locked Agent Parameters: {self.locked_agent.get_parameter_count():,}")
        print(f"   Action Agent Parameters: {self.action_agent.get_parameter_count():,}")
        print(f"   Total Parameters: {self.locked_agent.get_parameter_count() + self.action_agent.get_parameter_count():,}")
    
    def get_current_piece_state(self, env) -> Optional[Tuple[int, int, int]]:
        """
        Get current piece position and rotation
        
        Returns:
            (x, y, rotation) or None if no current piece
        """
        if hasattr(env, 'players') and len(env.players) > 0:
            player = env.players[0]
            if player and player.current_piece:
                return (player.current_piece.x, player.current_piece.y, player.current_piece.rotation)
        return None
    
    def is_target_reached(self, 
                         current_pos: Tuple[int, int, int], 
                         target_pos: Tuple[int, int, int, int],
                         tolerance: int = 1) -> bool:
        """
        Check if current position is close enough to target
        
        Args:
            current_pos: (x, y, rotation)
            target_pos: (x, y, rotation, lock_in)
            tolerance: Distance tolerance
            
        Returns:
            True if target is reached
        """
        if current_pos is None:
            return False
        
        cx, cy, cr = current_pos
        tx, ty, tr, _ = target_pos
        
        # Check if within tolerance
        pos_close = abs(cx - tx) <= tolerance and abs(cy - ty) <= tolerance
        rot_close = abs(cr - tr) <= 1 or abs(cr - tr) >= 3  # Handle rotation wrapping
        
        return pos_close and rot_close
    
    def shape_action_reward(self, 
                           current_pos: Tuple[int, int, int],
                           target_pos: Tuple[int, int, int, int],
                           action: int,
                           piece_placed: bool,
                           lines_cleared: int,
                           step_count: int) -> float:
        """
        Shape reward for action-level training
        
        Args:
            current_pos: Current piece position
            target_pos: Target position
            action: Action taken
            piece_placed: Whether piece was placed
            lines_cleared: Lines cleared
            step_count: Current step count
            
        Returns:
            Shaped reward
        """
        if not self.reward_shaping:
            return float(lines_cleared)  # Basic reward
        
        reward = 0.0
        
        if current_pos is not None:
            cx, cy, cr = current_pos
            tx, ty, tr, _ = target_pos
            
            # Distance-based reward
            pos_distance = abs(cx - tx) + abs(cy - ty)
            rot_distance = min(abs(cr - tr), 4 - abs(cr - tr))
            
            # Reward for getting closer
            reward -= pos_distance * 0.05
            reward -= rot_distance * 0.02
            
            # Strong bonus for reaching target
            if self.is_target_reached(current_pos, target_pos):
                reward += 5.0
        
        # Reward for clearing lines
        reward += lines_cleared * 10.0
        
        # Small step penalty for efficiency
        reward -= 0.01
        
        # Penalty for excessive steps
        if step_count > self.max_steps_per_target * 0.8:
            reward -= 0.1
        
        return reward
    
    def run_episode(self, episode: int, training: bool = True) -> Dict[str, Any]:
        """
        Run a single hierarchical episode
        
        Args:
            episode: Episode number
            training: Whether in training mode
            
        Returns:
            Episode statistics
        """
        observation = self.env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        
        episode_reward = 0.0
        episode_length = 0
        targets_reached = 0
        total_targets = 0
        
        locked_losses = []
        action_losses = []
        
        done = False
        max_episode_steps = 500
        
        while not done and episode_length < max_episode_steps:
            # UPPER LEVEL: Locked agent selects target position
            try:
                locked_state = self.locked_agent.encode_state_with_selection(observation)  
                locked_action = self.locked_agent.select_action(observation, 
                                                               training=training, 
                                                               env=self.env)
                
                # Decode action to target position
                target_components = self.locked_agent.decode_action_components(locked_action)
                target_x, target_y, target_rot, target_lock = target_components
                total_targets += 1
                
                # Set target for action agent
                self.action_agent.set_target_position(target_x, target_y, target_rot, target_lock)
                
                if episode < 5:  # Verbose logging for first few episodes
                    print(f"  Target {total_targets}: ({target_x}, {target_y}, rot={target_rot}, lock={target_lock})")
                
            except Exception as e:
                print(f"Error in locked agent selection: {e}")
                continue
            
            # LOWER LEVEL: Action agent tries to reach target
            target_start_pos = self.get_current_piece_state(self.env)
            steps_to_target = 0
            target_reached = False
            
            while (not done and 
                   steps_to_target < self.max_steps_per_target and 
                   episode_length < max_episode_steps and
                   not target_reached):
                
                try:
                    # Action agent selects action
                    action = self.action_agent.select_action(observation, training=training)
                    
                    # Execute action in environment
                    next_observation, reward, done, info = self.env.step(action)
                    if isinstance(next_observation, tuple):
                        next_observation = next_observation[0]
                    
                    # Get piece state after action
                    current_pos = self.get_current_piece_state(self.env)
                    piece_placed = info.get('piece_placed', False)
                    lines_cleared = info.get('lines_cleared', 0)
                    
                    # Check if target reached
                    target_reached = self.is_target_reached(current_pos, target_components)
                    if target_reached:
                        targets_reached += 1
                    
                    # Shape reward for action agent
                    shaped_reward = self.shape_action_reward(
                        current_pos, target_components, action, 
                        piece_placed, lines_cleared, steps_to_target
                    )
                    
                    # Store experience and train action agent
                    if training:
                        self.action_agent.store_experience(
                            observation, action, shaped_reward, 
                            next_observation, done or target_reached,
                            self.action_agent.current_target_position
                        )
                        
                        if len(self.action_agent.memory) >= self.action_agent.batch_size:
                            action_metrics = self.action_agent.train_batch()
                            if action_metrics['loss'] > 0:
                                action_losses.append(action_metrics['loss'])
                    
                    # Update state
                    observation = next_observation
                    episode_reward += reward
                    episode_length += 1
                    steps_to_target += 1
                    
                    # Break if piece was placed (target complete)
                    if piece_placed:
                        target_reached = True
                        break
                    
                except Exception as e:
                    print(f"Error in action execution: {e}")
                    break
            
            # Train locked agent on target completion
            if training and target_start_pos is not None:
                # Reward for locked agent based on whether target was reached
                locked_reward = 1.0 if target_reached else -0.5
                locked_reward += lines_cleared * 2.0  # Bonus for line clearing
                
                # Store and train locked agent
                try:
                    self.locked_agent.store_experience(
                        locked_state, locked_action, locked_reward,
                        self.locked_agent.encode_state_with_selection(observation), 
                        done
                    )
                    
                    if len(self.locked_agent.memory) >= self.locked_agent.batch_size:
                        locked_metrics = self.locked_agent.train_batch()
                        if locked_metrics['loss'] > 0:
                            locked_losses.append(locked_metrics['loss'])
                            
                except Exception as e:
                    print(f"Error in locked agent training: {e}")
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'targets_reached': targets_reached,
            'total_targets': total_targets,
            'target_success_rate': targets_reached / max(total_targets, 1),
            'avg_locked_loss': np.mean(locked_losses) if locked_losses else 0.0,
            'avg_action_loss': np.mean(action_losses) if action_losses else 0.0,
            'locked_epsilon': self.locked_agent.epsilon,
            'action_epsilon': self.action_agent.epsilon
        }
    
    def train(self, num_episodes: int = 100, save_freq: int = 50):
        """
        Train the hierarchical DQN system
        
        Args:
            num_episodes: Number of episodes to train
            save_freq: Frequency to save checkpoints
        """
        print(f"\nStarting Hierarchical DQN Training...")
        print(f"Episodes: {num_episodes}")
        
        episode_rewards = []
        target_success_rates = []
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            try:
                result = self.run_episode(episode, training=True)
                
                episode_rewards.append(result['episode_reward'])
                target_success_rates.append(result['target_success_rate'])
                
                # Logging
                if episode < 10 or (episode + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (episode + 1)
                    
                    if episode >= 9:
                        recent_rewards = episode_rewards[-10:]
                        recent_success = target_success_rates[-10:]
                        
                        print(f"Episodes {episode-8:3d}-{episode+1:3d}: "
                              f"Reward={np.mean(recent_rewards):6.1f}, "
                              f"Success={np.mean(recent_success):5.1%}, "
                              f"LockedEps={result['locked_epsilon']:.3f}, "
                              f"ActionEps={result['action_epsilon']:.3f}, "
                              f"Time={avg_time:.1f}s/ep")
                    else:
                        print(f"Episode {episode+1:3d}: "
                              f"Reward={result['episode_reward']:6.1f}, "
                              f"Targets={result['targets_reached']}/{result['total_targets']}, "
                              f"Success={result['target_success_rate']:5.1%}, "
                              f"Length={result['episode_length']}")
                
                # Save checkpoints
                if (episode + 1) % save_freq == 0:
                    self.save_checkpoint(f"hierarchical_dqn_episode_{episode+1}.pt")
                    
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        print(f"\nHierarchical Training Completed!")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Final Reward: {np.mean(episode_rewards[-10:]):.1f}")
        print(f"Final Target Success: {np.mean(target_success_rates[-10:]):.1%}")
        
        return {
            'episode_rewards': episode_rewards,
            'target_success_rates': target_success_rates,
            'total_time': total_time
        }
    
    def save_checkpoint(self, filepath: str):
        """Save both agents"""
        base_path = filepath.replace('.pt', '')
        self.locked_agent.save_checkpoint(f"{base_path}_locked.pt")
        self.action_agent.save_checkpoint(f"{base_path}_action.pt")
        print(f"Saved checkpoint: {base_path}_*.pt")
    
    def load_checkpoint(self, filepath: str):
        """Load both agents"""
        base_path = filepath.replace('.pt', '')
        locked_loaded = self.locked_agent.load_checkpoint(f"{base_path}_locked.pt")
        action_loaded = self.action_agent.load_checkpoint(f"{base_path}_action.pt")
        return locked_loaded and action_loaded


def setup_training_environment():
    """Setup training environment and device detection"""
    # GPU detection
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        print(f"GPU DETECTED: {gpu_name} ({gpu_memory}GB)")
    else:
        device = 'cpu'
        print("Using CPU (CUDA not available)")
    
    print(f"PyTorch Version: {torch.__version__}")
    return device


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Hierarchical DQN Training for Tetris')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--max-steps-per-target', type=int, default=50, help='Max steps to reach each target')
    parser.add_argument('--no-reward-shaping', action='store_true', help='Disable reward shaping')
    parser.add_argument('--save-freq', type=int, default=50, help='Checkpoint save frequency')
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_training_environment() if args.device == 'auto' else args.device
    
    print(f"\nHIERARCHICAL DQN CONFIGURATION:")
    print(f"   Episodes: {args.episodes}")
    print(f"   Device: {device}")
    print(f"   Max Steps per Target: {args.max_steps_per_target}")
    print(f"   Reward Shaping: {not args.no_reward_shaping}")
    
    try:
        # Initialize trainer
        trainer = HierarchicalDQNTrainer(
            device=device,
            max_steps_per_target=args.max_steps_per_target,
            reward_shaping=not args.no_reward_shaping
        )
        
        # Train
        results = trainer.train(
            num_episodes=args.episodes,
            save_freq=args.save_freq
        )
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 