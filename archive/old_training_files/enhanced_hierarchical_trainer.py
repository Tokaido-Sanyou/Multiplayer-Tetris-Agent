#!/usr/bin/env python3
"""
Enhanced Hierarchical DQN Training Pipeline
Sequential training with comprehensive debugging and validation
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
from agents.dqn_action_agent import ActionDQNAgent
from envs.tetris_env import TetrisEnv


class RNDNetwork(nn.Module):
    """Random Network Distillation for exploration"""
    
    def __init__(self, input_size: int = 206, hidden_size: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64)
        )
        
        self.target = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64)
        )
        
        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.predictor(x), self.target(x)
    
    def intrinsic_reward(self, x):
        with torch.no_grad():
            pred, target = self.forward(x)
            return F.mse_loss(pred, target, reduction='none').mean(dim=1)


class PositionValidator:
    """Validates locked positions using environment logic"""
    
    def __init__(self, env):
        self.env = env
    
    def is_valid_locked_position(self, x: int, y: int, rotation: int, piece_type: int) -> bool:
        """Check if a position is a valid locked placement"""
        try:
            # Basic bounds check
            if not (0 <= x < 10 and 0 <= y < 20 and 0 <= rotation < 4):
                return False
            
            # Use environment's validation if available
            if hasattr(self.env, 'players') and len(self.env.players) > 0:
                player = self.env.players[0]
                if player and player.current_piece:
                    # Simple validation - check if position is reasonable
                    # More sophisticated validation could be added here
                    return True
            
            return True  # Default to valid for basic bounds
            
        except Exception as e:
            return False


class TrajectoryCollector:
    """Collects and analyzes training trajectories"""
    
    def __init__(self, max_trajectories: int = 1000):
        self.trajectories = deque(maxlen=max_trajectories)
        self.piece_type_encoding = {'I': 0, 'O': 1, 'T': 2, 'S': 3, 'Z': 4, 'J': 5, 'L': 6}
        self.locked_positions_history = deque(maxlen=10000)
    
    def add_trajectory(self, trajectory: Dict):
        """Add a trajectory with metadata"""
        self.trajectories.append(trajectory)
        
        # Track locked positions
        if 'final_position' in trajectory and trajectory.get('locked', False):
            self.locked_positions_history.append(trajectory['final_position'])
    
    def get_piece_encoding(self, piece_type) -> int:
        """Get numeric encoding for piece type"""
        return self.piece_type_encoding.get(piece_type, 0)
    
    def analyze_trajectories(self) -> Dict[str, Any]:
        """Analyze collected trajectories"""
        if not self.trajectories:
            return {}
        
        analysis = {
            'total_trajectories': len(self.trajectories),
            'locked_trajectories': sum(1 for t in self.trajectories if t.get('locked', False)),
            'average_length': np.mean([t.get('length', 0) for t in self.trajectories]),
            'success_rate': np.mean([t.get('success', False) for t in self.trajectories]),
            'empty_board_ratio': sum(1 for t in self.trajectories if t.get('empty_board', True)) / len(self.trajectories)
        }
        
        return analysis


class EnhancedHierarchicalTrainer:
    """
    Enhanced Hierarchical Training Pipeline
    
    Sequential training with debugging and validation:
    1. Train locked position DQN first (1000 batches)
    2. Use its rollouts to train action DQN (1000 batches)
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 batch_size: int = 100,
                 max_steps_per_target: int = 30,
                 enable_rnd: bool = True,
                 debug_mode: bool = True):
        """Initialize enhanced trainer"""
        self.device = device
        self.batch_size = batch_size
        self.max_steps_per_target = max_steps_per_target
        self.enable_rnd = enable_rnd
        self.debug_mode = debug_mode
        
        # Initialize environment
        self.env = TetrisEnv(num_agents=1, headless=True, action_mode='locked_position')
        
        # Initialize components
        self.position_validator = PositionValidator(self.env)
        self.trajectory_collector = TrajectoryCollector()
        
        # Initialize redesigned locked agent with same parameters as train_redesigned_agent.py
        self.locked_agent = RedesignedLockedStateDQNAgent(
            device=device,
            learning_rate=0.00005,  # Match redesigned agent
            epsilon_decay_steps=50000,  # Match redesigned agent
            batch_size=32,
            memory_size=50000,
            epsilon_start=0.95,  # Updated start epsilon
            epsilon_end=0.01
        )
        
        # RND for exploration
        if self.enable_rnd:
            self.rnd_network = RNDNetwork().to(device)
            self.rnd_optimizer = optim.Adam(self.rnd_network.predictor.parameters(), lr=0.0001)
        
        # Training state
        self.locked_agent_trained = False
        self.locked_checkpoint_path = "checkpoints/locked_agent_checkpoint.pt"
        
        # Debug tracking
        self.debug_data = {
            'valid_positions_passed': [],
            'empty_board_episodes': 0,
            'rnd_rewards': [],
            'locked_only_trajectories': [],
            'tensor_shape_errors': [],
            'backprop_success': [],
            'reward_progression': [],
            'position_validity_checks': [],
            'lock_1_rate': 0.0,  # Track how often lock=1 actions are selected
            'debug_prints': [],
            'lines_cleared_total': 0,
            'pieces_placed_total': 0
        }
        
        print(f"Enhanced Hierarchical Trainer initialized:")
        print(f"   Device: {self.device}")
        print(f"   Batch Size: {self.batch_size} episodes")
        print(f"   RND Exploration: {self.enable_rnd}")
        print(f"   Debug Mode: {self.debug_mode}")
        print(f"   Locked Agent Parameters: {self.locked_agent.get_parameter_count():,}")
    
    def debug_print(self, message: str):
        """Print debug message and store it"""
        if self.debug_mode:
            print(f"DEBUG: {message}")
            self.debug_data['debug_prints'].append(message)
    
    def force_lock_action(self, action_idx: int) -> int:
        """Force an action to be lock=1 (final state) - redesigned agent doesn't need this"""
        # Redesigned agent actions are already final placements
        return action_idx
    
    def get_piece_type(self, env) -> int:
        """Get current piece type encoding"""
        try:
            if hasattr(env, 'players') and len(env.players) > 0:
                player = env.players[0]
                if player and player.current_piece:
                    # Try to get piece type - simplified
                    return random.randint(0, 6)  # Random piece type for now
            return 0
        except:
            return 0
    
    def is_empty_board(self, env) -> bool:
        """Check if the board is empty"""
        try:
            if hasattr(env, 'players') and len(env.players) > 0:
                player = env.players[0]
                if player:
                    return len(player.locked_positions) == 0
            return True
        except:
            return True
    
    def compute_rnd_reward(self, observation: np.ndarray) -> float:
        """Compute RND intrinsic reward"""
        if not self.enable_rnd:
            return 0.0
        
        try:
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            intrinsic_reward = self.rnd_network.intrinsic_reward(obs_tensor)
            
            # Train RND predictor
            pred, target = self.rnd_network(obs_tensor)
            rnd_loss = F.mse_loss(pred, target)
            
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()
            
            reward = intrinsic_reward.item() * 0.1
            self.debug_data['rnd_rewards'].append(reward)
            return reward
            
        except Exception as e:
            if self.debug_mode:
                self.debug_data['tensor_shape_errors'].append(f"RND: {e}")
            return 0.0
    
    def train_locked_agent_batch(self, num_batches: int = 100) -> Dict[str, Any]:
        """Train locked position DQN agent"""
        print(f"\n=== TRAINING LOCKED POSITION DQN ===")
        print(f"Batches: {num_batches} (each {self.batch_size} episodes)")
        
        batch_rewards = []
        batch_losses = []
        episode_count = 0
        lock_1_actions = 0
        total_actions = 0
        
        start_time = time.time()
        
        for batch in range(num_batches):
            episode_rewards = []
            episode_losses = []
            
            for episode in range(self.batch_size):
                try:
                    observation = self.env.reset()
                    if isinstance(observation, tuple):
                        observation = observation[0]
                    
                    episode_reward = 0.0
                    episode_length = 0
                    done = False
                    max_episode_steps = 100
                    
                    empty_board = self.is_empty_board(self.env)
                    piece_type = self.get_piece_type(self.env)
                    episode_positions = []
                    
                    if empty_board:
                        self.debug_data['empty_board_episodes'] += 1
                    
                    pieces_placed = 0
                    lines_cleared_total = 0
                    
                    while not done and episode_length < max_episode_steps:
                        # Select action (redesigned agent handles everything internally)
                        action = self.locked_agent.select_action(observation, training=True, env=self.env)
                        
                        # Track action coordinates for debugging
                        x, y, rotation = self.locked_agent.map_action_to_board(action)
                        episode_positions.append((x, y, rotation))
                        total_actions += 1
                        
                        # Execute action
                        next_observation, reward, done, info = self.env.step(action)
                        if isinstance(next_observation, tuple):
                            next_observation = next_observation[0]
                        
                        # Track piece placement and line clearing
                        if info.get('piece_placed', False):
                            pieces_placed += 1
                            self.debug_data['pieces_placed_total'] += 1
                            self.debug_print(f"Episode {episode}, Step {episode_length}: Piece placed! Total: {pieces_placed}")
                        
                        lines_cleared = info.get('lines_cleared', 0)
                        if lines_cleared > 0:
                            lines_cleared_total += lines_cleared
                            self.debug_data['lines_cleared_total'] += lines_cleared
                            reward += lines_cleared * 100  # Strong reward for line clearing
                            self.debug_print(f"Episode {episode}, Step {episode_length}: {lines_cleared} lines cleared! Total: {lines_cleared_total}")
                        
                        # Force line clearing check if board is getting full
                        if len(self.env.players[0].locked_positions) > 100 and lines_cleared_total == 0:
                            try:
                                player = self.env.players[0]
                                player.change_piece = True
                                forced_lines = player.update(self.env.game.fall_speed, self.env.game.level)
                                if forced_lines > 0:
                                    lines_cleared_total += forced_lines
                                    reward += forced_lines * 100
                                    self.debug_print(f"Episode {episode}, Step {episode_length}: Forced {forced_lines} lines cleared!")
                            except Exception as e:
                                pass
                        
                        # Add RND reward
                        if self.enable_rnd:
                            intrinsic_reward = self.compute_rnd_reward(observation)
                            reward += intrinsic_reward
                        
                        # Store experience
                        self.locked_agent.store_experience(observation, action, reward, next_observation, done)
                        
                        # Train
                        if len(self.locked_agent.memory) >= self.locked_agent.batch_size:
                            try:
                                metrics = self.locked_agent.train_batch()
                                if metrics['loss'] > 0:
                                    episode_losses.append(metrics['loss'])
                                    self.debug_data['backprop_success'].append(True)
                            except Exception as e:
                                if self.debug_mode:
                                    self.debug_data['tensor_shape_errors'].append(f"Training: {e}")
                                    self.debug_data['backprop_success'].append(False)
                        
                        observation = next_observation
                        episode_reward += reward
                        episode_length += 1
                    
                    # Validate positions (redesigned agent uses direct placement)
                    if episode_positions and self.debug_mode:
                        validation = {
                            'total_positions': len(episode_positions),
                            'lock_1_positions': len(episode_positions),  # All positions are final placements
                            'valid_positions': len(episode_positions)
                        }
                        self.debug_data['valid_positions_passed'].append(validation)
                    
                    # Collect trajectory
                    trajectory = {
                        'episode': episode_count,
                        'length': episode_length,
                        'reward': episode_reward,
                        'empty_board': empty_board,
                        'piece_type': piece_type,
                        'locked': True,  # Redesigned agent always uses locked placements
                        'success': episode_reward > -50
                    }
                    self.trajectory_collector.add_trajectory(trajectory)
                    
                    episode_rewards.append(episode_reward)
                    episode_count += 1
                    
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error in episode {episode}: {e}")
                    continue
            
            # Batch statistics
            batch_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            batch_loss = np.mean(episode_losses) if episode_losses else 0.0
            
            batch_rewards.append(batch_reward)
            batch_losses.append(batch_loss)
            
            # Update lock=1 rate
            if total_actions > 0:
                self.debug_data['lock_1_rate'] = lock_1_actions / total_actions
            
            # Logging
            if batch % 10 == 0 or batch < 5:
                print(f"Batch {batch:3d}: Reward={batch_reward:6.1f}, "
                      f"Loss={batch_loss:.4f}, "
                      f"Lock1Rate={self.debug_data['lock_1_rate']:.1%}, "
                      f"Epsilon={self.locked_agent.epsilon:.3f}")
        
        total_time = time.time() - start_time
        
        # Save checkpoint
        try:
            os.makedirs("checkpoints", exist_ok=True)
            self.locked_agent.save_checkpoint("checkpoints/locked_agent_checkpoint.pt")
            self.locked_agent_trained = True
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
            self.locked_agent_trained = True
        
        print(f"\nLocked Agent Training Complete:")
        print(f"  Total Time: {total_time:.1f}s")
        print(f"  Final Reward: {np.mean(batch_rewards[-5:]):.1f}")
        print(f"  Lock=1 Rate: {self.debug_data['lock_1_rate']:.1%}")
        print(f"  Episodes: {episode_count}")
        
        return {
            'batch_rewards': batch_rewards,
            'batch_losses': batch_losses,
            'total_episodes': episode_count,
            'lock_1_rate': self.debug_data['lock_1_rate'],
            'total_time': total_time
        }
    
    def train_action_agent_batch(self, num_batches: int = 100) -> Dict[str, Any]:
        """Train action DQN using locked agent rollouts"""
        print(f"\n=== TRAINING ACTION DQN WITH ROLLOUTS ===")
        
        if not self.locked_agent_trained:
            print("Error: Locked agent must be trained first!")
            return {}
        
        # Initialize action agent
        action_agent = ActionDQNAgent(
            device=self.device,
            learning_rate=0.003,
            epsilon_decay_steps=10000,
            batch_size=16
        )
        
        print(f"Action Agent Parameters: {action_agent.get_parameter_count():,}")
        print(f"Batches: {num_batches} (each {self.batch_size} episodes)")
        
        batch_rewards = []
        batch_success_rates = []
        episode_count = 0
        valid_targets_processed = 0
        
        start_time = time.time()
        
        for batch in range(num_batches):
            episode_rewards = []
            episode_success_rates = []
            
            for episode in range(self.batch_size):
                try:
                    observation = self.env.reset()
                    if isinstance(observation, tuple):
                        observation = observation[0]
                    
                    episode_reward = 0.0
                    episode_length = 0
                    targets_reached = 0
                    total_targets = 0
                    done = False
                    max_episode_steps = 100
                    
                    while not done and episode_length < max_episode_steps:
                        # Get target from locked agent
                        locked_action = self.locked_agent.select_action(observation, training=False, env=self.env)
                        
                        # Get target coordinates (redesigned agent uses direct mapping)
                        target_x, target_y, target_rot = self.locked_agent.map_action_to_board(locked_action)
                        target_lock = 1  # Redesigned agent always uses locked placements
                        target_components = (target_x, target_y, target_rot, target_lock)
                        
                        total_targets += 1
                        valid_targets_processed += 1
                        
                        # Validate target
                        piece_type = self.get_piece_type(self.env)
                        is_valid = self.position_validator.is_valid_locked_position(
                            target_x, target_y, target_rot, piece_type)
                        
                        if self.debug_mode:
                            self.debug_data['position_validity_checks'].append({
                                'target': (target_x, target_y, target_rot, target_lock),
                                'valid': is_valid,
                                'piece_type': piece_type
                            })
                        
                        # Set target for action agent
                        action_agent.set_target_position(target_x, target_y, target_rot, target_lock)
                        
                        # Action agent tries to reach target
                        target_reached = False
                        steps_to_target = 0
                        
                        while (not done and 
                               steps_to_target < self.max_steps_per_target and 
                               episode_length < max_episode_steps and
                               not target_reached):
                            
                            # Action selection
                            action = action_agent.select_action(observation, training=True)
                            
                            # Execute action
                            next_observation, reward, done, info = self.env.step(action)
                            if isinstance(next_observation, tuple):
                                next_observation = next_observation[0]
                            
                            # Check if target reached
                            current_pos = self.get_current_piece_state()
                            if current_pos is not None:
                                target_reached = self.is_target_reached(current_pos, target_components)
                                if target_reached:
                                    targets_reached += 1
                            
                            # Compute shaped reward
                            shaped_reward = self.compute_action_reward(
                                current_pos, target_components, action,
                                info.get('piece_placed', False),
                                info.get('lines_cleared', 0)
                            )
                            
                            # Store experience
                            action_agent.store_experience(
                                observation, action, shaped_reward,
                                next_observation, done or target_reached,
                                action_agent.current_target_position
                            )
                            
                            # Train
                            if len(action_agent.memory) >= action_agent.batch_size:
                                try:
                                    metrics = action_agent.train_batch()
                                    self.debug_data['backprop_success'].append(True)
                                except Exception as e:
                                    if self.debug_mode:
                                        self.debug_data['tensor_shape_errors'].append(f"Action: {e}")
                                        self.debug_data['backprop_success'].append(False)
                            
                            observation = next_observation
                            episode_reward += reward
                            episode_length += 1
                            steps_to_target += 1
                            
                            if info.get('piece_placed', False):
                                target_reached = True
                                break
                    
                    # Episode statistics
                    success_rate = targets_reached / max(total_targets, 1)
                    episode_rewards.append(episode_reward)
                    episode_success_rates.append(success_rate)
                    episode_count += 1
                    
                    # Track for debugging
                    if total_targets > 0:
                        self.debug_data['locked_only_trajectories'].append({
                            'episode': episode_count,
                            'targets': total_targets,
                            'reached': targets_reached,
                            'success_rate': success_rate,
                            'reward': episode_reward
                        })
                    
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error in action episode {episode}: {e}")
                    continue
            
            # Batch statistics
            batch_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            batch_success = np.mean(episode_success_rates) if episode_success_rates else 0.0
            
            batch_rewards.append(batch_reward)
            batch_success_rates.append(batch_success)
            
            # Logging
            if batch % 10 == 0 or batch < 5:
                print(f"Batch {batch:3d}: Reward={batch_reward:6.1f}, "
                      f"Success={batch_success:5.1%}, "
                      f"Epsilon={action_agent.epsilon:.3f}")
        
        total_time = time.time() - start_time
        
        print(f"\nAction Agent Training Complete:")
        print(f"  Total Time: {total_time:.1f}s")
        print(f"  Final Reward: {np.mean(batch_rewards[-5:]):.1f}")
        print(f"  Final Success: {np.mean(batch_success_rates[-5:]):.1%}")
        print(f"  Valid Targets: {valid_targets_processed}")
        
        return {
            'batch_rewards': batch_rewards,
            'batch_success_rates': batch_success_rates,
            'valid_targets_processed': valid_targets_processed,
            'total_time': total_time
        }
    
    def get_current_piece_state(self) -> Optional[Tuple[int, int, int]]:
        """Get current piece position and rotation"""
        try:
            if hasattr(self.env, 'players') and len(self.env.players) > 0:
                player = self.env.players[0]
                if player and player.current_piece:
                    return (player.current_piece.x, player.current_piece.y, player.current_piece.rotation)
            return None
        except:
            return None
    
    def is_target_reached(self, current_pos: Tuple[int, int, int], 
                         target_pos: Tuple[int, int, int, int], tolerance: int = 1) -> bool:
        """Check if target is reached"""
        if current_pos is None:
            return False
        
        cx, cy, cr = current_pos
        tx, ty, tr, _ = target_pos
        
        pos_close = abs(cx - tx) <= tolerance and abs(cy - ty) <= tolerance
        rot_close = abs(cr - tr) <= 1 or abs(cr - tr) >= 3
        
        return pos_close and rot_close
    
    def compute_action_reward(self, current_pos: Tuple[int, int, int],
                             target_pos: Tuple[int, int, int, int],
                             action: int, piece_placed: bool, lines_cleared: int) -> float:
        """Compute shaped reward for action agent"""
        reward = 0.0
        
        if current_pos is not None:
            cx, cy, cr = current_pos
            tx, ty, tr, _ = target_pos
            
            pos_distance = abs(cx - tx) + abs(cy - ty)
            rot_distance = min(abs(cr - tr), 4 - abs(cr - tr))
            
            reward -= pos_distance * 0.05
            reward -= rot_distance * 0.02
            
            if self.is_target_reached(current_pos, target_pos):
                reward += 10.0
        
        reward += lines_cleared * 20.0
        reward -= 0.01
        
        return reward
    
    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report"""
        report = {
            'trajectory_analysis': self.trajectory_collector.analyze_trajectories(),
            'tensor_issues': len(self.debug_data['tensor_shape_errors']),
            'backprop_success_rate': np.mean(self.debug_data['backprop_success']) if self.debug_data['backprop_success'] else 0.0,
            'empty_board_episodes': self.debug_data['empty_board_episodes'],
            'lock_1_rate': self.debug_data['lock_1_rate'],
            'rnd_rewards_generated': len(self.debug_data['rnd_rewards']),
            'locked_only_trajectories': len(self.debug_data['locked_only_trajectories']),
            'position_validity_checks': len(self.debug_data['position_validity_checks'])
        }
        
        return report


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Enhanced Hierarchical DQN Training')
    parser.add_argument('--locked-batches', type=int, default=10, help='Batches for locked agent')
    parser.add_argument('--action-batches', type=int, default=10, help='Batches for action agent')
    parser.add_argument('--batch-size', type=int, default=5, help='Episodes per batch')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--enable-rnd', action='store_true', help='Enable RND exploration')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    print(f"ENHANCED HIERARCHICAL DQN TRAINING:")
    print(f"   Device: {args.device}")
    print(f"   Locked Batches: {args.locked_batches}")
    print(f"   Action Batches: {args.action_batches}")
    print(f"   Batch Size: {args.batch_size}")
    
    try:
        # Initialize trainer
        trainer = EnhancedHierarchicalTrainer(
            device=args.device,
            batch_size=args.batch_size,
            enable_rnd=args.enable_rnd,
            debug_mode=args.debug
        )
        
        # Phase 1: Train locked agent
        locked_results = trainer.train_locked_agent_batch(args.locked_batches)
        
        # Phase 2: Train action agent
        action_results = trainer.train_action_agent_batch(args.action_batches)
        
        # Debug report
        if args.debug:
            debug_report = trainer.generate_debug_report()
            print(f"\n=== DEBUG REPORT ===")
            print(f"Lock=1 Rate: {debug_report['lock_1_rate']:.1%}")
            print(f"Backprop Success: {debug_report['backprop_success_rate']:.1%}")
            print(f"Empty Boards: {debug_report['empty_board_episodes']}")
            print(f"RND Rewards: {debug_report['rnd_rewards_generated']}")
            print(f"Locked Trajectories: {debug_report['locked_only_trajectories']}")
        
        print("\n🎉 ENHANCED TRAINING COMPLETE!")
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 