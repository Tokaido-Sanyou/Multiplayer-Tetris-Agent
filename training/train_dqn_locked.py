#!/usr/bin/env python3
"""
Comprehensive DQN Locked State Training with Batched Structure and Visualization
"""

import sys
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pickle
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from agents.dqn_locked_agent import LockedStateDQNAgent
from envs.tetris_env import TetrisEnv


class ComprehensiveDQNTrainer:
    """
    Comprehensive DQN Trainer with Batched Structure and Visualization
    Features:
    - Batched training every 10 episodes
    - Checkpoint saving and loading
    - Agent demonstration after each batch
    - Comprehensive logging and visualization
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 batch_size: int = 10,
                 episodes_per_batch: int = 10,
                 learning_rate: float = 0.0001,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay_episodes: int = 800,
                 target_update_freq: int = 1000,
                 memory_size: int = 100000,
                 save_freq: int = 1):  # Save every batch
        """
        Initialize Comprehensive DQN Trainer
        
        Args:
            device: Device to run on ('cpu' or 'cuda')
            batch_size: Batch size for training
            episodes_per_batch: Episodes per batch for batched training
            learning_rate: Learning rate for training
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_episodes: Episodes over which to decay epsilon
            target_update_freq: Target network update frequency
            memory_size: Experience replay buffer size
            save_freq: Frequency to save checkpoints (in batches)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.episodes_per_batch = episodes_per_batch
        self.save_freq = save_freq
        
        print(f"🚀 Comprehensive DQN Trainer Initialized")
        print(f"   Device: {self.device}")
        print(f"   Action Space: 1600 (200 coords × 4 rotations × 2 lock states)")
        print(f"   Episodes per batch: {episodes_per_batch}")
        print(f"   Batch size: {batch_size}")
        
        # Initialize environment in locked position mode with reward mode support
        self.reward_mode = getattr(self, 'reward_mode', 'standard')  # Default to standard
        self.env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='locked_position',
            reward_mode=self.reward_mode  # Support both 'standard' and 'lines_only'
        )
        
        # Initialize demo environment for visualization
        self.demo_env = TetrisEnv(
            num_agents=1,
            headless=False,  # Visual for demonstration
            action_mode='locked_position'
        )
        
        # Initialize agent
        self.agent = LockedStateDQNAgent(
            device=str(self.device),
            learning_rate=learning_rate,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_episodes,
            target_update_freq=target_update_freq,
            memory_size=memory_size,
            batch_size=batch_size
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episode_q_values = []
        self.episode_epsilons = []
        
        # Batch metrics
        self.batch_rewards = []
        self.batch_lengths = []
        self.batch_losses = []
        self.batch_q_values = []
        self.batch_epsilons = []
        
        # Create output directory
        os.makedirs('results/dqn_locked', exist_ok=True)
        
        # Logging
        self.log_file = f'results/dqn_locked/training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        self.log("DQN Locked State Training Started")
        self.log(f"Device: {self.device}")
        self.log(f"Action space: {self.agent.action_space_size}")
        self.log(f"State space: {self.agent.observation_space_shape[0]}")
    
    def log(self, message: str):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def get_valid_actions_for_state(self, observation: np.ndarray) -> List[int]:
        """Get valid action indices for current state"""
        valid_actions = []
        
        # Get valid positions from environment
        if hasattr(self.env, 'players') and len(self.env.players) > 0:
            player = self.env.players[0]
            if player.current_piece:
                valid_positions = self.env.get_valid_positions(player)
                
                # Convert environment positions to our action space
                for pos_idx in valid_positions:
                    x = pos_idx % 10
                    y = pos_idx // 10
                    
                    # Add actions for all rotations with both lock states
                    for rotation in range(4):
                        for lock_in in range(2):
                            action_idx = self.agent.encode_action_components(x, y, rotation, lock_in)
                            if action_idx < self.agent.action_space_size:
                                valid_actions.append(action_idx)
        
        # Fallback if no valid actions found
        if not valid_actions:
            valid_actions = list(range(min(200, self.agent.action_space_size)))
        
        return valid_actions
    
    def convert_action_to_env_format(self, action_info: Dict) -> int:
        """Convert agent action to environment format"""
        if action_info['lock_in']:
            # Convert (x, y) to environment position index
            x, y = action_info['x'], action_info['y']
            env_action = y * 10 + x
        else:
            env_action = 0  # Selection action
        
        return env_action
    
    def run_episode(self, training: bool = True) -> Tuple[float, int, Dict]:
        """Run a single episode"""
        observation = self.env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        
        total_reward = 0.0
        episode_length = 0
        losses = []
        q_values = []
        
        # Reset agent selection state
        self.agent.current_selection.fill(0.0)
        
        done = False
        while not done and episode_length < 1000:  # Safety limit
            # Get valid actions
            valid_actions = self.get_valid_actions_for_state(observation)
            
            # Get enhanced state
            enhanced_state = self.agent.encode_state_with_selection(observation)
            
            # Select action
            if training:
                action_info = self.agent.select_action_with_info(
                    observation, training=True, valid_actions=valid_actions
                )
            else:
                action_idx = self.agent.select_action(
                    observation, training=False, valid_actions=valid_actions
                )
                action_info = {
                    'action_idx': action_idx,
                    **dict(zip(['x', 'y', 'rotation', 'lock_in'], 
                              self.agent.decode_action_components(action_idx))),
                    'enhanced_state': enhanced_state
                }
            
            # Convert to environment action
            env_action = self.convert_action_to_env_format(action_info)
            
            # Execute action
            next_observation, reward, done, info = self.env.step(env_action)
            if isinstance(next_observation, tuple):
                next_observation = next_observation[0]
            
            total_reward += reward
            episode_length += 1
            
            # Store experience and train if training
            if training:
                next_enhanced_state = self.agent.encode_state_with_selection(next_observation)
                self.agent.store_experience(
                    enhanced_state, action_info, reward, next_enhanced_state, done
                )
                
                # Train on batch
                metrics = self.agent.train_batch()
                if metrics['loss'] > 0:
                    losses.append(metrics['loss'])
                    q_values.append(metrics['q_value'])
            
            observation = next_observation
        
        return total_reward, episode_length, {
            'avg_loss': np.mean(losses) if losses else 0.0,
            'avg_q_value': np.mean(q_values) if q_values else 0.0,
            'epsilon': self.agent.epsilon
        }
    
    def run_agent_demonstration(self, batch_num: int, num_demo_episodes: int = 1):
        """Run agent demonstration with visualization"""
        self.log(f"\n🎮 Running Agent Demonstration (Batch {batch_num})")
        self.agent.set_training_mode(False)
        
        demo_rewards = []
        demo_lengths = []
        
        for demo_ep in range(num_demo_episodes):
            observation = self.demo_env.reset()
            if isinstance(observation, tuple):
                observation = observation[0]
            
            total_reward = 0
            steps = 0
            done = False
            
            self.log(f"   Demo Episode {demo_ep + 1} - Starting...")
            
            while not done and steps < 500:
                # Get valid actions
                valid_actions = []
                if hasattr(self.demo_env, 'players') and len(self.demo_env.players) > 0:
                    player = self.demo_env.players[0]
                    if player.current_piece:
                        valid_positions = self.demo_env.get_valid_positions(player)
                        for pos_idx in valid_positions:
                            x, y = pos_idx % 10, pos_idx // 10
                            for rotation in range(4):
                                for lock_in in range(2):
                                    action_idx = self.agent.encode_action_components(x, y, rotation, lock_in)
                                    if action_idx < self.agent.action_space_size:
                                        valid_actions.append(action_idx)
                
                # Select action (evaluation mode)
                action_idx = self.agent.select_action(
                    observation, training=False, valid_actions=valid_actions
                )
                x, y, rotation, lock_in = self.agent.decode_action_components(action_idx)
                
                # Convert to environment action
                if lock_in:
                    env_action = y * 10 + x
                else:
                    env_action = 0
                
                # Execute action
                next_observation, reward, done, _ = self.demo_env.step(env_action)
                if isinstance(next_observation, tuple):
                    next_observation = next_observation[0]
                
                total_reward += reward
                steps += 1
                observation = next_observation
                
                # Render for visualization
                self.demo_env.render()
                time.sleep(0.1)  # Small delay for visibility
            
            demo_rewards.append(total_reward)
            demo_lengths.append(steps)
            self.log(f"   Demo Episode {demo_ep + 1}: {total_reward:.2f} reward, {steps} steps")
        
        self.agent.set_training_mode(True)
        
        avg_demo_reward = np.mean(demo_rewards)
        avg_demo_length = np.mean(demo_lengths)
        
        self.log(f"   Average Demo Performance: {avg_demo_reward:.2f} reward, {avg_demo_length:.1f} steps")
        self.log(f"   Current Epsilon: {self.agent.epsilon:.4f}")
        
        return avg_demo_reward, avg_demo_length
    
    def save_checkpoint(self, batch_num: int):
        """Save training checkpoint"""
        checkpoint_path = f'results/dqn_locked/checkpoint_batch_{batch_num:03d}.pth'
        self.agent.save_checkpoint(checkpoint_path)
        
        # Save training metrics
        metrics_path = f'results/dqn_locked/metrics_batch_{batch_num:03d}.pkl'
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_losses': self.episode_losses,
            'episode_q_values': self.episode_q_values,
            'episode_epsilons': self.episode_epsilons,
            'batch_rewards': self.batch_rewards,
            'batch_lengths': self.batch_lengths,
            'batch_losses': self.batch_losses,
            'batch_q_values': self.batch_q_values,
            'batch_epsilons': self.batch_epsilons,
            'batch_num': batch_num
        }
        
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        self.log(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def create_training_plots(self, batch_num: int):
        """Create and save training progress plots"""
        if len(self.episode_rewards) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Rewards')
        if len(self.episode_rewards) >= 10:
            # Moving average
            moving_avg = [np.mean(self.episode_rewards[max(0, i-9):i+1]) 
                         for i in range(len(self.episode_rewards))]
            axes[0, 0].plot(moving_avg, label='Moving Avg (10)', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Batch rewards
        if len(self.batch_rewards) > 0:
            axes[0, 1].plot(self.batch_rewards, 'o-', label='Batch Avg Rewards')
            axes[0, 1].set_title('Batch Average Rewards')
            axes[0, 1].set_xlabel('Batch')
            axes[0, 1].set_ylabel('Avg Reward')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Episode lengths
        axes[0, 2].plot(self.episode_lengths, alpha=0.6)
        axes[0, 2].set_title('Episode Lengths')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Steps')
        axes[0, 2].grid(True)
        
        # Training loss
        non_zero_losses = [loss for loss in self.episode_losses if loss > 0]
        if non_zero_losses:
            axes[1, 0].plot(non_zero_losses, alpha=0.6)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # Q-values
        non_zero_qvals = [qval for qval in self.episode_q_values if qval != 0]
        if non_zero_qvals:
            axes[1, 1].plot(non_zero_qvals, alpha=0.6)
            axes[1, 1].set_title('Average Q-Values')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Q-Value')
            axes[1, 1].grid(True)
        
        # Epsilon decay
        axes[1, 2].plot(self.episode_epsilons)
        axes[1, 2].set_title('Epsilon Decay')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Epsilon')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plot_path = f'results/dqn_locked/training_progress_batch_{batch_num:03d}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"📊 Training plots saved: {plot_path}")
    
    def train(self, total_episodes: int = 100):
        """Main training loop with batched structure"""
        self.log("="*80)
        self.log("STARTING COMPREHENSIVE DQN LOCKED STATE TRAINING")
        self.log("="*80)
        self.log(f"Total episodes: {total_episodes}")
        self.log(f"Episodes per batch: {self.episodes_per_batch}")
        self.log(f"Total batches: {total_episodes // self.episodes_per_batch}")
        self.log("="*80)
        
        start_time = time.time()
        best_reward = float('-inf')
        
        total_batches = total_episodes // self.episodes_per_batch
        
        for batch_num in range(1, total_batches + 1):
            batch_start_time = time.time()
            self.log(f"\n🔄 BATCH {batch_num}/{total_batches} - Training {self.episodes_per_batch} episodes...")
            
            batch_episode_rewards = []
            batch_episode_lengths = []
            batch_episode_losses = []
            batch_episode_q_values = []
            
            # Train episodes in this batch
            for episode_in_batch in range(self.episodes_per_batch):
                episode_num = (batch_num - 1) * self.episodes_per_batch + episode_in_batch + 1
                
                reward, length, metrics = self.run_episode(training=True)
                
                # Store episode metrics
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.episode_losses.append(metrics['avg_loss'])
                self.episode_q_values.append(metrics['avg_q_value'])
                self.episode_epsilons.append(metrics['epsilon'])
                
                # Store batch metrics
                batch_episode_rewards.append(reward)
                batch_episode_lengths.append(length)
                batch_episode_losses.append(metrics['avg_loss'])
                batch_episode_q_values.append(metrics['avg_q_value'])
                
                # Progress logging
                if episode_in_batch % 5 == 0 or episode_in_batch == self.episodes_per_batch - 1:
                    self.log(f"   Episode {episode_num}: Reward {reward:.2f}, Steps {length}, "
                           f"Loss {metrics['avg_loss']:.4f}, Q-val {metrics['avg_q_value']:.2f}, "
                           f"ε {metrics['epsilon']:.3f}")
            
            # Compute batch statistics
            avg_batch_reward = np.mean(batch_episode_rewards)
            avg_batch_length = np.mean(batch_episode_lengths)
            avg_batch_loss = np.mean([l for l in batch_episode_losses if l > 0]) if any(l > 0 for l in batch_episode_losses) else 0.0
            avg_batch_q_value = np.mean([q for q in batch_episode_q_values if q != 0]) if any(q != 0 for q in batch_episode_q_values) else 0.0
            
            # Store batch metrics
            self.batch_rewards.append(avg_batch_reward)
            self.batch_lengths.append(avg_batch_length)
            self.batch_losses.append(avg_batch_loss)
            self.batch_q_values.append(avg_batch_q_value)
            self.batch_epsilons.append(self.agent.epsilon)
            
            batch_time = time.time() - batch_start_time
            
            self.log(f"\n📈 BATCH {batch_num} SUMMARY:")
            self.log(f"   Average Reward: {avg_batch_reward:.2f}")
            self.log(f"   Average Length: {avg_batch_length:.1f}")
            self.log(f"   Average Loss: {avg_batch_loss:.4f}")
            self.log(f"   Average Q-Value: {avg_batch_q_value:.2f}")
            self.log(f"   Current Epsilon: {self.agent.epsilon:.4f}")
            self.log(f"   Batch Time: {batch_time:.1f}s")
            
            # Update best reward
            if avg_batch_reward > best_reward:
                best_reward = avg_batch_reward
                best_checkpoint = f'results/dqn_locked/best_model_batch_{batch_num:03d}.pth'
                self.agent.save_checkpoint(best_checkpoint)
                self.log(f"   🏆 NEW BEST REWARD: {best_reward:.2f} - Model saved!")
            
            # Save checkpoint every batch
            if batch_num % self.save_freq == 0:
                self.save_checkpoint(batch_num)
            
            # Create training plots
            self.create_training_plots(batch_num)
            
            # Run agent demonstration every batch
            self.log(f"\n🎮 AGENT DEMONSTRATION (Batch {batch_num}):")
            demo_reward, demo_length = self.run_agent_demonstration(batch_num, num_demo_episodes=1)
            
            self.log(f"   Demo Performance: {demo_reward:.2f} reward, {demo_length:.1f} steps")
        
        # Final summary
        total_time = time.time() - start_time
        self.log("\n" + "="*80)
        self.log("TRAINING COMPLETED")
        self.log("="*80)
        self.log(f"Total training time: {total_time:.1f}s")
        self.log(f"Total episodes: {len(self.episode_rewards)}")
        self.log(f"Total batches: {len(self.batch_rewards)}")
        self.log(f"Best batch reward: {best_reward:.2f}")
        self.log(f"Final epsilon: {self.agent.epsilon:.4f}")
        self.log(f"Final memory size: {len(self.agent.memory)}")
        
        # Save final checkpoint
        final_checkpoint = 'results/dqn_locked/final_model.pth'
        self.agent.save_checkpoint(final_checkpoint)
        self.log(f"Final model saved: {final_checkpoint}")
        
        # Close environments
        self.env.close()
        self.demo_env.close()
        
        self.log("="*80)


def main():
    """Main training function"""
    trainer = ComprehensiveDQNTrainer(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        episodes_per_batch=10,
        epsilon_decay_episodes=800
    )
    trainer.train(total_episodes=100)


if __name__ == "__main__":
    main() 