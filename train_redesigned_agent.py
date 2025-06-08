#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
import argparse
import json
import glob
from typing import Dict, Any, Optional

from envs.tetris_env import TetrisEnv
from agents.dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent


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

class RedesignedAgentTrainer:
    """Enhanced trainer with checkpoint resuming, configurable parameters, and RND mode"""
    
    def __init__(self, 
                 device: str = 'cuda',
                 batch_size: int = 32,
                 learning_rate: float = 0.00005,
                 gamma: float = 0.99,
                 epsilon_start: float = 0.95,  # Updated default
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 50000,
                 target_update_freq: int = 1000,
                 memory_size: int = 100000,
                 enable_rnd: bool = False,
                 rnd_reward_scale: float = 0.1,
                 show_visualization: bool = False,
                 visualization_interval: int = 10,
                 render_delay: float = 0.05,
                 update_per_episode: bool = False):
        self.device = device
        self.batch_size = batch_size
        self.enable_rnd = enable_rnd
        self.rnd_reward_scale = rnd_reward_scale
        self.show_visualization = show_visualization
        self.visualization_interval = visualization_interval
        self.render_delay = render_delay
        self.update_per_episode = update_per_episode
        
        # Training state for resuming
        self.start_episode = 0
        self.total_training_time = 0.0
        self.training_history = {
            'episode_rewards': [],
            'episode_pieces': [],
            'episode_lines': [],
            'episode_losses': [],
            'training_metrics': []
        }
        
        # Initialize environment and agent (headless unless visualization requested)
        self.env = TetrisEnv(action_mode='locked_position', headless=not show_visualization)
        self.agent = RedesignedLockedStateDQNAgent(
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            target_update_freq=target_update_freq,
            memory_size=memory_size,
            batch_size=batch_size
        )
        
        # Initialize RND network if enabled
        self.rnd_network = None
        if self.enable_rnd:
            self.rnd_network = RNDNetwork(input_size=206).to(device)
            # Setup optimizer for RND predictor
            self.rnd_optimizer = optim.Adam(self.rnd_network.predictor.parameters(), lr=learning_rate)
            # Disable epsilon-greedy when using RND exploration only
            self.agent.epsilon_start = 0.0
            self.agent.epsilon_end = 0.0
            self.agent.epsilon = 0.0
            self.agent.update_epsilon = lambda: None
            print(f"   RND Network initialized with {sum(p.numel() for p in self.rnd_network.parameters()):,} parameters")
        
        print(f"Redesigned Agent Trainer initialized:")
        print(f"   Device: {self.device}")
        print(f"   Batch Size: {self.batch_size}")  # Added batch size display
        print(f"   RND Mode: {self.enable_rnd}")    # Added RND mode display
        print(f"   Environment action space: {self.env.action_space}")
        print(f"   Environment observation space: {self.env.observation_space}")
        print(f"   Agent parameters: {self.agent.get_parameter_count():,}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Gamma: {gamma}")
        print(f"   Epsilon decay: {epsilon_start} -> {epsilon_end} over {epsilon_decay_steps} steps")
    
    def resume_from_checkpoint(self, checkpoint_pattern: str = "checkpoints/redesigned_agent_episode_*.pt") -> bool:
        """Resume training from the latest checkpoint"""
        try:
            # Find latest checkpoint
            checkpoint_files = glob.glob(checkpoint_pattern)
            if not checkpoint_files:
                print("No checkpoints found, starting fresh training")
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
            
            print(f"✅ Resumed from checkpoint: {latest_checkpoint}")
            print(f"   Starting from episode: {self.start_episode}")
            print(f"   Previous training time: {self.total_training_time:.1f}s")
            print(f"   Agent epsilon: {self.agent.epsilon:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to resume from checkpoint: {e}")
            return False
    
    def train(self, num_episodes: int = 1000, save_interval: int = 100, resume: bool = True) -> Dict[str, Any]:
        """Train the redesigned agent with checkpoint resuming"""
        
        # Try to resume from checkpoint if requested
        if resume:
            resumed = self.resume_from_checkpoint()
            if resumed:
                # Adjust episodes to continue from where we left off
                remaining_episodes = max(0, num_episodes - self.start_episode)
                if remaining_episodes == 0:
                    print(f"Training already completed {num_episodes} episodes!")
                    return self.training_history
                print(f"Continuing training for {remaining_episodes} more episodes")
                num_episodes = remaining_episodes
        
        print(f"\n=== TRAINING REDESIGNED AGENT ===")
        print(f"Episodes: {num_episodes} (starting from episode {self.start_episode})")
        
        episode_rewards = self.training_history['episode_rewards'].copy()
        episode_losses = self.training_history['episode_losses'].copy()
        episode_pieces = self.training_history['episode_pieces'].copy()
        episode_lines = self.training_history['episode_lines'].copy()
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            observation = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            pieces_placed = 0
            lines_cleared = 0
            
            while not done and episode_length < 200:  # Max 200 steps per episode
                # If using RND, enforce greedy selection
                if self.enable_rnd:
                    saved_eps = self.agent.epsilon
                    self.agent.epsilon = 0.0
                action = self.agent.select_action(observation, training=True, env=self.env)
                if self.enable_rnd:
                    self.agent.epsilon = saved_eps
                
                # Execute action
                next_observation, reward, done, info = self.env.step(action)
                
                # Add intrinsic RND reward and train predictor network
                if self.enable_rnd:
                    state_tensor = torch.FloatTensor(next_observation).unsqueeze(0).to(self.device)
                    intrinsic_reward = self.rnd_network.intrinsic_reward(state_tensor).item()
                    reward += self.rnd_reward_scale * intrinsic_reward
                    # Train RND predictor
                    pred, target = self.rnd_network(state_tensor)
                    rnd_loss = F.mse_loss(pred, target.detach())
                    self.rnd_optimizer.zero_grad()
                    rnd_loss.backward()
                    self.rnd_optimizer.step()
                # Visualization per step only on specified episodes
                if self.show_visualization and episode % self.visualization_interval == 0:
                    self.env.render()
                    time.sleep(self.render_delay)
                
                # Track statistics
                if info.get('piece_placed', False):
                    pieces_placed += 1
                
                lines_this_step = info.get('lines_cleared', 0)
                if lines_this_step > 0:
                    lines_cleared += lines_this_step
                
                # Store experience
                self.agent.store_experience(observation, action, reward, next_observation, done)
                
                # Train if we have enough experiences
                if not self.update_per_episode and len(self.agent.memory) >= self.agent.batch_size:
                    train_result = self.agent.train_batch()
                    if train_result and 'loss' in train_result:
                        episode_losses.append(train_result['loss'])
                
                observation = next_observation
                episode_reward += reward
                episode_length += 1
            
            # Per-episode training if requested
            if self.update_per_episode:
                for _ in range(episode_length):
                    if len(self.agent.memory) >= self.agent.batch_size:
                        train_result = self.agent.train_batch()
                        if train_result and 'loss' in train_result:
                            episode_losses.append(train_result['loss'])
            
            # Update epsilon
            self.agent.update_epsilon()
            
            # Update target network
            if episode % self.agent.target_update_freq == 0:
                self.agent.update_target_network()
            
            # Store episode statistics
            episode_rewards.append(episode_reward)
            episode_pieces.append(pieces_placed)
            episode_lines.append(lines_cleared)
            
            # Logging
            if episode % 10 == 0 or episode < 10:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_pieces = np.mean(episode_pieces[-10:])
                avg_lines = np.mean(episode_lines[-10:])
                avg_loss = np.mean(episode_losses[-100:]) if episode_losses else 0.0
                
                print(f"Episode {episode:4d}: "
                      f"Reward={episode_reward:6.1f} (avg={avg_reward:6.1f}), "
                      f"Pieces={pieces_placed:2d} (avg={avg_pieces:4.1f}), "
                      f"Lines={lines_cleared:2d} (avg={avg_lines:4.1f}), "
                      f"Loss={avg_loss:.4f}, "
                      f"Epsilon={self.agent.epsilon:.3f}")
            
            # Save checkpoint with training history
            actual_episode = self.start_episode + episode
            if episode > 0 and episode % save_interval == 0:
                self.save_checkpoint(f"checkpoints/redesigned_agent_episode_{actual_episode}.pt", episode_rewards, episode_pieces, episode_lines, episode_losses)
        
        total_time = time.time() - start_time
        
        # Final statistics
        final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        final_avg_pieces = np.mean(episode_pieces[-100:]) if len(episode_pieces) >= 100 else np.mean(episode_pieces)
        final_avg_lines = np.mean(episode_lines[-100:]) if len(episode_lines) >= 100 else np.mean(episode_lines)
        
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Total time: {total_time:.1f}s")
        print(f"Final average reward (last 100): {final_avg_reward:.1f}")
        print(f"Final average pieces (last 100): {final_avg_pieces:.1f}")
        print(f"Final average lines (last 100): {final_avg_lines:.1f}")
        print(f"Total episodes: {num_episodes}")
        
        # Save final checkpoint
        self.save_checkpoint("checkpoints/redesigned_agent_final.pt")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_pieces': episode_pieces,
            'episode_lines': episode_lines,
            'episode_losses': episode_losses,
            'final_avg_reward': final_avg_reward,
            'final_avg_pieces': final_avg_pieces,
            'final_avg_lines': final_avg_lines,
            'total_time': total_time
        }
    
    def save_checkpoint(self, filepath: str, episode_rewards=None, episode_pieces=None, episode_lines=None, episode_losses=None):
        """Save agent checkpoint with training history"""
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
                    'episode_losses': episode_losses or [],
                    'total_training_time': self.total_training_time + time.time(),
                    'start_episode': self.start_episode
                }
                
                history_file = filepath.replace('.pt', '_history.json')
                with open(history_file, 'w') as f:
                    json.dump(history_data, f, indent=2)
            
            print(f"Checkpoint saved: {filepath}")
        except Exception as e:
            print(f"Warning: Could not save checkpoint {filepath}: {e}")
    
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint"""
        try:
            self.agent.load_checkpoint(filepath)
            print(f"Checkpoint loaded: {filepath}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint {filepath}: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Redesigned Locked State DQN Agent')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train (default: 1000)')
    parser.add_argument('--save-interval', type=int, default=100, help='Save checkpoint every N episodes (default: 100)')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh training (ignore checkpoints)')
    
    # Agent parameters
    parser.add_argument('--learning-rate', type=float, default=0.00005, help='Learning rate (default: 0.00005)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon (default: 1.0)')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final epsilon (default: 0.01)')
    parser.add_argument('--epsilon-decay-steps', type=int, default=50000, help='Epsilon decay steps (default: 50000)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--memory-size', type=int, default=100000, help='Memory buffer size (default: 100000)')
    parser.add_argument('--target-update-freq', type=int, default=1000, help='Target network update frequency (default: 1000)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use (default: auto)')
    
    # RND parameters
    parser.add_argument('--enable-rnd', action='store_true', help='Enable Random Network Distillation for exploration')
    parser.add_argument('--rnd-reward-scale', type=float, default=0.1, help='RND reward scaling factor (default: 0.1)')
    
    # Visualization parameters
    parser.add_argument('--show-visualization', action='store_true', help='Show environment visualization during training')
    parser.add_argument('--visualization-interval', type=int, default=10, help='Show environment visualization every N episodes (default: 10)')
    parser.add_argument('--render-delay', type=float, default=0.05, help='Delay in seconds between renders during visualization')
    parser.add_argument('--update-per-episode', action='store_true', help='Update model after each episode instead of per step')
    
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
    print(f"Training configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg.replace('_', '-')}: {value}")
    
    # Initialize trainer with custom parameters
    trainer = RedesignedAgentTrainer(
        device=device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_freq=args.target_update_freq,
        memory_size=args.memory_size,
        enable_rnd=args.enable_rnd,
        rnd_reward_scale=args.rnd_reward_scale,
        show_visualization=args.show_visualization,
        visualization_interval=args.visualization_interval,
        render_delay=args.render_delay,
        update_per_episode=args.update_per_episode
    )
    
    # Train the agent
    results = trainer.train(
        num_episodes=args.episodes, 
        save_interval=args.save_interval,
        resume=not args.no_resume
    )
    
    print("\nTraining completed successfully!")
    if 'final_avg_reward' in results:
        print(f"Final performance: {results['final_avg_reward']:.1f} reward, "
              f"{results['final_avg_pieces']:.1f} pieces, "
              f"{results['final_avg_lines']:.1f} lines")

if __name__ == "__main__":
    main() 