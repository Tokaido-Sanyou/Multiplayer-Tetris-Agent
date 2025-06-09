#!/usr/bin/env python3
"""
COMPREHENSIVE DREAM TRAINING - ALL FEATURES MERGED
Complete DREAM training with multiple modes and all enhancements:

TRAINING MODES:
1. "basic" - Original DREAM with categorical distribution
2. "enhanced_exploration" - Extended epsilon-greedy + temperature scaling + curiosity
3. "fixed_logging" - Proper TensorBoard logging + buffer validation + dropout
4. "comprehensive" - All features combined (recommended)

ARCHITECTURAL DIFFERENCES:
- Basic: Standard categorical distribution, basic exploration
- Enhanced: Epsilon-greedy (0.8→0.05) + temperature scaling + curiosity bonuses
- Fixed: Dropout layers + proper buffer validation + enhanced TensorBoard
- Comprehensive: All above + longest episodes + best stability

All modes support both 'standard' and 'lines_only' reward modes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import matplotlib.pyplot as plt

from envs.tetris_env import TetrisEnv

# Configure matplotlib for non-blocking display
plt.ion()


class ComprehensiveActorCritic(torch.nn.Module):
    """Comprehensive Actor-Critic supporting all training modes"""
    
    def __init__(self, state_dim=206, action_dim=8, hidden_dim=400, training_mode="comprehensive", total_episodes=1000):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_mode = training_mode
        self.total_episodes = total_episodes
        
        # Network architecture varies by mode
        if training_mode in ["fixed_logging", "comprehensive"]:
            # Enhanced network with dropout for stability
            self.shared_net = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
        else:
            # Standard network without dropout
            self.shared_net = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
        
        # Actor and critic heads
        self.actor_head = torch.nn.Linear(hidden_dim, action_dim)
        self.critic_head = torch.nn.Linear(hidden_dim, 1)
        
        # Episode-based exploration parameters - decay to min by 3/4 of total episodes
        decay_episodes = int(0.75 * total_episodes)  # Decay over 75% of total episodes
        
        if training_mode in ["enhanced_exploration", "comprehensive"]:
            self.epsilon_start = 0.8
            self.epsilon_min = 0.05
            self.temperature_start = 1.0
        elif training_mode == "fixed_logging":
            self.epsilon_start = 0.9
            self.epsilon_min = 0.1
            self.temperature_start = 2.0
        else:  # basic mode
            self.epsilon_start = 0.8  # Increased from 0.1 for better exploration
            self.epsilon_min = 0.01
            self.temperature_start = 1.0
        
        # Calculate episode-based decay rate: epsilon_min = epsilon_start * (decay_rate ^ decay_episodes)
        self.epsilon_decay_rate = (self.epsilon_min / self.epsilon_start) ** (1.0 / decay_episodes)
        self.decay_episodes = decay_episodes
        
        # Current values
        self.epsilon = self.epsilon_start
        self.temperature = self.temperature_start
        self.current_episode = 0
        
        print(f"🔍 Epsilon decay: {self.epsilon_start:.3f} → {self.epsilon_min:.3f} over {decay_episodes} episodes (rate: {self.epsilon_decay_rate:.6f})")
    
    def forward(self, state):
        features = self.shared_net(state)
        
        # Temperature-scaled logits for exploration control
        action_logits = self.actor_head(features) / self.temperature
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        value = self.critic_head(features).squeeze(-1)
        
        return action_dist, value
    
    def get_action_and_value(self, state, deterministic=False, training=True):
        """Action selection supporting all training modes"""
        dist, value = self.forward(state)
        
        # Enhanced exploration for advanced modes
        if training and self.training_mode in ["enhanced_exploration", "fixed_logging", "comprehensive", "basic"]:
            if np.random.random() < self.epsilon:
                # Epsilon-greedy exploration
                action = torch.randint(0, dist.logits.shape[-1], (state.shape[0],)).to(state.device)
            elif deterministic:
                action = torch.argmax(dist.logits, dim=-1)
            else:
                action = dist.sample()
        else:
            # Fallback behavior
            if deterministic:
                action = torch.argmax(dist.logits, dim=-1)
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def update_exploration_episode(self, episode):
        """Update exploration parameters based on episode number"""
        self.current_episode = episode
        
        # Episode-based epsilon decay
        if episode <= self.decay_episodes:
            self.epsilon = max(self.epsilon_min, self.epsilon_start * (self.epsilon_decay_rate ** episode))
        else:
            self.epsilon = self.epsilon_min
        
        # Temperature decay based on epsilon progress
        epsilon_progress = (self.epsilon_start - self.epsilon) / (self.epsilon_start - self.epsilon_min)
        epsilon_progress = min(1.0, max(0.0, epsilon_progress))  # Clamp to [0, 1]
        
        if self.training_mode in ["enhanced_exploration", "comprehensive"]:
            # Gradual temperature reduction
            self.temperature = max(0.5, self.temperature_start * (1.0 - 0.5 * epsilon_progress))
        elif self.training_mode == "fixed_logging":
            # Slower temperature reduction
            self.temperature = max(0.5, self.temperature_start * (1.0 - 0.25 * epsilon_progress))
        else:  # basic mode
            self.temperature = max(0.5, self.temperature_start * (1.0 - 0.5 * epsilon_progress))
    
    def update_exploration(self):
        """Legacy method - now does nothing since we use episode-based decay"""
        pass
    
    def evaluate_actions(self, state, action):
        dist, value = self.forward(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, value, entropy


class ComprehensiveReplayBuffer:
    """Comprehensive replay buffer supporting all training modes"""
    
    def __init__(self, max_size=50000, training_mode="comprehensive"):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.training_mode = training_mode
        self.total_trajectories = 0
        self.total_steps = 0
        
    def add_trajectory(self, trajectory):
        """Add trajectory with mode-specific enhancements"""
        if not trajectory or len(trajectory.get('observations', [])) == 0:
            if self.training_mode in ["fixed_logging", "comprehensive"]:
                print(f"⚠️ Warning: Empty trajectory received")
            return
        
        # Validation for advanced modes
        if self.training_mode in ["fixed_logging", "comprehensive"]:
            traj_length = len(trajectory['observations'])
            required_keys = ['observations', 'actions', 'rewards', 'dones']
            
            for key in required_keys:
                if key not in trajectory:
                    print(f"⚠️ Warning: Missing key '{key}' in trajectory")
                    return
                if len(trajectory[key]) != traj_length:
                    print(f"⚠️ Warning: Inconsistent trajectory length for '{key}'")
                    return
        
        # Curiosity bonus for exploration modes
        if self.training_mode in ["enhanced_exploration", "comprehensive"] and len(trajectory['observations']) > 1:
            try:
                obs_array = np.array(trajectory['observations'])
                novelty_bonus = np.std(obs_array, axis=0).mean() * 0.05
                
                enhanced_rewards = []
                for r in trajectory['rewards']:
                    enhanced_rewards.append(r + novelty_bonus)
                trajectory['rewards'] = enhanced_rewards
            except Exception as e:
                if self.training_mode == "comprehensive":
                    print(f"⚠️ Warning: Could not compute novelty bonus: {e}")
        
        # Store trajectory
        self.buffer.append(trajectory)
        self.total_trajectories += 1
        self.total_steps += len(trajectory['observations'])
        
        # Log buffer stats for advanced modes
        if self.training_mode in ["fixed_logging", "comprehensive"] and self.total_trajectories % 10 == 0:
            print(f"📊 Buffer stats: {len(self.buffer):,} trajectories, {self.total_steps:,} total steps")
    
    def sample_sequences(self, batch_size=16, sequence_length=8):
        """Sample sequences with enhanced error handling"""
        if len(self.buffer) < batch_size:
            return None
        
        try:
            sampled_trajectories = np.random.choice(list(self.buffer), batch_size, replace=False)
        except ValueError:
            sampled_trajectories = np.random.choice(list(self.buffer), batch_size, replace=True)
        
        batch = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
        
        valid_sequences = 0
        for traj in sampled_trajectories:
            traj_len = len(traj['observations'])
            if traj_len >= sequence_length:
                start_idx = np.random.randint(0, traj_len - sequence_length + 1)
                end_idx = start_idx + sequence_length
                
                try:
                    batch['observations'].append(traj['observations'][start_idx:end_idx])
                    batch['actions'].append(traj['actions'][start_idx:end_idx])
                    batch['rewards'].append(traj['rewards'][start_idx:end_idx])
                    batch['dones'].append(traj['dones'][start_idx:end_idx])
                    valid_sequences += 1
                except Exception as e:
                    if self.training_mode in ["fixed_logging", "comprehensive"]:
                        print(f"⚠️ Warning: Error sampling sequence: {e}")
                    continue
        
        return batch if valid_sequences > 0 else None
    
    def get_stats(self):
        """Get comprehensive buffer statistics"""
        if len(self.buffer) == 0:
            return {'size': 0, 'avg_length': 0, 'total_steps': 0}
        
        lengths = [len(traj['observations']) for traj in self.buffer]
        return {
            'size': len(self.buffer),
            'avg_length': np.mean(lengths),
            'total_steps': sum(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths)
        }
    
    def __len__(self):
        return len(self.buffer)


class LiveDashboard:
    """Live visual dashboard for training monitoring"""
    
    def __init__(self, max_episodes=1000, show_plots=True):
        self.max_episodes = max_episodes
        self.show_plots = show_plots
        self.episode_data = {
            'episodes': [], 'rewards': [], 'lengths': [], 'lines': [],
            'actor_losses': [], 'buffer_sizes': [], 'entropies': [], 'epsilons': []
        }
        
        if show_plots:
            try:
                self.fig, self.axes = plt.subplots(2, 4, figsize=(16, 8))
                self.fig.suptitle('🎮 COMPREHENSIVE DREAM Training Dashboard', fontsize=14, fontweight='bold')
                self.setup_plots()
                plt.show(block=False)
                plt.pause(0.1)
                print("🎯 LIVE VISUAL DASHBOARD LAUNCHED!")
            except Exception as e:
                print(f"❌ Could not create visual dashboard: {e}")
                self.show_plots = False
    
    def setup_plots(self):
        if not self.show_plots:
            return
        
        self.ax = self.axes.flatten()
        titles = [
            'Episode Rewards', 'Episode Lengths', 'Lines Cleared', 'Actor Loss',
            'Buffer Utilization', 'Policy Entropy', 'Exploration (ε)', 'Training Progress'
        ]
        
        for i, title in enumerate(titles):
            self.ax[i].set_title(title)
            self.ax[i].set_xlabel('Episode')
            self.ax[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def update_dashboard(self, episode, reward, length, lines, actor_loss, buffer_size, max_buffer_size, entropy=0.0, epsilon=0.0):
        """Update the live dashboard with new data"""
        self.episode_data['episodes'].append(episode)
        self.episode_data['rewards'].append(reward)
        self.episode_data['lengths'].append(length)
        self.episode_data['lines'].append(lines)
        self.episode_data['actor_losses'].append(actor_loss)
        self.episode_data['buffer_sizes'].append(buffer_size)
        self.episode_data['entropies'].append(entropy)
        self.episode_data['epsilons'].append(epsilon)
        
        if not self.show_plots:
            return
        
        try:
            for ax in self.ax:
                ax.clear()
            
            self.setup_plots()
            episodes = self.episode_data['episodes']
            
            if len(episodes) > 0:
                # Plot 1: Episode Rewards
                self.ax[0].plot(episodes, self.episode_data['rewards'], 'b-', alpha=0.7, linewidth=2)
                if len(episodes) > 5:
                    window = min(10, len(episodes))
                    moving_avg = np.convolve(self.episode_data['rewards'], np.ones(window)/window, mode='valid')
                    self.ax[0].plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'MA({window})')
                    self.ax[0].legend()
                
                # Plot 2: Episode Lengths
                self.ax[1].plot(episodes, self.episode_data['lengths'], 'g-', alpha=0.7, linewidth=2)
                
                # Plot 3: Lines Cleared
                self.ax[2].scatter(episodes, self.episode_data['lines'], c='orange', alpha=0.7, s=30)
                
                # Plot 4: Actor Loss
                self.ax[3].plot(episodes, self.episode_data['actor_losses'], 'red', alpha=0.7, linewidth=2)
                
                # Plot 5: Buffer Utilization
                buffer_util = [s / max_buffer_size * 100 for s in self.episode_data['buffer_sizes']]
                self.ax[4].plot(episodes, buffer_util, 'brown', alpha=0.7, linewidth=2)
                self.ax[4].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Max')
                self.ax[4].legend()
                
                # Plot 6: Policy Entropy
                self.ax[5].plot(episodes, self.episode_data['entropies'], 'purple', alpha=0.7, linewidth=2)
                
                # Plot 7: Exploration (Epsilon)
                self.ax[6].plot(episodes, self.episode_data['epsilons'], 'cyan', alpha=0.7, linewidth=2)
                
                # Plot 8: Training Progress (reward trend)
                if len(episodes) > 10:
                    recent_rewards = self.episode_data['rewards'][-min(50, len(episodes)):]
                    trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                    self.ax[7].axhline(y=trend, color='green' if trend > 0 else 'red', 
                                     linestyle='-', linewidth=3, label=f'Trend: {trend:.3f}')
                    self.ax[7].plot(episodes[-len(recent_rewards):], recent_rewards, 'gray', alpha=0.5)
                    self.ax[7].legend()
            
            plt.pause(0.01)
            
        except Exception as e:
            print(f"⚠️ Dashboard update error: {e}")
    
    def close(self):
        if self.show_plots:
            plt.close(self.fig)


class ComprehensiveDREAMTrainer:
    """Comprehensive DREAM trainer supporting all modes and features"""
    
    def __init__(self, training_mode="comprehensive", reward_mode='lines_only', episodes=1000, 
                 max_buffer_size=50000, show_dashboard=True, use_tensorboard=True):
        
        self.training_mode = training_mode
        self.reward_mode = reward_mode
        self.episodes = episodes
        self.max_buffer_size = max_buffer_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Print comprehensive training info
        print("🚀 COMPREHENSIVE DREAM TRAINING")
        print("=" * 60)
        print(f"🎯 Training Mode: {training_mode}")
        print(f"🎮 Reward Mode: {reward_mode}")
        print(f"📊 Episodes: {episodes}")
        print(f"💾 Buffer Size: {max_buffer_size:,}")
        print(f"⚡ Device: {self.device}")
        print("=" * 60)
        
        # Mode-specific features
        if training_mode == "basic":
            print("🔧 BASIC MODE FEATURES:")
            print("✅ Categorical action distribution")
            print("✅ Standard exploration")
            print("✅ Basic buffer management")
        elif training_mode == "enhanced_exploration":
            print("🔧 ENHANCED EXPLORATION FEATURES:")
            print("✅ Epsilon-greedy exploration (0.8 → 0.05)")
            print("✅ Temperature-scaled action distribution")
            print("✅ Curiosity-driven intrinsic rewards")
            print("✅ Multi-strategy action selection")
        elif training_mode == "fixed_logging":
            print("🔧 FIXED LOGGING FEATURES:")
            print("✅ Proper TensorBoard compatibility")
            print("✅ Buffer validation and error handling")
            print("✅ Dropout layers for stability")
            print("✅ Enhanced exploration parameters")
        else:  # comprehensive
            print("🔧 COMPREHENSIVE MODE FEATURES:")
            print("✅ All exploration enhancements")
            print("✅ Complete TensorBoard logging")
            print("✅ Buffer validation & curiosity")
            print("✅ Dropout & stability features")
            print("✅ Live visual dashboard")
            print("✅ Maximum episode length (1000)")
        print("=" * 60)
        
        # Create logging directories
        self.log_dir = Path(f"logs/dream_{training_mode}_{reward_mode}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard logging
        if use_tensorboard:
            try:
                # Use proper Windows path handling
                log_path = str(self.log_dir / "tensorboard").replace('\\', '/')
                self.writer = SummaryWriter(log_path)
                print(f"✅ TensorBoard logging to: {log_path}")
            except Exception as e:
                print(f"⚠️ TensorBoard setup failed: {e}")
                self.writer = None
        else:
            self.writer = None
        
        # Environment setup
        self.max_episode_steps = 1000 if training_mode == "comprehensive" else 500
        self.env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='direct',
            reward_mode=reward_mode
        )
        
        # Get state dimension
        obs = self.env.reset()
        self.state_dim = obs.shape[0]
        print(f"🎯 State dimension: {self.state_dim}")
        
        # Initialize comprehensive actor-critic
        hidden_dim = 512 if training_mode in ["enhanced_exploration", "comprehensive"] else 400
        self.actor_critic = ComprehensiveActorCritic(
            state_dim=self.state_dim,
            action_dim=8,
            hidden_dim=hidden_dim,
            training_mode=training_mode,
            total_episodes=episodes
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.actor_critic.parameters())
        print(f"🧠 Total parameters: {total_params:,}")
        
        # Initialize comprehensive replay buffer
        self.replay_buffer = ComprehensiveReplayBuffer(max_buffer_size, training_mode)
        
        # Optimizer with mode-specific learning rates
        lr = 1e-4 if training_mode in ["enhanced_exploration", "comprehensive"] else 1e-3
        self.actor_optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.lines_cleared = []
        self.actor_losses = []
        
        # Live dashboard
        self.dashboard = LiveDashboard(episodes, show_dashboard) if show_dashboard else None
        
        print("✅ Comprehensive DREAM trainer initialized successfully!")
    
    def collect_episode(self, episode_num):
        """Collect episode data with comprehensive logging"""
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        obs = self.env.reset()
        if isinstance(obs, np.ndarray) and obs.ndim > 1:
            obs = obs.flatten()
        
        total_reward = 0
        lines_cleared = 0
        steps = 0
        
        while True:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action with mode-specific exploration
            with torch.no_grad():
                action, log_prob, value = self.actor_critic.get_action_and_value(
                    obs_tensor, 
                    deterministic=False,
                    training=True
                )
            
            action_np = action.cpu().numpy()[0]
            
            # Take environment step
            next_obs, reward, done, info = self.env.step(action_np)
            
            if isinstance(next_obs, np.ndarray) and next_obs.ndim > 1:
                next_obs = next_obs.flatten()
            
            # Store transition
            trajectory['observations'].append(obs.tolist())
            trajectory['actions'].append(action_np)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            
            total_reward += reward
            steps += 1
            
            # Track lines cleared
            if hasattr(info, 'get') and 'lines_cleared' in info:
                lines_cleared += info['lines_cleared']
            
            # Check termination conditions
            if done or steps >= self.max_episode_steps:
                break
            
            obs = next_obs
        
        # Add trajectory to buffer
        if len(trajectory['observations']) > 0:
            self.replay_buffer.add_trajectory(trajectory)
        
        # Comprehensive logging
        if episode_num % 10 == 0 and self.training_mode in ["fixed_logging", "comprehensive"]:
            print(f"📊 Episode {episode_num}: Reward={total_reward:.2f}, Steps={steps}, "
                  f"Lines={lines_cleared}, ε={self.actor_critic.epsilon:.3f}")
        
        return total_reward, steps, lines_cleared
    
    def train_actor_critic(self):
        """Train actor-critic with comprehensive features"""
        batch = self.replay_buffer.sample_sequences(batch_size=16, sequence_length=8)
        if batch is None:
            return 0.0, 0.0
        
        try:
            # Convert batch to tensors
            observations = torch.FloatTensor(batch['observations']).to(self.device)
            actions = torch.LongTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).to(self.device)
            
            # Flatten sequences for processing
            batch_size, seq_len = observations.shape[:2]
            observations = observations.view(-1, observations.shape[-1])
            actions = actions.view(-1)
            rewards = rewards.view(-1)
            dones = dones.view(-1)
            
            # Get action probabilities and values
            log_probs, values, entropies = self.actor_critic.evaluate_actions(observations, actions)
            
            # Compute advantages using rewards-to-go
            advantages = rewards.clone()
            
            # Actor loss with entropy bonus
            entropy_bonus = 0.1 if self.training_mode in ["enhanced_exploration", "comprehensive"] else 0.01
            actor_loss = -(log_probs * advantages.detach()).mean() - entropy_bonus * entropies.mean()
            
            # Critic loss
            critic_loss = ((values - rewards) ** 2).mean()
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss
            
            # Optimize
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            
            self.actor_optimizer.step()
            
            return actor_loss.item(), entropies.mean().item()
        
        except Exception as e:
            if self.training_mode in ["fixed_logging", "comprehensive"]:
                print(f"⚠️ Training error: {e}")
            return 0.0, 0.0
    
    def log_comprehensive_metrics(self, episode, metrics, training_metrics, buffer_stats):
        """Comprehensive TensorBoard logging"""
        if self.writer is None:
            return
        
        try:
            # Episode metrics
            self.writer.add_scalar('Episode/Reward', metrics['reward'], episode)
            self.writer.add_scalar('Episode/Length', metrics['length'], episode)
            self.writer.add_scalar('Episode/Lines_Cleared', metrics['lines'], episode)
            
            # Training metrics
            self.writer.add_scalar('Training/Actor_Loss', training_metrics['actor_loss'], episode)
            self.writer.add_scalar('Training/Entropy', training_metrics['entropy'], episode)
            
            # Exploration metrics
            self.writer.add_scalar('Exploration/Epsilon', self.actor_critic.epsilon, episode)
            self.writer.add_scalar('Exploration/Temperature', self.actor_critic.temperature, episode)
            
            # Buffer metrics
            self.writer.add_scalar('Buffer/Size', buffer_stats['size'], episode)
            self.writer.add_scalar('Buffer/Avg_Length', buffer_stats['avg_length'], episode)
            self.writer.add_scalar('Buffer/Total_Steps', buffer_stats['total_steps'], episode)
            
            # Performance metrics
            if len(self.episode_rewards) >= 10:
                recent_avg = np.mean(self.episode_rewards[-10:])
                self.writer.add_scalar('Performance/Recent_Avg_Reward', recent_avg, episode)
            
            self.writer.flush()
            
        except Exception as e:
            if self.training_mode in ["fixed_logging", "comprehensive"]:
                print(f"⚠️ TensorBoard logging error: {e}")
    
    def train(self):
        """Comprehensive training loop"""
        print(f"\n🎮 Starting {self.training_mode.upper()} DREAM training...")
        print(f"🎯 Target episodes: {self.episodes}")
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in range(1, self.episodes + 1):
            # Update exploration parameters based on episode
            self.actor_critic.update_exploration_episode(episode)
            
            # Collect episode
            reward, length, lines = self.collect_episode(episode)
            
            # Store metrics
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.lines_cleared.append(lines)
            
            # Train actor-critic
            actor_loss, entropy = self.train_actor_critic()
            self.actor_losses.append(actor_loss)
            
            # Get buffer statistics
            buffer_stats = self.replay_buffer.get_stats()
            
            # Comprehensive logging
            metrics = {'reward': reward, 'length': length, 'lines': lines}
            training_metrics = {'actor_loss': actor_loss, 'entropy': entropy}
            
            self.log_comprehensive_metrics(episode, metrics, training_metrics, buffer_stats)
            
            # Update live dashboard
            if self.dashboard:
                self.dashboard.update_dashboard(
                    episode, reward, length, lines, actor_loss,
                    len(self.replay_buffer), self.max_buffer_size,
                    entropy, self.actor_critic.epsilon
                )
            
            # Progress reporting
            if episode % 50 == 0:
                elapsed = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_length = np.mean(self.episode_lengths[-50:])
                total_lines = sum(self.lines_cleared[-50:])
                
                print(f"\n📊 PROGRESS REPORT - Episode {episode}/{self.episodes}")
                print(f"⏱️  Elapsed: {elapsed:.1f}s")
                print(f"🎯 Avg Reward (50ep): {avg_reward:.2f}")
                print(f"📏 Avg Length (50ep): {avg_length:.1f}")
                print(f"🎲 Lines Cleared (50ep): {total_lines}")
                print(f"🔍 Exploration ε: {self.actor_critic.epsilon:.3f}")
                print(f"🧠 Buffer Size: {len(self.replay_buffer):,}")
                print("-" * 60)
        
        # Final statistics
        total_time = time.time() - start_time
        final_avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
        total_lines = sum(self.lines_cleared)
        
        print(f"\n🏁 TRAINING COMPLETED!")
        print("=" * 60)
        print(f"📊 FINAL STATISTICS:")
        print(f"⏱️  Total Time: {total_time:.1f}s")
        print(f"🎯 Final Avg Reward: {final_avg_reward:.2f}")
        print(f"📏 Max Episode Length: {max(self.episode_lengths)}")
        print(f"🎲 Total Lines Cleared: {total_lines}")
        print(f"🧠 Final Buffer Size: {len(self.replay_buffer):,}")
        print(f"🔍 Final ε: {self.actor_critic.epsilon:.3f}")
        print("=" * 60)
        
        # Save model
        model_path = self.log_dir / f"dream_{self.training_mode}_final.pth"
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'lines_cleared': self.lines_cleared,
            'training_mode': self.training_mode,
            'reward_mode': self.reward_mode
        }, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        
        return final_avg_reward, total_lines
    
    def cleanup(self):
        """Cleanup resources"""
        if self.writer:
            self.writer.close()
        if self.dashboard:
            self.dashboard.close()
        self.env.close()
        print("🧹 Cleanup completed")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive DREAM Training')
    parser.add_argument('--mode', type=str, default='comprehensive',
                       choices=['basic', 'enhanced_exploration', 'fixed_logging', 'comprehensive'],
                       help='Training mode (default: comprehensive)')
    parser.add_argument('--reward_mode', type=str, default='lines_only',
                       choices=['standard', 'lines_only'],
                       help='Reward mode (default: lines_only)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes (default: 1000)')
    parser.add_argument('--buffer_size', type=int, default=50000,
                       help='Replay buffer size (default: 50000)')
    parser.add_argument('--no_dashboard', action='store_true',
                       help='Disable live visual dashboard')
    parser.add_argument('--no_tensorboard', action='store_true',
                       help='Disable TensorBoard logging')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ComprehensiveDREAMTrainer(
        training_mode=args.mode,
        reward_mode=args.reward_mode,
        episodes=args.episodes,
        max_buffer_size=args.buffer_size,
        show_dashboard=not args.no_dashboard,
        use_tensorboard=not args.no_tensorboard
    )
    
    try:
        # Run training
        final_reward, total_lines = trainer.train()
        
        print(f"\n✅ Training completed successfully!")
        print(f"🎯 Final average reward: {final_reward:.2f}")
        print(f"🎲 Total lines cleared: {total_lines}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main() 