"""
Optimized DREAM Implementation for Tetris - Addressing All Identified Issues

Key Fixes:
1. Proper reward system that encourages learning
2. Better exploration strategy
3. Improved training frequency and batch processing
4. Streamlined architecture for efficiency
5. Fixed agent demonstration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.tetris_env import TetrisEnv
import time
import matplotlib.pyplot as plt
from collections import deque
import random

# ============================================================================
# OPTIMIZED WORLD MODEL - SMALLER BUT MORE EFFECTIVE
# ============================================================================

class OptimizedTetrisWorldModel(nn.Module):
    """Optimized world model with streamlined architecture"""
    
    def __init__(self, obs_dim=425, action_dim=8, hidden_dim=256, state_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Streamlined encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Simplified RNN
        self.rnn = nn.GRU(state_dim + action_dim, state_dim, batch_first=True)
        
        # Streamlined prediction heads
        self.obs_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.continue_head = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def encode(self, observations):
        """Encode observations to state representation"""
        return self.obs_encoder(observations)
    
    def forward(self, observations, actions):
        """Forward pass with proper tensor handling"""
        # Ensure observations have batch dimension
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = observations.shape[0]
        
        # Encode observations
        states = self.encode(observations)
        
        # Ensure actions have proper shape
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        if actions.shape[0] != batch_size:
            if actions.shape[0] == 1 and batch_size > 1:
                actions = actions.repeat(batch_size)
            elif batch_size == 1:
                actions = actions[:1]
        
        # One-hot encode actions
        actions_one_hot = F.one_hot(actions.long(), self.action_dim).float()
        
        # Combine states and actions
        combined = torch.cat([states, actions_one_hot], dim=-1)
        rnn_input = combined.unsqueeze(1)
        
        # Initial hidden state
        h0 = torch.zeros(1, batch_size, self.state_dim, device=observations.device)
        
        # Run through RNN
        rnn_output, _ = self.rnn(rnn_input, h0)
        rnn_output = rnn_output.squeeze(1)
        
        # Make predictions
        next_observations = self.obs_decoder(rnn_output)
        rewards = self.reward_head(rnn_output).squeeze(-1)
        continues = torch.sigmoid(self.continue_head(rnn_output)).squeeze(-1)
        
        # Handle single input case
        if single_input:
            next_observations = next_observations.squeeze(0)
            rewards = rewards.squeeze(0) if rewards.dim() > 0 else rewards
            continues = continues.squeeze(0) if continues.dim() > 0 else continues
            rnn_output = rnn_output.squeeze(0)
        
        return {
            'next_observations': next_observations,
            'rewards': rewards,
            'continues': continues,
            'states': rnn_output
        }

# ============================================================================
# OPTIMIZED ACTOR-CRITIC - SIMPLIFIED BUT EFFECTIVE
# ============================================================================

class OptimizedTetrisActorCritic(nn.Module):
    """Optimized Actor-Critic with better action selection"""
    
    def __init__(self, obs_dim=425, action_dim=8, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
        # Policy head
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Value head
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, observations, temperature=1.0):
        """Forward pass with temperature scaling"""
        features = self.backbone(observations)
        
        # Policy with temperature
        logits = self.actor(features) / temperature
        action_probs = F.softmax(logits, dim=-1)
        
        # Value
        values = self.critic(features)
        
        return action_probs, values
    
    def get_action_and_value(self, observation, epsilon=0.1, temperature=1.0):
        """Get action with epsilon-greedy exploration"""
        with torch.no_grad():
            if random.random() < epsilon:
                # Random exploration
                action = torch.randint(0, self.action_dim, (1,)).item()
                action_probs, values = self.forward(observation.unsqueeze(0), temperature)
                log_prob = torch.log(action_probs[0, action] + 1e-8)
            else:
                # Policy action
                action_probs, values = self.forward(observation.unsqueeze(0), temperature)
                action = torch.multinomial(action_probs, 1).item()
                log_prob = torch.log(action_probs[0, action] + 1e-8)
        
        return action, log_prob, values[0]

# ============================================================================
# OPTIMIZED REPLAY BUFFER
# ============================================================================

class OptimizedReplayBuffer:
    """Optimized replay buffer with better sampling"""
    
    def __init__(self, capacity=5000, sequence_length=15):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.trajectories = deque(maxlen=capacity)
    
    def add_trajectory(self, trajectory):
        """Add trajectory to buffer"""
        if len(trajectory['observations']) > 1:
            self.trajectories.append(trajectory)
    
    def sample_batch(self, batch_size):
        """Sample batch of sequences"""
        if len(self.trajectories) < batch_size:
            return None
        
        sampled_trajectories = random.sample(list(self.trajectories), batch_size)
        
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_continues = []
        
        for traj in sampled_trajectories:
            traj_len = len(traj['observations']) - 1
            if traj_len < self.sequence_length:
                continue
            
            start_idx = random.randint(0, max(0, traj_len - self.sequence_length))
            end_idx = start_idx + self.sequence_length
            
            # Extract sequence
            obs_seq = traj['observations'][start_idx:end_idx]
            action_seq = traj['actions'][start_idx:end_idx-1]
            reward_seq = traj['rewards'][start_idx:end_idx-1]
            done_seq = traj['dones'][start_idx:end_idx-1]
            
            # Convert to tensors
            obs_tensor = torch.FloatTensor(obs_seq)
            action_tensor = torch.LongTensor(action_seq)
            reward_tensor = torch.FloatTensor(reward_seq)
            continue_tensor = torch.FloatTensor([0.0 if done else 1.0 for done in done_seq])
            
            batch_obs.append(obs_tensor)
            batch_actions.append(action_tensor)
            batch_rewards.append(reward_tensor)
            batch_continues.append(continue_tensor)
        
        if not batch_obs:
            return None
        
        return {
            'observations': torch.stack(batch_obs),
            'actions': torch.stack(batch_actions),
            'rewards': torch.stack(batch_rewards),
            'continues': torch.stack(batch_continues)
        }
    
    def __len__(self):
        return len(self.trajectories)

# ============================================================================
# OPTIMIZED DREAM TRAINER
# ============================================================================

class OptimizedDREAMTrainer:
    """Optimized DREAM trainer with all fixes applied"""
    
    def __init__(self, device='cuda', enable_visualization=False, batch_size=8):
        # Device setup
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        # Environment
        self.env = TetrisEnv(num_agents=1, headless=True, step_mode='action', action_mode='direct')
        
        # Optimized models
        self.world_model = OptimizedTetrisWorldModel().to(self.device)
        self.actor_critic = OptimizedTetrisActorCritic().to(self.device)
        
        # Better optimizers with higher learning rates
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.actor_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=5e-3, weight_decay=1e-5)
        
        # Learning rate schedulers
        self.world_scheduler = torch.optim.lr_scheduler.StepLR(self.world_optimizer, step_size=25, gamma=0.9)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=25, gamma=0.9)
        
        # Optimized replay buffer
        self.replay_buffer = OptimizedReplayBuffer(capacity=2000, sequence_length=12)
        
        # Training parameters
        self.batch_size = batch_size
        self.imagination_horizon = 20
        self.world_model_train_steps = 5  # Reduced for efficiency
        self.actor_train_steps = 3
        
        # Better exploration parameters
        self.epsilon = 0.7  # Higher initial exploration
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.1
        
        self.temperature = 2.5  # Higher initial temperature
        self.temperature_decay = 0.998
        self.temperature_min = 1.2
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.world_losses = []
        self.actor_losses = []
        
        print(f"Optimized DREAM Trainer initialized")
        print(f"World Model params: {sum(p.numel() for p in self.world_model.parameters()):,}")
        print(f"Actor-Critic params: {sum(p.numel() for p in self.actor_critic.parameters()):,}")
        print(f"Batch size: {batch_size}, Higher LR, Better exploration")
    
    def collect_trajectory(self):
        """Collect trajectory with proper termination"""
        obs = self.env.reset()
        trajectory = {
            'observations': [obs],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        max_steps = 1000  # Reasonable limit
        step = 0
        
        while step < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action, log_prob, value = self.actor_critic.get_action_and_value(
                obs_tensor, epsilon=self.epsilon, temperature=self.temperature
            )
            
            next_obs, reward, done, info = self.env.step(action)
            
            # Better reward shaping for Tetris
            shaped_reward = reward
            if reward == 0:  # Survival bonus
                shaped_reward = 0.1
            elif reward > 0:  # Line clearing reward
                shaped_reward = reward * 2.0  # Amplify positive rewards
            else:  # Death penalty
                shaped_reward = max(-10.0, reward * 0.1)  # Cap negative rewards
            
            trajectory['actions'].append(action)
            trajectory['rewards'].append(shaped_reward)
            trajectory['dones'].append(done)
            trajectory['observations'].append(next_obs)
            
            if done:
                break
            
            obs = next_obs
            step += 1
        
        return trajectory
    
    def train_world_model(self):
        """Train world model efficiently"""
        if len(self.replay_buffer) < self.batch_size:
            return {'world_loss': 0.0}
        
        total_loss = 0.0
        
        for _ in range(self.world_model_train_steps):
            batch = self.replay_buffer.sample_batch(self.batch_size)
            if batch is None:
                continue
            
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            rewards = batch['rewards'].to(self.device)
            continues = batch['continues'].to(self.device)
            
            batch_size, seq_len = observations.shape[:2]
            if seq_len <= 1:
                continue
            
            total_obs_loss = 0
            total_reward_loss = 0
            total_continue_loss = 0
            num_steps = 0
            
            for t in range(seq_len - 1):
                curr_obs = observations[:, t]
                curr_act = actions[:, t]
                next_obs = observations[:, t + 1]
                curr_rew = rewards[:, t]
                curr_cont = continues[:, t]
                
                predictions = self.world_model(curr_obs, curr_act)
                
                obs_loss = F.mse_loss(predictions['next_observations'], next_obs)
                reward_loss = F.mse_loss(predictions['rewards'], curr_rew)
                continue_loss = F.binary_cross_entropy(predictions['continues'], curr_cont)
                
                total_obs_loss += obs_loss
                total_reward_loss += reward_loss
                total_continue_loss += continue_loss
                num_steps += 1
            
            if num_steps > 0:
                avg_obs_loss = total_obs_loss / num_steps
                avg_reward_loss = total_reward_loss / num_steps
                avg_continue_loss = total_continue_loss / num_steps
                
                loss = avg_obs_loss + 2.0 * avg_reward_loss + avg_continue_loss
                
                self.world_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
                self.world_optimizer.step()
                
                total_loss += loss.item()
        
        return {'world_loss': total_loss / max(1, self.world_model_train_steps)}
    
    def generate_imagined_trajectories(self):
        """Generate imagined trajectories for training"""
        if len(self.replay_buffer) == 0:
            return []
        
        # Sample real observations as starting points
        real_trajectories = random.sample(list(self.replay_buffer.trajectories), 
                                        min(4, len(self.replay_buffer.trajectories)))
        
        imagined_trajectories = []
        
        for traj in real_trajectories:
            if len(traj['observations']) < 2:
                continue
            
            start_obs = random.choice(traj['observations'][:-1])
            start_obs_tensor = torch.FloatTensor(start_obs).unsqueeze(0).to(self.device)
            
            # Generate action sequence
            actions = []
            obs = start_obs_tensor
            
            for _ in range(self.imagination_horizon):
                with torch.no_grad():
                    action_probs, _ = self.actor_critic(obs, temperature=self.temperature)
                    action = torch.multinomial(action_probs, 1).item()
                    actions.append(action)
                
                # Predict next observation
                action_tensor = torch.tensor([action], device=self.device)
                predictions = self.world_model(obs.squeeze(0), action_tensor)
                obs = predictions['next_observations'].unsqueeze(0)
            
            # Create imagined trajectory
            imagined_trajectory = {
                'observations': [start_obs],
                'actions': actions,
                'rewards': [0.1] * len(actions),  # Optimistic rewards for imagination
                'dones': [False] * len(actions)
            }
            
            imagined_trajectories.append(imagined_trajectory)
        
        return imagined_trajectories
    
    def train_actor_critic(self, imagined_trajectories):
        """Train actor-critic on imagined trajectories"""
        if not imagined_trajectories:
            return {'actor_loss': 0.0}
        
        total_loss = 0.0
        
        for traj in imagined_trajectories:
            if len(traj['actions']) == 0:
                continue
            
            # Convert to tensors
            observations = torch.FloatTensor(traj['observations'][:-1]).to(self.device)
            actions = torch.LongTensor(traj['actions']).to(self.device)
            rewards = torch.FloatTensor(traj['rewards']).to(self.device)
            
            # Forward pass
            action_probs, values = self.actor_critic(observations)
            
            # Compute advantages (simple version)
            advantages = rewards - values.squeeze(-1).detach()
            
            # Policy loss
            action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
            policy_loss = -(action_log_probs.squeeze() * advantages).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), rewards)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            self.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
            self.actor_optimizer.step()
            
            total_loss += loss.item()
        
        return {'actor_loss': total_loss / max(1, len(imagined_trajectories))}
    
    def train(self, num_episodes=100):
        """Main training loop with optimizations"""
        print(f"Starting Optimized DREAM training for {num_episodes} episodes")
        print("=" * 60)
        
        best_reward = float('-inf')
        episodes_without_improvement = 0
        
        for episode in range(num_episodes):
            # Collect trajectory
            trajectory = self.collect_trajectory()
            self.replay_buffer.add_trajectory(trajectory)
            
            episode_reward = sum(trajectory['rewards'])
            episode_length = len(trajectory['actions'])
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Train models
            world_losses = self.train_world_model()
            imagined_trajectories = self.generate_imagined_trajectories()
            actor_losses = self.train_actor_critic(imagined_trajectories)
            
            self.world_losses.append(world_losses['world_loss'])
            self.actor_losses.append(actor_losses['actor_loss'])
            
            # Update exploration parameters
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
            
            # Learning rate scheduling
            if episode % 10 == 9:
                self.world_scheduler.step()
                self.actor_scheduler.step()
            
            # Track improvement
            if episode_reward > best_reward:
                best_reward = episode_reward
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 1
            
            # Progress reporting
            if episode % 10 == 9:
                recent_rewards = self.episode_rewards[-10:]
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(self.episode_lengths[-10:])
                
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Avg Length: {avg_length:5.1f} | "
                      f"Best: {best_reward:7.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Temp: {self.temperature:.2f}")
                
                # Run demonstration every 20 episodes
                if episode % 20 == 19:
                    self.run_demonstration()
        
        print("=" * 60)
        print("✅ Optimized DREAM training completed!")
        self.env.close()
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'world_losses': self.world_losses,
            'actor_losses': self.actor_losses,
            'best_reward': best_reward
        }
    
    def run_demonstration(self):
        """Run a quick demonstration"""
        print("🎮 Running demonstration...")
        
        try:
            demo_env = TetrisEnv(num_agents=1, headless=True, step_mode='action', action_mode='direct')
            obs = demo_env.reset()
            
            total_reward = 0
            step_count = 0
            max_demo_steps = 200
            
            while step_count < max_demo_steps:
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                with torch.no_grad():
                    action_probs, _ = self.actor_critic(obs_tensor.unsqueeze(0))
                    action = torch.multinomial(action_probs, 1).item()
                
                obs, reward, done, info = demo_env.step(action)
                total_reward += reward
                step_count += 1
                
                if done:
                    break
            
            demo_env.close()
            
            score = info.get('score', 0) if isinstance(info, dict) else 0
            lines = info.get('lines_cleared', 0) if isinstance(info, dict) else 0
            
            print(f"   Demo: Score={score}, Lines={lines}, Reward={total_reward:.1f}, Steps={step_count}")
            
        except Exception as e:
            print(f"   Demo failed: {e}")
    
    def estimate_learning_time(self):
        """Estimate when the model will learn to clear blocks"""
        if len(self.episode_rewards) < 20:
            return "Need more episodes"
        
        recent_rewards = self.episode_rewards[-20:]
        early_rewards = self.episode_rewards[:10] if len(self.episode_rewards) >= 10 else self.episode_rewards
        
        improvement_rate = (np.mean(recent_rewards) - np.mean(early_rewards)) / len(self.episode_rewards)
        
        if improvement_rate > 0.1:
            # Positive learning trend
            episodes_to_positive = max(10, int((5.0 - np.mean(recent_rewards)) / improvement_rate))
            return f"~{episodes_to_positive} episodes to positive rewards"
        elif improvement_rate > -0.05:
            # Slow improvement
            return "~100-200 episodes (slow learning)"
        else:
            # No improvement
            return "Learning not detected - check hyperparameters"


def main():
    """Main function for testing"""
    trainer = OptimizedDREAMTrainer(device='cuda', batch_size=6)
    
    print("Running optimization test...")
    results = trainer.train(num_episodes=20)
    
    print(f"\nLearning estimate: {trainer.estimate_learning_time()}")
    
    return results


if __name__ == "__main__":
    main() 