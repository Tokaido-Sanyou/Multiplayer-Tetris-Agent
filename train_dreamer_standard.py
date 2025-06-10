import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
import random
from collections import deque, namedtuple, Counter
import logging
from datetime import datetime
import os
import sys
import argparse
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.tetris_env import TetrisEnv
from torch.utils.tensorboard import SummaryWriter

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

class RepresentationModel(nn.Module):
    """Encodes observations into latent representations"""
    def __init__(self, obs_dim=206, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # Mean + logvar for VAE
        )
        self.latent_dim = latent_dim
        
    def forward(self, obs):
        encoded = self.encoder(obs)
        mean, logvar = encoded.chunk(2, dim=-1)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, -10.0, 10.0)
        return mean, logvar
    
    def encode(self, obs):
        """Get latent representation"""
        mean, logvar = self.forward(obs)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std, mean, logvar

class DynamicsModel(nn.Module):
    """Predicts next latent state given current latent state and action"""
    def __init__(self, latent_dim=128, action_dim=8, hidden_dim=256):
        super().__init__()
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean + logvar
        )
        self.latent_dim = latent_dim
        
    def forward(self, latent_state, action):
        # One-hot encode action
        action_onehot = F.one_hot(action.long(), num_classes=8).float()
        input_tensor = torch.cat([latent_state, action_onehot], dim=-1)
        output = self.dynamics(input_tensor)
        mean, logvar = output.chunk(2, dim=-1)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, -10.0, 10.0)
        return mean, logvar
    
    def predict(self, latent_state, action):
        """Get next latent state prediction"""
        mean, logvar = self.forward(latent_state, action)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std, mean, logvar

class RewardModel(nn.Module):
    """Predicts reward given latent state and action"""
    def __init__(self, latent_dim=128, action_dim=8, hidden_dim=256):
        super().__init__()
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, latent_state, action):
        action_onehot = F.one_hot(action.long(), num_classes=8).float()
        input_tensor = torch.cat([latent_state, action_onehot], dim=-1)
        return self.reward_head(input_tensor)

class ContinueModel(nn.Module):
    """Predicts whether episode continues (1-done)"""
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.continue_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, latent_state):
        return self.continue_head(latent_state)

class WorldModel(nn.Module):
    """Complete world model: representation + dynamics + reward + continue"""
    def __init__(self, obs_dim=206, latent_dim=128, action_dim=8):
        super().__init__()
        self.representation = RepresentationModel(obs_dim, latent_dim)
        self.dynamics = DynamicsModel(latent_dim, action_dim)
        self.reward = RewardModel(latent_dim, action_dim)
        self.continue_model = ContinueModel(latent_dim)
        # Observation decoder for reconstruction loss
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, obs_dim)
        )
        
    def encode_obs(self, obs):
        return self.representation.encode(obs)
    
    def decode(self, latent):
        """Reconstruct observations from latent space"""
        return self.decoder(latent)
    
    def predict_step(self, latent_state, action):
        """Predict one step in latent space"""
        next_latent, next_mean, next_logvar = self.dynamics.predict(latent_state, action)
        reward = self.reward(latent_state, action)
        continue_prob = self.continue_model(next_latent)
        
        return {
            'next_latent': next_latent,
            'next_mean': next_mean,
            'next_logvar': next_logvar,
            'reward': reward,
            'continue': continue_prob
        }

class Actor(nn.Module):
    """Policy network operating in latent space"""
    def __init__(self, latent_dim=128, action_dim=8, hidden_dim=256):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, latent_state):
        logits = self.policy(latent_state)
        return Categorical(logits=logits)

class Critic(nn.Module):
    """Value network operating in latent space"""
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, latent_state):
        return self.value(latent_state)

class DreamerAgent:
    def __init__(self, obs_dim=206, action_dim=8, latent_dim=128, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # World model components
        self.world_model = WorldModel(obs_dim, latent_dim, action_dim).to(device)
        
        # Actor-Critic components
        self.actor = Actor(latent_dim, action_dim).to(device)
        self.critic = Critic(latent_dim).to(device)
        
        # Optimizers - separate for world model and policy
        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=1e-3)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Training parameters
        self.imagination_horizon = 15  # Length of imagined rollouts
        self.batch_size = 32
        self.gamma = 0.99
        self.lambda_gae = 0.95
        # Loss coefficients
        self.recon_coeff = 1.0  # Weight for reconstruction loss
        self.entropy_coeff = 0.01  # Weight for policy entropy bonus
        
    def select_action(self, obs, explore=True):
        """Select action using current policy"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            latent, _, _ = self.world_model.encode_obs(obs_tensor)
            action_dist = self.actor(latent)
            
            if explore:
                action = action_dist.sample()
            else:
                action = action_dist.probs.argmax()
                
            return action.item()
    
    def train_world_model(self, batch):
        """Train world model on real experience"""
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Encode observations
        latent, mean, logvar = self.world_model.encode_obs(states)
        next_latent_target, next_mean_target, next_logvar_target = self.world_model.encode_obs(next_states)
        
        # Predict next state in latent space
        pred_next_mean, pred_next_logvar = self.world_model.dynamics(latent, actions)
        
        # Predict rewards
        pred_rewards = self.world_model.reward(latent, actions)
        
        # Predict continue probabilities
        continue_probs = self.world_model.continue_model(next_latent_target)
        continue_targets = 1.0 - dones.float()
        
        # Compute losses
        # 1. Representation loss (VAE regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()
        
        # 2. Dynamics loss (predict next latent state)
        dynamics_loss = F.mse_loss(pred_next_mean, next_mean_target.detach()) + \
                       F.mse_loss(pred_next_logvar, next_logvar_target.detach())
        
        # 3. Reward loss
        reward_loss = F.mse_loss(pred_rewards.squeeze(), rewards)
        
        # 4. Continue loss
        continue_loss = F.binary_cross_entropy(continue_probs.squeeze(), continue_targets)
        
        # 5. Reconstruction loss: decode latent to reconstruct observations
        recon_states = self.world_model.decode(latent)
        recon_loss = F.mse_loss(recon_states, states)
        
        # Total world model loss
        world_loss = kl_loss + self.recon_coeff * recon_loss + dynamics_loss + reward_loss + continue_loss
        
        # Update world model
        self.world_optimizer.zero_grad()
        world_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_optimizer.step()
        
        return {
            'world_loss': world_loss.item(),
            'kl_loss': kl_loss.item(),
            'recon_loss': recon_loss.item(),
            'dynamics_loss': dynamics_loss.item(),
            'reward_loss': reward_loss.item(),
            'continue_loss': continue_loss.item()
        }
    
    def imagine_rollout(self, initial_latent, horizon):
        """Generate imagined trajectories using world model"""
        batch_size = initial_latent.shape[0]
        # Debug: check for NaNs in initial latent
        if torch.isnan(initial_latent).any():
            print(f"⚠️ initial_latent contains NaNs at start of imagined rollout")
        
        # Storage for imagined trajectory
        latents = [initial_latent]
        actions = []
        rewards = []
        continues = []
        
        current_latent = initial_latent
        
        for step in range(horizon):
            # Debug: sanitize any NaNs in current_latent
            if torch.isnan(current_latent).any():
                print(f"⚠️ NaN detected in current_latent at rollout step {step}, sanitizing")
                current_latent = torch.nan_to_num(current_latent, nan=0.0, posinf=1e6, neginf=-1e6)
            # Sample action from current policy
            action_dist = self.actor(current_latent)
            action = action_dist.sample()
            
            # Predict next step using world model
            with torch.no_grad():  # Don't backprop through world model during policy training
                prediction = self.world_model.predict_step(current_latent, action)
                next_latent = prediction['next_latent']
                # Debug: sanitize NaNs in next_latent
                if torch.isnan(next_latent).any():
                    print(f"⚠️ NaN detected in next_latent at rollout step {step}, sanitizing")
                    next_latent = torch.nan_to_num(next_latent, nan=0.0, posinf=1e6, neginf=-1e6)
                reward = prediction['reward']
                continue_prob = prediction['continue']
            
            # Store trajectory
            latents.append(next_latent)
            actions.append(action)
            rewards.append(reward.squeeze())
            continues.append(continue_prob.squeeze())
            
            current_latent = next_latent
        
        return {
            'latents': torch.stack(latents),  # [horizon+1, batch, latent_dim]
            'actions': torch.stack(actions),  # [horizon, batch]
            'rewards': torch.stack(rewards),  # [horizon, batch]
            'continues': torch.stack(continues)  # [horizon, batch]
        }
    
    def compute_lambda_returns(self, rewards, values, continues):
        """Compute λ-returns for policy gradient"""
        horizon = rewards.shape[0]
        batch_size = rewards.shape[1]
        
        returns = torch.zeros_like(values)
        last_value = values[-1]
        
        for t in reversed(range(horizon)):
            if t == horizon - 1:
                returns[t] = rewards[t] + self.gamma * continues[t] * last_value
            else:
                delta = rewards[t] + self.gamma * continues[t] * values[t + 1] - values[t]
                returns[t] = values[t] + delta + self.gamma * self.lambda_gae * continues[t] * (returns[t + 1] - values[t + 1])
        
        return returns
    
    def train_policy(self, batch):
        """Train actor-critic using imagined rollouts"""
        states, _, _, _, _ = batch
        states = states.to(self.device)
        
        # Encode initial states
        with torch.no_grad():
            initial_latent, _, _ = self.world_model.encode_obs(states)
        
        # Generate imagined rollouts
        rollout = self.imagine_rollout(initial_latent, self.imagination_horizon)
        
        # Get values for all states in rollout
        values = self.critic(rollout['latents'][:-1].view(-1, self.latent_dim))
        values = values.view(self.imagination_horizon, -1).squeeze(-1)  # [horizon, batch]
        
        last_value = self.critic(rollout['latents'][-1]).squeeze(-1)  # [batch]
        
        # Compute λ-returns
        all_values = torch.cat([values, last_value.unsqueeze(0)])  # [horizon+1, batch]
        returns = self.compute_lambda_returns(
            rollout['rewards'], 
            all_values,
            rollout['continues']
        )[:-1]  # Remove bootstrap, now [horizon, batch]
        
        # Policy gradient loss with entropy bonus
        flat_latents = rollout['latents'][:-1].view(-1, self.latent_dim)
        flat_actions = rollout['actions'].view(-1)
        
        action_dist = self.actor(flat_latents)
        log_probs = action_dist.log_prob(flat_actions).view(self.imagination_horizon, -1)
        
        advantages = returns - values
        entropy = action_dist.entropy().mean()
        actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coeff * entropy
        
        # Value loss
        critic_loss = F.mse_loss(values, returns.detach())
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'avg_return': returns.mean().item(),
            'avg_advantage': advantages.mean().item(),
            'entropy': entropy.item()
        }

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, *args):
        self.buffer.append(Experience(*args))
        
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to numpy first, then to torch tensors for efficiency
        states = np.array(batch.state, dtype=np.float32)
        actions = np.array(batch.action, dtype=np.int64)
        rewards = np.array(batch.reward, dtype=np.float32)
        next_states = np.array(batch.next_state, dtype=np.float32)
        dones = np.array(batch.done, dtype=bool)
        
        # Convert to tensors
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

def train_dreamer(episodes=1000, world_model_pretrain=100, reward_mode='lines_only', visualize_interval=0, step_mode='block_placed'):
    """
    Standard Dreamer training:
    1. Pretrain world model on random data
    2. Alternate between world model training and policy training
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = TetrisEnv(reward_mode=reward_mode, step_mode=step_mode)
    agent = DreamerAgent(device=device)
    buffer = ReplayBuffer(capacity=100000)
    
    # Debug counters for random pretrain data
    pretrain_steps = 0
    pretrain_piece_placed = 0
    pretrain_game_overs = 0
    
    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/dreamer_standard_{timestamp}"
    writer = SummaryWriter(log_dir)
    
    # Performance tracking
    episode_rewards = []
    episode_lines_cleared = []
    episode_steps_history = []
    
    print(f"Training Standard Dreamer on {device}")
    print(f"Logs: {log_dir}")
    print(f"TensorBoard logs are at: {log_dir} (run: tensorboard --logdir {log_dir})")
    print(f"World Model Parameters: {sum(p.numel() for p in agent.world_model.parameters())}")
    print(f"Policy Parameters: {sum(p.numel() for p in agent.actor.parameters()) + sum(p.numel() for p in agent.critic.parameters())}")
    
    total_steps = 0
    world_model_updates = 0
    policy_updates = 0
    
    # Prepare Phase 2 instrumentation
    phase2_start = None
    random_actions = 0
    policy_actions = 0
    action_counts = Counter()
    
    # Visualization setup
    if visualize_interval > 0:
        print(f"🔍 Rendering every {visualize_interval} episodes")
        # Quick test: reset then perform one random action to lock a piece for visualization
        print("🔍 Testing render after one placement... (window should show a locked piece for 1s)")
        state = env.reset()
        # Take a random action to place the first piece
        rnd_act = env.action_space.sample()
        _obs, _rwd, _done, _info = env.step(rnd_act)
        env.render()
        time.sleep(1.0)
    
    # Phase 1: Collect initial random data and pretrain world model
    print("Phase 1: Collecting random data and pretraining world model...")
    
    for episode in range(world_model_pretrain):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        lines_cleared = 0
        
        while episode_steps < 500:  # Max episode length
            # Random action for initial data collection
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            # Instrument data stats
            pretrain_steps += 1
            if info.get('piece_placed', False):
                pretrain_piece_placed += 1
            if done:
                pretrain_game_overs += 1
            
            # Track lines cleared
            if 'lines_cleared' in info:
                lines_cleared += info['lines_cleared']
            
            buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if done:
                break
        
        # Store episode stats
        episode_rewards.append(episode_reward)
        episode_lines_cleared.append(lines_cleared)
        episode_steps_history.append(episode_steps)
        
        # Train world model every few episodes once we have enough data
        if len(buffer) >= agent.batch_size and episode % 5 == 0:
            batch = buffer.sample(agent.batch_size)
            world_losses = agent.train_world_model(batch)
            world_model_updates += 1
            
            # Log world model training
            for key, value in world_losses.items():
                writer.add_scalar(f'WorldModel/{key}', value, world_model_updates)
        
        if episode % 20 == 0:
            # Calculate stats for last 20 episodes (or all if less than 20)
            recent_episodes = min(20, len(episode_lines_cleared))
            recent_lines = episode_lines_cleared[-recent_episodes:]
            recent_rewards = episode_rewards[-recent_episodes:]
            
            avg_lines = np.mean(recent_lines) if recent_lines else 0
            max_lines = max(recent_lines) if recent_lines else 0
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            
            print(f"Pretrain Episode {episode}, Buffer: {len(buffer)}, "
                  f"Lines: {lines_cleared}, Reward: {episode_reward:.2f}")
            print(f"  Last {recent_episodes} episodes - Avg Lines: {avg_lines:.1f}, "
                  f"Max Lines: {max_lines}, Avg Reward: {avg_reward:.2f}")
    
    # After pretraining, print debug stats
    print(f"📊 Pretrain summary: {pretrain_steps} steps, {pretrain_piece_placed} placements ({(pretrain_piece_placed/pretrain_steps*100 if pretrain_steps>0 else 0):.1f}%), {pretrain_game_overs} episodes ended by game over")
    
    # Phase 2: Joint training with imagined rollouts
    phase2_start = time.time()
    print("Phase 2: Joint training with imagination...")
    
    for episode in range(episodes):
        # Should we visualize this episode?
        visualize_episode = (visualize_interval > 0 and episode % visualize_interval == 0)
        state = env.reset()
        if visualize_episode:
            print(f"🔍 Visualizing Episode {episode}")
            env.render()
            time.sleep(0.5)
        episode_reward = 0
        episode_steps = 0
        lines_cleared = 0
        
        while episode_steps < 500:
            # Epsilon-greedy action selection
            epsilon_start = 1.0
            epsilon_end = 0.01
            epsilon_decay = 0.995
            epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
            if random.random() < epsilon:
                action = env.action_space.sample()
                random_actions += 1
            else:
                action = agent.select_action(state, explore=True)
                policy_actions += 1
            action_counts[action] += 1
            next_state, reward, done, info = env.step(action)
            
            if visualize_episode:
                env.render()
                time.sleep(0.05)
            
            # Track lines cleared
            if 'lines_cleared' in info:
                lines_cleared += info['lines_cleared']
            
            buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if done:
                break
        
        # Store episode stats
        episode_rewards.append(episode_reward)
        episode_lines_cleared.append(lines_cleared)
        episode_steps_history.append(episode_steps)
        
        # Training updates
        if len(buffer) >= agent.batch_size:
            batch = buffer.sample(agent.batch_size)

            # Always train world model
            world_losses = agent.train_world_model(batch)
            world_model_updates += 1
            for key, value in world_losses.items():
                writer.add_scalar(f'WorldModel/{key}', value, world_model_updates)

            # Train policy using imagination
            policy_losses = agent.train_policy(batch)
            policy_updates += 1
            for key, value in policy_losses.items():
                writer.add_scalar(f'Policy/{key}', value, policy_updates)
        
        # Logging
        writer.add_scalar('Environment/episode_reward', episode_reward, episode)
        writer.add_scalar('Environment/episode_steps', episode_steps, episode)
        writer.add_scalar('Environment/lines_cleared', lines_cleared, episode)
        writer.add_scalar('Environment/total_steps', total_steps, episode)
        
        # Performance reporting every 25 episodes
        if episode % 25 == 0:
            # Calculate stats for last 50 episodes
            total_episodes = len(episode_lines_cleared)
            recent_count = min(50, total_episodes)
            recent_lines = episode_lines_cleared[-recent_count:]
            recent_rewards = episode_rewards[-recent_count:]
            recent_steps = episode_steps_history[-recent_count:]
            
            # Statistics
            avg_lines = np.mean(recent_lines)
            max_lines = max(recent_lines)
            avg_reward = np.mean(recent_rewards)
            max_reward = max(recent_rewards)
            avg_steps = np.mean(recent_steps)
            
            # Find best performing episodes
            sorted_indices = np.argsort(recent_lines)[::-1]  # Descending order
            top_5_lines = [recent_lines[i] for i in sorted_indices[:5]]
            top_5_rewards = [recent_rewards[i] for i in sorted_indices[:5]]
            
            print(f"\n📊 Episode {episode} Performance Report (Last {recent_count} episodes):")
            print(f"  🏆 Highest Lines Cleared: {max_lines}")
            print(f"  📈 Average Lines/Episode: {avg_lines:.2f}")
            print(f"  💰 Average Reward: {avg_reward:.2f}")
            print(f"  🎯 Max Reward: {max_reward:.2f}")
            print(f"  ⏱️  Average Steps: {avg_steps:.1f}")
            print(f"  🔥 Top 5 Line Performances: {top_5_lines}")
            print(f"  📋 Updates - World Model: {world_model_updates}, Policy: {policy_updates}")
            
            # Log to tensorboard
            writer.add_scalar('Performance/avg_lines_last_50', avg_lines, episode)
            writer.add_scalar('Performance/max_lines_last_50', max_lines, episode)
            writer.add_scalar('Performance/avg_reward_last_50', avg_reward, episode)
            writer.add_scalar('Performance/max_reward_last_50', max_reward, episode)
            # Instrumentation: timing and exploration
            elapsed = time.time() - phase2_start
            avg_sec = elapsed / (episode + 1)
            print(f"⏱️  Avg sec per placement: {avg_sec:.3f}s")
            print(f"📊 Exploration so far: random={random_actions}, policy={policy_actions}, random%={(random_actions/(random_actions+policy_actions)*100 if random_actions+policy_actions>0 else 0):.1f}%")
            print(f"📋 Action counts: {dict(action_counts)}")
            writer.add_scalar('Timing/avg_sec_per_placement', avg_sec, episode)
            writer.add_scalar('Exploration/random_frac', random_actions/(random_actions+policy_actions) if random_actions+policy_actions>0 else 0, episode)
        
        # Save model periodically
        if episode % 200 == 0 and episode > 0:
            checkpoint_dir = f"checkpoints/dreamer_standard_{timestamp}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save performance stats too
            stats = {
                'episode_rewards': episode_rewards,
                'episode_lines_cleared': episode_lines_cleared,
                'episode_steps': episode_steps_history,
                'world_model_updates': world_model_updates,
                'policy_updates': policy_updates
            }
            
            torch.save({
                'world_model': agent.world_model.state_dict(),
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'episode': episode,
                'total_steps': total_steps,
                'performance_stats': stats
            }, f"{checkpoint_dir}/checkpoint_episode_{episode}.pt")
            
            print(f"💾 Saved checkpoint at episode {episode}")
    
    # Final performance summary
    print("\n🎉 Training Completed! Final Performance Summary:")
    final_50 = episode_lines_cleared[-50:] if len(episode_lines_cleared) >= 50 else episode_lines_cleared
    print(f"  🏆 Best Lines Cleared (All Time): {max(episode_lines_cleared)}")
    print(f"  📊 Average Lines (Last 50): {np.mean(final_50):.2f}")
    print(f"  🔥 Best Lines (Last 50): {max(final_50)}")
    print(f"  📈 Total Episodes: {len(episode_lines_cleared)}")
    print(f"  🎯 Total Steps: {total_steps}")
    
    # Final instrumentation summary
    elapsed_total = time.time() - phase2_start
    print(f"\n📈 Total Phase 2 time: {elapsed_total:.1f}s for {episodes} placements (~{elapsed_total/episodes:.3f}s per placement)")
    print(f"Random actions: {random_actions}, Policy actions: {policy_actions}")
    print(f"Final action distribution: {dict(action_counts)}")
    
    writer.close()
    
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Standard Dreamer Agent")
    parser.add_argument("--episodes", "-e", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--pretrain", "-p", type=int, default=100,
                        help="Number of world-model pretrain episodes")
    parser.add_argument("--visualize_interval", "-v", type=int, default=0,
                        help="Interval (in episodes) at which to render gameplay (0=off)")
    parser.add_argument("--step_mode", "-s", type=str, choices=["action","block_placed"], default="block_placed",
                        help="Environment.step_mode: action or block_placed")
    args = parser.parse_args()
    print(f"Starting training: pretrain={args.pretrain}, episodes={args.episodes}, visualize_interval={args.visualize_interval}, step_mode={args.step_mode}")
    agent = train_dreamer(episodes=args.episodes,
                          world_model_pretrain=args.pretrain,
                          visualize_interval=args.visualize_interval,
                          step_mode=args.step_mode)