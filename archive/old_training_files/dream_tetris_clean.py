"""
Improved DREAM Implementation for Tetris

Incorporates findings from debugging:
- Reward shaping to handle large negative rewards
- Improved world model architecture with LayerNorm and Dropout
- Learning rate scheduling for stability
- Better imagination-reality gap handling
- Real-time visualization and batched updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.tetris_env import TetrisEnv
from dream_visualizer import DREAMVisualizer, BatchTracker

# ============================================================================
# IMPROVED WORLD MODEL
# ============================================================================

class ImprovedTetrisWorldModel(nn.Module):
    """Improved world model with better architecture and stability"""
    
    def __init__(self, obs_dim=212, action_dim=8, hidden_dim=256, state_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Improved observation encoder with residual connections
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Efficient RNN with GRU (faster than LSTM)
        self.rnn = nn.GRU(state_dim + action_dim, state_dim, batch_first=True)
        
        # Improved prediction heads
        self.obs_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),  # Use Tanh instead of ReLU for better negative value learning
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),  # Another Tanh layer
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.continue_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize reward head for Tetris-specific reward range
        self._init_reward_head()
    
    def _init_reward_head(self):
        """Initialize reward head for realistic Tetris reward prediction"""
        # CRITICAL FIX: Tetris rewards are mostly negative (-1 to -100)
        # Initialize to predict realistic negative values
        final_layer = self.reward_head[-1]
        if hasattr(final_layer, 'weight'):
            # Larger weights to capture full reward range
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.5)
        if hasattr(final_layer, 'bias') and final_layer.bias is not None:
            # CRITICAL: Bias toward realistic negative rewards
            nn.init.constant_(final_layer.bias, -2.0)  # Matches real mean of -2.17
    
    def encode(self, observations):
        """Encode observations to state representation"""
        return self.obs_encoder(observations)
    
    def forward(self, observations, actions):
        """
        Forward pass through world model
        observations: [batch, obs_dim] (2D tensor)
        actions: [batch] (1D LongTensor)
        """
        batch_size = observations.shape[0]
        
        # Ensure actions are LongTensor
        actions = actions.long()
        
        # Encode observations: [batch, obs_dim] -> [batch, state_dim]
        states = self.encode(observations)
        
        # One-hot encode actions: [batch] -> [batch, action_dim]
        actions_one_hot = F.one_hot(actions, self.action_dim).float()
        
        # Combine states and actions: [batch, state_dim + action_dim]
        combined = torch.cat([states, actions_one_hot], dim=-1)
        
        # Add sequence dimension for RNN: [batch, 1, combined_dim]
        rnn_input = combined.unsqueeze(1)
        
        # Initial hidden state for GRU
        h0 = torch.zeros(1, batch_size, self.state_dim, device=observations.device)
        
        # Run through RNN
        rnn_output, _ = self.rnn(rnn_input, h0)
        
        # Remove sequence dimension: [batch, 1, state_dim] -> [batch, state_dim]
        rnn_output = rnn_output.squeeze(1)
        
        # Make predictions
        next_observations = self.obs_decoder(rnn_output)
        rewards = self.reward_head(rnn_output)
        continues = torch.sigmoid(self.continue_head(rnn_output))
        
        return {
            'next_observations': next_observations,
            'rewards': rewards,
            'continues': continues,
            'states': rnn_output
        }
    
    def imagine(self, initial_state, actions):
        """Generate imagined trajectory with LSTM hidden states"""
        batch_size, seq_len = actions.shape
        device = actions.device
        
        # Initialize GRU hidden state
        h = initial_state.unsqueeze(0)  # [1, batch, state_dim]
        
        imagined_observations = []
        imagined_rewards = []
        imagined_continues = []
        
        for t in range(seq_len):
            action_t = actions[:, t]
            action_one_hot = F.one_hot(action_t, self.action_dim).float()
            
            # Zero state encoding for imagination
            state_encoding = torch.zeros(batch_size, self.state_dim, device=device)
            
            # Combine state and action
            rnn_input = torch.cat([state_encoding, action_one_hot], dim=-1).unsqueeze(1)
            
            # Step through RNN
            rnn_output, h = self.rnn(rnn_input, h)
            state = rnn_output.squeeze(1)
            
            # Make predictions
            obs = self.obs_decoder(state)
            reward = self.reward_head(state).squeeze(-1)
            continue_prob = torch.sigmoid(self.continue_head(state)).squeeze(-1)
            
            imagined_observations.append(obs)
            imagined_rewards.append(reward)
            imagined_continues.append(continue_prob)
        
        return {
            'observations': torch.stack(imagined_observations, dim=1),
            'rewards': torch.stack(imagined_rewards, dim=1),
            'continues': torch.stack(imagined_continues, dim=1)
        }

# ============================================================================
# IMPROVED ACTOR-CRITIC
# ============================================================================

class ImprovedTetrisActorCritic(nn.Module):
    """Improved Actor-Critic with better architecture"""
    
    def __init__(self, obs_dim=212, action_dim=8, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Improved shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head with better initialization
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observations, temperature=1.0):
        """Forward pass"""
        features = self.features(observations)
        
        # Actor output (action probabilities)
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits / temperature, dim=-1)
        
        # Critic output (value)
        values = self.critic(features).squeeze(-1)
        
        return action_probs, values
    
    def get_action_and_value(self, observation, epsilon=0.1, temperature=1.0, rnd_network=None):
        """Get action and value for single observation with RND exploration"""
        observation = observation.unsqueeze(0)  # Add batch dimension
        action_probs, value = self.forward(observation, temperature=temperature)
        
        # Use RND for exploration if provided
        if rnd_network is not None:
            # Get intrinsic reward from RND
            intrinsic_reward = rnd_network(observation)
            
            # Boost action probabilities based on intrinsic reward
            # Higher intrinsic reward = more exploration of that state
            exploration_boost = torch.exp(intrinsic_reward.unsqueeze(-1) * 0.1)  # Scale factor
            action_probs = action_probs * exploration_boost
            action_probs = F.softmax(action_probs, dim=-1)  # Renormalize
        
        # Sample action from modified probabilities
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob, value

# ============================================================================
# RANDOM NETWORK DISTILLATION (RND) FOR EXPLORATION
# ============================================================================

class RNDNetwork(nn.Module):
    """Random Network Distillation for exploration"""
    
    def __init__(self, obs_dim=425, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        
        # Target network (frozen, randomly initialized)
        self.target_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Predictor network (trainable)
        self.predictor_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Freeze target network
        for param in self.target_network.parameters():
            param.requires_grad = False
    
    def forward(self, observations):
        """Compute intrinsic reward based on prediction error"""
        with torch.no_grad():
            target_features = self.target_network(observations)
        
        predicted_features = self.predictor_network(observations)
        
        # Intrinsic reward is prediction error (normalized)
        intrinsic_reward = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=-1)
        return intrinsic_reward
    
    def get_prediction_loss(self, observations):
        """Get loss for training the predictor network"""
        with torch.no_grad():
            target_features = self.target_network(observations)
        
        predicted_features = self.predictor_network(observations)
        return F.mse_loss(predicted_features, target_features)

# ============================================================================
# IMPROVED REPLAY BUFFER
# ============================================================================

class ImprovedTetrisReplayBuffer:
    """Improved replay buffer with better sampling"""
    
    def __init__(self, capacity=3000, sequence_length=8):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.trajectories = []
        self.position = 0
        
        # Track buffer statistics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def add_trajectory(self, trajectory):
        """Add trajectory with statistics tracking"""
        if len(self.trajectories) < self.capacity:
            self.trajectories.append(trajectory)
        else:
            self.trajectories[self.position] = trajectory
            self.position = (self.position + 1) % self.capacity
        
        # Update statistics
        episode_reward = sum(trajectory['rewards'])
        episode_length = len(trajectory['actions'])
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Keep only recent statistics
        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]
            self.episode_lengths = self.episode_lengths[-1000:]
    
    def sample_batch(self, batch_size):
        """Sample batch with improved sampling strategy"""
        if not self.trajectories:
            return None
        
        import random
        
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_continues = []
        
        for _ in range(batch_size):
            # Prioritize longer trajectories for better learning
            weights = [len(traj['actions']) for traj in self.trajectories]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Weighted random selection
            traj = np.random.choice(self.trajectories, p=weights)
            traj_len = len(traj['observations']) - 1
            
            if traj_len >= self.sequence_length:
                start_idx = random.randint(0, traj_len - self.sequence_length)
                end_idx = start_idx + self.sequence_length
            else:
                start_idx = 0
                end_idx = traj_len
            
            # Extract sequences
            obs_seq = []
            act_seq = []
            rew_seq = []
            cont_seq = []
            
            for i in range(start_idx, end_idx):
                obs_seq.append(torch.tensor(traj['observations'][i], dtype=torch.float32))
                act_seq.append(traj['actions'][i])
                rew_seq.append(traj['rewards'][i])
                cont_seq.append(1.0 - float(traj['dones'][i]))
            
            # Pad if necessary
            while len(obs_seq) < self.sequence_length:
                obs_seq.append(obs_seq[-1].clone())
                act_seq.append(7)  # No-op
                rew_seq.append(0.0)
                cont_seq.append(0.0)
            
            batch_obs.append(torch.stack(obs_seq))
            batch_actions.append(torch.tensor(act_seq, dtype=torch.long))
            batch_rewards.append(torch.tensor(rew_seq, dtype=torch.float32))
            batch_continues.append(torch.tensor(cont_seq, dtype=torch.float32))
        
        return {
            'observations': torch.stack(batch_obs),
            'actions': torch.stack(batch_actions),
            'rewards': torch.stack(batch_rewards),
            'continues': torch.stack(batch_continues)
        }
    
    def __len__(self):
        return len(self.trajectories)

# ============================================================================
# IMPROVED DREAM TRAINER
# ============================================================================

class ImprovedDREAMTrainer:
    """Improved DREAM trainer with reward shaping and LR scheduling"""
    
    def __init__(self, device='cuda', enable_visualization=True, batch_size=6):
        # Auto-detect and properly initialize device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available, falling back to CPU")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Environment - Fixed to use direct action mode for proper scalar actions
        self.env = TetrisEnv(num_agents=1, headless=True, step_mode='action', action_mode='direct')
        
        # Improved models
        self.world_model = ImprovedTetrisWorldModel().to(self.device)
        self.actor_critic = ImprovedTetrisActorCritic().to(self.device)
        self.rnd_network = RNDNetwork().to(self.device)
        
        # Optimizers with better learning rates
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=3e-3)
        self.actor_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=1e-2)
        self.rnd_optimizer = torch.optim.Adam(self.rnd_network.predictor_network.parameters(), lr=1e-3)
        
        # Learning rate schedulers - FIXED: Less aggressive decay
        self.world_scheduler = torch.optim.lr_scheduler.StepLR(self.world_optimizer, step_size=25, gamma=0.9)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=25, gamma=0.95)
        
        # Improved replay buffer
        self.replay_buffer = ImprovedTetrisReplayBuffer(capacity=3000, sequence_length=8)
        
        # Training parameters
        self.batch_size = 4
        self.imagination_horizon = 500  # Match real episode lengths (400+)
        self.world_model_train_steps = 25  # Intensive world model training for better reward prediction
        self.actor_train_steps = 3  # Efficient actor training
        
        # Use environment rewards directly - no artificial shaping
        self.reward_scale = 1.0  # No scaling - use original rewards
        self.survival_bonus = 0.0  # No artificial survival bonus
        self.penalty_cap = 0.0  # No penalty capping
        
        # Reward normalization for improved convergence
        self.reward_history = []
        self.reward_norm_window = 1000  # Window for reward normalization
        
        # Visualization and batching
        self.enable_visualization = enable_visualization
        self.visualizer = DREAMVisualizer(enable_plots=enable_visualization) if enable_visualization else None
        self.batch_tracker = BatchTracker(batch_size)
        self.episode_batch_size = batch_size
        
        print(f"Optimized DREAM Trainer initialized on {self.device}")
        print(f"Architecture: Efficient GRU-based (79% parameter reduction)")
        print(f"Learning Rates: World 3e-3, Actor 1e-2 (aggressive)")
        print(f"Reward Handling: Using original environment rewards (no shaping)")
        print(f"Exploration: ε=0.9→0.1, T=3.5→1.0 (aggressive)")
        print(f"Visualization: {'Enabled' if enable_visualization else 'Disabled'}")
        print(f"Batch Size: {batch_size} episodes")
        
        # Proportional exploration parameters - adapt to training length
        self.epsilon_initial = 0.9  # High initial exploration
        self.epsilon = 0.9
        self.epsilon_min = 0.1
        
        # Temperature parameters - proportional decay
        self.temperature_initial = 3.5  # High initial temperature
        self.temperature = 3.5
        self.temperature_min = 1.0
        
        # Proportional decay constants for smooth curves
        self.epsilon_decay_rate = 3.0  # Higher = more aggressive decay
        self.temperature_decay_rate = 2.0  # Lower = slower decay for diversity
    
    def update_exploration_proportional(self, episode, total_episodes):
        """Update exploration parameters proportional to training progress"""
        import math
        
        if total_episodes <= 0:
            return
        
        progress = episode / total_episodes
        
        # Epsilon decay (more aggressive exploration reduction)
        epsilon_decay_factor = 1 - math.exp(-self.epsilon_decay_rate * progress)
        self.epsilon = max(self.epsilon_min, 
                          self.epsilon_initial - (self.epsilon_initial - self.epsilon_min) * epsilon_decay_factor)
        
        # Temperature decay (slower decay to maintain diversity)
        temp_decay_factor = 1 - math.exp(-self.temperature_decay_rate * progress)
        self.temperature = max(self.temperature_min,
                              self.temperature_initial - (self.temperature_initial - self.temperature_min) * temp_decay_factor)
    
    def get_proportional_termination_rate(self, episode, total_episodes):
        """Calculate proportional termination rate for imagination trajectories"""
        if total_episodes <= 0:
            return 0.08  # Default rate
        
        progress = episode / total_episodes
        
        # Start with low termination rate (0.02) and increase to higher rate (0.15)
        # Early training: Lower termination = longer imagination trajectories 
        # Late training: Higher termination = more realistic trajectory lengths
        min_rate = 0.02  # 2% chance early in training
        max_rate = 0.15  # 15% chance late in training
        
        # Exponential increase in termination rate
        import math
        rate_factor = 1 - math.exp(-2.0 * progress)  # Similar curve to exploration
        termination_rate = min_rate + (max_rate - min_rate) * rate_factor
        
        return termination_rate
    
    def shape_rewards(self, rewards):
        """Use environment rewards directly - no artificial shaping"""
        # Simply return the original rewards without any modification
        return rewards.copy() if isinstance(rewards, list) else list(rewards)
    
    def unshape_rewards(self, shaped_rewards):
        """No unshaping needed since we use original rewards directly"""
        # Since we're not shaping rewards, just return them as-is
        return shaped_rewards.copy() if isinstance(shaped_rewards, list) else list(shaped_rewards)
    
    def normalize_reward(self, reward):
        """Optional reward normalization for improved convergence"""
        # Track reward history
        self.reward_history.append(reward)
        if len(self.reward_history) > self.reward_norm_window:
            self.reward_history.pop(0)
        
        # Optional normalization (currently disabled to use original rewards)
        # if len(self.reward_history) > 10:
        #     mean_reward = np.mean(self.reward_history)
        #     std_reward = np.std(self.reward_history)
        #     if std_reward > 0:
        #         return (reward - mean_reward) / (std_reward + 1e-8)
        
        # Return original reward (no normalization applied)
        return reward
    
    def collect_trajectory(self):
        """Collect trajectory with proper reward handling for world model"""
        obs = self.env.reset()
        trajectory = {
            'observations': [obs],
            'actions': [],
            'rewards': [],  # Store environment rewards directly
            'dones': []
        }
        
        # No step limit - natural episode termination only  
        step = 0
        
        while True:  # Continue until natural episode termination
            # Get action from current policy with RND exploration
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action, log_prob, value = self.actor_critic.get_action_and_value(
                obs_tensor, epsilon=self.epsilon, temperature=self.temperature, rnd_network=self.rnd_network
            )
            action = action.item()
            
            # Take environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # Store transition
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)  # Environment reward used directly
            trajectory['dones'].append(done)
            trajectory['observations'].append(next_obs)
            
            if done:
                break
            
            obs = next_obs
            step += 1
        
        return trajectory
    
    def train_world_model(self):
        """Train world model with improved loss computation"""
        if len(self.replay_buffer) < 1:  # Changed from self.batch_size to 1
            return {'world_loss': 0.0}
        
        total_loss = 0.0
        total_reward_loss = 0.0
        total_obs_loss = 0.0
        valid_steps = 0
        
        for step in range(self.world_model_train_steps):
            batch = self.replay_buffer.sample_batch(self.batch_size)
            if batch is None:
                continue
            
            # Ensure all batch data is on the correct device
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            rewards = batch['rewards'].to(self.device)
            continues = batch['continues'].to(self.device)
            
            # Ensure correct shapes and types
            batch_size, seq_len = observations.shape[:2]
            if seq_len <= 1:
                continue  # Skip sequences that are too short
                
            # Process sequences step by step
            step_obs_loss = 0
            step_reward_loss = 0
            step_continue_loss = 0
            num_steps = 0
            
            for t in range(seq_len - 1):
                # Get current step data
                curr_obs = observations[:, t]     # [batch, obs_dim]
                curr_act = actions[:, t].long()   # [batch] - ensure LongTensor
                next_obs = observations[:, t + 1] # [batch, obs_dim]
                curr_rew = rewards[:, t]          # [batch]
                curr_cont = continues[:, t]       # [batch]
                
                # Forward pass through world model
                predictions = self.world_model(curr_obs, curr_act)
                
                # Compute step losses with proper tensor shapes
                obs_loss = F.mse_loss(predictions['next_observations'], next_obs)
                reward_loss = F.mse_loss(predictions['rewards'].squeeze(-1), curr_rew)
                continue_loss = F.binary_cross_entropy(predictions['continues'].squeeze(-1), curr_cont)
                
                step_obs_loss += obs_loss
                step_reward_loss += reward_loss
                step_continue_loss += continue_loss
                num_steps += 1
            
            if num_steps > 0:
                # Average the losses for this batch
                avg_obs_loss = step_obs_loss / num_steps
                avg_reward_loss = step_reward_loss / num_steps
                avg_continue_loss = step_continue_loss / num_steps
                
                # Weighted combination - EXTREME: Prioritize reward prediction above all else
                total_loss_step = 0.1 * avg_obs_loss + 100.0 * avg_reward_loss + 5.0 * avg_continue_loss
                
                # Ensure loss is not zero
                if total_loss_step.item() < 1e-8:
                    # Add small regularization to prevent zero loss
                    reg_loss = 1e-6 * sum(p.pow(2).sum() for p in self.world_model.parameters())
                    total_loss_step = total_loss_step + reg_loss
                
                # Backward pass
                self.world_optimizer.zero_grad()
                total_loss_step.backward()
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
                self.world_optimizer.step()
                
                # Accumulate losses
                total_loss += total_loss_step.item()
                total_reward_loss += avg_reward_loss.item()
                total_obs_loss += avg_obs_loss.item()
                valid_steps += 1
        
        # Return average losses
        if valid_steps > 0:
            avg_loss = total_loss / valid_steps
            avg_reward_loss = total_reward_loss / valid_steps
            avg_obs_loss = total_obs_loss / valid_steps
        else:
            avg_loss = 0.0
            avg_reward_loss = 0.0
            avg_obs_loss = 0.0
        
        return {
            'world_loss': avg_loss,
            'reward_loss': avg_reward_loss,
            'obs_loss': avg_obs_loss
        }
    
    def generate_imagined_trajectories(self, current_episode=0, total_episodes=100):
        """Generate realistic imagined trajectories with proportional randomization"""
        if len(self.replay_buffer) < 1:  # Changed from self.batch_size to 1
            return []
        
        batch = self.replay_buffer.sample_batch(self.batch_size)
        if batch is None:
            return []
        
        # Ensure all batch data is moved to device immediately
        batch = {k: v.to(self.device) for k, v in batch.items()}
        initial_observations = batch['observations'][:, 0]  # [batch, obs_dim]
        
        imagined_trajectories = []
        
        for i in range(min(self.batch_size, initial_observations.shape[0])):
            trajectory = {
                'observations': [initial_observations[i].cpu().numpy()],
                'actions': [],
                'rewards': [],
                'dones': []
            }
            
            current_obs = initial_observations[i:i+1]  # [1, obs_dim]
            
            for step in range(self.imagination_horizon):
                # Get action from current policy with exploration
                with torch.no_grad():
                    action, log_prob, value = self.actor_critic.get_action_and_value(current_obs.squeeze(0), epsilon=0.1)
                    action = action.item()
                
                # Create action tensor
                action_tensor = torch.tensor([action], device=self.device, dtype=torch.long)  # [1]
                
                # Use world model for single step prediction
                with torch.no_grad():
                    predictions = self.world_model(current_obs, action_tensor)
                
                # Extract predictions
                next_obs = predictions['next_observations'][0]  # [obs_dim]
                raw_reward = predictions['rewards'][0].item()  # Raw reward from world model
                continue_prob = predictions['continues'][0].item()
                
                # CRITICAL: Use world model prediction directly with NO modifications
                reward = raw_reward  # Direct from world model - no processing
                
                # Fixed termination logic - allow much longer trajectories
                continue_threshold = 0.01  # Very low threshold - trust world model more
                
                # Only terminate if:
                # 1. World model strongly predicts termination (< 1% continue probability)
                # 2. We've reached the full imagination horizon
                # 3. Random early termination (very low probability)
                done = (continue_prob < continue_threshold or 
                       step >= self.imagination_horizon - 1 or
                       (step > 200 and np.random.rand() < 0.01))  # Very rare early termination
                
                # Store transition with environment reward
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)  # Use environment reward directly
                trajectory['dones'].append(done)
                trajectory['observations'].append(next_obs.detach().cpu().numpy())
                
                if done:
                    break
                
                current_obs = next_obs.unsqueeze(0)  # [1, obs_dim]
            
            # Only add trajectories with reasonable length (at least 3 steps)
            if len(trajectory['actions']) >= 3:
                imagined_trajectories.append(trajectory)
        
        return imagined_trajectories
    
    def train_actor_critic(self, imagined_trajectories):
        """Train actor-critic with improved loss computation"""
        if not imagined_trajectories:
            return {'actor_loss': 0.0}
        
        total_loss = 0.0
        
        for _ in range(self.actor_train_steps):
            batch_obs = []
            batch_returns = []
            batch_actions = []
            
            for traj in imagined_trajectories:
                # Compute returns with GAE
                returns = []
                G = 0.0
                gamma = 0.99
                
                for reward in reversed(traj['rewards']):
                    G = reward + gamma * G
                    returns.insert(0, G)
                
                # Add to batch
                for i, obs in enumerate(traj['observations'][:-1]):
                    batch_obs.append(torch.tensor(obs, dtype=torch.float32))
                    batch_returns.append(returns[i])
                    batch_actions.append(traj['actions'][i])
            
            if not batch_obs:
                continue
            
            # Convert to tensors
            observations = torch.stack(batch_obs).to(self.device)
            returns = torch.tensor(batch_returns, dtype=torch.float32).to(self.device)
            actions = torch.tensor(batch_actions, dtype=torch.long).to(self.device)
            
            # Forward pass
            action_probs, values = self.actor_critic(observations)
            
            # Compute advantages with better normalization
            advantages = returns - values.detach()
            if advantages.numel() > 1:  # Only normalize if we have multiple samples
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Actor loss (policy gradient) with clipping for stability
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
            actor_loss = -(log_probs * advantages).mean()
            
            # Critic loss with Huber loss for robustness
            critic_loss = F.smooth_l1_loss(values, returns)
            
            # Entropy bonus for exploration
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            
            # Balanced loss with proper weighting
            total_loss_step = actor_loss + 0.5 * critic_loss - 0.01 * entropy  # Reduced entropy weight
            
            # Backward pass
            self.actor_optimizer.zero_grad()
            total_loss_step.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 5.0)
            self.actor_optimizer.step()
            
            total_loss += total_loss_step.item()
        
        return {'actor_loss': total_loss / self.actor_train_steps}
    
    def train_rnd_network(self, observations):
        """Train RND network for exploration"""
        self.rnd_optimizer.zero_grad()
        
        # Convert observations to tensor efficiently
        if not isinstance(observations[0], torch.Tensor):
            # Convert list of numpy arrays to single numpy array first (much faster)
            obs_array = np.array(observations)
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(self.device)
        else:
            obs_tensor = torch.stack([torch.as_tensor(obs, dtype=torch.float32) for obs in observations]).to(self.device)
        
        # Get prediction loss
        rnd_loss = self.rnd_network.get_prediction_loss(obs_tensor)
        
        # Backpropagation
        rnd_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnd_network.predictor_network.parameters(), 1.0)
        self.rnd_optimizer.step()
        
        return rnd_loss.item()
    
    def train(self, num_episodes=100):
        """Main training loop with batched updates and visualization"""
        print(f"Starting Improved DREAM training for {num_episodes} episodes")
        print(f"Batch size: {self.episode_batch_size} episodes")
        print("=" * 60)
        
        batch_trajectories = []
        
        for episode in range(num_episodes):
            # 1. Collect real trajectory
            trajectory = self.collect_trajectory()
            self.replay_buffer.add_trajectory(trajectory)
            batch_trajectories.append(trajectory)
            
            episode_reward = sum(trajectory['rewards'])
            episode_length = len(trajectory['actions'])
            
            # Get current learning rates
            world_lr = self.world_optimizer.param_groups[0]['lr']
            actor_lr = self.actor_optimizer.param_groups[0]['lr']
            
            # Enhanced single episode training (for immediate feedback)
            # Multiple world model training rounds for better accuracy
            world_losses = None
            for _ in range(3):
                world_losses = self.train_world_model()
            
            imagined_trajectories = self.generate_imagined_trajectories(episode, num_episodes)
            actor_losses = self.train_actor_critic(imagined_trajectories)
            
            # Train RND network with collected observations
            rnd_loss = self.train_rnd_network(trajectory['observations'])
            
            # Add to batch tracker
            self.batch_tracker.add_episode(episode, episode_reward, 
                                         world_losses['world_loss'], 
                                         actor_losses['actor_loss'])
            
            # Update visualization per episode
            if self.visualizer:
                self.visualizer.update_episode_data(
                    episode, episode_reward, episode_length,
                    world_losses['world_loss'], actor_losses['actor_loss'],
                    world_lr, actor_lr
                )
                
                # Update game state visualization
                if hasattr(self.env, 'get_board'):
                    board = self.env.get_board()
                    self.visualizer.update_game_state(board, None, 0, 0)
            
            # Batch processing when batch is complete
            if self.batch_tracker.is_batch_complete():
                batch_metrics = self.batch_tracker.finalize_batch()
                
                # Batch training (more intensive updates)
                print(f"\n🔄 Processing Batch {batch_metrics['batch_idx']} "
                      f"(Episodes {batch_metrics['episodes'][0]}-{batch_metrics['episodes'][-1]})")
                
                # Intensive batch training
                for _ in range(3):  # Extra training rounds per batch
                    self.train_world_model()
                    imagined_trajectories = self.generate_imagined_trajectories(episode, num_episodes)
                    self.train_actor_critic(imagined_trajectories)
                
                # Update visualization with batch metrics
                if self.visualizer:
                    self.visualizer.update_batch_data(
                        batch_metrics['batch_idx'],
                        batch_metrics['episodes'],
                        batch_metrics['avg_reward'],
                        batch_metrics['avg_world_loss'],
                        batch_metrics['avg_actor_loss']
                    )
                
                # Visual demonstration every 10 batches
                if batch_metrics['batch_idx'] % 10 == 0:
                    print(f"\n🎮 Running visual Tetris demonstration for batch {batch_metrics['batch_idx']}")
                    self.run_visual_tetris_demonstration()
                
                # Batch performance report
                print(f"✅ Batch {batch_metrics['batch_idx']} Complete:")
                print(f"   Average Reward: {batch_metrics['avg_reward']:.2f}")
                print(f"   World Loss: {batch_metrics['avg_world_loss']:.4f}")
                print(f"   Actor Loss: {batch_metrics['avg_actor_loss']:.4f}")
                
                # Run agent evaluation and integrate into dashboard
                print(f"\nRunning agent evaluation for Batch {batch_metrics['batch_idx']}...")
                batch_eval = self.run_agent_evaluation(episode_count=1)
                if batch_eval and self.visualizer:
                    # Integrate evaluation into dashboard instead of pop-up
                    self.visualizer.update_agent_evaluation(batch_eval, batch_metrics['batch_idx'])
                
                # Generate game state visualization for this batch
                if self.visualizer:
                    self.visualizer.visualize_game_state(
                        save_path=f"batch_{batch_metrics['batch_idx']}_game_state.png"
                    )
                
                batch_trajectories = []  # Reset for next batch
            
            # Update exploration parameters proportionally
            self.update_exploration_proportional(episode, num_episodes)
            
            # Learning rate scheduling - FIXED: Less frequent stepping
            if episode % 15 == 0 and episode > 0:
                self.world_scheduler.step()
                self.actor_scheduler.step()
            
            # Regular logging
            if episode % 10 == 0:
                progress = episode / num_episodes
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Length: {episode_length:3d} | "
                      f"World Loss: {world_losses['world_loss']:.4f} | "
                      f"Actor Loss: {actor_losses['actor_loss']:.4f} | "
                      f"WLR: {world_lr:.2e} | "
                      f"ALR: {actor_lr:.2e} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"T: {self.temperature:.2f} | "
                      f"Prog: {progress:.1%}")
        
        # Final batch processing if incomplete
        if len(self.batch_tracker.current_batch_episodes) > 0:
            print(f"\n🔄 Processing Final Incomplete Batch...")
            # Force finalize the last batch
            self.batch_tracker.batch_size = len(self.batch_tracker.current_batch_episodes)
            batch_metrics = self.batch_tracker.finalize_batch()
            if batch_metrics and self.visualizer:
                self.visualizer.update_batch_data(
                    batch_metrics['batch_idx'],
                    batch_metrics['episodes'],
                    batch_metrics['avg_reward'],
                    batch_metrics['avg_world_loss'],
                    batch_metrics['avg_actor_loss']
                )
        
        # Generate final performance report
        if self.visualizer:
            report = self.visualizer.generate_performance_report()
            print("\n" + report)
            
            # Save training data
            self.visualizer.save_training_data("dream_training_data.pkl")
            
            # Final agent demonstration
            print("\n🎮 Running final agent demonstration...")
            final_eval = self.run_agent_evaluation(episode_count=3)
            if final_eval:
                self.visualizer.visualize_agent_demo(final_eval, "final_agent_demo.png")
            
            # Keep plots open for inspection
            input("\nPress Enter to close visualization and exit...")
            self.visualizer.close()
        
        print("=" * 60)
        print("✅ Improved DREAM training completed!")
        self.env.close()
    
    def run_visual_tetris_demonstration(self):
        """Run a visual Tetris demonstration showing the agent playing"""
        print("🎮 Starting visual Tetris demonstration...")
        
        try:
            # Create visual environment (not headless)
            from envs.tetris_env import TetrisEnv
            visual_env = TetrisEnv(num_agents=1, headless=False, step_mode='action', action_mode='direct')
            
            obs = visual_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            total_reward = 0
            step_count = 0
            game_states = []
            
            print("   🔥 WATCH: Agent is now playing Tetris visually!")
            print("   ⏱️  Duration: ~30 seconds or until game over")
            
            import time
            start_time = time.time()
            
            while True:
                # Get agent action using current policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    if obs_tensor.numel() == 0 or obs_tensor.shape[-1] == 0:
                        action = visual_env.action_space.sample()
                    else:
                        action_probs, _ = self.actor_critic(obs_tensor)
                        action = torch.multinomial(action_probs, 1).item()
                
                # Store game state for analysis
                if hasattr(visual_env, 'get_board'):
                    try:
                        board_state = visual_env.get_board()
                        if board_state is not None:
                            game_states.append(board_state.copy())
                    except:
                        pass
                
                # Take action and render
                step_result = visual_env.step(action)
                visual_env.render()  # Show the actual game
                
                # Handle step result
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                    truncated = False
                elif len(step_result) == 5:
                    obs, reward, done, truncated, info = step_result
                else:
                    obs, reward, done, info = step_result[:4]
                    truncated = False
                
                total_reward += reward
                step_count += 1
                
                # Control demonstration duration (30 seconds max)
                elapsed_time = time.time() - start_time
                if done or truncated or elapsed_time > 30:
                    break
                
                # Slow down for visibility
                time.sleep(0.1)
            
            visual_env.close()
            
            print(f"   🏆 Demonstration complete!")
            print(f"      Final Score: {info.get('score', 0) if isinstance(info, dict) else 0}")
            print(f"      Total Reward: {total_reward:.2f}")
            print(f"      Steps Taken: {step_count}")
            print(f"      Duration: {elapsed_time:.1f}s")
            
            return {
                'score': info.get('score', 0) if isinstance(info, dict) else 0,
                'reward': total_reward,
                'steps': step_count,
                'duration': elapsed_time,
                'game_states': game_states[-5:] if game_states else []
            }
            
        except Exception as e:
            print(f"   ❌ Visual demonstration failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_agent_evaluation(self, episode_count=1):
        """Run agent evaluation games for dashboard integration"""
        print(f"📊 Running {episode_count} agent evaluation game(s)...")
        
        # Create headless environment for evaluation (no pop-up window)
        from envs.tetris_env import TetrisEnv
        eval_env = TetrisEnv(num_agents=1, headless=True, step_mode='action', action_mode='direct')
        
        best_demo = None
        best_reward = float('-inf')
        
        for episode in range(episode_count):
            try:
                obs = eval_env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]  # Handle different gym versions
                
                total_reward = 0
                step_count = 0
                game_states = []
                actions_taken = []
                
                print(f"  Demo episode {episode + 1}/{episode_count}")
                
                step = 0
                while True:  # Unlimited steps - run until natural termination
                    step += 1
                    # Get agent action using current policy
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        # Handle empty observation case
                        if obs_tensor.numel() == 0 or obs_tensor.shape[-1] == 0:
                            action = eval_env.action_space.sample()  # Random action if obs is empty
                        else:
                            action_probs, _ = self.actor_critic(obs_tensor)
                            action = torch.multinomial(action_probs, 1).item()
                    
                    # Store state for visualization
                    if hasattr(eval_env, 'get_board'):
                        try:
                            board_state = eval_env.get_board()
                            game_states.append(board_state.copy())
                        except:
                            pass  # If get_board fails, continue without state
                    
                    actions_taken.append(action)
                    
                    # Take action
                    step_result = eval_env.step(action)
                    
                    # Store game state for dashboard visualization
                    try:
                        if hasattr(eval_env, 'get_board'):
                            board_state = eval_env.get_board()
                            if board_state is not None:
                                game_states.append(board_state.copy())
                    except Exception:
                        pass  # Continue if state capture fails
                    
                    # Handle different step return formats
                    if len(step_result) == 4:
                        obs, reward, done, info = step_result
                        truncated = False
                    elif len(step_result) == 5:
                        obs, reward, done, truncated, info = step_result
                    else:
                        print(f"⚠️  Unexpected step result length: {len(step_result)}")
                        obs, reward, done, info = step_result[:4]
                        truncated = False
                    total_reward += reward
                    step_count += 1
                    
                    if done or truncated:
                        break
                
                # Create demo result
                demo_result = {
                    'episode': episode + 1,
                    'total_reward': total_reward,
                    'steps': step_count,
                    'final_score': info.get('score', 0) if isinstance(info, dict) else 0,
                    'lines_cleared': info.get('lines_cleared', 0) if isinstance(info, dict) else 0,
                    'game_states': game_states[-10:] if game_states else [],
                    'actions': actions_taken
                }
                
                print(f"    Reward: {total_reward:.2f}, Steps: {step_count}, Score: {demo_result['final_score']}")
                
                # Keep best demo for visualization
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_demo = demo_result
                    
            except Exception as e:
                print(f"    ⚠️  Demo episode {episode + 1} failed: {e}")
                continue
        
        eval_env.close()
        
        if best_demo:
            print(f"🏆 Best demo: Reward={best_demo['total_reward']:.2f}, Score={best_demo['final_score']}")
            return best_demo
        else:
            print("❌ No successful demo episodes")
            return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run improved DREAM training with argument parsing and visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DREAM Tetris Training with Visualization')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for batched updates')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    print("IMPROVED DREAM IMPLEMENTATION WITH VISUALIZATION")
    print("Starting enhanced Tetris DREAM training...")
    print(f"Architecture: LayerNorm + Dropout + LSTM")
    print(f"LR Scheduling: StepLR with gamma 0.8/0.9 (FIXED ORDER)")
    print(f"Reward Shaping: Scale 0.1, Survival 0.05")
    print(f"Batched Updates: {args.batch_size} episodes per batch")
    print(f"Visualization: {'Disabled' if args.no_viz else 'Enabled'}")
    print("=" * 60)
    
    trainer = ImprovedDREAMTrainer(
        device=args.device, 
        enable_visualization=not args.no_viz,
        batch_size=args.batch_size
    )
    trainer.train(num_episodes=args.episodes)

if __name__ == "__main__":
    main() 