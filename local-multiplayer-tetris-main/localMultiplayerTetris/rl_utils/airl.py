import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

class AIRLConfig:
    """Configuration class for AIRL parameters"""
    
    def __init__(self):
        # Network architecture
        self.state_dim = 207  # Grid (200) + scalars (7)
        self.action_dim = 41  # 40 placements + 1 hold (original action space)
        self.hidden_dim = 128
        self.feature_dim = 128  # Multiple of common dimensions
        
        # Training parameters
        self.batch_size = 128
        self.learning_rate_discriminator = 1e-4
        self.learning_rate_policy = 1e-4
        self.learning_rate_value = 1e-3
        self.gamma = 0.99
        self.tau = 0.95  # GAE parameter
        
        # AIRL specific
        self.discriminator_steps = 5
        self.gradient_penalty_coeff = 10.0
        self.entropy_coeff = 0.01
        self.value_loss_coeff = 0.5
        
        # Training schedule
        self.max_episodes = 5000
        self.eval_interval = 100
        self.save_interval = 200
        self.log_interval = 10
        
        # Parallel training
        self.num_workers = max(1, mp.cpu_count() - 1)
        self.parallel_envs = 4
        
        # Expert data
        self.expert_data_path = "expert_trajectories/expert_dataset.pkl"
        self.min_expert_episodes = 50
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")

class CompactFeatureExtractor(nn.Module):
    """
    Compact feature extractor maintaining ≥200 nodes as requested
    State: 207 dimensions (Grid: 200 + Scalars: 7)
    Output: 128 dimensions for efficient computation
    """
    
    def __init__(self, config: AIRLConfig):
        super(CompactFeatureExtractor, self).__init__()
        
        # Grid processing: maintain ≥200 nodes as requested
        # 200 -> 400 -> 200 progression
        self.grid_net = nn.Sequential(
            nn.Linear(200, 400),    # Expand for richer representation
            nn.ReLU(),
            nn.Linear(400, 200),    # Back to ≥200 as requested
            nn.ReLU()
        )
        
        # Scalar processing: multiples of input dimensions as requested
        # 7 -> 14 -> 28 progression (multiples of 7)
        self.scalar_net = nn.Sequential(
            nn.Linear(7, 14),       # 2x input
            nn.ReLU(),
            nn.Linear(14, 28),      # 4x input
            nn.ReLU()
        )
        
        # Final combination layer: 200 + 28 = 228 -> 128 for compact representation
        self.combine_net = nn.Sequential(
            nn.Linear(228, 128),    # Compact feature dimension
            nn.ReLU()
        )
        
    def forward(self, state):
        """
        Extract features from state
        Args:
            state: (batch_size, 207) - Grid (200) + Scalars (7)
        Returns:
            features: (batch_size, 128)
        """
        # Ensure state is on the correct device
        if hasattr(self, 'training') and self.training:
            device = next(self.parameters()).device
            if state.device != device:
                state = state.to(device)
        
        grid = state[:, :200]       # Grid portion
        scalars = state[:, 200:]    # Scalar portion (7 dimensions)
        
        # Process each component
        grid_features = self.grid_net(grid)      # 200 -> 200
        scalar_features = self.scalar_net(scalars)  # 7 -> 28
        
        # Combine features
        combined = torch.cat([grid_features, scalar_features], dim=1)  # 228
        features = self.combine_net(combined)  # 128
        
        return features

class AIRLDiscriminator(nn.Module):
    """
    AIRL Discriminator: D(s,a,s') that learns to distinguish expert from policy
    Under 300k parameters total constraint
    """
    
    def __init__(self, config: AIRLConfig):
        super(AIRLDiscriminator, self).__init__()
        
        self.config = config
        self.feature_extractor = CompactFeatureExtractor(config)
        
        # Action embedding: 41 -> compact representation
        self.action_embed = nn.Sequential(
            nn.Embedding(config.action_dim, 32),
            nn.ReLU()
        )
        
        # Combined discriminator: state features + action + next state features
        # 128 + 32 + 128 = 288 input
        self.discriminator = nn.Sequential(
            nn.Linear(288, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state, action, next_state):
        """
        Discriminator forward pass
        Args:
            state: (batch_size, state_dim)
            action: (batch_size,) - integer actions
            next_state: (batch_size, state_dim)
        Returns:
            discrimination: (batch_size, 1) - probability of being expert
        """
        # Ensure all tensors are on the same device
        device = next(self.parameters()).device
        if state.device != device:
            state = state.to(device)
        if action.device != device:
            action = action.to(device)
        if next_state.device != device:
            next_state = next_state.to(device)
        
        # Extract features
        state_features = self.feature_extractor(state)
        next_state_features = self.feature_extractor(next_state)
        
        # Embed actions
        action_features = self.action_embed(action.long())
        
        # Combine all features
        combined = torch.cat([state_features, action_features, next_state_features], dim=1)
        
        # Discriminate
        disc_output = self.discriminator(combined)
        
        return disc_output

class AIRLPolicy(nn.Module):
    """
    AIRL Policy Network
    Lightweight policy for imitation learning
    """
    
    def __init__(self, config: AIRLConfig):
        super(AIRLPolicy, self).__init__()
        
        self.config = config
        self.feature_extractor = CompactFeatureExtractor(config)
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        """
        Policy forward pass
        Args:
            state: (batch_size, state_dim)
        Returns:
            action_probs: (batch_size, action_dim)
        """
        # Ensure state is on the correct device
        device = next(self.parameters()).device
        if state.device != device:
            state = state.to(device)
        
        features = self.feature_extractor(state)
        action_probs = self.policy(features)
        return action_probs

class ParallelTrainingBuffer:
    """Buffer for collecting parallel training data"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = mp.Lock()
    
    def add(self, data_batch: List[Dict]):
        """Add batch of experiences"""
        with self.lock:
            self.buffer.extend(data_batch)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample batch of experiences"""
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

class AIRLAgent:
    """Adversarial Imitation Learning Agent with Parallel Training Support"""
    
    def __init__(self, config: AIRLConfig):
        self.config = config
        
        # Initialize networks
        self.discriminator = AIRLDiscriminator(
            config.state_dim, 
            config.action_dim,  # 41 actions
            config.feature_dim
        ).to(config.device)
        
        self.policy = AIRLPolicy(
            config.state_dim,
            config.action_dim,  # 41 actions
            config.hidden_dim
        ).to(config.device)
        
        # Optimizers
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate_discriminator
        )
        
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate_policy
        )
        
        # Experience buffers
        self.policy_buffer = ParallelTrainingBuffer(10000)
        self.expert_data = None
        
        # Training metrics
        self.training_stats = {
            'discriminator_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'expert_accuracy': [],
            'policy_accuracy': []
        }
        
        logging.info(f"AIRL Agent initialized on {config.device}")
        logging.info(f"Action space: {config.action_dim} actions (0-{config.action_dim-1})")
        logging.info(f"Discriminator parameters: {self.count_parameters(self.discriminator)}")
        logging.info(f"Policy parameters: {self.count_parameters(self.policy)}")
    
    def count_parameters(self, model):
        """Count trainable parameters in a model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def load_expert_data(self, expert_data_path: str):
        """Load expert demonstrations"""
        if not os.path.exists(expert_data_path):
            raise FileNotFoundError(f"Expert data not found: {expert_data_path}")
        
        with open(expert_data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Use actions as-is (should be 41-action space)
        actions = data['actions']
        
        # Validate action range
        if np.max(actions) >= 41 or np.min(actions) < 0:
            logging.warning(f"Action range {np.min(actions)}-{np.max(actions)} not in expected 0-40 range")
            # Clip to valid range
            actions = np.clip(actions, 0, 40)
        
        self.expert_data = {
            'states': torch.FloatTensor(data['states']).to(self.config.device),
            'actions': torch.LongTensor(actions).to(self.config.device),
            'rewards': torch.FloatTensor(data['rewards']).to(self.config.device)
        }
        
        logging.info(f"Loaded expert data: {len(data['states'])} transitions")
        logging.info(f"Action range: {actions.min()}-{actions.max()}")
        logging.info(f"Expert performance: {data.get('avg_reward', 'N/A')}")
    
    def select_action(self, state, deterministic=False):
        """Select action using current policy"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            logits, value = self.policy(state)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
        
        return action.cpu().item()
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to policy buffer"""
        self.policy_buffer.add([{
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }])
    
    def sample_expert_batch(self, batch_size):
        """Sample batch from expert data"""
        if self.expert_data is None:
            raise ValueError("Expert data not loaded")
        
        total_size = len(self.expert_data['states'])
        indices = torch.randperm(total_size)[:batch_size]
        
        return {
            'states': self.expert_data['states'][indices],
            'actions': self.expert_data['actions'][indices],
            'rewards': self.expert_data['rewards'][indices]
        }
    
    def sample_policy_batch(self, batch_size):
        """Sample batch from policy buffer"""
        batch = self.policy_buffer.sample(batch_size)
        if batch is None:
            return None
        
        states = torch.FloatTensor([b['state'] for b in batch]).to(self.config.device)
        actions = torch.LongTensor([b['action'] for b in batch]).to(self.config.device)
        rewards = torch.FloatTensor([b['reward'] for b in batch]).to(self.config.device)
        next_states = torch.FloatTensor([b['next_state'] for b in batch]).to(self.config.device)
        dones = torch.BoolTensor([b['done'] for b in batch]).to(self.config.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def update_discriminator(self):
        """Update discriminator to distinguish expert from policy"""
        if self.expert_data is None:
            return None
        
        total_loss = 0
        expert_acc = 0
        policy_acc = 0
        
        for _ in range(self.config.discriminator_steps):
            # Sample batches
            expert_batch = self.sample_expert_batch(self.config.batch_size // 2)
            policy_batch = self.sample_policy_batch(self.config.batch_size // 2)
            
            if policy_batch is None:
                continue
            
            # Expert forward pass
            expert_rewards, expert_values = self.discriminator(
                expert_batch['states'], 
                expert_batch['actions']
            )
            expert_logits = expert_rewards
            
            # Policy forward pass
            policy_rewards, policy_values = self.discriminator(
                policy_batch['states'],
                policy_batch['actions']
            )
            policy_logits = policy_rewards
            
            # Discriminator loss (binary classification)
            expert_labels = torch.ones_like(expert_logits)
            policy_labels = torch.zeros_like(policy_logits)
            
            expert_loss = F.binary_cross_entropy_with_logits(expert_logits, expert_labels)
            policy_loss = F.binary_cross_entropy_with_logits(policy_logits, policy_labels)
            
            loss = expert_loss + policy_loss
            
            # Update discriminator
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.discriminator_optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracies
            expert_pred = (torch.sigmoid(expert_logits) > 0.5).float()
            policy_pred = (torch.sigmoid(policy_logits) > 0.5).float()
            expert_acc += (expert_pred == expert_labels).float().mean().item()
            policy_acc += (policy_pred == policy_labels).float().mean().item()
        
        # Average over discriminator steps
        avg_loss = total_loss / self.config.discriminator_steps
        avg_expert_acc = expert_acc / self.config.discriminator_steps
        avg_policy_acc = policy_acc / self.config.discriminator_steps
        
        self.training_stats['discriminator_loss'].append(avg_loss)
        self.training_stats['expert_accuracy'].append(avg_expert_acc)
        self.training_stats['policy_accuracy'].append(avg_policy_acc)
        
        return {
            'discriminator_loss': avg_loss,
            'expert_accuracy': avg_expert_acc,
            'policy_accuracy': avg_policy_acc
        }
    
    def update_policy(self):
        """Update policy using AIRL rewards"""
        policy_batch = self.sample_policy_batch(self.config.batch_size)
        if policy_batch is None:
            return None
        
        states = policy_batch['states']
        actions = policy_batch['actions']
        next_states = policy_batch['next_states']
        dones = policy_batch['dones']
        
        # Get AIRL rewards
        with torch.no_grad():
            airl_rewards, _, _ = self.discriminator(states, actions, next_states)
            airl_rewards = airl_rewards.squeeze()
        
        # Policy forward pass
        action_probs = self.policy(states)
        
        # Compute advantages using GAE
        advantages = self.compute_advantages(airl_rewards, action_probs, dones)
        returns = advantages + action_probs.squeeze()
        
        # Policy loss
        policy_loss = -(advantages.detach() * action_probs.log()).mean()
        
        # Value loss
        value_loss = F.mse_loss(action_probs.squeeze(), returns.detach())
        
        # Entropy loss
        entropy_loss = -action_probs.log().mean()
        
        # Total loss
        total_loss = (policy_loss + 
                      self.config.value_loss_coeff * value_loss + 
                      self.config.entropy_coeff * entropy_loss)
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Store stats
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['value_loss'].append(value_loss.item())
        self.training_stats['entropy_loss'].append(entropy_loss.item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def compute_advantages(self, rewards, values, dones):
        """Compute GAE advantages"""
        advantages = torch.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            advantage = delta + self.config.gamma * self.config.tau * advantage
            advantages[t] = advantage
        
        return advantages
    
    def save(self, filepath: str):
        """Save agent state"""
        torch.save({
            'discriminator_state_dict': self.discriminator.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, filepath)
        logging.info(f"AIRL agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        logging.info(f"AIRL agent loaded from {filepath}")
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        stats = {}
        for key, values in self.training_stats.items():
            if values:
                stats[f"{key}_mean"] = np.mean(values[-100:])  # Last 100 updates
                stats[f"{key}_latest"] = values[-1]
        return stats 