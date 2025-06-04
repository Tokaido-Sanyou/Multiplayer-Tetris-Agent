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
        self.action_dim = 41  # 40 placements + 1 hold
        self.hidden_dim = 128
        self.feature_dim = 128  # Compact feature dimension
        
        # Training parameters
        self.batch_size = 128
        self.learning_rate_discriminator = 1e-4
        self.learning_rate_policy = 1e-4
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
        
        # Value network (for actor-critic)
        self.value = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
    def forward(self, state):
        """
        Policy forward pass
        Args:
            state: (batch_size, state_dim)
        Returns:
            action_probs: (batch_size, action_dim)
            value: (batch_size, 1)
        """
        features = self.feature_extractor(state)
        action_probs = self.policy(features)
        value = self.value(features)
        return action_probs, value
    
    def get_action_and_value(self, state, action=None):
        """Get action and value with log probability"""
        action_probs, value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value

class AIRLAgent:
    """Adversarial Imitation Learning Agent"""
    
    def __init__(self, config: AIRLConfig):
        self.config = config
        
        # Initialize networks
        self.discriminator = AIRLDiscriminator(config).to(config.device)
        self.policy = AIRLPolicy(config).to(config.device)
        
        # Initialize optimizers
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=config.learning_rate_discriminator
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=config.learning_rate_policy
        )
        
        # Training statistics
        self.training_stats = {
            'discriminator_loss': deque(maxlen=100),
            'policy_loss': deque(maxlen=100),
            'expert_discrimination': deque(maxlen=100),
            'policy_discrimination': deque(maxlen=100),
        }
        
        logging.info(f"AIRL Agent initialized on {config.device}")
        logging.info(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters())}")
        logging.info(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters())}")
    
    def train_discriminator(self, expert_batch: Dict, policy_batch: Dict) -> float:
        """Train discriminator to distinguish expert from policy"""
        self.discriminator.train()
        
        # Move to device
        device = self.config.device
        for batch in [expert_batch, policy_batch]:
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
        
        # Forward pass
        expert_disc = self.discriminator(
            expert_batch['states'], 
            expert_batch['actions'], 
            expert_batch['next_states']
        )
        
        policy_disc = self.discriminator(
            policy_batch['states'], 
            policy_batch['actions'], 
            policy_batch['next_states']
        )
        
        # Binary classification loss
        expert_labels = torch.ones_like(expert_disc)
        policy_labels = torch.zeros_like(policy_disc)
        
        expert_loss = F.binary_cross_entropy(expert_disc, expert_labels)
        policy_loss = F.binary_cross_entropy(policy_disc, policy_labels)
        
        discriminator_loss = expert_loss + policy_loss
        
        # Gradient penalty for stability
        if self.config.gradient_penalty_coeff > 0:
            gp = self.compute_gradient_penalty(expert_batch, policy_batch)
            discriminator_loss += self.config.gradient_penalty_coeff * gp
        
        # Backward pass
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        # Update statistics
        self.training_stats['discriminator_loss'].append(discriminator_loss.item())
        self.training_stats['expert_discrimination'].append(expert_disc.mean().item())
        self.training_stats['policy_discrimination'].append(policy_disc.mean().item())
        
        return discriminator_loss.item()
    
    def train_policy(self, policy_batch: Dict) -> float:
        """Train policy using AIRL rewards"""
        self.policy.train()
        self.discriminator.eval()
        
        # Move to device
        device = self.config.device
        for key in policy_batch:
            if torch.is_tensor(policy_batch[key]):
                policy_batch[key] = policy_batch[key].to(device)
        
        # Get AIRL rewards from discriminator
        with torch.no_grad():
            disc_output = self.discriminator(
                policy_batch['states'],
                policy_batch['actions'],
                policy_batch['next_states']
            )
            # AIRL reward: log(D(s,a,s')) - log(1 - D(s,a,s'))
            airl_rewards = torch.log(disc_output + 1e-8) - torch.log(1 - disc_output + 1e-8)
            airl_rewards = airl_rewards.squeeze()
        
        # Policy forward pass
        action, log_prob, entropy, value = self.policy.get_action_and_value(
            policy_batch['states'], 
            policy_batch['actions']
        )
        
        # Compute advantages (simplified GAE)
        advantages = airl_rewards - value.squeeze()
        returns = airl_rewards
        
        # Policy loss (PPO-style)
        policy_loss = -(log_prob * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(value.squeeze(), returns.detach())
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.config.value_loss_coeff * value_loss + 
                     self.config.entropy_coeff * entropy_loss)
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        
        # Update statistics
        self.training_stats['policy_loss'].append(total_loss.item())
        
        return total_loss.item()
    
    def compute_gradient_penalty(self, expert_batch: Dict, policy_batch: Dict) -> torch.Tensor:
        """Compute gradient penalty for discriminator stability"""
        batch_size = expert_batch['states'].size(0)
        device = self.config.device
        
        # Interpolate between expert and policy data
        alpha = torch.rand(batch_size, 1).to(device)
        
        # Interpolated states
        expert_states = expert_batch['states']
        policy_states = policy_batch['states']
        interpolated_states = alpha * expert_states + (1 - alpha) * policy_states
        interpolated_states.requires_grad_(True)
        
        # Use expert actions and next states for simplicity
        interpolated_actions = expert_batch['actions']
        interpolated_next_states = expert_batch['next_states']
        
        # Discriminator forward pass
        disc_interpolated = self.discriminator(
            interpolated_states, 
            interpolated_actions, 
            interpolated_next_states
        )
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated_states,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def save(self, filepath: str):
        """Save agent state"""
        checkpoint = {
            'discriminator_state_dict': self.discriminator.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'training_stats': dict(self.training_stats),
            'config': self.config
        }
        torch.save(checkpoint, filepath)
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
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_latest"] = values[-1]
        return stats 