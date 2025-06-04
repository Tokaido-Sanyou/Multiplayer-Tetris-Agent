import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Change relative import to module-based import
try:
    from localMultiplayerTetris.rl_utils.replay_buffer import ReplayBuffer
except ImportError:
    from rl_utils.replay_buffer import ReplayBuffer

class Discriminator(nn.Module):
    """
    AIRL Discriminator Network that learns to distinguish between expert and learner trajectories.
    
    The discriminator takes state-action pairs and outputs a reward-like signal.
    In AIRL, the discriminator is trained to output high values for expert transitions
    and low values for learner transitions.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 128]):
        super(Discriminator, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Process state features
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        
        # Process action (one-hot encoded for discrete actions)
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_sizes[1] // 2),
            nn.ReLU()
        )
        
        # Combined processing
        combined_dim = hidden_sizes[1] + hidden_sizes[1] // 2
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output logit for expert vs learner classification
        )
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of discriminator.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim] (one-hot encoded)
            
        Returns:
            Discriminator logits [batch_size, 1]
        """
        state_features = self.state_net(states)
        action_features = self.action_net(actions)
        
        combined = torch.cat([state_features, action_features], dim=-1)
        return self.combined_net(combined)
    
    def get_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get AIRL reward from discriminator output.
        
        The AIRL reward is: r(s,a) = log(D(s,a)) - log(1 - D(s,a))
        where D(s,a) is the discriminator output (probability of being expert)
        """
        logits = self.forward(states, actions)
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        # Compute AIRL reward: log(D/(1-D))
        reward = torch.log(probs + 1e-8) - torch.log(1 - probs + 1e-8)
        return reward

class AIRLAgent:
    """
    Adversarial Inverse Reinforcement Learning Agent.
    
    Combines a policy network (actor-critic) with a discriminator network
    to learn from expert demonstrations.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 policy_network,  # Actor-critic network from existing implementation
                 lr_discriminator: float = 3e-4,
                 lr_policy: float = 3e-4,
                 discriminator_update_freq: int = 1,
                 policy_update_freq: int = 1,
                 gamma: float = 0.99,
                 device: str = 'cpu'):
        
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Initialize discriminator
        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_discriminator)
        
        # Policy network (use existing actor-critic)
        self.policy = policy_network.to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        
        # Update frequencies
        self.discriminator_update_freq = discriminator_update_freq
        self.policy_update_freq = policy_update_freq
        
        # Training counters
        self.update_count = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def update_discriminator(self, 
                           expert_states: torch.Tensor,
                           expert_actions: torch.Tensor,
                           learner_states: torch.Tensor,
                           learner_actions: torch.Tensor) -> Dict[str, float]:
        """
        Update discriminator to distinguish expert from learner trajectories.
        
        Args:
            expert_states: Expert state batch [batch_size, state_dim]
            expert_actions: Expert action batch [batch_size, action_dim] (one-hot)
            learner_states: Learner state batch [batch_size, state_dim]
            learner_actions: Learner action batch [batch_size, action_dim] (one-hot)
            
        Returns:
            Dictionary with discriminator training metrics
        """
        batch_size = expert_states.shape[0]
        
        # Get discriminator predictions
        expert_logits = self.discriminator(expert_states, expert_actions)
        learner_logits = self.discriminator(learner_states, learner_actions)
        
        # Labels: 1 for expert, 0 for learner
        expert_labels = torch.ones(batch_size, 1, device=self.device)
        learner_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Binary cross-entropy loss
        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, expert_labels)
        learner_loss = F.binary_cross_entropy_with_logits(learner_logits, learner_labels)
        
        discriminator_loss = expert_loss + learner_loss
        
        # Update discriminator
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        # Calculate accuracy
        expert_preds = (torch.sigmoid(expert_logits) > 0.5).float()
        learner_preds = (torch.sigmoid(learner_logits) < 0.5).float()
        expert_accuracy = expert_preds.mean().item()
        learner_accuracy = learner_preds.mean().item()
        overall_accuracy = (expert_accuracy + learner_accuracy) / 2
        
        return {
            'discriminator_loss': discriminator_loss.item(),
            'expert_loss': expert_loss.item(),
            'learner_loss': learner_loss.item(),
            'expert_accuracy': expert_accuracy,
            'learner_accuracy': learner_accuracy,
            'overall_accuracy': overall_accuracy
        }
    
    def update_policy(self, 
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     next_states: torch.Tensor,
                     dones: torch.Tensor) -> Dict[str, float]:
        """
        Update policy using AIRL rewards from discriminator.
        
        Args:
            states: State batch [batch_size, state_dim]
            actions: Action batch [batch_size, action_dim] (one-hot)
            next_states: Next state batch [batch_size, state_dim]
            dones: Done flags [batch_size, 1]
            
        Returns:
            Dictionary with policy training metrics
        """
        # Get AIRL rewards from discriminator
        with torch.no_grad():
            airl_rewards = self.discriminator.get_reward(states, actions)
        
        # Get policy predictions
        action_probs, state_values = self.policy(states)
        next_state_values = self.policy(next_states)[1]
        
        # Calculate advantages using AIRL rewards
        targets = airl_rewards + self.gamma * next_state_values * (1 - dones)
        advantages = targets - state_values
        
        # Actor loss (policy gradient with advantages)
        action_indices = actions.argmax(dim=-1)
        log_probs = torch.log(action_probs.gather(1, action_indices.unsqueeze(1)) + 1e-8)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value function approximation)
        critic_loss = F.mse_loss(state_values, targets.detach())
        
        # Combined loss
        policy_loss = actor_loss + 0.5 * critic_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_airl_reward': airl_rewards.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            deterministic: If True, select action deterministically
            
        Returns:
            Selected action index
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)
            
            if deterministic:
                action = action_probs.argmax(dim=-1).item()
            else:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
                
        return action
    
    def train_step(self, 
                   expert_batch: Dict,
                   learner_batch: Dict) -> Dict[str, float]:
        """
        Perform one training step of AIRL.
        
        Args:
            expert_batch: Batch of expert transitions
            learner_batch: Batch of learner transitions
            
        Returns:
            Dictionary with all training metrics
        """
        metrics = {}
        
        # Update discriminator
        if self.update_count % self.discriminator_update_freq == 0:
            disc_metrics = self.update_discriminator(
                expert_batch['states'],
                expert_batch['actions'],
                learner_batch['states'],
                learner_batch['actions']
            )
            metrics.update(disc_metrics)
        
        # Update policy
        if self.update_count % self.policy_update_freq == 0:
            policy_metrics = self.update_policy(
                learner_batch['states'],
                learner_batch['actions'],
                learner_batch['next_states'],
                learner_batch['dones']
            )
            metrics.update(policy_metrics)
        
        self.update_count += 1
        return metrics
    
    def save(self, filepath: str):
        """Save AIRL agent."""
        torch.save({
            'discriminator_state_dict': self.discriminator.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'update_count': self.update_count
        }, filepath)
    
    def load(self, filepath: str):
        """Load AIRL agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0) 