"""PPO agent for Tetris environment.

This implements a PPO (Proximal Policy Optimization) agent that was trained using AIRL.
The agent uses an actor-critic architecture with shared feature extraction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOAgent(nn.Module):
    """
    PPO agent with actor-critic architecture matching Stable Baselines 3.
    
    Architecture:
    - MLP extractor with separate policy and value networks
    - Action network for policy head
    - Value network for critic head
    """
    def __init__(self, 
                 state_dim: int = 206,
                 action_dim: int = 41,
                 hidden_dim: int = 64,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 c1: float = 1.0,
                 c2: float = 0.01):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space (206 in saved model)
            action_dim: Number of possible actions
            hidden_dim: Size of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: PPO clipping parameter
            c1: Value loss coefficient
            c2: Entropy bonus coefficient
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        
        # MLP Extractor (separate networks for policy and value)
        self.mlp_extractor = nn.ModuleDict({
            'policy_net': nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ),
            'value_net': nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
        })
        
        # Policy head (action network)
        self.action_net = nn.Linear(hidden_dim, action_dim)
        
        # Value head
        self.value_net = nn.Linear(hidden_dim, 1)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        self.to(self.device)
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        # Extract features
        policy_features = self.mlp_extractor['policy_net'](state)
        value_features = self.mlp_extractor['value_net'](state)
        
        # Get action probabilities and value
        action_logits = self.action_net(policy_features)
        state_value = self.value_net(value_features)
        
        return F.softmax(action_logits, dim=-1), state_value
    
    def act(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """
        Select an action given a state.
        
        Args:
            state: Tensor of shape (state_dim,)
            deterministic: If True, select action with highest probability
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            # Get action probabilities and value
            action_probs, _ = self.forward(state)
            
            if deterministic:
                action = torch.argmax(action_probs).item()
            else:
                # Sample from probability distribution
                dist = Categorical(action_probs)
                action = dist.sample().item()
            
            return action
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.
        
        Args:
            states: Tensor of shape (batch_size, state_dim)
            actions: Tensor of shape (batch_size,)
            
        Returns:
            Tuple of (log_probs, state_values, entropy)
        """
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)
        
        # Get log probabilities of taken actions
        log_probs = dist.log_prob(actions)
        
        # Compute entropy for exploration bonus
        entropy = dist.entropy().mean()
        
        return log_probs, state_values.squeeze(), entropy
    
    def compute_loss(self, 
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    old_log_probs: torch.Tensor,
                    returns: torch.Tensor,
                    advantages: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Compute PPO loss.
        
        Args:
            states: Tensor of shape (batch_size, state_dim)
            actions: Tensor of shape (batch_size,)
            old_log_probs: Log probabilities of actions from old policy
            returns: Tensor of shape (batch_size,)
            advantages: Tensor of shape (batch_size,)
            
        Returns:
            Tuple of (total_loss, dict of loss components)
        """
        # Evaluate actions under current policy
        log_probs, values, entropy = self.evaluate_actions(states, actions)
        
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Compute surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        
        # Policy loss (PPO clipped objective)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
        
        return total_loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        } 