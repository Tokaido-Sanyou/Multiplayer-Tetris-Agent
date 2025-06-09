"""
DREAM Actor-Critic Networks

Implements the policy (actor) and value function (critic) networks for DREAM.
Supports both direct and locked position action modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from dream.models.observation_model import MLP


class Actor(nn.Module):
    """
    Policy network for DREAM
    
    Takes state representation and outputs action probabilities.
    Supports different action modes for Tetris.
    """
    
    def __init__(self,
                 state_dim: int = 212,
                 action_dim: int = 8,
                 hidden_dim: int = 400,
                 action_mode: str = 'direct',
                 num_layers: int = 2):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_mode = action_mode
        
        # Feature extraction
        self.feature_net = MLP(
            input_dim=state_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation='relu'
        )
        
        if action_mode == 'direct':
            # Direct mode: 8 binary actions (independent Bernoulli)
            self.action_head = nn.Linear(hidden_dim, action_dim)
            self.action_activation = nn.Sigmoid
        elif action_mode == 'locked_position':
            # Locked position mode: 200 discrete positions (categorical)
            self.action_head = nn.Linear(hidden_dim, action_dim)
            self.action_activation = nn.Softmax
        else:
            raise ValueError(f"Unsupported action mode: {action_mode}")
    
    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        """
        Forward pass through actor network
        
        Args:
            state: State representation [batch, state_dim]
            
        Returns:
            Action distribution
        """
        features = self.feature_net(state)
        logits = self.action_head(features)
        
        if self.action_mode == 'direct':
            # Independent Bernoulli distributions for each action
            probs = torch.sigmoid(logits)
            # Clamp probabilities to avoid numerical issues
            probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
            return torch.distributions.Independent(
                torch.distributions.Bernoulli(probs), 1
            )
        elif self.action_mode == 'locked_position':
            # Categorical distribution over positions
            return torch.distributions.Categorical(logits=logits)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from state
        
        Args:
            state: State representation
            deterministic: If True, use mode/mean instead of sampling
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        dist = self.forward(state)
        
        if deterministic:
            if self.action_mode == 'direct':
                action = (dist.mean > 0.5).float()
            else:  # locked_position
                action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states
        
        Args:
            state: State representation
            action: Actions to evaluate
            
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of action distribution
        """
        dist = self.forward(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class Critic(nn.Module):
    """
    Value function network for DREAM
    
    Takes state representation and outputs value estimate.
    """
    
    def __init__(self,
                 state_dim: int = 212,
                 hidden_dim: int = 400,
                 num_layers: int = 2):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Value network
        self.value_net = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation='relu'
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network
        
        Args:
            state: State representation [batch, state_dim]
            
        Returns:
            Value estimate [batch]
        """
        value = self.value_net(state).squeeze(-1)
        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for DREAM
    
    Combines policy and value networks with shared feature extraction.
    """
    
    def __init__(self,
                 state_dim: int = 212,
                 action_dim: int = 8,
                 hidden_dim: int = 400,
                 action_mode: str = 'direct',
                 num_layers: int = 2,
                 shared_features: bool = True):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_mode = action_mode
        self.shared_features = shared_features
        
        if shared_features:
            # Shared feature extraction
            self.shared_net = MLP(
                input_dim=state_dim,
                output_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation='relu'
            )
            
            # Separate heads
            if action_mode == 'direct':
                self.actor_head = nn.Linear(hidden_dim, action_dim)
            else:  # locked_position
                self.actor_head = nn.Linear(hidden_dim, action_dim)
            
            self.critic_head = nn.Linear(hidden_dim, 1)
            
        else:
            # Separate networks
            self.actor = Actor(state_dim, action_dim, hidden_dim, action_mode, num_layers)
            self.critic = Critic(state_dim, hidden_dim, num_layers)
        
        # Target critic for stability (updated periodically)
        if shared_features:
            self.target_critic_head = nn.Linear(hidden_dim, 1)
            self.target_critic_head.load_state_dict(self.critic_head.state_dict())
        else:
            self.target_critic = Critic(state_dim, hidden_dim, num_layers)
            self.target_critic.load_state_dict(self.critic.state_dict())
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass through both actor and critic
        
        Args:
            state: State representation [batch, state_dim]
            
        Returns:
            action_dist: Action distribution from actor
            value: Value estimate from critic
        """
        if self.shared_features:
            features = self.shared_net(state)
            
            # Actor output
            logits = self.actor_head(features)
            if self.action_mode == 'direct':
                probs = torch.sigmoid(logits)
                probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
                action_dist = torch.distributions.Independent(
                    torch.distributions.Bernoulli(probs), 1
                )
            else:  # locked_position
                action_dist = torch.distributions.Categorical(logits=logits)
            
            # Critic output
            value = self.critic_head(features).squeeze(-1)
            
        else:
            action_dist = self.actor(state)
            value = self.critic(state)
        
        return action_dist, value
    
    def get_action_and_value(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and value estimate from state
        
        Args:
            state: State representation
            deterministic: If True, use deterministic action selection
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: Value estimate
        """
        action_dist, value = self.forward(state)
        
        if deterministic:
            if self.action_mode == 'direct':
                action = (action_dist.mean > 0.5).float()
            else:  # locked_position
                action = torch.argmax(action_dist.logits, dim=-1)
        else:
            action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action)
        return action, log_prob, value
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states
        
        Args:
            state: State representation
            action: Actions to evaluate
            
        Returns:
            log_prob: Log probability of actions
            value: Value estimate
            entropy: Entropy of action distribution
        """
        action_dist, value = self.forward(state)
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return log_prob, value, entropy
    
    def get_value(self, state: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """
        Get value estimate (optionally from target network)
        
        Args:
            state: State representation
            use_target: If True, use target critic
            
        Returns:
            Value estimate
        """
        if use_target:
            if self.shared_features:
                with torch.no_grad():
                    features = self.shared_net(state)
                    value = self.target_critic_head(features).squeeze(-1)
            else:
                with torch.no_grad():
                    value = self.target_critic(state)
        else:
            if self.shared_features:
                features = self.shared_net(state)
                value = self.critic_head(features).squeeze(-1)
            else:
                value = self.critic(state)
        
        return value
    
    def update_target_critic(self, tau: float = 1.0):
        """
        Update target critic network
        
        Args:
            tau: Soft update parameter (1.0 for hard update)
        """
        if self.shared_features:
            # Soft update of target critic head
            for target_param, param in zip(self.target_critic_head.parameters(), self.critic_head.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        else:
            # Soft update of target critic network
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class DynamicActorCritic(ActorCritic):
    """
    Actor-Critic with dynamic action space adaptation
    
    Can switch between different action modes during training or evaluation.
    Useful for curriculum learning or multi-task scenarios.
    """
    
    def __init__(self, 
                 state_dim: int = 212,
                 hidden_dim: int = 400,
                 num_layers: int = 2):
        
        # Initialize with direct mode first
        super().__init__(
            state_dim=state_dim,
            action_dim=8,  # Will be overridden
            hidden_dim=hidden_dim,
            action_mode='direct',
            num_layers=num_layers,
            shared_features=True
        )
        
        # Create heads for both action modes
        self.direct_head = nn.Linear(hidden_dim, 8)  # 8 binary actions
        self.locked_position_head = nn.Linear(hidden_dim, 200)  # 200 positions
        
        # Current mode
        self.current_mode = 'direct'
        self.current_action_dim = 8
    
    def set_action_mode(self, mode: str):
        """Switch action mode"""
        if mode not in ['direct', 'locked_position']:
            raise ValueError(f"Unsupported action mode: {mode}")
        
        self.current_mode = mode
        self.action_mode = mode
        self.current_action_dim = 8 if mode == 'direct' else 200
        self.action_dim = self.current_action_dim
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """Forward pass with dynamic action head selection"""
        features = self.shared_net(state)
        
        # Select action head based on current mode
        if self.current_mode == 'direct':
            logits = self.direct_head(features)
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
            action_dist = torch.distributions.Independent(
                torch.distributions.Bernoulli(probs), 1
            )
        else:  # locked_position
            logits = self.locked_position_head(features)
            action_dist = torch.distributions.Categorical(logits=logits)
        
        # Critic output (shared)
        value = self.critic_head(features).squeeze(-1)
        
        return action_dist, value 