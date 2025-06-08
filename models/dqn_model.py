"""
DQN Model Implementation
Deep Q-Network model specifically designed for Tetris DQN agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .tetris_cnn import TetrisCNN


class DQNModel(TetrisCNN):
    """
    DQN Model for Tetris
    
    Extends TetrisCNN with DQN-specific functionality including:
    - Q-value output layer configuration
    - Optional dueling architecture
    - Noisy layers for exploration
    - Double DQN support
    """
    
    def __init__(self,
                 action_space_size: int = 8,
                 dueling: bool = False,
                 noisy: bool = False,
                 distributional: bool = False,
                 num_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0,
                 **kwargs):
        """
        Initialize DQN Model
        
        Args:
            action_space_size: Number of possible actions
            dueling: Whether to use dueling architecture
            noisy: Whether to use noisy layers for exploration
            distributional: Whether to use distributional DQN (C51)
            num_atoms: Number of atoms for distributional DQN
            v_min: Minimum value for distributional DQN
            v_max: Maximum value for distributional DQN
            **kwargs: Additional arguments passed to TetrisCNN
        """
        # Configure base CNN parameters
        cnn_config = {
            'output_size': action_space_size,
            'activation_type': 'identity',  # For Q-values
            'use_dropout': kwargs.get('use_dropout', True),
            'dropout_rate': kwargs.get('dropout_rate', 0.1),
            **kwargs
        }
        
        super().__init__(**cnn_config)
        
        # DQN-specific configuration
        self.action_space_size = action_space_size
        self.dueling = dueling
        self.noisy = noisy
        self.distributional = distributional
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Replace the final layer based on architecture
        if self.dueling:
            self._build_dueling_layers()
        elif self.distributional:
            self._build_distributional_layers()
        elif self.noisy:
            self._build_noisy_layers()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_dueling_layers(self):
        """Build dueling DQN architecture"""
        # Get the size of the feature layer (fc1 output is 256)
        feature_size = 256
        
        # Remove the final layer and replace with dueling streams
        self.fc_out = None  # Remove original output layer
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space_size)
        )
    
    def _build_distributional_layers(self):
        """Build distributional DQN (C51) architecture"""
        # Replace final layer for distributional output
        self.fc_out = nn.Linear(256, self.action_space_size * self.num_atoms)
        
        # Create support for value distribution
        self.register_buffer('support', torch.linspace(self.v_min, self.v_max, self.num_atoms))
    
    def _build_noisy_layers(self):
        """Build noisy layers for exploration"""
        # Replace linear layers with noisy versions
        self.fc1 = NoisyLinear(self.fc1.in_features, self.fc1.out_features)
        self.fc_out = NoisyLinear(self.fc_out.in_features, self.fc_out.out_features)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Q-values or distribution logits
        """
        if self.dueling:
            return self._forward_dueling(x)
        elif self.distributional:
            return self._forward_distributional(x)
        else:
            return super().forward(x)
    
    def _forward_dueling(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for dueling architecture"""
        # Extract features using parent's feature extraction (up to fc1)
        features = self.extract_features(x)
        
        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine using dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def _forward_distributional(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for distributional DQN"""
        # Get logits for each action-atom pair
        logits = super().forward(x)
        
        # Reshape to [batch_size, num_actions, num_atoms]
        batch_size = logits.size(0)
        logits = logits.view(batch_size, self.action_space_size, self.num_atoms)
        
        # Apply softmax over atoms dimension
        probabilities = F.softmax(logits, dim=2)
        
        return probabilities
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values from the network output
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values tensor [batch_size, num_actions]
        """
        if self.distributional:
            # For distributional DQN, compute expected Q-values
            probabilities = self.forward(x)
            q_values = torch.sum(probabilities * self.support.view(1, 1, -1), dim=2)
            return q_values
        else:
            return self.forward(x)
    
    def get_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get value distribution (only for distributional DQN)
        
        Args:
            x: Input tensor
            
        Returns:
            Distribution probabilities [batch_size, num_actions, num_atoms]
        """
        if not self.distributional:
            raise ValueError("Distribution only available for distributional DQN")
        
        return self.forward(x)
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary containing model configuration and statistics
        """
        info = {
            'model_type': 'DQNModel',
            'action_space_size': self.action_space_size,
            'dueling': self.dueling,
            'noisy': self.noisy,
            'distributional': self.distributional,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        if self.distributional:
            info.update({
                'num_atoms': self.num_atoms,
                'v_min': self.v_min,
                'v_max': self.v_max
            })
        
        return info


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for NoisyNet DQN
    
    Implements factorised Gaussian noise as described in:
    "Noisy Networks for Exploration" (Fortunato et al., 2017)
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        """
        Initialize noisy linear layer
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            sigma_init: Initial value for noise parameter
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Noise buffers
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / (self.in_features ** 0.5)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5))
    
    def reset_noise(self):
        """Generate new noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias) 