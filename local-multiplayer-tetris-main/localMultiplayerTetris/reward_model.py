"""Reward model for Tetris environment.

This implements a reward network that was trained using AIRL (Adversarial Inverse Reinforcement Learning).
The network takes state-action pairs and outputs reward values.
"""
from __future__ import annotations

import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """
    Neural network that predicts rewards from state-action pairs.
    
    Architecture:
    - Simple MLP that processes concatenated state-action pairs
    """
    def __init__(self, state_dim: int = 207, action_dim: int = 41, hidden_dim: int = 32):
        """
        Initialize reward network.
        
        Args:
            state_dim: Dimension of state space (207 = 200 grid + 7 metadata)
            action_dim: Number of possible actions (41 = 40 placements + hold)
            hidden_dim: Size of hidden layers (32 in the saved model)
        """
        super().__init__()
        
        # Simple MLP architecture
        self.mlp = nn.ModuleDict({
            'dense0': nn.Linear(state_dim + action_dim - 1, hidden_dim),  # -1 because action is scalar
            'dense1': nn.Linear(hidden_dim, hidden_dim),
            'dense_final': nn.Linear(hidden_dim, 1)
        })
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards for state-action pairs.
        
        Args:
            states: Tensor of shape (batch_size, state_dim)
            actions: Tensor of shape (batch_size,) with integer actions
            
        Returns:
            Tensor of shape (batch_size, 1) containing predicted rewards
        """
        # Convert actions to one-hot
        action_one_hot = torch.zeros(actions.shape[0], 40, device=states.device)  # 40 actions (no hold)
        action_one_hot.scatter_(1, actions.unsqueeze(1), 1)
        
        # Concatenate state and action
        x = torch.cat([states, action_one_hot], dim=1)
        
        # Forward through MLP
        x = torch.relu(self.mlp.dense0(x))
        x = torch.relu(self.mlp.dense1(x))
        rewards = self.mlp.dense_final(x)
        
        return rewards
    
    def predict_reward(self, state: torch.Tensor, action: int) -> float:
        """
        Predict reward for a single state-action pair.
        
        Args:
            state: Tensor of shape (state_dim,)
            action: Integer action
            
        Returns:
            Predicted reward value
        """
        with torch.no_grad():
            # Add batch dimension
            state = state.unsqueeze(0)
            action = torch.tensor([action], device=state.device)
            
            # Get reward
            reward = self.forward(state, action)
            return reward.item() 