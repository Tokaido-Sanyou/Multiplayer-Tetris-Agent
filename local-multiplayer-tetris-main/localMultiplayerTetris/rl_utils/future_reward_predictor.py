"""
Future Reward Predictor: Predicts long-term rewards for state-action pairs
Inspired by Dream/AIRL architectures for model-based RL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle both direct execution and module import
try:
    from ..config import TetrisConfig  # Import centralized config
except ImportError:
    # Direct execution - add parent directory to path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TetrisConfig  # Import centralized config

class FutureRewardPredictor(nn.Module):
    """
    Predicts future rewards given current state and intended action
    Uses centralized configuration for all network dimensions
    """
    def __init__(self, state_dim=None, action_dim=None):
        super(FutureRewardPredictor, self).__init__()
        
        # Get centralized config
        self.config = TetrisConfig()
        self.net_config = self.config.NetworkConfig.FutureRewardPredictor
        
        # Use centralized dimensions
        self.state_dim = state_dim or self.config.STATE_DIM  # 410
        self.action_dim = action_dim or self.config.ACTION_DIM  # 8
        
        # State encoder with centralized config
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.net_config.STATE_ENCODER_LAYERS[1]),
            nn.ReLU(),
            nn.Linear(self.net_config.STATE_ENCODER_LAYERS[1], self.net_config.STATE_ENCODER_LAYERS[2]),
            nn.ReLU()
        )
        
        # Action encoder with centralized config  
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.net_config.ACTION_ENCODER_LAYERS[1]),
            nn.ReLU()
        )
        
        # Combined encoder with centralized config
        self.combined_encoder = nn.Sequential(
            nn.Linear(self.net_config.COMBINED_INPUT, self.net_config.COMBINED_LAYERS[1]),
            nn.ReLU(),
            nn.Linear(self.net_config.COMBINED_LAYERS[1], self.net_config.COMBINED_LAYERS[2]),
            nn.ReLU()
        )
        
        # Output heads with centralized config
        self.reward_head = nn.Linear(self.net_config.COMBINED_LAYERS[2], self.net_config.REWARD_HEAD_OUTPUT)
        self.value_head = nn.Linear(self.net_config.COMBINED_LAYERS[2], self.net_config.VALUE_HEAD_OUTPUT)
        
    def forward(self, state, action):
        """
        Args:
            state: Tensor of shape (batch_size, state_dim)
            action: Tensor of shape (batch_size, action_dim) - one-hot encoded
        Returns:
            reward_pred: Predicted immediate reward (batch_size, 1)
            value_pred: Predicted future value (batch_size, 1)
        """
        # Encode state and action
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(action)
        
        # Combine features
        combined = torch.cat([state_features, action_features], dim=1)
        combined_features = self.combined_encoder(combined)
        
        # Predict rewards and values
        reward_pred = self.reward_head(combined_features)
        value_pred = self.value_head(combined_features)
        
        return reward_pred, value_pred
    
    def predict_trajectory_value(self, states, actions):
        """
        Predict cumulative value for a trajectory of state-action pairs
        Args:
            states: Tensor of shape (batch_size, seq_len, state_dim)
            actions: Tensor of shape (batch_size, seq_len, action_dim)
        Returns:
            trajectory_value: Predicted cumulative value (batch_size, 1)
        """
        batch_size, seq_len = states.shape[:2]
        
        total_value = torch.zeros(batch_size, 1, device=states.device)
        
        for t in range(seq_len):
            reward_pred, value_pred = self.forward(states[:, t], actions[:, t])
            # Discount future values
            discount = 0.99 ** t
            total_value += discount * (reward_pred + value_pred)
            
        return total_value
