"""
Future Reward Predictor: Predicts long-term rewards for terminal block placements
Blends with state model to predict future rewards of optimal placements
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
    Predicts future rewards for terminal block placements
    Blends with state model to predict long-term consequences of optimal placements
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
    
    def predict_terminal_placement_value(self, state, placement_goal):
        """
        Predict the future reward of a terminal block placement given a goal from state model
        Args:
            state: Tensor of shape (batch_size, state_dim) - current state
            placement_goal: Tensor of shape (batch_size, goal_dim) - goal from state model
        Returns:
            terminal_value: Predicted long-term value of the terminal placement
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Extract placement information from goal vector
        # Goal format: [rotation_one_hot(4) + x_position_one_hot(10) + y_position_one_hot(20) + value(1) + confidence(1)]
        
        # Convert placement goal to action representation for prediction
        # For simplicity, map placement to the most relevant action (hard drop)
        action_one_hot = torch.zeros(batch_size, self.action_dim, device=device)
        action_one_hot[:, 5] = 1.0  # Hard drop action
        
        # Get reward and value predictions
        reward_pred, value_pred = self.forward(state, action_one_hot)
        
        # Weight the prediction by the confidence from the state model (index 35 in goal)
        confidence = placement_goal[:, 35:36]  # Shape: (batch_size, 1)
        state_model_value = placement_goal[:, 34:35]  # Shape: (batch_size, 1)
        
        # Blend state model value with our future prediction
        # High confidence → trust state model more, low confidence → trust our prediction more
        blended_value = confidence * state_model_value + (1 - confidence) * value_pred
        
        # Terminal value is combination of immediate reward and blended future value
        terminal_value = reward_pred + self.config.RewardConfig.DISCOUNT_FACTOR * blended_value
        
        return terminal_value
    
    def train_on_terminal_placements(self, placement_data, optimizer, num_epochs=5):
        """
        Train the reward predictor specifically on terminal placement data
        Args:
            placement_data: List of dicts with 'state', 'placement', 'terminal_reward', 'resulting_state'
            optimizer: Optimizer for training
            num_epochs: Number of training epochs
        Returns:
            Dictionary with loss information
        """
        if not placement_data:
            return {}
        
        device = next(self.parameters()).device
        
        total_losses = []
        reward_losses = []
        value_losses = []
        
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            epoch_total_loss = 0
            epoch_reward_loss = 0
            epoch_value_loss = 0
            
            for data in placement_data:
                state = torch.FloatTensor(data['state']).unsqueeze(0).to(device)
                terminal_reward = data['terminal_reward']
                
                # Create dummy action (hard drop) for terminal placement
                action = torch.zeros(1, self.action_dim, device=device)
                action[0, 5] = 1.0  # Hard drop
                
                # Forward pass
                reward_pred, value_pred = self.forward(state, action)
                
                # Calculate losses
                target_reward = torch.FloatTensor([[terminal_reward]]).to(device)
                # For terminal placements, future value should be 0
                target_value = torch.zeros(1, 1, device=device)
                
                reward_loss = criterion(reward_pred, target_reward)
                value_loss = criterion(value_pred, target_value)
                total_loss = reward_loss + value_loss
                
                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Accumulate losses
                epoch_total_loss += total_loss.item()
                epoch_reward_loss += reward_loss.item()
                epoch_value_loss += value_loss.item()
            
            # Average losses for this epoch
            num_samples = len(placement_data)
            total_losses.append(epoch_total_loss / num_samples)
            reward_losses.append(epoch_reward_loss / num_samples)
            value_losses.append(epoch_value_loss / num_samples)
        
        return {
            'total_loss': total_losses[-1],
            'reward_loss': reward_losses[-1],
            'value_loss': value_losses[-1],
            'all_total_losses': total_losses,
            'all_reward_losses': reward_losses,
            'all_value_losses': value_losses
        }

    def predict_trajectory_value(self, states, actions):
        """
        Predict cumulative value for a trajectory of state-action pairs
        Updated to handle terminal placements better
        """
        batch_size, seq_len = states.shape[:2]
        
        total_value = torch.zeros(batch_size, 1, device=states.device)
        
        for t in range(seq_len):
            reward_pred, value_pred = self.forward(states[:, t], actions[:, t])
            # Discount future values
            discount = self.config.RewardConfig.DISCOUNT_FACTOR ** t
            total_value += discount * (reward_pred + value_pred)
            
        return total_value
