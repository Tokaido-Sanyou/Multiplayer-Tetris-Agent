"""
State transition model: predicts optimal piece placements from terminal rewards.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Handle both direct execution and module import
try:
    from ..config import TetrisConfig  # Import centralized config
except ImportError:
    # Direct execution - add parent directory to path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TetrisConfig  # Import centralized config

class StateModel(nn.Module):
    """
    State model that learns to predict optimal piece placements from state vectors
    Uses centralized configuration for all network dimensions
    """
    def __init__(self, state_dim=None):
        super(StateModel, self).__init__()
        
        # Get centralized config
        self.config = TetrisConfig()
        self.net_config = self.config.NetworkConfig.StateModel
        
        # Use centralized state dimension
        self.state_dim = state_dim or self.config.STATE_DIM  # 410
        
        # MLP encoder with dropout (using centralized config)
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.net_config.ENCODER_LAYERS[1]),
            nn.ReLU(),
            nn.Dropout(self.net_config.DROPOUT_RATE),
            nn.Linear(self.net_config.ENCODER_LAYERS[1], self.net_config.ENCODER_LAYERS[2]),
            nn.ReLU(),
            nn.Dropout(self.net_config.DROPOUT_RATE),
            nn.Linear(self.net_config.ENCODER_LAYERS[2], self.net_config.ENCODER_LAYERS[3]),
            nn.ReLU()
        )
        
        # Output heads (using centralized config)
        self.rotation_head = nn.Linear(self.net_config.ENCODER_LAYERS[3], self.net_config.ROTATION_CLASSES)
        self.x_position_head = nn.Linear(self.net_config.ENCODER_LAYERS[3], self.net_config.X_POSITION_CLASSES)
        self.y_position_head = nn.Linear(self.net_config.ENCODER_LAYERS[3], self.net_config.Y_POSITION_CLASSES)
        self.value_head = nn.Linear(self.net_config.ENCODER_LAYERS[3], self.net_config.VALUE_OUTPUT)

    def forward(self, state):
        """
        Args:
            state: Tensor of shape (batch_size, state_dim)
        Returns:
            rot_logits: (batch_size, num_rotations)
            x_logits: (batch_size, board_width)
            y_logits: (batch_size, board_height)
            value: (batch_size, 1) - predicted terminal reward
        """
        h = self.encoder(state)
        rot_logits = self.rotation_head(h)
        x_logits = self.x_position_head(h)
        y_logits = self.y_position_head(h)
        value = self.value_head(h)
        return rot_logits, x_logits, y_logits, value
    
    def get_placement_distribution(self, state):
        """
        Get probability distributions over placements
        Returns:
            rot_probs: (batch_size, num_rotations)
            x_probs: (batch_size, board_width)
            y_probs: (batch_size, board_height)
        """
        rot_logits, x_logits, y_logits, _ = self.forward(state)
        rot_probs = F.softmax(rot_logits, dim=1)
        x_probs = F.softmax(x_logits, dim=1)
        y_probs = F.softmax(y_logits, dim=1)
        return rot_probs, x_probs, y_probs
    
    def train_from_placements(self, placement_data, optimizer, num_epochs=10):
        """
        Train the model from exploration placement data with terminal rewards
        Args:
            placement_data: List of dicts with 'state', 'placement', 'terminal_reward'
            optimizer: Optimizer for training
            num_epochs: Number of training epochs
        Returns:
            Dictionary with detailed loss information
        """
        if not placement_data:
            return {}
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        total_losses = []
        rot_losses = []
        x_losses = []
        value_losses = []
        
        criterion = nn.CrossEntropyLoss()
        value_criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            epoch_total_loss = 0
            epoch_rot_loss = 0
            epoch_x_loss = 0
            epoch_value_loss = 0
            
            np.random.shuffle(placement_data)
            
            for data in placement_data:
                state = torch.FloatTensor(data['state']).unsqueeze(0).to(device)
                rotation, x_pos = data['placement']
                terminal_reward = data['terminal_reward']
                
                # Forward pass
                rot_logits, x_logits, y_logits, value_pred = self.forward(state)
                
                # Calculate losses (move targets to device)
                rot_loss = criterion(rot_logits, torch.LongTensor([rotation]).to(device))
                x_loss = criterion(x_logits, torch.LongTensor([x_pos]).to(device))
                value_loss = value_criterion(value_pred, torch.FloatTensor([[terminal_reward]]).to(device))
                
                # Weight losses by terminal reward (higher rewards get more weight)
                reward_weight = max(0.1, (terminal_reward + 100) / 200)  # Normalize to [0.1, 1]
                total_loss = reward_weight * (rot_loss + x_loss) + value_loss
                
                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Accumulate losses for averaging
                epoch_total_loss += total_loss.item()
                epoch_rot_loss += rot_loss.item()
                epoch_x_loss += x_loss.item()
                epoch_value_loss += value_loss.item()
            
            # Average losses for this epoch
            num_samples = len(placement_data)
            total_losses.append(epoch_total_loss / num_samples)
            rot_losses.append(epoch_rot_loss / num_samples)
            x_losses.append(epoch_x_loss / num_samples)
            value_losses.append(epoch_value_loss / num_samples)
        
        return {
            'total_loss': total_losses[-1],  # Final epoch loss
            'rotation_loss': rot_losses[-1],
            'x_position_loss': x_losses[-1],
            'value_loss': value_losses[-1],
            'all_total_losses': total_losses,
            'all_rotation_losses': rot_losses,
            'all_x_position_losses': x_losses,
            'all_value_losses': value_losses
        }

    def get_optimal_placement(self, state):
        """
        Get the optimal placement directly from the model predictions
        Returns:
            optimal_placement: Dict with 'rotation', 'x_position', 'y_position', and 'confidence'
        """
        with torch.no_grad():
            rot_logits, x_logits, y_logits, value = self.forward(state)
            
            # Get the most likely placement
            optimal_rotation = torch.argmax(rot_logits, dim=1)
            optimal_x = torch.argmax(x_logits, dim=1)
            optimal_y = torch.argmax(y_logits, dim=1)
            
            # Calculate confidence scores (softmax probabilities)
            rot_probs = F.softmax(rot_logits, dim=1)
            x_probs = F.softmax(x_logits, dim=1)
            y_probs = F.softmax(y_logits, dim=1)
            
            # Get confidence for the optimal placement
            batch_indices = torch.arange(rot_logits.shape[0])
            rot_confidence = rot_probs[batch_indices, optimal_rotation]
            x_confidence = x_probs[batch_indices, optimal_x]
            y_confidence = y_probs[batch_indices, optimal_y]
            
            # Overall confidence is the product of individual confidences
            overall_confidence = rot_confidence * x_confidence * y_confidence
            
            return {
                'rotation': optimal_rotation,
                'x_position': optimal_x,
                'y_position': optimal_y,
                'value': value.squeeze(-1),
                'confidence': overall_confidence
            }

    def get_placement_goal_vector(self, state):
        """
        Get optimal placement as a vector that can be used as goal for the actor
        Returns:
            goal_vector: Tensor of shape (batch_size, goal_dim) encoding the optimal placement
        """
        optimal_placement = self.get_optimal_placement(state)
        
        batch_size = state.shape[0]
        device = state.device
        
        # Encode placement as a concatenated vector
        # [rotation_one_hot(4) + x_position_one_hot(10) + y_position_one_hot(20) + value(1) + confidence(1)]
        goal_dim = 4 + 10 + 20 + 1 + 1  # 36 total
        goal_vector = torch.zeros(batch_size, goal_dim, device=device)
        
        # One-hot encode rotation (indices 0-3)
        rot_indices = optimal_placement['rotation']
        goal_vector[torch.arange(batch_size), rot_indices] = 1.0
        
        # One-hot encode x position (indices 4-13)
        x_indices = optimal_placement['x_position'] + 4
        goal_vector[torch.arange(batch_size), x_indices] = 1.0
        
        # One-hot encode y position (indices 14-33)  
        y_indices = optimal_placement['y_position'] + 14
        goal_vector[torch.arange(batch_size), y_indices] = 1.0
        
        # Add value and confidence (indices 34-35)
        goal_vector[:, 34] = optimal_placement['value']
        goal_vector[:, 35] = optimal_placement['confidence']
        
        return goal_vector
