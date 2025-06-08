"""
Tetris CNN Model for Board State Representation
Standard ML implementation for Deep Reinforcement Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TetrisCNN(nn.Module):
    """
    CNN for processing Tetris board states with specified architecture.
    
    Input: 20×10 binary board → shape = (batch_size, 1, 20, 10)
    Output: Feature vector of size 256 or action values/policy
    """
    
    def __init__(self, output_size=8, activation_type='relu', use_dropout=True, dropout_rate=0.1):
        """
        Initialize the Tetris CNN.
        
        Args:
            output_size (int): Number of output units (8 for Tetris actions)
            activation_type (str): 'relu', 'identity', or 'softmax'
            use_dropout (bool): Whether to use dropout
            dropout_rate (float): Dropout probability
        """
        super(TetrisCNN, self).__init__()
        
        self.output_size = output_size
        self.activation_type = activation_type
        self.use_dropout = use_dropout
        
        # Conv Layer 1: filters=16, kernel=4x4, stride=2, padding=same
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=16, 
            kernel_size=4, 
            stride=2, 
            padding=1  # Adjusted padding for correct output size
        )
        
        # Conv Layer 2: filters=32, kernel=3x3, stride=1, padding=same  
        self.conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1  # To achieve 'same' padding
        )
        
        # Conv Layer 3: filters=32, kernel=2x2, stride=1, padding=same
        self.conv3 = nn.Conv2d(
            in_channels=32, 
            out_channels=32, 
            kernel_size=2, 
            stride=1, 
            padding=0  # No padding for 2x2 kernel to reduce size
        )
        
        # Calculate flattened size after convolutions
        # Input: (20, 10) -> Conv1 (stride=2, pad=1): (10, 5) -> Conv2 (stride=1, pad=1): (10, 5) -> Conv3 (stride=1, pad=0): (9, 4)
        self.flattened_size = 9 * 4 * 32  # = 1152
        
        # FC Layer 1: units=256, activation=ReLU
        self.fc1 = nn.Linear(self.flattened_size, 256)
        
        # Optional dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer: units=output_size (8 for actions)
        self.fc_out = nn.Linear(256, output_size)
        
    def forward(self, x):
        """
        Forward pass through the CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 20, 10)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Ensure input has correct shape
        if len(x.shape) == 3:  # (batch_size, 20, 10)
            x = x.unsqueeze(1)  # Add channel dimension -> (batch_size, 1, 20, 10)
        
        # Conv Layer 1: ReLU activation
        x = F.relu(self.conv1(x))
        
        # Conv Layer 2: ReLU activation
        x = F.relu(self.conv2(x))
        
        # Conv Layer 3: ReLU activation  
        x = F.relu(self.conv3(x))
        
        # Flatten: (batch_size, 10, 5, 32) -> (batch_size, 1600)
        x = x.view(x.size(0), -1)
        
        # FC Layer 1: ReLU activation
        x = F.relu(self.fc1(x))
        
        # Optional dropout
        if self.use_dropout and self.training:
            x = self.dropout(x)
        
        # Output layer with specified activation
        x = self.fc_out(x)
        
        if self.activation_type == 'softmax':
            x = F.softmax(x, dim=1)
        elif self.activation_type == 'identity':
            pass  # No activation (for DQN Q-values)
        
        return x
    
    def extract_features(self, x):
        """
        Extract intermediate features from FC layer 1 (256-dim vector).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 20, 10)
            
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, 256)
        """
        # Ensure input has correct shape
        if len(x.shape) == 3:  # (batch_size, 20, 10)
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Forward through conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten and FC1
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc1(x))
        
        return features


class TetrisDQN(TetrisCNN):
    """DQN variant of Tetris CNN for Q-learning."""
    
    def __init__(self, use_dropout=True, dropout_rate=0.1):
        super().__init__(output_size=8, activation_type='identity', 
                         use_dropout=use_dropout, dropout_rate=dropout_rate)


class TetrisPolicyNet(TetrisCNN):
    """Policy network variant for Actor-Critic methods."""
    
    def __init__(self, use_dropout=True, dropout_rate=0.1):
        super().__init__(output_size=8, activation_type='softmax', 
                         use_dropout=use_dropout, dropout_rate=dropout_rate)


def board_tuple_to_tensor(board_tuple, device='cpu'):
    """
    Convert binary board tuple to tensor format for CNN.
    
    Args:
        board_tuple (tuple): Binary tuple of 200 bits (20x10 board)
        device (str): PyTorch device
        
    Returns:
        torch.Tensor: Tensor of shape (1, 1, 20, 10) for single sample
    """
    # Extract board portion (first 200 bits)
    board_bits = board_tuple[:200]
    
    # Reshape to 20x10 grid
    board_array = np.array(board_bits, dtype=np.float32).reshape(20, 10)
    
    # Convert to tensor and add batch/channel dimensions
    board_tensor = torch.tensor(board_array, device=device).unsqueeze(0).unsqueeze(0)
    
    return board_tensor


def batch_board_tuples_to_tensor(board_tuples, device='cpu'):
    """
    Convert batch of binary board tuples to tensor format for CNN.
    
    Args:
        board_tuples (list): List of binary tuples
        device (str): PyTorch device
        
    Returns:
        torch.Tensor: Tensor of shape (batch_size, 1, 20, 10)
    """
    batch_size = len(board_tuples)
    batch_tensor = torch.zeros(batch_size, 1, 20, 10, device=device)
    
    for i, board_tuple in enumerate(board_tuples):
        board_bits = board_tuple[:200]
        board_array = np.array(board_bits, dtype=np.float32).reshape(20, 10)
        batch_tensor[i, 0] = torch.tensor(board_array, device=device)
    
    return batch_tensor


def test_model():
    """Test the CNN model with sample input."""
    print("Testing Tetris CNN Model...")
    
    # Create model
    model = TetrisCNN(output_size=8, activation_type='identity')
    model.eval()
    
    # Create sample input (batch_size=2, 1 channel, 20x10 board)
    sample_input = torch.randn(2, 1, 20, 10)
    
    # Forward pass
    output = model(sample_input)
    features = model.extract_features(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test with board tuple
    sample_tuple = tuple([0] * 200 + [1, 0, 0, 0, 0, 0, 0] + [0] * 7 + [0, 0, 0, 0, 0] + [0] * 200)
    tensor_input = board_tuple_to_tensor(sample_tuple)
    tuple_output = model(tensor_input)
    
    print(f"Tuple input shape: {tensor_input.shape}")
    print(f"Tuple output shape: {tuple_output.shape}")
    print("CNN model test completed successfully!")


if __name__ == "__main__":
    test_model() 