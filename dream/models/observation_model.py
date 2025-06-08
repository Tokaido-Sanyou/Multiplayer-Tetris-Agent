"""
Tetris Observation Encoder/Decoder for DREAM

Provides specialized neural networks for encoding Tetris observations
into latent representations and decoding them back for the world model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class TetrisEncoder(nn.Module):
    """
    Tetris-specific observation encoder for world model
    
    Encodes the Tetris game state including:
    - Grid state (20x10)
    - Current piece information
    - Next piece information
    - Hold piece information
    """
    
    def __init__(self, 
                 observation_shape: Tuple[int, int, int] = (1, 20, 10), 
                 latent_dim: int = 256,
                 piece_embedding_dim: int = 32):
        super().__init__()
        
        self.observation_shape = observation_shape
        self.latent_dim = latent_dim
        self.piece_embedding_dim = piece_embedding_dim
        
        # Convolutional layers for grid processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2))  # Reduce to manageable size
        )
        
        # Piece embedding layers
        self.piece_embedding = nn.Embedding(8, piece_embedding_dim)  # 7 pieces + empty
        
        # Calculate conv output size
        conv_output_size = 128 * 4 * 2  # 128 channels * 4 * 2 from adaptive pooling
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + piece_embedding_dim * 3, latent_dim),  # +3 for current, next, hold
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode observation to latent representation
        
        Args:
            observation: Dict containing grid, current_piece, next_piece, hold_piece
            
        Returns:
            Latent representation tensor
        """
        # Handle both dict and tensor inputs
        if isinstance(observation, dict):
            batch_size = observation['empty_grid'].shape[0]
        else:
            batch_size = observation.shape[0]
            # Convert tensor to dict format if needed
            observation = {'empty_grid': observation}
        
        # Process grid through conv layers
        grid = observation['empty_grid']  # Use empty grid as primary grid state
        if len(grid.shape) == 3:
            grid = grid.unsqueeze(1)  # Add channel dimension
        
        conv_features = self.conv_layers(grid)
        conv_features = conv_features.view(batch_size, -1)
        
        # Process piece information
        current_piece = observation.get('current_piece', torch.zeros(batch_size, 7, device=grid.device))
        next_piece = observation.get('next_piece', torch.zeros(batch_size, 7, device=grid.device))
        hold_piece = observation.get('hold_piece', torch.zeros(batch_size, 7, device=grid.device))
        
        # Convert one-hot to indices
        current_idx = torch.argmax(current_piece, dim=-1)
        next_idx = torch.argmax(next_piece, dim=-1)
        hold_idx = torch.argmax(hold_piece, dim=-1)
        
        # Get embeddings
        current_emb = self.piece_embedding(current_idx)
        next_emb = self.piece_embedding(next_idx)
        hold_emb = self.piece_embedding(hold_idx)
        
        # Concatenate all features
        combined_features = torch.cat([
            conv_features,
            current_emb,
            next_emb,
            hold_emb
        ], dim=-1)
        
        # Final encoding
        latent = self.fc_layers(combined_features)
        return latent


class TetrisDecoder(nn.Module):
    """
    Tetris-specific observation decoder for world model
    
    Decodes latent representations back to Tetris observations
    for training the world model reconstruction loss.
    """
    
    def __init__(self, 
                 latent_dim: int = 256,
                 observation_shape: Tuple[int, int, int] = (1, 20, 10),
                 piece_embedding_dim: int = 32):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.observation_shape = observation_shape
        self.piece_embedding_dim = piece_embedding_dim
        
        # Grid reconstruction - Reduced for 500k limit
        self.grid_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, observation_shape[1] * observation_shape[2]),  # 20 * 10
            nn.Sigmoid()  # Output between 0 and 1 for grid occupancy
        )
        
        # Piece prediction heads - Reduced for 500k limit
        self.current_piece_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Softmax(dim=-1)
        )
        
        self.next_piece_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Softmax(dim=-1)
        )
        
        self.hold_piece_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode latent representation to observation
        
        Args:
            latent: Latent representation tensor
            
        Returns:
            Dict containing reconstructed observation components
        """
        batch_size = latent.shape[0]
        
        # Reconstruct grid
        grid_flat = self.grid_decoder(latent)
        grid = grid_flat.view(batch_size, self.observation_shape[1], self.observation_shape[2])
        
        # Predict pieces
        current_piece = self.current_piece_head(latent)
        next_piece = self.next_piece_head(latent)
        hold_piece = self.hold_piece_head(latent)
        
        return {
            'empty_grid': grid,
            'current_piece': current_piece,
            'next_piece': next_piece,
            'hold_piece': hold_piece
        }


class MLP(nn.Module):
    """Multi-layer perceptron utility class"""
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int = 400, 
                 num_layers: int = 2,
                 activation: str = 'relu',
                 dropout: float = 0.0):
        super().__init__()
        
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) 