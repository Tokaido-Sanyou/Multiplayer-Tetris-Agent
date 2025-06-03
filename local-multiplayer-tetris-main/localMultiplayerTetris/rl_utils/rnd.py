import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RandomNetwork(nn.Module):
    """
    Random network that generates fixed random features for states
    """
    def __init__(self, input_dim):
        super(RandomNetwork, self).__init__()
        
        # CNN for grid processing (20x10 input)
        self.grid_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Scalar embedding for next and hold pieces
        self.piece_embed = nn.Sequential(
            nn.Linear(2, 32),  # next + hold scalar IDs
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Combined network
        self.combined = nn.Sequential(
            nn.Linear(64 * 20 * 10 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # Output dimension for random features
        )
        
        # Initialize with random weights and freeze them
        self.apply(self._init_weights)
        for param in self.parameters():
            param.requires_grad = False
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        # Reshape input
        batch_size = x.size(0)
        grid = x[:, :200].view(batch_size, 1, 20, 10)  # Grid: 20x10
        pieces = x[:, 200:]  # Scalars: next_piece + hold_piece
        
        # Process grid with CNN
        grid_features = self.grid_conv(grid)
        grid_features = grid_features.view(batch_size, -1)
        
        # Process piece scalars with embedding
        piece_features = self.piece_embed(pieces)
        
        # Combine features
        combined = torch.cat([grid_features, piece_features], dim=1)
        
        # Output random features
        return self.combined(combined)

class PredictorNetwork(nn.Module):
    """
    Predictor network that tries to predict the random network's output
    """
    def __init__(self, input_dim):
        super(PredictorNetwork, self).__init__()
        
        # CNN for grid processing (20x10 input)
        self.grid_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Scalar embedding for next and hold pieces
        self.piece_embed = nn.Sequential(
            nn.Linear(2, 32),  # next + hold scalar IDs
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Combined network
        self.combined = nn.Sequential(
            nn.Linear(64 * 20 * 10 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # Output dimension matches random network
        )
    
    def forward(self, x):
        # Reshape input
        batch_size = x.size(0)
        grid = x[:, :200].view(batch_size, 1, 20, 10)  # Grid: 20x10
        pieces = x[:, 200:]  # Scalars: next_piece + hold_piece
        
        # Process grid with CNN
        grid_features = self.grid_conv(grid)
        grid_features = grid_features.view(batch_size, -1)
        
        # Process piece scalars with embedding
        piece_features = self.piece_embed(pieces)
        
        # Combine features
        combined = torch.cat([grid_features, piece_features], dim=1)
        
        # Output predicted features
        return self.combined(combined)

class RND:
    """
    Random Network Distillation for exploration
    """
    def __init__(self, input_dim, device):
        self.device = device
        self.random_net = RandomNetwork(input_dim).to(device)
        self.predictor_net = PredictorNetwork(input_dim).to(device)
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=1e-4)
        self.mse_loss = nn.MSELoss()
    
    def compute_intrinsic_reward(self, state):
        """
        Compute intrinsic reward based on prediction error
        """
        with torch.no_grad():
            # Get random features
            random_features = self.random_net(state)
            # Get predicted features
            predicted_features = self.predictor_net(state)
            # Compute prediction error
            error = self.mse_loss(predicted_features, random_features)
            
            # Normalize error to prevent extreme values
            error = torch.clamp(error, min=0.0, max=1.0)
            
            # Add bonus for states with pieces placed
            grid = state[:, :200].view(-1, 1, 20, 10)
            pieces_placed = torch.sum(grid > 0).float() / 200.0  # Normalize by grid size
            
            # Higher bonus for states with more pieces
            bonus = 0.2 * pieces_placed  # Increased from 0.1 to 0.2
            
            # Add extra bonus for states with pieces in the bottom rows
            bottom_rows = grid[:, :, -4:, :]  # Look at bottom 4 rows
            bottom_pieces = torch.sum(bottom_rows > 0).float() / 40.0  # Normalize by bottom area
            bottom_bonus = 0.3 * bottom_pieces  # Bonus for pieces near bottom
            
            # Return combined reward
            return error + bonus + bottom_bonus
    
    def update(self, state):
        """
        Update predictor network
        """
        # Get random features
        random_features = self.random_net(state)
        # Get predicted features
        predicted_features = self.predictor_net(state)
        
        # Compute loss with L2 regularization
        loss = self.mse_loss(predicted_features, random_features)
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., device=self.device)
        for param in self.predictor_net.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg
        
        # Add extra loss term to encourage piece placement
        grid = state[:, :200].view(-1, 1, 20, 10)
        pieces_placed = torch.sum(grid > 0).float() / 200.0
        placement_loss = 0.1 * (1.0 - pieces_placed)  # Loss increases when fewer pieces are placed
        loss += placement_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(self.predictor_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path):
        """Save RND networks"""
        torch.save({
            'predictor_net_state_dict': self.predictor_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load RND networks"""
        checkpoint = torch.load(path)
        self.predictor_net.load_state_dict(checkpoint['predictor_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 