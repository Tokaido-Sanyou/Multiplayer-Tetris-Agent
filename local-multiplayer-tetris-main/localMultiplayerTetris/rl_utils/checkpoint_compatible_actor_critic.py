import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from .replay_buffer import ReplayBuffer

class CheckpointCompatibleFeatureExtractor(nn.Module):
    """
    Feature extractor compatible with the saved checkpoint
    """
    def __init__(self):
        super(CheckpointCompatibleFeatureExtractor, self).__init__()

        # CNN for grid processing - 8 channels to match checkpoint
        self.grid_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 1->8, keeps 20x10
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),  # 8->8, keeps 20x10
            nn.ReLU()
        )

        # MLP for piece metadata - 32 dimensions to match checkpoint
        self.piece_embed = nn.Sequential(
            nn.Linear(7, 32),  # 7 metadata scalars -> 32
            nn.ReLU(),
            nn.Linear(32, 32), # 32 -> 32
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass through the feature extractor
        Args:
            x: Input tensor of shape (batch_size, 207)
        Returns:
            Extracted features of dimension 1632 (8*20*10 + 32)
        """
        batch_size = x.size(0)
        grid = x[:, :200].view(batch_size, 1, 20, 10)  # Grid: 20x10
        pieces = x[:, 200:]  # Scalars: next_piece, hold_piece, current_shape, rotation, x, y, can_hold

        # Process grid with CNN
        grid_features = self.grid_conv(grid)
        grid_features = grid_features.view(batch_size, -1)  # Flatten to (batch, 8*20*10=1600)

        # Process piece scalars with embedding
        piece_features = self.piece_embed(pieces)  # (batch, 32)

        # Combine features
        return torch.cat([grid_features, piece_features], dim=1)  # (batch, 1632)

class CheckpointCompatibleActorCritic(nn.Module):
    """
    Actor-Critic network compatible with the saved checkpoint
    Feature dimension: 1632 (8*20*10 + 32)
    Action space: 41 actions
    """
    def __init__(self, input_dim=207, output_dim=41):
        super(CheckpointCompatibleActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = CheckpointCompatibleFeatureExtractor()
        
        # Actor network (6 layers to match checkpoint)
        self.actor = nn.Sequential(
            nn.Linear(1632, 512),  # Layer 0
            nn.ReLU(),
            nn.Linear(512, 256),   # Layer 2
            nn.ReLU(),
            nn.Linear(256, 256),   # Layer 4
            nn.ReLU(),
            nn.Linear(256, 64),    # Layer 6
            nn.ReLU(),
            nn.Linear(64, output_dim),  # Layer 8 -> 41 actions
            nn.Softmax(dim=-1)
        )
        
        # Critic network (6 layers to match checkpoint)
        self.critic = nn.Sequential(
            nn.Linear(1632, 512),  # Layer 0
            nn.ReLU(),
            nn.Linear(512, 256),   # Layer 2
            nn.ReLU(),
            nn.Linear(256, 256),   # Layer 4
            nn.ReLU(),
            nn.Linear(256, 64),    # Layer 6
            nn.ReLU(),
            nn.Linear(64, 1)       # Layer 8 -> value
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, 207)
        Returns:
            Tuple of (action_probs, state_value)
        """
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class CheckpointCompatibleAgent:
    """
    Agent class for loading and using saved checkpoints
    """
    def __init__(self, state_dim=207, action_dim=41):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize network
        self.network = CheckpointCompatibleActorCritic(state_dim, action_dim)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        self.network.to(self.device)
        
        print(f"CheckpointCompatibleAgent initialized on {self.device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters() if p.requires_grad)}")
    
    def load(self, filepath: str):
        """Load checkpoint"""
        print(f"Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network state
        self.network.load_state_dict(checkpoint['network'])
        self.network.eval()
        
        print(f"Checkpoint loaded successfully")
        print(f"Checkpoint episode: {checkpoint.get('episode', 'N/A')}")
        print(f"Checkpoint epsilon: {checkpoint.get('epsilon', 'N/A')}")
    
    def select_action(self, state, deterministic=True):
        """
        Select action using the loaded policy
        Args:
            state: State vector (207 dimensions)
            deterministic: Whether to use greedy action selection
        Returns:
            Action integer (0-40)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.network(state)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
        
        return action.cpu().item()
    
    def select_action_with_value(self, state):
        """
        Select action and get value estimate
        Returns:
            Tuple of (action, value)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.network(state)
            action = torch.argmax(action_probs, dim=-1)
        
        return action.cpu().item(), value.cpu().item() 