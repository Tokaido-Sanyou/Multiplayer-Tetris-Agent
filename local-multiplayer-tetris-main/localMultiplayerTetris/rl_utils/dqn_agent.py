import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .replay_buffer import ReplayBuffer

class DQN(nn.Module):
    """
    Deep Q-Network for Tetris
    
    Input Structure (from tetris_env.py):
    - Grid: 20x10 matrix (200 values)
    - Next piece: scalar ID
    - Hold piece: scalar ID
    Total input dimension: 202
    
    Output Structure:
    - 41 Q-values corresponding to actions:
        0-39: Placement actions (rotation * 10 + column)
            rotation ∈ [0,3] for 4 possible rotations
            column ∈ [0,9] for 10 possible columns
        40: Hold piece
    
    Related Files:
    - tetris_env.py: Defines action space and state structure
    - action_handler.py: Implements action mechanics
    - game.py: Contains game state and piece movement logic
    - piece.py: Defines piece shapes and rotation logic
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize DQN network
        Args:
            input_dim: Dimension of input state (202)
            output_dim: Number of possible actions (41)
        """
        super(DQN, self).__init__()
        
        # CNN for processing the grid
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Piece embedding
        self.piece_embedding = nn.Embedding(8, 32)  # 8 = 7 piece types + 1 for empty
        
        # Calculate flattened dimensions
        self.conv_out_size = 64 * 20 * 10  # channels * height * width
        self.fc_input_size = self.conv_out_size + 32  # conv output + piece embedding
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, state):
        """
        Forward pass through the network
        Args:
            state: Dictionary containing:
                - grid: 20x10 matrix of piece colors
                - current_shape: scalar ID of the current piece
        Returns:
            Q-values for each action of shape (batch_size, 41)
        """
        # Extract components from state
        grid = state['grid'].float().unsqueeze(1)  # Add channel dimension
        current_piece = state['current_shape'].long()
        
        # Process grid through CNN
        conv_out = self.conv(grid)
        conv_out = conv_out.view(-1, self.conv_out_size)
        
        # Process current piece
        piece_out = self.piece_embedding(current_piece)
        
        # Combine features
        combined = torch.cat([conv_out, piece_out], dim=1)
        
        # Final layers
        q_values = self.fc(combined)
        return q_values

class DQNAgent:
    """
    DQN Agent for Tetris
    
    State Space (from tetris_env.py):
    - Grid: 20x10 matrix (0 for empty, 1-7 for different piece colors)
    - Next piece: scalar ID
    - Hold piece: scalar ID
    
    Action Space (from tetris_env.py):
    - 0-39: Placement actions (rotation * 10 + column)
        rotation ∈ [0,3] for 4 possible rotations
        column ∈ [0,9] for 10 possible columns
    - 40: Hold piece
    
    Total actions: 41 (40 placement actions + 1 hold action)
    
    Related Files:
    - tetris_env.py: Defines action space and state structure
    - action_handler.py: Implements action mechanics
    - game.py: Contains game state and piece movement logic
    - piece.py: Defines piece shapes and rotation logic
    """
    def __init__(self, state_dim, action_dim=41, learning_rate=1e-4, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, top_k_ac=3):
        """
        Initialize DQN agent
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions (default 41: 4 rotations × 10 columns + hold)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            top_k_ac: Number of top actions to consider for sampling
        """
        self.state_dim = state_dim
        self.action_dim = action_dim  # Should be 41 (4 rotations × 10 columns + hold)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.top_k = min(top_k_ac, action_dim)  # Ensure top_k doesn't exceed action space
        
        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(100000)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        # Training parameters
        self.batch_size = 64
        self.target_update = 10
        self.gradient_clip = 1.0
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy
        Args:
            state: Dictionary containing:
                - grid: 20x10 matrix of piece colors
                - current_shape: scalar ID of the current piece
        Returns:
            Integer (0-40) representing the selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state = {
                'grid': torch.FloatTensor(state['grid']).unsqueeze(0).to(self.device),
                'current_shape': torch.LongTensor([state['current_shape']]).to(self.device)
            }
            q_values = self.policy_net(state)
            top_k_q_values, top_k_indices = torch.topk(q_values, self.top_k)  # watch for batching, debug shapes for later
            return int(np.random.choice(top_k_indices.cpu().numpy()))
            # return q_values.argmax().item()
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, batch_size=None):
        """
        Train the agent on a batch of experiences
        Args:
            batch_size: Size of batch to train on (optional)
        Returns:
            Loss value if training occurred, None otherwise
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Sample from replay buffer
        batch = self.memory.sample(batch_size)
        if batch is None:
            return None
            
        states, actions, rewards, next_states, dones, info, indices, weights = batch
        
        # Move tensors to device
        states = {
            'grid': states['grid'].to(self.device),
            'current_shape': states['current_shape'].to(self.device)
        }
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = {
            'grid': next_states['grid'].to(self.device),
            'current_shape': next_states['current_shape'].to(self.device)
        }
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values with Double DQN
        with torch.no_grad():
            # Select actions using policy network
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # Evaluate actions using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss with importance sampling weights
        loss = (weights.unsqueeze(1) * (current_q_values - target_q_values) ** 2).mean()
        
        # Update priorities
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] 
