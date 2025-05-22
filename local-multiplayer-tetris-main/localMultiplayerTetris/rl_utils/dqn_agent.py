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
    - Current piece: 4x4 matrix (16 values)
    - Next piece: 4x4 matrix (16 values)
    - Hold piece: 4x4 matrix (16 values)
    Total input dimension: 248
    
    Output Structure:
    - 7 Q-values corresponding to actions:
        0: Move Left (action_handler.py: move_left)
        1: Move Right (action_handler.py: move_right)
        2: Move Down (action_handler.py: move_down)
        3: Rotate Clockwise (action_handler.py: rotate_cw)
        4: Rotate Counter-clockwise (action_handler.py: rotate_ccw)
        5: Hard Drop (action_handler.py: hard_drop)
        6: Hold Piece (action_handler.py: hold_piece)
    
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
            input_dim: Dimension of input state (248)
            output_dim: Number of possible actions (7)
        """
        super(DQN, self).__init__()
        
        # CNN for grid processing (20x10 input)
        self.grid_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # MLP for piece processing (3 pieces * 16 features)
        self.piece_mlp = nn.Sequential(
            nn.Linear(48, 128),  # 3 pieces * 16 features
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Combined network
        self.combined = nn.Sequential(
            nn.Linear(64 * 20 * 10 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, 248)
        Returns:
            Q-values for each action of shape (batch_size, 7)
        """
        # Reshape input
        batch_size = x.size(0)
        grid = x[:, :200].view(batch_size, 1, 20, 10)  # Grid: 20x10
        pieces = x[:, 200:]  # Pieces: 3 * 16 features
        
        # Process grid with CNN
        grid_features = self.grid_conv(grid)
        grid_features = grid_features.view(batch_size, -1)
        
        # Process pieces with MLP
        piece_features = self.piece_mlp(pieces)
        
        # Combine features
        combined = torch.cat([grid_features, piece_features], dim=1)
        
        # Output Q-values
        return self.combined(combined)

class DQNAgent:
    """
    DQN Agent for Tetris
    
    State Space (from tetris_env.py):
    - Grid: 20x10 matrix (0 for empty, 1-7 for different piece colors)
    - Current piece: 4x4 matrix (0 for empty, 1 for filled)
    - Next piece: 4x4 matrix (0 for empty, 1 for filled)
    - Hold piece: 4x4 matrix (0 for empty, 1 for filled)
    
    Action Space (from tetris_env.py):
    - 0: Move Left
    - 1: Move Right
    - 2: Move Down
    - 3: Rotate Clockwise
    - 4: Rotate Counter-clockwise
    - 5: Hard Drop
    - 6: Hold Piece
    
    Related Files:
    - tetris_env.py: Defines action space and state structure
    - action_handler.py: Implements action mechanics
    - game.py: Contains game state and piece movement logic
    - piece.py: Defines piece shapes and rotation logic
    """
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, top_k_ac=3):
        """
        Initialize DQN agent
        Args:
            state_dim: Dimension of state space (248)
            action_dim: Number of possible actions (7)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.top_k = top_k_ac
        
        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(100000)  # Increased buffer size
        
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
                - current_piece: 4x4 matrix of current piece
                - next_piece: 4x4 matrix of next piece
                - hold_piece: 4x4 matrix of hold piece
        Returns:
            Integer (0-6) representing the selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            top_k_q_values, top_k_indices = torch.topk(q_values, self.top_k)  # watch for batching, debug shapes for later
            return int(np.random.choice(top_k_indices.cpu().numpy())
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
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
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
