import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    """
    Deep Q-Network for Tetris
    
    Input Structure (from tetris_env.py):
    - Grid: 20x10 matrix (200 values)
    - Next piece: scalar ID
    - Hold piece: scalar ID
    Total input dimension: 202
    
    Output Structure:
    - 8 Q-values corresponding to actions:
        0: Move Left
        1: Move Right
        2: Move Down
        3: Rotate Clockwise
        4: Rotate Counter-clockwise
        5: Hard Drop
        6: Hold Piece
        7: No-op
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize DQN network
        Args:
            input_dim: Dimension of input state (202)
            output_dim: Number of possible actions (8)
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
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, 202)
        Returns:
            Q-values for each action of shape (batch_size, 8)
        """
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
        
        # Output Q-values
        return self.combined(combined)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, info):
        self.buffer.append((state, action, reward, next_state, done, info))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    DQN Agent for Tetris
    
    State Space (from tetris_env.py):
    - Grid: 20x10 matrix (0 for empty, 1-7 for different piece colors)
    - Next piece: scalar ID
    - Hold piece: scalar ID
    
    Action Space (from tetris_env.py):
    - 0: Move Left
    - 1: Move Right
    - 2: Move Down
    - 3: Rotate Clockwise
    - 4: Rotate Counter-clockwise
    - 5: Hard Drop
    - 6: Hold Piece
    - 7: No-op
    
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
            state_dim: Dimension of state space (202)
            action_dim: Number of possible actions (8)
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
        self.train_step = 0
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy
        Args:
            state: Dictionary containing:
                - grid: 20x10 matrix of piece colors
                - next_piece: scalar ID
                - hold_piece: scalar ID
        Returns:
            Integer (0-7) representing the selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            # Convert state dict to tensor
            state_tensor = torch.FloatTensor([
                state['grid'].flatten(),
                state['next_piece'],
                state['hold_piece']
            ]).unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(state_tensor)
            top_k_values, top_k_indices = torch.topk(q_values, self.top_k, dim=1)
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
            
        # Unpack batch
        states, actions, rewards, next_states, dones, infos = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor([
            [s['grid'].flatten(), s['next_piece'], s['hold_piece']]
            for s in states
        ]).to(self.device)
        
        next_states = torch.FloatTensor([
            [s['grid'].flatten(), s['next_piece'], s['hold_piece']]
            for s in next_states
        ]).to(self.device)
        
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values with Double DQN
        with torch.no_grad():
            # Select actions using policy network
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # Evaluate actions using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.update_target_network()
        
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
