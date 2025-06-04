import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """
    Simple replay buffer for storing experience tuples.
    
    Each experience contains:
    - state: numpy array of shape (207,) containing:
        - grid: first 200 values (20x10 flattened)
        - metadata: last 7 values (next_piece, hold_piece, curr_shape, curr_rot, curr_x, curr_y, can_hold)
    - action: integer in range [0, 40]
    - reward: float
    - next_state: same format as state
    - done: boolean
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        # Validate state shapes
        assert isinstance(state, np.ndarray) and state.shape == (207,), f"Expected state shape (207,), got {state.shape}"
        assert isinstance(next_state, np.ndarray) and next_state.shape == (207,), f"Expected next_state shape (207,), got {next_state.shape}"
        
        # Convert to float32 for consistency
        state = state.astype(np.float32)
        next_state = next_state.astype(np.float32)
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to torch tensors with proper shapes
        states = torch.FloatTensor(np.array(states))  # shape: (batch_size, 207)
        actions = torch.LongTensor(actions)  # shape: (batch_size,)
        rewards = torch.FloatTensor(rewards)  # shape: (batch_size,)
        next_states = torch.FloatTensor(np.array(next_states))  # shape: (batch_size, 207)
        dones = torch.FloatTensor(dones)  # shape: (batch_size,)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """
    Deep Q-Network for Tetris
    
    Architecture:
    1. Grid Processing Branch:
        - Input: (batch_size, 1, 20, 10) - Reshaped grid
        - 3x Conv2d layers with ReLU
        - Flattened output
        
    2. Metadata Processing Branch:
        - Input: (batch_size, 7) - Piece information
        - 2x Linear layers with ReLU
        
    3. Combined Processing:
        - Concatenated features from both branches
        - 3x Linear layers with ReLU
        - Output: Q-values for 41 actions
    """
    def __init__(self):
        super(DQN, self).__init__()
        
        # Grid processing (CNN)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Metadata processing (MLP)
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # Combined processing
        self.fc3 = nn.Linear(64 * 20 * 10 + 64, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 41)  # 40 placements + 1 hold
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through network
        
        Args:
            x: Tensor of shape (batch_size, 207)
                - First 200 values: Flattened grid
                - Last 7 values: Piece metadata
                
        Returns:
            Tensor of shape (batch_size, 41) containing Q-values for each action
        """
        batch_size = x.size(0)
        
        # Split input into grid and metadata
        grid = x[:, :200].view(batch_size, 1, 20, 10)  # Reshape to (batch_size, 1, 20, 10)
        metadata = x[:, 200:]  # Shape: (batch_size, 7)
        
        # Process grid
        grid = F.relu(self.conv1(grid))  # Shape: (batch_size, 32, 20, 10)
        grid = F.relu(self.conv2(grid))  # Shape: (batch_size, 64, 20, 10)
        grid = F.relu(self.conv3(grid))  # Shape: (batch_size, 64, 20, 10)
        grid = grid.view(batch_size, -1)  # Flatten: (batch_size, 64 * 20 * 10)
        
        # Process metadata
        metadata = F.relu(self.fc1(metadata))  # Shape: (batch_size, 64)
        metadata = F.relu(self.fc2(metadata))  # Shape: (batch_size, 64)
        
        # Combine features
        combined = torch.cat([grid, metadata], dim=1)  # Shape: (batch_size, 64 * 20 * 10 + 64)
        
        # Final processing
        x = F.relu(self.fc3(combined))  # Shape: (batch_size, 512)
        x = F.relu(self.fc4(x))  # Shape: (batch_size, 256)
        q_values = self.fc5(x)  # Shape: (batch_size, 41)
        
        return q_values

class DQNAgent:
    """
    DQN Agent with Double Q-learning and Experience Replay
    
    Features:
    - Double Q-learning to reduce overestimation
    - Experience replay for sample efficiency
    - Epsilon-greedy exploration with decay
    - Target network updates for stability
    - Gradient clipping to prevent exploding gradients
    """
    def __init__(self, 
                 learning_rate=1e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 target_update=10,
                 buffer_size=100000,
                 batch_size=64,
                 device=None):
        
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Set hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        
        # Initialize step counter
        self.steps_done = 0
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: numpy array of shape (207,)
            
        Returns:
            action: integer in range [0, 40]
        """
        # Validate state shape
        assert isinstance(state, np.ndarray) and state.shape == (207,), f"Expected state shape (207,), got {state.shape}"
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, 40)
        
        # Convert state to tensor
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Add batch dimension
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()  # Get action with highest Q-value
    
    def train_step(self):
        """
        Perform one step of training
        
        Returns:
            loss: float or None if batch size not met
        """
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None
            
        states, actions, rewards, next_states, dones = batch
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using target network (Double Q-learning)
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # Get Q-values from target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            # Compute target Q values
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network if needed
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path):
        """Save model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done'] 