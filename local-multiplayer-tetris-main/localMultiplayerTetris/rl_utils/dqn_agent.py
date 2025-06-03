import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from .rnd import RND

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
    DQN Agent for Tetris with RND exploration
    
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
    """
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, top_k_ac=3, rnd_weight=0.1):
        """
        Initialize DQN agent with RND
        Args:
            state_dim: Dimension of state space (202)
            action_dim: Number of possible actions (8)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            top_k_ac: Number of top actions to sample from
            rnd_weight: Weight for intrinsic reward
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.top_k = top_k_ac
        self.rnd_weight = rnd_weight
        
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
        
        # Initialize RND
        self.rnd = RND(state_dim, self.device)
        
        # Training parameters
        self.batch_size = 64
        self.target_update = 10
        self.gradient_clip = 1.0
        self.train_step = 0
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy with RND exploration
        Args:
            state: Dictionary containing:
                - grid: 20x10 matrix of piece colors
                - next_piece: scalar ID
                - hold_piece: scalar ID
        Returns:
            Integer (0-7) representing the selected action
        """
        # Convert state dict to tensor
        state_tensor = torch.FloatTensor(np.concatenate([
            state['grid'].flatten(),
            [state['next_piece']],
            [state['hold_piece']]
        ])).unsqueeze(0).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
            # Get intrinsic reward
            intrinsic_reward = self.rnd.compute_intrinsic_reward(state_tensor)
            
            # Combine Q-values with intrinsic reward
            q_values = q_values + self.rnd_weight * intrinsic_reward
            
            # During exploration, prioritize actions that place pieces
            if np.random.random() < self.epsilon:
                # 70% chance to choose from actions that place pieces
                if np.random.random() < 0.7:
                    # Prioritize hard drop and down movement
                    piece_placing_actions = [2, 5]  # Move Down and Hard Drop
                    return np.random.choice(piece_placing_actions)
                return np.random.randint(self.action_dim)
            
            # During exploitation
            if self.top_k > 1:
                # Get top-k actions
                top_k_values, top_k_indices = torch.topk(q_values[0], self.top_k)
                
                # Ensure at least one piece-placing action is in top-k
                piece_placing_actions = torch.tensor([2, 5], device=self.device)  # Move Down and Hard Drop
                if not any(action in top_k_indices for action in piece_placing_actions):
                    # Replace the lowest value action with hard drop
                    top_k_indices[-1] = torch.tensor(5, device=self.device)
                
                # Randomly select from top-k actions
                selected_idx = np.random.randint(self.top_k)
                return top_k_indices[selected_idx].item()
            else:
                # If top_k=1, use hard drop if it's close to best action
                best_action = q_values.argmax().item()
                if best_action not in [2, 5] and q_values[0][5] > q_values[0][best_action] * 0.8:
                    return 5
                return best_action
    
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
            Tuple of (q_loss, rnd_loss) if training occurred, None otherwise
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
        states = torch.FloatTensor(np.array([
            np.concatenate([
                s['grid'].flatten(),
                [s['next_piece']],
                [s['hold_piece']]
            ])
            for s in states
        ])).to(self.device)
        
        next_states = torch.FloatTensor(np.array([
            np.concatenate([
                s['grid'].flatten(),
                [s['next_piece']],
                [s['hold_piece']]
            ])
            for s in next_states
        ])).to(self.device)
        
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
        
        # Compute Q-learning loss
        q_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        q_loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()
        
        # Update RND predictor and get loss
        rnd_loss = self.rnd.update(states)
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.update_target_network()
        
        return q_loss.item(), rnd_loss
    
    def save(self, path):
        """Save model weights and RND networks"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        # Save RND networks
        self.rnd.save(path.replace('.pt', '_rnd.pt'))
    
    def load(self, path):
        """Load model weights and RND networks"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        # Load RND networks
        self.rnd.load(path.replace('.pt', '_rnd.pt')) 
