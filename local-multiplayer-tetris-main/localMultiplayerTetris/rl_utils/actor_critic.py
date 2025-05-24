import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from .replay_buffer import ReplayBuffer

class SharedFeatureExtractor(nn.Module):
    """
    Shared feature extractor for both actor and critic networks
    """
    def __init__(self):
        super(SharedFeatureExtractor, self).__init__()
        
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
    
    def forward(self, x):
        """
        Forward pass through the feature extractor
        Args:
            x: Input tensor of shape (batch_size, 202)
        Returns:
            Extracted features
        """
        batch_size = x.size(0)
        grid = x[:, :200].view(batch_size, 1, 20, 10)  # Grid: 20x10
        pieces = x[:, 200:]  # Scalars: next_piece + hold_piece
        
        # Process grid with CNN
        grid_features = self.grid_conv(grid)
        grid_features = grid_features.view(batch_size, -1)
        
        # Process piece scalars with embedding
        piece_features = self.piece_embed(pieces)
        
        # Combine features
        return torch.cat([grid_features, piece_features], dim=1)

class ActorCritic(nn.Module):
    """
    Actor-Critic network for Tetris
    
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
    - 8: No-op
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize Actor-Critic network
        Args:
            input_dim: Dimension of input state (202)
            output_dim: Number of possible actions (9)
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = SharedFeatureExtractor()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(64 * 20 * 10 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=1)
        )
        
        # Critic network (value)
        self.critic = nn.Sequential(
            nn.Linear(64 * 20 * 10 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, 202)
        Returns:
            Tuple of (action_probs, state_value)
        """
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class ActorCriticAgent:
    """
    Actor-Critic agent with epsilon-greedy exploration
    """
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, top_k_ac = 3):
        """
        Initialize Actor-Critic agent
        Args:
            state_dim: Dimension of state space (202)
            action_dim: Number of possible actions (9)
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
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
        
        # Initialize network
        self.network = ActorCritic(state_dim, action_dim)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.network.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.network.critic.parameters(), lr=critic_lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(100000)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        self.network.to(self.device)
        
        # Training parameters
        self.batch_size = 64
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
            Integer (0-8) representing the selected action
        """
        # Throttle actor to at least 50 ms per action
        start = time.perf_counter()
        
        # exploration (epsilon)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # exploitation(top k values)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, _ = self.network(state_tensor)
                top_k_ac, top_k_indices = torch.topk(action_probs, self.top_k)
                top_k_indices = top_k_indices[0].cpu().numpy()
                action = int(np.random.choice(top_k_indices))
        # Ensure minimum 50 ms per call
        # elapsed = time.perf_counter() - start
        # if elapsed < 0.05:
        #     time.sleep(0.05 - elapsed)
        return action
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, batch_size=None):
        """
        Train the agent on a batch of experiences
        Args:
            batch_size: Size of batch to train on (optional)
        Returns:
            Tuple of (actor_loss, critic_loss) if training occurred, None otherwise
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
        
        # Get current action probabilities and state values
        action_probs, state_values = self.network(states)
        
        # Get next state values
        with torch.no_grad():
            _, next_state_values = self.network(next_states)
            next_state_values = next_state_values.squeeze()
        
        # Calculate returns and advantages
        returns = rewards + (1 - dones) * self.gamma * next_state_values
        advantages = returns - state_values.squeeze()
        
        # Calculate actor loss (policy gradient)
        action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        action_probs = action_probs.clamp(min=1e-6, max=1.0)
        actor_loss = -(torch.log(action_probs) * advantages * weights).mean()
        
        # Calculate critic loss (value function)
        critic_loss = (weights * (returns - state_values.squeeze()) ** 2).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.network.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        
        # Update priorities
        td_errors = torch.abs(returns - state_values.squeeze()).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] 
