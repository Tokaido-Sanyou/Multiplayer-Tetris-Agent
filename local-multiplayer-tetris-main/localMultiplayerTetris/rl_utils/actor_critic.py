import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F  # for losses
from .replay_buffer import ReplayBuffer
from .state_model import StateModel  # add import for state model

class SharedFeatureExtractor(nn.Module):
    """
    Shared feature extractor for both actor and critic networks
    """
    def __init__(self):
        super(SharedFeatureExtractor, self).__init__()

        # CNN for grid processing (20×10 input) with no pooling
        # two conv layers reducing channels to 4, full resolution maintained
        self.grid_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),   # 1×20×10 → 4×20×10
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),   # 4×20×10 → 4×20×10
            nn.ReLU()
        )

        # Scalar embedding for piece metadata: current_shape, rotation, x, y, next, hold (6 dims)
        self.piece_embed = nn.Sequential(
            nn.Linear(6, 64),  # embed 6-dimensional metadata
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass through the feature extractor
        Args:
            x: Input tensor of shape (batch_size, state_dim)
        Returns:
            Extracted features
        """
        batch_size = x.size(0)
        grid = x[:, :200].view(batch_size, 1, 20, 10)  # Grid: 20x10
        pieces = x[:, 200:206]  # 6 scalars: cur_shape, cur_rot, cur_x, cur_y, next, hold

        # Process grid with CNN
        grid_features = self.grid_conv(grid)
        grid_features = grid_features.view(batch_size, -1)  # Flatten

        # Process piece scalars with embedding
        piece_features = self.piece_embed(pieces)

        # Combine features
        return torch.cat([grid_features, piece_features], dim=1)

class ActorCritic(nn.Module):
    """
    Conditioned Actor-Critic network: selects actions to achieve a given instruction state.
    """
    def __init__(self, state_dim, instr_dim, output_dim):
        super(ActorCritic, self).__init__()
        # Shared feature extractor for current state
        self.feature_extractor = SharedFeatureExtractor()
        # Embedding for instruction state (e.g., target grid mask + piece IDs)
        instr_feat_dim = 128
        self.instr_embed = nn.Sequential(
            nn.Linear(instr_dim, instr_feat_dim),
            nn.ReLU()
        )
        # Combined feature dimension
        # grid conv features: full resolution 4×20×10 = 800, piece_embed outputs 64
        base_feat_dim = (4 * 20 * 10) + 64      # 800 + 64 = 864
        combined_dim = base_feat_dim + instr_feat_dim
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=1)
        )
        # Critic network (value)
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, instruction):
        """
        Forward pass with conditioning
        Args:
            state: tensor (batch, state_dim)
            instruction: tensor (batch, instr_dim)
        Returns:
            action_probs, state_value
        """
        # state features
        feat = self.feature_extractor(state)
        # instruction embedding
        instr_feat = self.instr_embed(instruction)
        # combine and predict
        combined = torch.cat([feat, instr_feat], dim=1)
        action_probs = self.actor(combined)
        state_value = self.critic(combined)
        return action_probs, state_value

class ActorCriticAgent:
    """
    Actor-Critic agent with epsilon-greedy exploration
    """
    def __init__(self, state_dim, action_dim, instr_dim=None,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, top_k_ac=3,
                 state_model=None, state_rules=None):
        """
        Initialize Actor-Critic agent
        Args:
            state_dim: Dimension of input state (202)
            instr_dim: Dimension of instruction vector
            action_dim: Number of possible actions
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
        """
        # Core dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Instruction dimension defaults to state_dim
        self.instr_dim = instr_dim or state_dim
         # state instruction model
        self.state_model = state_model
        self.state_rules = state_rules or {}
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.top_k = top_k_ac
        
        # Initialize network (state_dim, instr_dim, action_dim)
        self.network = ActorCritic(self.state_dim, self.instr_dim, self.action_dim)
        
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
        if self.state_model:
            self.state_model.to(self.device)
        
        # Training parameters
        self.batch_size = 64
        self.gradient_clip = 1.0
    
    def set_state_model(self, state_model, state_rules):
        """
        Attach a pretrained state transition model and validity rules.
        """
        self.state_model = state_model.to(self.device)
        self.state_rules = state_rules
        # Optimizer for state model
        self.state_model_optimizer = optim.Adam(self.state_model.parameters(), lr=1e-3)

    def select_action(self, state, instruction=None):
        """
        Select action using epsilon-greedy policy
        Args:
            state: flattened array or tensor of shape (state_dim)
            instruction: flattened array or tensor of shape (instr_dim)
        Returns:
            Integer representing selected action
        """
        # exploration (epsilon)
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        # exploitation (policy conditioned on instruction)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if instruction is None:
                instr_tensor = torch.zeros((1, self.instr_dim), device=self.device)
            else:
                instr_tensor = torch.FloatTensor(instruction).unsqueeze(0).to(self.device)
            action_probs, _ = self.network(state_tensor, instr_tensor)
            action = action_probs.argmax(dim=1).item()
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
        # Use next_states as instruction target
        instructions = next_states.detach()
 
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        instructions = instructions.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Get current action probabilities and state values conditioned on instruction
        action_probs, state_values = self.network(states, instructions)
        
        # Get next state values (no instruction)
        with torch.no_grad():
            zero_instr = torch.zeros_like(instructions)
            _, next_state_values = self.network(next_states, zero_instr)
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
        
        # Auxiliary state-model training
        if hasattr(self, 'state_model') and self.state_model is not None:
            # Predict next-state from current state and action
            grid_pred, piece_pred = self.state_model(states, actions)
            # Targets
            grid_target = next_states[:, :200]
            piece_target = next_states[:, 200:202]
            # Losses
            loss_grid = F.mse_loss(grid_pred, grid_target)
            loss_piece = F.mse_loss(piece_pred, piece_target)
            aux_loss = loss_grid + loss_piece
            # Update state_model
            self.state_model_optimizer.zero_grad()
            aux_loss.backward()
            self.state_model_optimizer.step()
 
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
