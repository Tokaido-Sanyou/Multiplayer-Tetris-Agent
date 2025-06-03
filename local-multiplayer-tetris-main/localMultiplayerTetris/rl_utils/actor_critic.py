import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F  # for losses

# Handle both direct execution and module import
try:
    from ..config import TetrisConfig  # Import centralized config
    from .replay_buffer import ReplayBuffer
    from .state_model import StateModel  # add import for state model
    from .future_reward_predictor import FutureRewardPredictor  # add import for future reward predictor
except ImportError:
    # Direct execution - imports without relative paths
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TetrisConfig  # Import centralized config
    from rl_utils.replay_buffer import ReplayBuffer  # Fix: use correct path
    from rl_utils.state_model import StateModel  # Fix: use correct path
    from rl_utils.future_reward_predictor import FutureRewardPredictor  # Fix: use correct path

class SharedFeatureExtractor(nn.Module):
    """
    Shared feature extractor using pure MLP for simplified state vector
    Uses centralized configuration for all dimensions
    """
    def __init__(self, input_dim=None):
        super(SharedFeatureExtractor, self).__init__()
        
        # Get centralized config
        self.config = TetrisConfig()
        self.net_config = self.config.NetworkConfig.SharedFeatureExtractor
        
        # Use centralized dimensions
        self.input_dim = input_dim or self.config.STATE_DIM  # 410
        
        # MLP feature extractor with centralized dimensions
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.net_config.HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Dropout(self.net_config.DROPOUT_RATE),
            nn.Linear(self.net_config.HIDDEN_LAYERS[0], self.net_config.HIDDEN_LAYERS[1]),
            nn.ReLU(),
            nn.Dropout(self.net_config.DROPOUT_RATE),
            nn.Linear(self.net_config.HIDDEN_LAYERS[1], self.net_config.HIDDEN_LAYERS[2]),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass through the MLP feature extractor
        Args:
            x: Input tensor of shape (batch_size, 410)
        Returns:
            Extracted features of shape (batch_size, 128)
        """
        return self.mlp(x)

class ActorCritic(nn.Module):
    """
    Actor-Critic network with centralized configuration and goal conditioning
    """
    def __init__(self, state_dim=None, action_dim=None, goal_dim=None):
        super(ActorCritic, self).__init__()
        
        # Get centralized config
        self.config = TetrisConfig()
        self.net_config = self.config.NetworkConfig.ActorCritic
        
        # Use centralized dimensions
        self.state_dim = state_dim or self.config.STATE_DIM  # 410
        self.action_dim = action_dim or self.config.ACTION_DIM  # 8
        self.goal_dim = goal_dim or self.config.GOAL_DIM  # 36 (from centralized config)
        
        # Shared feature extractor for current state
        self.feature_extractor = SharedFeatureExtractor(input_dim=self.state_dim)
        
        # Goal encoder for processing state model goals (using centralized config)
        self.goal_encoder = nn.Sequential(
            nn.Linear(self.goal_dim, self.net_config.GOAL_ENCODER_LAYERS[1]),
            nn.ReLU(),
            nn.Linear(self.net_config.GOAL_ENCODER_LAYERS[1], self.net_config.GOAL_FEATURES),
            nn.ReLU()
        )
        
        # Get dimensions from centralized config
        shared_feat_dim = self.config.NetworkConfig.SharedFeatureExtractor.OUTPUT_FEATURES  # 128
        goal_feat_dim = self.net_config.GOAL_FEATURES  # 64
        combined_feat_dim = self.net_config.COMBINED_FEATURES  # 192
        
        # Actor network (policy) - outputs binary decisions with goal conditioning
        self.actor = nn.Sequential(
            nn.Linear(combined_feat_dim, self.net_config.ACTOR_HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Linear(self.net_config.ACTOR_HIDDEN_LAYERS[0], self.net_config.ACTOR_HIDDEN_LAYERS[1]),
            nn.ReLU(),
            nn.Linear(self.net_config.ACTOR_HIDDEN_LAYERS[1], self.net_config.ACTOR_OUTPUT_DIM),
            nn.Sigmoid()  # For binary outputs as specified in config
        )
        
        # Critic network (value) with goal conditioning
        self.critic = nn.Sequential(
            nn.Linear(combined_feat_dim, self.net_config.CRITIC_HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Linear(self.net_config.CRITIC_HIDDEN_LAYERS[0], self.net_config.CRITIC_HIDDEN_LAYERS[1]),
            nn.ReLU(),
            nn.Linear(self.net_config.CRITIC_HIDDEN_LAYERS[1], self.net_config.CRITIC_OUTPUT_DIM)
        )
    
    def forward(self, state, goal=None):
        """
        Forward pass with optional goal conditioning
        Args:
            state: tensor (batch, state_dim)
            goal: tensor (batch, goal_dim) - optional goal from state model
        Returns:
            action_probs, state_value
        """
        # Extract features from state
        state_feat = self.feature_extractor(state)
        
        if goal is not None:
            # Encode goal and concatenate with state features
            goal_feat = self.goal_encoder(goal)
            combined_feat = torch.cat([state_feat, goal_feat], dim=1)
        else:
            # Use state features only (pad with zeros for goal features)
            batch_size = state_feat.shape[0]
            zero_goal_feat = torch.zeros(batch_size, self.net_config.GOAL_FEATURES, device=state_feat.device)
            combined_feat = torch.cat([state_feat, zero_goal_feat], dim=1)
        
        # Get action probabilities and state value
        action_probs = self.actor(combined_feat)
        state_value = self.critic(combined_feat)
        return action_probs, state_value

class ActorCriticAgent:
    """
    Actor-Critic agent with epsilon-greedy exploration and centralized configuration
    """
    def __init__(self, state_dim=None, action_dim=None,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 state_model=None, clip_ratio=0.2):
        """
        Initialize Actor-Critic agent with PPO clipping and centralized config
        Args:
            state_dim: Dimension of input state (410: simplified representation)
            action_dim: Number of possible actions (8: one-hot encoded)
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            clip_ratio: PPO clipping ratio
        """
        # Get centralized config
        self.config = TetrisConfig()
        
        # Core dimensions
        self.state_dim = state_dim or self.config.STATE_DIM  # 410
        self.action_dim = action_dim or self.config.ACTION_DIM  # 8
        
        # State model reference
        self.state_model = state_model
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.clip_ratio = clip_ratio
        
        # Initialize network with centralized config
        self.network = ActorCritic(self.state_dim, self.action_dim)
        
        # Initialize future reward predictor with centralized config
        self.future_reward_predictor = FutureRewardPredictor(self.state_dim, self.action_dim)
        
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
        # Optimizer for state model
        self.state_model_optimizer = optim.Adam(self.state_model.parameters(), lr=1e-3)

    def select_action(self, state):
        """
        Select action using epsilon-greedy policy with goal conditioning from state model
        Args:
            state: flattened array or tensor of shape (state_dim)
        Returns:
            One-hot encoded action vector (8-dimensional)
        """
        # exploration (epsilon)
        if np.random.random() < self.epsilon:
            # Random action as one-hot vector
            action_idx = np.random.randint(self.action_dim)
            action_one_hot = np.zeros(self.action_dim, dtype=np.int8)
            action_one_hot[action_idx] = 1
            return action_one_hot
            
        # exploitation (policy with goal conditioning)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get goal from state model if available
            goal_tensor = None
            if self.state_model is not None:
                goal_tensor = self.state_model.get_placement_goal_vector(state_tensor)
            
            # Get action probabilities with goal conditioning
            action_probs, _ = self.network(state_tensor, goal_tensor)
            
            # Convert sigmoid outputs to binary action
            # Take the action with highest probability
            action_idx = torch.argmax(action_probs, dim=1).item()
            action_one_hot = np.zeros(self.action_dim, dtype=np.int8)
            action_one_hot[action_idx] = 1
            return action_one_hot
    
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
        action_probs, state_values = self.network(states)
        
        # Get next state values (no instruction)
        with torch.no_grad():
            zero_instr = torch.zeros_like(instructions)
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
        
        # Auxiliary state-model training (predict placement parameters)
        aux_loss_value = 0.0
        if hasattr(self, 'state_model') and self.state_model is not None and hasattr(self, 'state_model_optimizer'):
            # Predict rotation, x-position, y-position logits, and value
            rot_logits, x_logits, y_logits, value_pred = self.state_model(states)
            # Extract true labels from next_states tensor
            # next_states layout: grid (0-199), cur_shape(200), cur_rot(201), cur_x(202), cur_y(203), next_piece(204), hold_piece(205)
            rot_labels = next_states[:, 201].long()
            x_labels = next_states[:, 202].long()
            y_labels = next_states[:, 203].long()
            
            # Compute individual cross-entropy losses for detailed logging
            rot_loss = F.cross_entropy(rot_logits, rot_labels)
            x_loss = F.cross_entropy(x_logits, x_labels)
            y_loss = F.cross_entropy(y_logits, y_labels)
            aux_loss = rot_loss + x_loss + y_loss
            aux_loss_value = aux_loss.item()
            
            # Update state_model parameters
            self.state_model_optimizer.zero_grad()
            aux_loss.backward()
            self.state_model_optimizer.step()
            
            # Store individual losses for logging (if writer is available)
            if hasattr(self, 'writer') and self.writer is not None:
                step = getattr(self, 'training_step', 0)
                self.writer.add_scalar('StateModel/AuxiliaryTotalLoss', aux_loss_value, step)
                self.writer.add_scalar('StateModel/AuxiliaryRotationLoss', rot_loss.item(), step)
                self.writer.add_scalar('StateModel/AuxiliaryXPositionLoss', x_loss.item(), step)
                self.writer.add_scalar('StateModel/AuxiliaryYPositionLoss', y_loss.item(), step)
                self.training_step = step + 1
 
        # Update priorities
        td_errors = torch.abs(returns - state_values.squeeze()).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        return actor_loss.item(), critic_loss.item(), aux_loss_value
    
    def train_ppo(self, batch_size=None, ppo_epochs=4):
        """
        Train the agent using PPO clipping mechanism
        Args:
            batch_size: Size of batch to train on
            ppo_epochs: Number of PPO update epochs
        Returns:
            Tuple of (actor_loss, critic_loss, reward_loss) if training occurred, None otherwise
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Sample from replay buffer
        batch = self.memory.sample(batch_size)
        if batch is None:
            return None
            
        states, actions, rewards, next_states, dones, info, indices, weights = batch
        instructions = next_states.detach()
 
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        instructions = instructions.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Get old policy probabilities
        with torch.no_grad():
            old_action_probs, old_state_values = self.network(states)
            old_action_log_probs = F.log_softmax(old_action_probs, dim=1)
            old_selected_log_probs = old_action_log_probs.gather(1, actions.long().unsqueeze(1)).squeeze()
        
        # Calculate returns and advantages
        with torch.no_grad():
            zero_instr = torch.zeros_like(instructions)
            _, next_state_values = self.network(next_states)
            returns = rewards + (1 - dones) * self.gamma * next_state_values.squeeze()
            advantages = returns - old_state_values.squeeze()
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training loop
        total_actor_loss = 0
        total_critic_loss = 0
        total_reward_loss = 0
        
        for _ in range(ppo_epochs):
            # Get current policy probabilities
            action_probs, state_values = self.network(states)
            action_log_probs = F.log_softmax(action_probs, dim=1)
            selected_log_probs = action_log_probs.gather(1, actions.long().unsqueeze(1)).squeeze()
            
            # Calculate ratio for PPO
            ratio = torch.exp(selected_log_probs - old_selected_log_probs)
            
            # Calculate clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(state_values.squeeze(), returns)
            
            # Future reward prediction loss
            action_one_hot = F.one_hot(actions.long(), self.action_dim).float()
            reward_pred, value_pred = self.future_reward_predictor(states, action_one_hot)
            reward_loss = F.mse_loss(reward_pred.squeeze(), rewards) + F.mse_loss(value_pred.squeeze(), returns)
            
            # Backpropagation
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.network.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.network.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            # Update future reward predictor
            if hasattr(self, 'reward_optimizer'):
                self.reward_optimizer.zero_grad()
                reward_loss.backward()
                self.reward_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_reward_loss += reward_loss.item()        # Add auxiliary state model loss if available
        aux_loss_value = 0.0
        if self.state_model is not None and hasattr(self, 'state_model_optimizer'):
            # Predict rotation, x-position, y-position logits, and value
            rot_logits, x_logits, y_logits, value_pred = self.state_model(states)
            # Extract true labels from next_states tensor
            rot_labels = next_states[:, 201].long()
            x_labels = next_states[:, 202].long()
            y_labels = next_states[:, 203].long()
            
            # Compute individual cross-entropy losses for detailed logging
            rot_loss = F.cross_entropy(rot_logits, rot_labels)
            x_loss = F.cross_entropy(x_logits, x_labels)
            y_loss = F.cross_entropy(y_logits, y_labels)
            aux_loss = rot_loss + x_loss + y_loss
            aux_loss_value = aux_loss.item()
            
            # Update state_model parameters
            self.state_model_optimizer.zero_grad()
            aux_loss.backward()
            self.state_model_optimizer.step()
            
            # Store individual losses for logging (if writer is available)
            if hasattr(self, 'writer') and self.writer is not None:
                step = getattr(self, 'training_step', 0)
                self.writer.add_scalar('StateModel/PPOAuxiliaryTotalLoss', aux_loss_value, step)
                self.writer.add_scalar('StateModel/PPOAuxiliaryRotationLoss', rot_loss.item(), step)
                self.writer.add_scalar('StateModel/PPOAuxiliaryXPositionLoss', x_loss.item(), step)
                self.writer.add_scalar('StateModel/PPOAuxiliaryYPositionLoss', y_loss.item(), step)
                self.training_step = step + 1
        
        return (total_actor_loss / ppo_epochs, 
                total_critic_loss / ppo_epochs, 
                total_reward_loss / ppo_epochs,
                aux_loss_value)

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
    
    def set_writer(self, writer):
        """Set TensorBoard writer for logging"""
        self.writer = writer
        self.training_step = 0
