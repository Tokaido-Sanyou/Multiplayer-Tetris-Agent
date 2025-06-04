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

        # CNN for grid processing without pooling (preserving 20x10 resolution)
        self.grid_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 1->8, keeps 20x10
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),  # 8->8, keeps 20x10 (reduced)
            nn.ReLU()
        )

        # MLP for piece metadata: next, hold, current_shape, rotation, x, y, can_hold
        self.piece_embed = nn.Sequential(
            nn.Linear(7, 32),  # 7 metadata scalars
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass through the feature extractor
        Args:
            x: Input tensor of shape (batch_size, 207)
        Returns:
            Extracted features
        """
        batch_size = x.size(0)
        grid = x[:, :200].view(batch_size, 1, 20, 10)  # Grid: 20x10
        pieces = x[:, 200:]  # Scalars: next_piece, hold_piece, current_shape, rotation, x, y, can_hold

        # Process grid with CNN
        grid_features = self.grid_conv(grid)
        grid_features = grid_features.view(batch_size, -1)  # Flatten

        # Process piece scalars with embedding
        piece_features = self.piece_embed(pieces)

        # Combine features
        return torch.cat([grid_features, piece_features], dim=1)

class ActorCritic(nn.Module):
    """
    Actor-Critic network for Tetris

    State Space:
    - Grid, next_piece, hold_piece, current_shape, rotation, x, y, can_hold

    Action Space:
    - 0-39: Flattened placement index = rotation*10 + column (rot∈[0-3], col∈[0-9])
    - 40 : hold current piece
    - 41 total actions
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize Actor-Critic network
        Args:
            input_dim: Dimension of input state (207)
            output_dim: Number of possible actions (41)
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = SharedFeatureExtractor()
        
        # Actor network (policy)
        # Feature dimension: grid conv output (8×20×10 = 1600) + piece embed (32)
        self.feature_dim = 1600 + 32
        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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

class ActorCriticAgent:
    """
    Actor-Critic agent with epsilon-greedy exploration
    """
    def __init__(self, state_dim, action_dim,
                 actor_lr=1e-4, critic_lr=1e-3,
                 gamma_start=0.9, gamma_end=0.99,
                 epsilon_start=1.0, epsilon_end=0.05,
                 schedule_episodes=10000,
                 top_k_ac=3):
        """
        Initialize Actor-Critic agent
        Args:
            state_dim: Dimension of state space (207)
            action_dim: Number of possible actions (41)
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma_start: Starting discount factor (e.g., 0.9)
            gamma_end: Final discount factor after schedule_episodes (e.g., 0.99)
            epsilon_start: Starting exploration rate
            epsilon_end: Final exploration rate after schedule_episodes
            schedule_episodes: Number of episodes over which to schedule epsilon and gamma
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Scheduling parameters
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.schedule_episodes = schedule_episodes  # total training episodes
        # Episodes over which ε decays to ε_end (first half of training)
        self._eps_decay_episodes = max(1, schedule_episodes // 2)
        
        # Modified epsilon decay to maintain more exploration
        if epsilon_start > 0 and epsilon_end > 0 and epsilon_end < epsilon_start:
            # Slower decay rate to maintain exploration
            self._eps_decay_rate = (epsilon_end / epsilon_start) ** (1.0 / self.schedule_episodes)
        else:
            self._eps_decay_rate = 0.998  # Slower default decay
        
        # Initialize current epsilon, gamma, and episode count
        self.epsilon = epsilon_start
        self.gamma = self.gamma_start
        self.current_episode = 0
        self.top_k = top_k_ac
        
        # Entropy regularization coefficient (increases with episodes to prevent convergence)
        self.entropy_coef_start = 0.01
        self.entropy_coef_end = 0.05
        self.entropy_coef = self.entropy_coef_start
        
        # Initialize network
        self.network = ActorCritic(state_dim, action_dim)
        
        # Initialize optimizers with slightly higher learning rates
        self.actor_optimizer = optim.Adam(self.network.actor.parameters(), lr=actor_lr * 1.5)
        self.critic_optimizer = optim.Adam(self.network.critic.parameters(), lr=critic_lr * 1.5)
        
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
    
    def select_action(self, state):  # returns integer in [0,40]
        """
        Select action using epsilon-greedy policy with proper top-K sampling
        Args:
            state: Dictionary containing state information
        Returns:
            Integer (0-40) representing the selected action
        """
        if np.random.random() < self.epsilon:
            # During exploration, ensure we sample all rotations
            if np.random.random() < 0.8:  # 80% chance to try different rotations
                rot = np.random.randint(4)  # 0-3 rotations
                col = np.random.randint(10)  # 0-9 columns
                return rot * 10 + col
            else:
                return np.random.randint(self.action_dim)  # Include hold action
        
        # Exploitation with proper top-K sampling
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, _ = self.network(state_tensor)
            action_probs = action_probs[0]  # Remove batch dimension
            
            # Get top K actions and their probabilities
            top_k_probs, top_k_indices = torch.topk(action_probs, self.top_k)
            
            # Normalize the probabilities of top K actions
            top_k_probs = torch.softmax(top_k_probs, dim=0)
            
            # Sample from top K actions using normalized probabilities
            chosen_idx = torch.multinomial(top_k_probs, 1).item()
            chosen_action = top_k_indices[chosen_idx].item()
            
            # Handle hold action availability
            can_hold_flag = bool(state[-1]) if isinstance(state, (list, np.ndarray)) else True
            if (not can_hold_flag) and chosen_action == 40:
                # If hold is not available but was chosen, pick the next best action
                sorted_actions = torch.argsort(action_probs, descending=True)
                for action in sorted_actions:
                    if action != 40:
                        chosen_action = action.item()
                        break
            
            return chosen_action

    def select_actions_batch(self, states, eval_mode=False): # Added eval_mode
        """
        Select actions for a batch of states.
        Args:
            states: A list of state dictionaries, or a pre-batched tensor.
            eval_mode (bool): If True, select actions greedily without exploration.
        Returns:
            A numpy array of actions for each state.
        """
        if isinstance(states, list): # If list of dicts, convert to batch tensor
            state_tensors = torch.stack([self.memory._state_to_tensor(s) for s in states]).to(self.device)
        else: # Assuming states is already a batched tensor
            state_tensors = states.to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.network(state_tensors)
        
        batch_size = state_tensors.size(0)
        actions = []
        for i in range(batch_size):
            if eval_mode:
                # Greedy action for evaluation
                actions.append(torch.argmax(action_probs[i]).item())
            else:
                # Epsilon-greedy exploration or top-k sampling for training
                if np.random.random() < self.epsilon:
                    actions.append(np.random.randint(self.action_dim))
                else:
                    # top-k sampling for exploitation
                    # Ensure action_probs[i] is 1D before topk if it's not already
                    current_action_probs = action_probs[i].squeeze() # Ensure it's 1D
                    if current_action_probs.ndim == 0: # if only one action possible after squeeze (e.g. action_dim=1)
                        chosen_action = 0 # or appropriate single action
                    elif self.top_k == 1:
                        chosen_action = torch.argmax(current_action_probs).item()
                    else:
                        # Ensure self.top_k is not greater than the number of available actions
                        k = min(self.top_k, current_action_probs.size(0))
                        if k <= 0: # Handle cases where k might become non-positive
                            k = 1 
                        top_k_probs, top_k_indices = torch.topk(current_action_probs, k, dim=-1)
                        # Multinomial expects probabilities, ensure top_k_probs are normalized if they aren't already
                        # For sampling from top-k, it's common to use the original probabilities of the top-k actions
                        # or re-normalize them. Here, we directly sample from the indices based on their original probabilities (via top_k_probs).
                        # If top_k_probs are logits, they should be passed through a softmax. Assuming they are probabilities or can be treated as weights.
                        if k == 1: # If k is 1, top_k_indices will have one element
                            chosen_action = top_k_indices[0].item()
                        else:
                            # Normalize probabilities for multinomial sampling if they are not already
                            # If action_probs are logits, apply softmax first. Assuming they are already probabilities.
                            # For simplicity, if top_k_probs are not normalized, multinomial might not behave as expected.
                            # However, torch.multinomial can take unnormalized weights.
                            chosen_action_index_in_top_k = torch.multinomial(torch.softmax(top_k_probs, dim=-1), 1).item()
                            chosen_action = top_k_indices[chosen_action_index_in_top_k].item()
                    actions.append(chosen_action)
        return np.array(actions)

    def update_schedules(self, total_completed_episodes):
        """Update epsilon, gamma, and entropy coefficient schedules based on total completed episodes."""
        # More gradual epsilon decay
        progress = total_completed_episodes / self.schedule_episodes
        
        # Modified epsilon schedule to maintain exploration
        if progress < 0.7:  # First 70% of training
            self.epsilon = self.epsilon_start * (self._eps_decay_rate ** total_completed_episodes)
        else:  # Last 30% - maintain higher minimum exploration
            self.epsilon = max(self.epsilon_end * 2, self.epsilon_start * (self._eps_decay_rate ** total_completed_episodes))
        
        # Linear gamma schedule
        self.gamma = self.gamma_start + progress * (self.gamma_end - self.gamma_start)
        
        # Increase entropy regularization coefficient over time to prevent convergence
        self.entropy_coef = self.entropy_coef_start + progress * (self.entropy_coef_end - self.entropy_coef_start)

    def train(self):
        """Single training step: update networks if buffer has enough samples"""
        if len(self.memory) < self.batch_size:
            return None
        return self.update_networks()
    
    def store_transition(self, *args):
        """
        Store transition in replay buffer
        Args:
            *args: Arguments from env.step() and agent's select_action()
        """
        self.memory.add(*args)
    
    def compute_returns(self, rewards, dones):
        """
        Compute discounted returns for TD(0) and TD(λ)
        Args:
            rewards: List of rewards
            dones: List of done flags (1 if episode ended, 0 otherwise)
        Returns:
            List of returns
        """
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return returns
    
    def update_networks(self):
        """
        Update actor and critic networks using replay buffer
        """
        # Sample batch from replay buffer
        sample = self.memory.sample(self.batch_size)
        if sample is None:
            return None
        states, actions, rewards, next_states, dones, *_ = sample
         
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Extract shared features
        features = self.network.feature_extractor(states)
        next_features = self.network.feature_extractor(next_states)
        
        # Compute targets for critic (TD target)
        with torch.no_grad():
            next_state_values = self.network.critic(next_features)
            targets = rewards + self.gamma * next_state_values * (1 - dones)
        
        # ---------------- Critic update ---------------- #
        self.critic_optimizer.zero_grad()
        state_values = self.network.critic(features)
        critic_loss = nn.MSELoss()(state_values, targets)
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.network.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()

        # ---------------- Actor update ---------------- #
        self.actor_optimizer.zero_grad()
        action_probs = self.network.actor(features)
        
        # Avoid log(0) by clamping
        selected_action_probs = action_probs.gather(1, actions).clamp(min=1e-8)
        log_probs = torch.log(selected_action_probs)
        
        # Advantage = TD-target − V(s)
        advantages = (targets - state_values).detach()
        
        # Policy gradient loss
        actor_loss = -(log_probs * advantages).mean()
        
        # Enhanced entropy regularization
        entropy = -(action_probs * torch.log(action_probs.clamp(min=1e-8))).sum(dim=1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Add distribution regularization
        probs_per_rotation = torch.zeros(4, 10).to(self.device)  # 4 rotations, 10 columns
        for rot in range(4):
            probs_per_rotation[rot] = action_probs[:, rot*10:(rot+1)*10].mean(dim=0)
        
        # Penalize uneven distribution across columns for each rotation
        distribution_loss = 0.1 * torch.var(probs_per_rotation, dim=1).mean()
        
        # Combined loss
        total_loss = actor_loss + entropy_loss + distribution_loss
        
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

    # ---------------- Persistence helpers ---------------- #
    def save(self, filepath: str):
        """Save model and optimizer states to a checkpoint file.

        Args:
            filepath (str): Path to the .pt/.pth file to save.
        """
        checkpoint = {
            'network': self.network.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'episode': self.current_episode
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str, map_location=None):
        """Load model and optimizer states from a checkpoint file.

        Args:
            filepath (str): Path to checkpoint file.
            map_location: Optional device mapping for torch.load.
        """
        checkpoint = torch.load(filepath, map_location=self.device if map_location is None else map_location)
        self.network.load_state_dict(checkpoint['network'])
        if 'actor_optimizer' in checkpoint and checkpoint['actor_optimizer']:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if 'critic_optimizer' in checkpoint and checkpoint['critic_optimizer']:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.current_episode = checkpoint.get('episode', self.current_episode)
