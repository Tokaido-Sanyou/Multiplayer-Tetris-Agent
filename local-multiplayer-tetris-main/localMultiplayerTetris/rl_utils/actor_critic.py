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

        # MLP for piece metadata: next, hold, current_shape, rotation, x, y
        self.piece_embed = nn.Sequential(
            nn.Linear(6, 32),  # 6 metadata scalars
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
        grid_features = grid_features.view(batch_size, -1)  # Flatten

        # Process piece scalars with embedding
        piece_features = self.piece_embed(pieces)

        # Combine features
        return torch.cat([grid_features, piece_features], dim=1)

class ActorCritic(nn.Module):
    """
    Actor-Critic network for Tetris

    State Space:
    - Grid, next_piece, hold_piece, current_shape, rotation, x, y

    Action Space:
    - Flattened placement index = rotation*10 + column (rot∈[0-3], col∈[0-9])
    - 4 rotations × 10 columns = 40 actions
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize Actor-Critic network
        Args:
            input_dim: Dimension of input state (202)
            output_dim: Number of possible actions (40)
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
    def __init__(self, state_dim, action_dim,
                 actor_lr=1e-4, critic_lr=1e-3,
                 gamma_start=0.9, gamma_end=0.99,
                 epsilon_start=1.0, epsilon_end=0.05,
                 schedule_episodes=10000,
                 top_k_ac=3):
        """
        Initialize Actor-Critic agent
        Args:
            state_dim: Dimension of state space (202)
            action_dim: Number of possible actions (40)
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
        self.schedule_episodes = schedule_episodes
        # Initialize current epsilon, gamma, and episode count
        self.epsilon = epsilon_start
        self.epsilon_decay = (self.epsilon - self.epsilon_end) / self.schedule_episodes
        self.gamma = self.gamma_start
        self.gamma_decay = (self.gamma_start - self.gamma_end) / self.schedule_episodes
        self.current_episode = 0
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
    
    def select_action(self, state):  # returns integer in [0,39]
        """
        Select action using epsilon-greedy policy
        Args:
            state: Dictionary containing:
                - grid: 20x10 matrix of piece colors
                - current_piece: 4x4 matrix of current piece
                - next_piece: 4x4 matrix of next piece
                - hold_piece: 4x4 matrix of hold piece
        Returns:
            Integer (0-39) representing the selected action
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

    def update_schedules(self, total_completed_episodes): # Renamed from update_epsilon and added total_completed_episodes
        """Update epsilon and gamma schedules based on total completed episodes."""
        # Compute scheduling fraction
        # self.current_episode += 1 # Removed: episode count managed by training loop
        frac = min(1.0, total_completed_episodes / self.schedule_episodes)
        # Update epsilon (exploration rate)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
        # Update gamma (discount factor)
        self.gamma = self.gamma_start + frac * (self.gamma_end - self.gamma_start)
    
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
        # Sample batch from replay buffer (sample returns states, actions, rewards, next_states, dones, info, indices, weights)
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
        
        # Update critic network on V(s)
        self.critic_optimizer.zero_grad()
        state_values = self.network.critic(features)
        critic_loss = nn.MSELoss()(state_values, targets)
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.network.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        
        # Update actor network on π(a|s)
        self.actor_optimizer.zero_grad()
        action_probs = self.network.actor(features)
        log_probs = torch.log(action_probs.gather(1, actions))
        actor_loss = -(log_probs * targets.detach()).mean()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.network.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        # return losses: (actor_loss, critic_loss)
        return actor_loss.item(), critic_loss.item()
