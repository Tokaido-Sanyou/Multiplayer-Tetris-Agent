"""
Random Network Distillation (RND) for Exploration
Implements curiosity-driven exploration using prediction error on random network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Handle both direct execution and module import
try:
    from ..config import TetrisConfig  # Import centralized config
except ImportError:
    # Direct execution - add parent directory to path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TetrisConfig  # Import centralized config

class RandomTargetNetwork(nn.Module):
    """
    Random target network that generates fixed random features
    This network is never trained - its weights remain fixed
    """
    def __init__(self, state_dim=None):
        super(RandomTargetNetwork, self).__init__()
        
        # Get centralized config
        self.config = TetrisConfig()
        self.state_dim = state_dim or self.config.STATE_DIM  # 410
        
        # Fixed random network (never trained)
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Output feature dimension
        )
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, state):
        """
        Generate fixed random features for the given state
        Args:
            state: Tensor of shape (batch_size, state_dim)
        Returns:
            Random features of shape (batch_size, 64)
        """
        return self.network(state)

class PredictorNetwork(nn.Module):
    """
    Predictor network that learns to predict the random network's output
    This network is trained to minimize prediction error
    """
    def __init__(self, state_dim=None):
        super(PredictorNetwork, self).__init__()
        
        # Get centralized config
        self.config = TetrisConfig()
        self.state_dim = state_dim or self.config.STATE_DIM  # 410
        
        # Trainable predictor network
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Must match target network output
        )
    
    def forward(self, state):
        """
        Predict the random network's features
        Args:
            state: Tensor of shape (batch_size, state_dim)
        Returns:
            Predicted features of shape (batch_size, 64)
        """
        return self.network(state)

class RNDExploration(nn.Module):
    """
    Random Network Distillation for exploration
    Provides intrinsic motivation based on prediction error
    """
    def __init__(self, state_dim=None):
        super(RNDExploration, self).__init__()
        
        # Get centralized config
        self.config = TetrisConfig()
        self.state_dim = state_dim or self.config.STATE_DIM  # 410
        
        # Initialize networks
        self.target_network = RandomTargetNetwork(self.state_dim)
        self.predictor_network = PredictorNetwork(self.state_dim)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Normalization parameters for intrinsic reward
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.update_count = 0
        
    def forward(self, state):
        """
        Compute intrinsic reward based on prediction error
        Args:
            state: Tensor of shape (batch_size, state_dim)
        Returns:
            intrinsic_reward: Tensor of shape (batch_size, 1)
            prediction_error: Tensor of shape (batch_size, 1) - for logging
        """
        with torch.no_grad():
            target_features = self.target_network(state)
        
        predicted_features = self.predictor_network(state)
        
        # Calculate prediction error as intrinsic reward
        prediction_error = F.mse_loss(predicted_features, target_features, reduction='none')
        prediction_error = prediction_error.mean(dim=1, keepdim=True)  # (batch_size, 1)
        
        # Normalize intrinsic reward
        intrinsic_reward = self._normalize_reward(prediction_error)
        
        return intrinsic_reward, prediction_error
    
    def train_predictor(self, state_batch, optimizer):
        """
        Train the predictor network to minimize prediction error
        Args:
            state_batch: Tensor of shape (batch_size, state_dim)
            optimizer: Optimizer for predictor network
        Returns:
            loss: Training loss value
        """
        # Get target features (fixed)
        with torch.no_grad():
            target_features = self.target_network(state_batch)
        
        # Get predicted features
        predicted_features = self.predictor_network(state_batch)
        
        # Calculate loss
        loss = self.criterion(predicted_features, target_features)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.predictor_network.parameters(), 1.0)
        
        optimizer.step()
        
        return loss.item()
    
    def _normalize_reward(self, reward):
        """
        Normalize intrinsic rewards using running statistics
        Args:
            reward: Tensor of shape (batch_size, 1)
        Returns:
            normalized_reward: Tensor of shape (batch_size, 1)
        """
        self.update_count += 1
        
        # Update running statistics
        current_mean = reward.mean().item()
        current_std = reward.std().item() + 1e-8
        
        # Exponential moving average
        alpha = min(1.0 / self.update_count, 0.01)
        self.reward_mean = (1 - alpha) * self.reward_mean + alpha * current_mean
        self.reward_std = (1 - alpha) * self.reward_std + alpha * current_std
        
        # Normalize
        normalized_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        
        # Clip to reasonable range
        normalized_reward = torch.clamp(normalized_reward, -5.0, 5.0)
        
        return normalized_reward
    
    def get_exploration_stats(self):
        """
        Get current exploration statistics
        Returns:
            Dictionary with exploration metrics
        """
        return {
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'update_count': self.update_count
        }

class RNDExplorationActor:
    """
    Enhanced exploration actor using Random Network Distillation
    Replaces simple reward-based exploration with curiosity-driven exploration
    """
    def __init__(self, env, state_dim=None):
        self.env = env
        
        # Get centralized config
        self.config = TetrisConfig()
        self.state_dim = state_dim or self.config.STATE_DIM  # 410
        
        # Initialize RND exploration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            
        self.rnd_exploration = RNDExploration(self.state_dim).to(self.device)
        self.rnd_optimizer = torch.optim.Adam(
            self.rnd_exploration.predictor_network.parameters(), 
            lr=1e-4
        )
        
        # Experience storage for RND training
        self.state_buffer = []
        self.buffer_size = 10000
        
    def collect_placement_data(self, num_episodes=100):
        """
        Collect data using RND-driven exploration
        Focus on terminal states with intrinsic motivation
        """
        placement_data = []
        
        print(f"Starting RND-driven placement data collection for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            print(f"Episode {episode+1}/{num_episodes}...")
            try:
                obs = self.env.reset()
                episode_placements = 0
                max_steps_per_episode = 20
                
                for step in range(max_steps_per_episode):
                    print(f"  Step {step+1}/{max_steps_per_episode}", end="")
                    try:
                        # Get current state
                        current_state = self._obs_to_state_vector(obs)
                        print(f" - got state", end="")
                        
                        # Calculate intrinsic motivation using RND
                        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
                        intrinsic_reward, prediction_error = self.rnd_exploration(state_tensor)
                        intrinsic_reward_value = intrinsic_reward.item()
                        
                        print(f" - intrinsic reward: {intrinsic_reward_value:.3f}", end="")
                        
                        # Add state to buffer for RND training
                        self.state_buffer.append(current_state)
                        if len(self.state_buffer) > self.buffer_size:
                            self.state_buffer.pop(0)
                        
                        # Train RND predictor if we have enough data
                        if len(self.state_buffer) >= 64:
                            self._train_rnd_predictor()
                        
                        # Extract piece information
                        if np.any(obs['next_piece']):
                            next_piece_idx = np.argmax(obs['next_piece'])
                        else:
                            next_piece_idx = 0
                        piece_shape = next_piece_idx + 1
                        print(f" - piece {piece_shape}", end="")
                        
                        # Generate terminal placements with RND bias
                        print(f" - generating RND-guided placements", end="")
                        
                        # Use intrinsic reward to guide exploration intensity
                        num_placements = max(2, min(6, int(2 + intrinsic_reward_value * 2)))
                        
                        for i in range(num_placements):
                            # RND-guided placement generation
                            terminal_rotation, terminal_x_pos = self._generate_rnd_guided_placement(
                                current_state, intrinsic_reward_value
                            )
                            
                            # Simulate terminal state
                            terminal_state, terminal_reward = self._simulate_terminal_placement(
                                current_state, terminal_rotation, terminal_x_pos, piece_shape, 
                                intrinsic_reward_value
                            )
                            
                            if terminal_state is not None:
                                placement_data.append({
                                    'state': current_state,
                                    'placement': (terminal_rotation, terminal_x_pos),
                                    'terminal_reward': terminal_reward,
                                    'resulting_state': terminal_state,
                                    'intrinsic_reward': intrinsic_reward_value,
                                    'prediction_error': prediction_error.item()
                                })
                                episode_placements += 1
                        
                        print(f" - added {num_placements} RND-guided placements", end="")
                        
                        # Take action to continue episode
                        action_one_hot = np.zeros(8, dtype=np.int8)
                        action_one_hot[np.random.randint(0, 7)] = 1
                        obs, reward, done, info = self.env.step(action_one_hot)
                        print(f" - done={done}")
                        
                        if done:
                            print(f"    Episode terminated at step {step+1}")
                            # Record final terminal state with intrinsic reward
                            final_state = self._obs_to_state_vector(obs)
                            placement_data.append({
                                'state': current_state,
                                'placement': (0, 0),
                                'terminal_reward': reward + intrinsic_reward_value,  # Combine extrinsic + intrinsic
                                'resulting_state': final_state,
                                'intrinsic_reward': intrinsic_reward_value,
                                'prediction_error': prediction_error.item()
                            })
                            break
                            
                    except Exception as e:
                        print(f"    Error in step {step+1}: {e}")
                        break
                
                print(f"  Episode {episode+1} completed with {episode_placements} RND-guided placements")
                
            except Exception as e:
                print(f"  Error in episode {episode+1}: {e}")
                continue
        
        # Get exploration statistics
        rnd_stats = self.rnd_exploration.get_exploration_stats()
        print(f"RND Collection completed. Total placements: {len(placement_data)}")
        print(f"RND Stats - Mean: {rnd_stats['reward_mean']:.3f}, Std: {rnd_stats['reward_std']:.3f}")
        
        return placement_data
    
    def _generate_rnd_guided_placement(self, state, intrinsic_reward):
        """
        Generate placements guided by RND intrinsic reward
        Higher intrinsic reward encourages more diverse exploration
        """
        if intrinsic_reward > 0.5:  # High intrinsic reward -> explore more
            # More random/diverse placements
            rotation = np.random.randint(0, 4)
            x_pos = np.random.randint(0, 10)
        else:  # Low intrinsic reward -> exploit more
            # More centered/conservative placements
            rotation = np.random.choice([0, 1], p=[0.7, 0.3])  # Prefer standard orientations
            x_pos = np.random.choice(range(3, 8))  # Prefer center columns
        
        return rotation, x_pos
    
    def _train_rnd_predictor(self):
        """Train the RND predictor network on collected states"""
        if len(self.state_buffer) < 64:
            return
        
        # Sample random batch from buffer
        batch_indices = np.random.choice(len(self.state_buffer), 64, replace=False)
        state_batch = torch.FloatTensor([self.state_buffer[i] for i in batch_indices]).to(self.device)
        
        # Train predictor
        loss = self.rnd_exploration.train_predictor(state_batch, self.rnd_optimizer)
        
        return loss
    
    def _simulate_terminal_placement(self, current_state, rotation, x_pos, piece_shape, intrinsic_reward):
        """
        Simulate terminal placement with intrinsic reward integration
        """
        try:
            terminal_state = current_state.copy()
            
            # Base terminal reward
            terminal_reward = self._evaluate_terminal_placement(rotation, x_pos, piece_shape)
            
            # Add intrinsic reward component
            intrinsic_bonus = intrinsic_reward * 10.0  # Scale intrinsic reward
            terminal_reward += intrinsic_bonus
            
            # Modify state to reflect terminal placement with some randomness
            state_noise = np.random.randn(len(terminal_state)) * 0.01
            terminal_state = terminal_state + state_noise
            
            return terminal_state, terminal_reward
            
        except Exception as e:
            print(f"[SIM_ERR:{e}]", end="")
            return None, 0
    
    def _evaluate_terminal_placement(self, rotation, x_pos, piece_shape):
        """Evaluate terminal placement quality"""
        # Use updated reward weights from config
        config = self.config.RewardConfig
        
        center_bonus = -abs(x_pos - 4.5) * 5
        rotation_bonus = -abs(rotation - 1) * 2
        piece_bonus = piece_shape * 2
        random_bonus = np.random.uniform(-10, 10)
        
        terminal_reward = center_bonus + rotation_bonus + piece_bonus + random_bonus
        return max(-100, min(100, terminal_reward))
    
    def _obs_to_state_vector(self, obs):
        """Convert observation to state vector (410-dimensional)"""
        current_piece_flat = obs['current_piece_grid'].flatten()  # 200
        empty_grid_flat = obs['empty_grid'].flatten()  # 200
        next_piece = obs['next_piece']  # 7
        metadata = np.array([
            obs['current_rotation'],
            obs['current_x'], 
            obs['current_y']
        ])  # 3
        
        return np.concatenate([
            current_piece_flat, 
            empty_grid_flat,
            next_piece,
            metadata
        ]) 