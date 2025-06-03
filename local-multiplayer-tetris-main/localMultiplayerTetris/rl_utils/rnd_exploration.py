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
        self.reward_std = 1.0  # Initialize to 1.0 instead of 0
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
        
        # Add small minimum error to avoid zero variance issues
        prediction_error = prediction_error + 1e-6
        
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
        current_std = reward.std().item()
        
        # Handle NaN and zero std cases
        if torch.isnan(reward).any() or np.isnan(current_mean) or np.isnan(current_std):
            # Return small positive values if we have NaN
            return torch.ones_like(reward) * 0.1
        
        if current_std < 1e-8:
            current_std = 1.0  # Set to reasonable default
        
        # Exponential moving average
        alpha = min(1.0 / self.update_count, 0.01)
        
        # Update running mean
        if np.isnan(self.reward_mean):
            self.reward_mean = current_mean
        else:
            self.reward_mean = (1 - alpha) * self.reward_mean + alpha * current_mean
        
        # Update running std  
        if np.isnan(self.reward_std) or self.reward_std < 1e-8:
            self.reward_std = max(current_std, 1.0)
        else:
            self.reward_std = (1 - alpha) * self.reward_std + alpha * current_std
        
        # Ensure std is never too small
        self.reward_std = max(self.reward_std, 1e-4)
        
        # Normalize
        normalized_reward = (reward - self.reward_mean) / self.reward_std
        
        # Handle any remaining NaN values
        if torch.isnan(normalized_reward).any():
            normalized_reward = torch.where(torch.isnan(normalized_reward), 
                                          torch.ones_like(normalized_reward) * 0.1, 
                                          normalized_reward)
        
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
    Now focuses on terminal value prediction and unvisited terminal states
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
        
        # NEW: Terminal state tracking for novelty detection
        self.visited_terminal_states = set()  # Store hash of terminal states
        self.terminal_value_history = []  # Store terminal values for analysis
        self.novelty_bonus_scale = 5.0  # Scale for novelty bonus
        self.prev_unique_count = 0  # Track previous batch count for delta calculation
        
    def collect_placement_data(self, num_episodes=100):
        """
        Collect ONLY terminal state data using RND-driven exploration
        Focus on unvisited terminal states with terminal value-based intrinsic rewards
        Each episode focuses on the same piece type for consistent exploration
        """
        placement_data = []
        
        # Adjust episodes to ensure good coverage of all 7 piece types
        # Use multiples of 7 to ensure even distribution
        adjusted_episodes = ((num_episodes + 6) // 7) * 7  # Round up to nearest multiple of 7
        piece_types = 7  # I, O, T, S, Z, J, L pieces
        episodes_per_piece = adjusted_episodes // piece_types
        
        print(f"Starting RND exploration: {adjusted_episodes} episodes ({episodes_per_piece} per piece type)...")
        
        for piece_type in range(piece_types):
            piece_shape = piece_type + 1  # 1-7 for piece types
            
            for episode_in_piece in range(episodes_per_piece):
                episode_num = piece_type * episodes_per_piece + episode_in_piece
                try:
                    obs = self.env.reset()
                    
                    # Force the specific piece type for this episode
                    # We'll work with whatever piece appears but focus our exploration on this piece type
                    target_piece_shape = piece_shape
                    
                    episode_terminal_placements = 0
                    max_steps_per_episode = 20
                    
                    for step in range(max_steps_per_episode):
                        try:
                            # Get current state
                            current_state = self._obs_to_state_vector(obs)
                            
                            # Calculate RND intrinsic reward for this state
                            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
                            intrinsic_reward, prediction_error = self.rnd_exploration(state_tensor)
                            intrinsic_reward_value = intrinsic_reward.item()
                            
                            # Handle NaN or invalid intrinsic reward values
                            if np.isnan(intrinsic_reward_value) or np.isinf(intrinsic_reward_value):
                                intrinsic_reward_value = 0.1  # Default small positive value
                            
                            # Add state to buffer for RND training
                            self.state_buffer.append(current_state)
                            if len(self.state_buffer) > self.buffer_size:
                                self.state_buffer.pop(0)
                            
                            # Train RND predictor if we have enough data
                            if len(self.state_buffer) >= 64:
                                self._train_rnd_predictor()
                            
                            # Generate ONLY terminal placements with RND + novelty bias
                            # Focus on the target piece type for this episode
                            num_terminal_trials = max(4, min(10, int(4 + intrinsic_reward_value * 4)))
                            
                            for i in range(num_terminal_trials):
                                # Generate terminal placement with novelty bias for target piece
                                terminal_data = self._generate_terminal_placement_with_novelty(
                                    current_state, intrinsic_reward_value, target_piece_shape
                                )
                                
                                if terminal_data is not None:
                                    # Mark which piece type this exploration was focused on
                                    terminal_data['target_piece_type'] = target_piece_shape
                                    placement_data.append(terminal_data)
                                    episode_terminal_placements += 1
                            
                            # Take action to continue episode
                            action_one_hot = np.zeros(8, dtype=np.int8)
                            action_one_hot[np.random.randint(0, 7)] = 1
                            obs, reward, done, info = self.env.step(action_one_hot)
                            
                            if done:
                                # Record FINAL terminal state with enhanced intrinsic reward
                                final_state = self._obs_to_state_vector(obs)
                                final_terminal_data = self._create_terminal_data(
                                    current_state, final_state, reward, intrinsic_reward_value, is_final=True
                                )
                                final_terminal_data['target_piece_type'] = target_piece_shape
                                placement_data.append(final_terminal_data)
                                break
                                
                        except Exception as e:
                            print(f"Error in piece {piece_shape} episode {episode_in_piece+1}, step {step+1}: {e}")
                            break
                    
                except Exception as e:
                    print(f"Error in piece {piece_shape} episode {episode_in_piece+1}: {e}")
                    continue
        
        # Get exploration statistics
        rnd_stats = self.rnd_exploration.get_exploration_stats()
        novelty_stats = self._get_novelty_stats()
        
        # DEBUG: Add detailed uniqueness analysis
        if len(placement_data) > 0:
            # Check actual diversity in placements
            placements = [d.get('placement', (0, 0, 10)) for d in placement_data]
            rotations = [p[0] for p in placements]
            x_positions = [p[1] for p in placements]
            y_positions = [p[2] if len(p) > 2 else 10 for p in placements]
            
            unique_rotations = len(set(rotations))
            unique_x_positions = len(set(x_positions))
            unique_y_positions = len(set(y_positions))
            unique_placements = len(set(placements))
            
            # Sample state diversity check
            if len(placement_data) >= 10:
                sample_states = [placement_data[i]['resulting_state'] for i in range(0, min(10, len(placement_data)))]
                state_hashes = []
                for state in sample_states:
                    grid_hash = hash(tuple(state[:400:5]))
                    piece_hash = hash(tuple(state[400:407]))
                    pos_hash = hash(tuple(state[407:410]))
                    combined_hash = hash((grid_hash, piece_hash, pos_hash))
                    state_hashes.append(combined_hash)
                sample_unique = len(set(state_hashes))
                
                print(f"   • DEBUG: Sample uniqueness check: {sample_unique}/10 unique in first 10 states")
                print(f"   • DEBUG: Placement diversity: {unique_rotations} rotations, {unique_x_positions} x-pos, {unique_y_positions} y-pos")
                print(f"   • DEBUG: Total unique placements: {unique_placements}/{len(placement_data)}")
                
                # Show rotation distribution
                rotation_counts = {}
                for r in rotations[:100]:  # First 100 for debugging
                    rotation_counts[r] = rotation_counts.get(r, 0) + 1
                print(f"   • DEBUG: Rotation distribution (first 100): {dict(sorted(rotation_counts.items()))}")
        
        print(f"✅ RND Exploration completed:")
        print(f"   • Total terminal placements: {len(placement_data)}")
        print(f"   • Episodes per piece type: {episodes_per_piece}")
        print(f"   • Unique terminal states discovered: {novelty_stats['unique_terminals']}")
        print(f"   • New distinct states this batch: {novelty_stats['new_terminals_this_batch']}")
        print(f"   • Average terminal value: {novelty_stats.get('avg_terminal_value', 0):.2f}")
        print(f"   • RND learning progress - Mean: {rnd_stats['reward_mean']:.4f}, Std: {rnd_stats['reward_std']:.4f}")
        
        return placement_data
    
    def _generate_terminal_placement_with_novelty(self, state, intrinsic_reward, piece_shape):
        """
        Generate terminal placement with strong bias toward unvisited terminal states
        """
        max_attempts = 10
        best_terminal_data = None
        best_novelty_score = -float('inf')
        
        for attempt in range(max_attempts):
            # Generate much more diverse potential terminal placements
            if intrinsic_reward > 0.5:  # High intrinsic reward → explore more
                terminal_rotation = np.random.randint(0, 4)
                terminal_x_pos = np.random.randint(0, 10)
                # Add Y position variation for more diversity
                terminal_y_pos = np.random.randint(0, 20)
            else:  # Low intrinsic reward → exploit more, but still diverse
                terminal_rotation = np.random.randint(0, 4)  # All rotations possible
                terminal_x_pos = np.random.randint(0, 10)   # All positions possible
                terminal_y_pos = np.random.randint(10, 20)  # Prefer lower positions
            
            # Simulate this terminal placement with more parameters
            terminal_state, terminal_value = self._simulate_terminal_placement(
                state, terminal_rotation, terminal_x_pos, piece_shape, intrinsic_reward, terminal_y_pos
            )
            
            if terminal_state is not None:
                # Calculate novelty score for this terminal state
                novelty_score = self._calculate_terminal_novelty(terminal_state, terminal_value)
                
                if novelty_score > best_novelty_score:
                    best_novelty_score = novelty_score
                    # Store placement parameters for accurate recording
                    placement_params = {
                        'rotation': terminal_rotation,
                        'x_pos': terminal_x_pos,
                        'y_pos': terminal_y_pos
                    }
                    best_terminal_data = self._create_terminal_data(
                        state, terminal_state, terminal_value, intrinsic_reward, 
                        novelty_score=novelty_score, placement_params=placement_params
                    )
        
        return best_terminal_data
    
    def _calculate_terminal_novelty(self, terminal_state, terminal_value):
        """
        Calculate novelty score for a terminal state
        Higher score = more novel/unvisited
        """
        # Create more comprehensive hash using multiple components
        # Use grid state (first 400 elements), piece info (next 7), and position info (last 3)
        grid_hash = hash(tuple(terminal_state[:400:5]))  # Grid every 5th element (80 elements)
        piece_hash = hash(tuple(terminal_state[400:407]))  # All piece info (7 elements)
        pos_hash = hash(tuple(terminal_state[407:410]))   # All position info (3 elements)
        
        # Combine hashes for better uniqueness
        state_hash = hash((grid_hash, piece_hash, pos_hash))
        
        # Check if this terminal state pattern has been visited
        if state_hash in self.visited_terminal_states:
            novelty_bonus = 0.1  # Small bonus for revisited states
        else:
            novelty_bonus = self.novelty_bonus_scale  # Large bonus for new states
            self.visited_terminal_states.add(state_hash)
        
        # Combine terminal value with novelty
        # Higher terminal values + novelty get higher scores
        novelty_score = terminal_value + novelty_bonus
        
        return novelty_score
    
    def _create_terminal_data(self, state, terminal_state, terminal_value, intrinsic_reward, 
                             novelty_score=0.0, is_final=False, placement_params=None):
        """
        Create terminal placement data with enhanced rewards
        """
        # Use actual placement parameters if provided, otherwise default
        if placement_params is not None:
            placement = (placement_params['rotation'], placement_params['x_pos'], placement_params['y_pos'])
        else:
            placement = (0, 0, 10)  # Default placement
        
        # Enhanced terminal reward combines multiple signals
        enhanced_terminal_reward = (
            terminal_value +  # Base terminal value
            intrinsic_reward * 2.0 +  # RND curiosity bonus
            novelty_score * 0.5  # Novelty exploration bonus
        )
        
        terminal_data = {
            'state': state,
            'placement': placement,  # Now includes (rotation, x_pos, y_pos)
            'terminal_reward': enhanced_terminal_reward,  # Enhanced with RND + novelty
            'resulting_state': terminal_state,
            'intrinsic_reward': intrinsic_reward,
            'terminal_value': terminal_value,  # NEW: Actual terminal value
            'novelty_score': novelty_score,  # NEW: Novelty bonus
            'is_final_state': is_final,  # NEW: Flag for episode-ending states
        }
        
        # Track terminal value for analysis
        self.terminal_value_history.append(terminal_value)
        
        return terminal_data
    
    def _get_novelty_stats(self):
        """Get statistics about terminal state novelty exploration"""
        current_unique = len(self.visited_terminal_states)
        new_this_batch = current_unique - self.prev_unique_count
        
        stats = {
            'unique_terminals': current_unique,
            'prev_unique_terminals': self.prev_unique_count,
            'new_terminals_this_batch': new_this_batch,
            'total_terminals_seen': len(self.terminal_value_history),
            'avg_terminal_value': np.mean(self.terminal_value_history) if self.terminal_value_history else 0.0,
            'avg_novelty_bonus': self.novelty_bonus_scale if self.visited_terminal_states else 0.0,
        }
        
        # Update previous count for next batch
        self.prev_unique_count = current_unique
        
        return stats
    
    def _train_rnd_predictor(self):
        """Train the RND predictor network on collected states"""
        if len(self.state_buffer) < 64:
            return
        
        # Sample random batch from buffer - fix performance warning
        batch_indices = np.random.choice(len(self.state_buffer), 64, replace=False)
        # Convert to numpy array first to avoid PyTorch warning
        state_array = np.array([self.state_buffer[i] for i in batch_indices])
        state_batch = torch.FloatTensor(state_array).to(self.device)
        
        # Train predictor
        loss = self.rnd_exploration.train_predictor(state_batch, self.rnd_optimizer)
        
        return loss
    
    def _simulate_terminal_placement(self, current_state, rotation, x_pos, piece_shape, intrinsic_reward, terminal_y_pos=10):
        """
        Simulate terminal placement and return terminal state + terminal VALUE (not reward)
        """
        try:
            terminal_state = current_state.copy()
            
            # Calculate actual terminal VALUE based on placement quality
            terminal_value = self._evaluate_terminal_placement_value(rotation, x_pos, piece_shape, current_state, terminal_y_pos)
            
            # Modify state to reflect terminal placement with realistic changes
            # Simulate piece placement effects on the grid
            placement_effects = self._simulate_piece_placement_effects(current_state, rotation, x_pos, piece_shape, terminal_y_pos)
            terminal_state = terminal_state + placement_effects
            
            return terminal_state, terminal_value
            
        except Exception as e:
            print(f"[SIM_ERR:{e}]", end="")
            return None, 0.0
    
    def _evaluate_terminal_placement_value(self, rotation, x_pos, piece_shape, current_state, terminal_y_pos=10):
        """
        Evaluate terminal placement VALUE including LINE CLEARING POTENTIAL
        CRITICAL FIX: Now includes line clearing rewards - the core Tetris objective!
        """
        # Use updated reward weights from config
        config = self.config.RewardConfig
        
        # CRITICAL ADDITION: Calculate line clearing potential and rewards
        line_clear_value = self._calculate_line_clearing_value(current_state, rotation, x_pos, piece_shape, terminal_y_pos)
        
        # Base value calculations with more diverse factors
        center_distance = abs(x_pos - 4.5)  # Distance from center
        rotation_optimality = min(rotation, 4 - rotation)  # Prefer 0 or minimal rotations
        height_factor = (20 - terminal_y_pos) / 20.0  # Height preference (0-1)
        
        # Advanced terminal value calculation with more diversity
        height_penalty = -center_distance * config.MAX_HEIGHT_WEIGHT * 0.1
        rotation_bonus = -rotation_optimality * 2.0
        piece_type_bonus = piece_shape * 1.5  # Some pieces are more valuable
        position_bonus = height_factor * 10.0  # Reward lower placements
        
        # Add placement-specific randomness for diversity
        placement_id = hash((rotation, x_pos, piece_shape, terminal_y_pos)) % 1000
        placement_noise = (placement_id / 1000.0) * 10 - 5  # -5 to +5 based on placement
        
        # Add some game-state specific evaluation
        grid_state_value = self._evaluate_grid_state_value(current_state)
        
        # ENHANCED: Combine all factors with LINE CLEARING as primary objective
        terminal_value = (
            line_clear_value * 3.0 +  # 🔥 LINE CLEARING VALUE (3x weight - most important!)
            grid_state_value +        # Strategic grid position value
            rotation_bonus +          # Rotation optimality
            piece_type_bonus -        # Piece type value
            height_penalty +          # Height strategy penalty
            position_bonus +          # Position-based bonus
            placement_noise           # Deterministic but diverse noise
        )
        
        # Clamp to reasonable range
        terminal_value = max(-100, min(200, terminal_value))  # Expanded range for line clears
        
        return terminal_value
    
    def _calculate_line_clearing_value(self, current_state, rotation, x_pos, piece_shape, terminal_y_pos):
        """
        CRITICAL NEW METHOD: Calculate line clearing potential and rewards for this placement
        This is the core Tetris objective that was missing from terminal rewards!
        
        Args:
            current_state: Game state vector
            rotation, x_pos, piece_shape, terminal_y_pos: Placement parameters
        Returns:
            line_clear_value: Value based on line clearing potential and rewards
        """
        try:
            # Extract board state
            current_piece_grid = current_state[:200].reshape(20, 10)
            empty_grid = current_state[200:400].reshape(20, 10)
            
            # Create combined board state (1 = occupied, 0 = empty)
            board = 1 - empty_grid  # Invert empty grid to get occupied cells
            
            # Simulate placing the piece to see line clearing effects
            simulated_board = board.copy()
            
            # Simple piece placement simulation (approximate)
            # Place a small footprint around the target position
            piece_positions = self._get_approximate_piece_positions(rotation, x_pos, terminal_y_pos, piece_shape)
            
            for row, col in piece_positions:
                if 0 <= row < 20 and 0 <= col < 10:
                    simulated_board[row, col] = 1
            
            # Calculate lines that would be cleared
            lines_cleared = 0
            cleared_lines = []
            
            for row in range(20):
                if np.sum(simulated_board[row, :]) == 10:  # Full line
                    lines_cleared += 1
                    cleared_lines.append(row)
            
            # Calculate line clearing rewards (using game config)
            line_clear_reward = 0
            if lines_cleared > 0:
                # Use actual Tetris scoring from config
                line_clear_base = {1: 100, 2: 200, 3: 400, 4: 1600}
                line_clear_reward = line_clear_base.get(lines_cleared, lines_cleared * 400)
                
                # Add bonus for clearing multiple lines simultaneously
                if lines_cleared >= 2:
                    line_clear_reward *= 1.5  # Bonus for double/triple/tetris
                
                print(f"     🔥 LINE CLEAR DETECTED! {lines_cleared} lines, reward: {line_clear_reward}")
            
            # Calculate line clearing POTENTIAL (near-complete lines)
            line_potential = 0
            for row in range(20):
                filled_cells = np.sum(simulated_board[row, :])
                if filled_cells >= 8:  # Close to being complete
                    potential_value = (filled_cells - 7) * 10  # 10, 20, 30 for 8, 9, 10 filled
                    line_potential += potential_value
            
            # Calculate setup value (creating good line clearing opportunities)
            setup_value = self._calculate_line_clearing_setup_value(simulated_board, cleared_lines)
            
            # Total line clearing value
            total_line_value = line_clear_reward + line_potential + setup_value
            
            return total_line_value
            
        except Exception as e:
            print(f"Error calculating line clearing value: {e}")
            return 0.0  # Fallback to no line clearing value
    
    def _get_approximate_piece_positions(self, rotation, x_pos, y_pos, piece_shape):
        """
        Get approximate positions where a piece would be placed
        This is a simplified version - the actual piece shape depends on the piece type
        """
        positions = []
        
        # Basic piece footprints (simplified)
        if piece_shape == 0:  # I-piece
            if rotation % 2 == 0:  # Horizontal
                positions = [(y_pos, x_pos), (y_pos, x_pos+1), (y_pos, x_pos+2), (y_pos, x_pos+3)]
            else:  # Vertical
                positions = [(y_pos, x_pos), (y_pos+1, x_pos), (y_pos+2, x_pos), (y_pos+3, x_pos)]
        elif piece_shape == 1:  # O-piece
            positions = [(y_pos, x_pos), (y_pos, x_pos+1), (y_pos+1, x_pos), (y_pos+1, x_pos+1)]
        elif piece_shape == 2:  # T-piece
            if rotation == 0:
                positions = [(y_pos, x_pos-1), (y_pos, x_pos), (y_pos, x_pos+1), (y_pos+1, x_pos)]
            elif rotation == 1:
                positions = [(y_pos, x_pos), (y_pos+1, x_pos-1), (y_pos+1, x_pos), (y_pos+2, x_pos)]
            elif rotation == 2:
                positions = [(y_pos, x_pos), (y_pos+1, x_pos-1), (y_pos+1, x_pos), (y_pos+1, x_pos+1)]
            else:  # rotation == 3
                positions = [(y_pos, x_pos), (y_pos+1, x_pos), (y_pos+1, x_pos+1), (y_pos+2, x_pos)]
        else:  # Other pieces - simplified footprint
            positions = [(y_pos, x_pos), (y_pos, x_pos+1), (y_pos+1, x_pos), (y_pos+1, x_pos+1)]
        
        # Filter valid positions
        valid_positions = [(r, c) for r, c in positions if 0 <= r < 20 and 0 <= c < 10]
        return valid_positions
    
    def _calculate_line_clearing_setup_value(self, board, cleared_lines):
        """
        Calculate the value of setting up future line clearing opportunities
        """
        setup_value = 0
        
        # After line clears, check if we've created good opportunities
        if cleared_lines:
            # Remove cleared lines and shift board down
            new_board = board.copy()
            for line in sorted(cleared_lines, reverse=True):
                new_board = np.delete(new_board, line, axis=0)
                new_board = np.vstack([np.zeros((1, 10)), new_board])
            
            # Check for new line clearing opportunities
            for row in range(20):
                filled_cells = np.sum(new_board[row, :])
                if filled_cells >= 7:  # Setup for future line clear
                    setup_value += (filled_cells - 6) * 5
        
        # Check for creating wells (good for I-pieces)
        column_heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if board[row, col] == 1:
                    height = 20 - row
                    break
            column_heights.append(height)
        
        # Reward creating wells (height differences of 3+)
        for i in range(len(column_heights) - 1):
            height_diff = abs(column_heights[i] - column_heights[i + 1])
            if height_diff >= 3:
                setup_value += height_diff * 2
        
        return setup_value
    
    def _evaluate_grid_state_value(self, state):
        """
        Evaluate the strategic value of the current grid state
        This helps determine how valuable different terminal states are
        """
        # Extract grid information from state vector
        current_piece_grid = state[:200].reshape(20, 10)  # First 200 elements
        empty_grid = state[200:400].reshape(20, 10)  # Next 200 elements
        
        # Calculate strategic metrics
        occupied_rows = np.where(np.sum(current_piece_grid + (1 - empty_grid), axis=1) > 0)[0]
        max_height = 20 - (occupied_rows[0] if len(occupied_rows) > 0 else 20)
        
        # Count holes (empty cells with filled cells above)
        holes = 0
        for col in range(10):
            column = current_piece_grid[:, col] + (1 - empty_grid[:, col])
            filled_found = False
            for row in range(20):
                if column[row] > 0:
                    filled_found = True
                elif filled_found and column[row] == 0:
                    holes += 1
        
        # Calculate bumpiness (height differences between adjacent columns)
        column_heights = []
        for col in range(10):
            column = current_piece_grid[:, col] + (1 - empty_grid[:, col])
            filled_rows = np.where(column > 0)[0]
            height = 20 - (filled_rows[0] if len(filled_rows) > 0 else 20)
            column_heights.append(height)
        
        bumpiness = sum(abs(column_heights[i] - column_heights[i+1]) for i in range(9))
        
        # Strategic value calculation (inverted penalties become values)
        grid_value = (
            -max_height * 0.5 +     # Lower height is better
            -holes * 2.0 +          # Fewer holes is better  
            -bumpiness * 0.1 +      # Smoother surface is better
            np.random.uniform(-2, 2)  # Small noise
        )
        
        return grid_value
    
    def _simulate_piece_placement_effects(self, current_state, rotation, x_pos, piece_shape, terminal_y_pos=10):
        """
        Simulate the effects of placing a piece on the game state
        Returns the change vector to apply to current_state
        """
        # Create a change vector the same size as current_state
        state_changes = np.zeros_like(current_state)
        
        # Create much more realistic and diverse placement effects
        # Base effect magnitude on placement parameters
        rotation_factor = 1.0 + rotation * 0.3  # Different rotations have different effects
        position_factor = 1.0 + abs(x_pos - 4.5) * 0.2  # Edge positions differ from center
        height_factor = 1.0 + (20 - terminal_y_pos) * 0.1  # Height affects impact
        piece_factor = 1.0 + piece_shape * 0.15  # Different pieces have different effects
        
        overall_magnitude = 0.5 * rotation_factor * position_factor * height_factor * piece_factor
        
        # Effect on current piece grid (200 elements, indices 0-199)
        # Create pattern based on placement parameters for diversity
        for i in range(200):
            grid_row = i // 10
            grid_col = i % 10
            
            # Distance from placement position affects the change
            col_distance = abs(grid_col - x_pos) + 1
            row_distance = abs(grid_row - terminal_y_pos) + 1
            spatial_factor = 1.0 / (col_distance * row_distance)
            
            # Create deterministic but diverse changes
            change_seed = hash((rotation, x_pos, piece_shape, terminal_y_pos, i)) % 10000
            change_value = (change_seed / 10000.0 - 0.5) * overall_magnitude * spatial_factor
            state_changes[i] = change_value
        
        # Effect on empty grid (200 elements, indices 200-399) - different pattern
        for i in range(200, 400):
            grid_idx = i - 200
            grid_row = grid_idx // 10
            grid_col = grid_idx % 10
            
            # Empty grid changes differently than piece grid
            col_distance = abs(grid_col - x_pos) + 1
            row_distance = abs(grid_row - terminal_y_pos) + 1
            spatial_factor = 1.0 / (col_distance * row_distance)
            
            # Create different deterministic pattern for empty grid
            change_seed = hash((rotation, x_pos, piece_shape, terminal_y_pos, i, 'empty')) % 10000
            change_value = (change_seed / 10000.0 - 0.5) * overall_magnitude * spatial_factor * 0.7
            state_changes[i] = change_value
        
        # Effect on next piece (indices 400-406) - rotation affects what might come next
        for i in range(400, 407):
            piece_idx = i - 400
            # Different rotations might affect piece sequence differently
            change_seed = hash((rotation, piece_shape, piece_idx)) % 1000
            change_value = (change_seed / 1000.0 - 0.5) * 0.1
            state_changes[i] = change_value
        
        # Strong effect on metadata (indices 407-409) - direct placement influence
        state_changes[407] = rotation * 0.25 + (hash((rotation, x_pos)) % 100) / 400.0  # Rotation influence + noise
        state_changes[408] = (x_pos - 5) * 0.2 + (hash((x_pos, piece_shape)) % 100) / 500.0  # X position influence + noise
        state_changes[409] = (terminal_y_pos - 10) * 0.15 + (hash((terminal_y_pos, rotation)) % 100) / 600.0  # Y position influence + noise
        
        return state_changes
    
    def _obs_to_state_vector(self, obs):
        """
        Convert observation to 410-dimensional state vector
        ENHANCED: Validates complete active block representation
        """
        # Current piece grid contains ALL active block coordinates (not just center)
        current_piece_flat = obs['current_piece_grid'].flatten()  # 20*10 = 200
        empty_grid_flat = obs['empty_grid'].flatten()  # 20*10 = 200 
        next_piece = obs['next_piece']  # 7 values
        metadata = np.array([
            obs['current_rotation'],
            obs['current_x'], 
            obs['current_y']
        ])  # 3 values
        
        # VALIDATION: Check that we have complete block representation
        active_blocks = np.sum(current_piece_flat > 0)
        if active_blocks > 0:
            # Validate that we have a reasonable number of active blocks (1-4 for Tetris pieces)
            if active_blocks < 1 or active_blocks > 4:
                print(f"     🔍 Validation: Unusual active block count: {active_blocks}")
            
            # Find active block positions for validation
            current_piece_grid = current_piece_flat.reshape(20, 10)
            active_positions = np.where(current_piece_grid > 0)
            if len(active_positions[0]) > 0:
                min_y, max_y = np.min(active_positions[0]), np.max(active_positions[0])
                min_x, max_x = np.min(active_positions[1]), np.max(active_positions[1])
                span_x, span_y = max_x - min_x, max_y - min_y
                
                # Tetris pieces should span at most 3 units in any direction
                if span_x > 3 or span_y > 3:
                    print(f"     🔍 Validation: Large piece span: x={span_x}, y={span_y}")
        
        # Concatenate: 200 + 200 + 7 + 3 = 410 (complete spatial representation)
        state_vector = np.concatenate([
            current_piece_flat, 
            empty_grid_flat,
            next_piece,
            metadata
        ])
        
        # Final validation
        assert len(state_vector) == 410, f"State vector should be 410 dimensions, got {len(state_vector)}"
        
        return state_vector

class DeterministicTerminalExplorer:
    """
    Sequential Deterministic exploration that generates terminal states in chains
    For each piece placement, generates ALL possible terminal states of the next piece
    Creates a sequence of 10 pieces with exponentially growing terminal state possibilities
    """
    def __init__(self, env, state_dim=None):
        self.env = env
        
        # Get centralized config
        self.config = TetrisConfig()
        self.state_dim = state_dim or self.config.STATE_DIM  # 410
        
        # Import necessary classes for validation
        try:
            from ..piece import Piece
            from ..constants import shapes
            from ..utils import create_grid
            from ..piece_utils import valid_space, convert_shape_format
        except ImportError:
            from piece import Piece
            from constants import shapes
            from utils import create_grid
            from piece_utils import valid_space, convert_shape_format
        
        self.Piece = Piece
        self.shapes = shapes
        self.create_grid = create_grid
        self.valid_space = valid_space
        self.convert_shape_format = convert_shape_format
        
        # Tetris game constraints
        self.num_rotations = 4  # 0, 1, 2, 3
        self.board_width = 10   # X positions 0-9
        self.board_height = 20  # Y positions 0-19
        self.piece_types = 7    # I, O, T, S, Z, J, L
        
        # Sequential chain parameters
        self.sequence_length = 10  # 10 pieces in sequence
        self.max_states_per_piece = 100  # Limit to prevent explosion
        
        # Track generated terminal states
        self.generated_terminals = []
        self.state_hash_set = set()
        
    def generate_all_terminal_states(self, sequence_length=3):
        """
        Generate terminal states through a sequential chain of pieces
        Each piece placement leads to ALL possible terminal states of the next piece
        ENHANCED: Now starts with partially filled boards for realistic line clearing opportunities
        
        Args:
            sequence_length: Number of pieces in the sequence (default 3, max recommended)
        Returns:
            List of terminal placement data (variable size)
        """
        print(f"🎯 Generating sequential terminal states for {sequence_length} pieces...")
        print(f"   Each piece placement → ALL terminal states of next piece")
        print(f"   ⚠️  Limited to {sequence_length} pieces to prevent exponential explosion")
        
        # Reset tracking
        self.generated_terminals = []
        self.state_hash_set = set()
        self.sequence_length = sequence_length
        
        # CRITICAL FIX: Start with PARTIALLY FILLED boards to enable line clearing
        # Generate multiple starting scenarios with different fill levels
        starting_scenarios = self._generate_line_clearing_scenarios()
        
        all_terminal_states = []
        
        for scenario_idx, initial_state in enumerate(starting_scenarios):
            print(f"   📋 Scenario {scenario_idx + 1}/{len(starting_scenarios)}: {self._describe_board_state(initial_state)}")
            
            # Generate sequence of random pieces
            piece_sequence = [np.random.randint(0, self.piece_types) for _ in range(sequence_length)]
            print(f"   🎲 Piece sequence: {[['S','Z','I','O','J','L','T'][p] for p in piece_sequence]}")
            
            # Start recursive chain generation
            scenario_terminals = self._generate_sequential_chain(
                state=initial_state,
                piece_sequence=piece_sequence,
                chain_depth=0,
                placement_history=[],
                scenario_id=scenario_idx
            )
            
            all_terminal_states.extend(scenario_terminals)
            
            print(f"   📊 Scenario {scenario_idx + 1} generated {len(scenario_terminals)} terminal states")
        
        print(f"✅ Sequential exploration completed:")
        print(f"   • Total terminal states generated: {len(all_terminal_states)}")
        print(f"   • Starting scenarios: {len(starting_scenarios)}")
        print(f"   • Chain depth reached: {max([t.get('chain_depth', 0) for t in all_terminal_states]) if all_terminal_states else 0}")
        print(f"   • Final sequence length: {sequence_length}")
        
        # Show distribution by chain depth
        depth_distribution = {}
        for terminal in all_terminal_states:
            depth = terminal.get('chain_depth', 0)
            depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
        print(f"   • States by chain depth: {dict(sorted(depth_distribution.items()))}")
        
        return all_terminal_states
    
    def _generate_line_clearing_scenarios(self):
        """
        CRITICAL NEW METHOD: Generate multiple starting board scenarios with line clearing potential
        This replaces empty board starts with realistic game states that can achieve line clears
        
        Returns:
            List of state vectors representing different board fill scenarios
        """
        scenarios = []
        
        # Scenario 1: Near-complete lines (8-9 filled cells per row)
        scenario_1 = self._create_near_complete_line_scenario()
        scenarios.append(scenario_1)
        
        # Scenario 2: Well setup for I-piece Tetris
        scenario_2 = self._create_tetris_well_scenario()
        scenarios.append(scenario_2)
        
        # Scenario 3: Multiple partial lines
        scenario_3 = self._create_multiple_partial_lines_scenario()
        scenarios.append(scenario_3)
        
        # Scenario 4: T-spin setup
        scenario_4 = self._create_t_spin_scenario()
        scenarios.append(scenario_4)
        
        # Scenario 5: Random mid-game state
        scenario_5 = self._create_random_mid_game_scenario()
        scenarios.append(scenario_5)
        
        return scenarios
    
    def _create_near_complete_line_scenario(self):
        """Create scenario with near-complete lines (8-9 filled cells)"""
        state = np.zeros(410)
        
        # Create empty grid (1 = empty, 0 = occupied)
        empty_grid = np.ones((20, 10))
        
        # Fill bottom 3 rows with near-complete lines
        for row in [19, 18, 17]:  # Bottom 3 rows
            filled_positions = np.random.choice(10, 8, replace=False)  # 8 random positions
            for pos in filled_positions:
                empty_grid[row, pos] = 0  # Mark as occupied
        
        # Add some scattered blocks above
        for row in [16, 15, 14]:
            filled_positions = np.random.choice(10, np.random.randint(3, 6), replace=False)
            for pos in filled_positions:
                empty_grid[row, pos] = 0
        
        state[200:400] = empty_grid.flatten()
        
        # Random next piece
        next_piece_idx = np.random.randint(0, 7)
        state[400 + next_piece_idx] = 1.0
        
        # Random metadata
        state[407] = np.random.uniform(0, 1)  # rotation
        state[408] = np.random.uniform(0, 1)  # x_pos
        state[409] = np.random.uniform(0, 1)  # y_pos
        
        return state
    
    def _create_tetris_well_scenario(self):
        """Create scenario with a well setup for I-piece Tetris"""
        state = np.zeros(410)
        empty_grid = np.ones((20, 10))
        
        # Create a well in column 9 (rightmost)
        well_column = 9
        well_height = 4
        
        # Fill all columns except the well column for bottom rows
        for row in range(20 - well_height, 20):  # Bottom 4 rows
            for col in range(10):
                if col != well_column:
                    empty_grid[row, col] = 0  # Occupied
        
        # Add some random fill above
        for row in range(20 - well_height - 3, 20 - well_height):
            filled_positions = np.random.choice([c for c in range(10) if c != well_column], 
                                              np.random.randint(4, 7), replace=False)
            for pos in filled_positions:
                empty_grid[row, pos] = 0
        
        state[200:400] = empty_grid.flatten()
        
        # Bias toward I-piece for Tetris potential
        state[400 + 2] = 1.0  # I-piece is index 2
        
        state[407] = 0.25  # rotation 1 (vertical I-piece)
        state[408] = 0.9   # x_pos near well
        state[409] = 0.2   # y_pos near top
        
        return state
    
    def _create_multiple_partial_lines_scenario(self):
        """Create scenario with multiple partial lines at different fill levels"""
        state = np.zeros(410)
        empty_grid = np.ones((20, 10))
        
        # Create lines with different fill levels
        fill_patterns = [6, 7, 8, 9, 7, 6]  # Different fill amounts
        
        for i, fill_amount in enumerate(fill_patterns):
            row = 19 - i  # Start from bottom
            if row >= 0:
                filled_positions = np.random.choice(10, fill_amount, replace=False)
                for pos in filled_positions:
                    empty_grid[row, pos] = 0
        
        state[200:400] = empty_grid.flatten()
        
        # Random next piece
        next_piece_idx = np.random.randint(0, 7)
        state[400 + next_piece_idx] = 1.0
        
        state[407] = np.random.uniform(0, 1)
        state[408] = np.random.uniform(0, 1)
        state[409] = np.random.uniform(0, 1)
        
        return state
    
    def _create_t_spin_scenario(self):
        """Create scenario set up for T-spin opportunities"""
        state = np.zeros(410)
        empty_grid = np.ones((20, 10))
        
        # Create T-spin setup pattern
        bottom_row = 19
        
        # Fill most of bottom row except T-spin hole
        t_spin_x = 5  # Middle position
        for col in range(10):
            if col not in [t_spin_x - 1, t_spin_x, t_spin_x + 1]:  # Leave T-shape space
                empty_grid[bottom_row, col] = 0
                
        # Create overhang for T-spin
        empty_grid[bottom_row - 1, t_spin_x] = 0  # Block above center
        
        # Fill surrounding areas
        for row in [bottom_row - 2, bottom_row - 3]:
            filled_positions = np.random.choice(10, np.random.randint(6, 8), replace=False)
            for pos in filled_positions:
                empty_grid[row, pos] = 0
        
        state[200:400] = empty_grid.flatten()
        
        # Bias toward T-piece
        state[400 + 6] = 1.0  # T-piece is index 6
        
        state[407] = 0.0   # rotation 0
        state[408] = 0.5   # x_pos centered
        state[409] = 0.1   # y_pos near bottom
        
        return state
    
    def _create_random_mid_game_scenario(self):
        """Create random mid-game scenario with varied fill"""
        state = np.zeros(410)
        empty_grid = np.ones((20, 10))
        
        # Random mid-game fill pattern
        for row in range(10, 20):  # Bottom half
            fill_probability = 0.3 + (20 - row) * 0.05  # More fill toward bottom
            for col in range(10):
                if np.random.random() < fill_probability:
                    empty_grid[row, col] = 0
        
        state[200:400] = empty_grid.flatten()
        
        # Random next piece
        next_piece_idx = np.random.randint(0, 7)
        state[400 + next_piece_idx] = 1.0
        
        state[407] = np.random.uniform(0, 1)
        state[408] = np.random.uniform(0, 1)
        state[409] = np.random.uniform(0, 1)
        
        return state
    
    def _describe_board_state(self, state):
        """Describe the board state for logging"""
        empty_grid = state[200:400].reshape(20, 10)
        occupied_cells = np.sum(1 - empty_grid)
        
        # Count near-complete lines
        near_complete = 0
        for row in range(20):
            filled = np.sum(1 - empty_grid[row, :])
            if filled >= 8:
                near_complete += 1
        
        return f"{occupied_cells:.0f} filled cells, {near_complete} near-complete lines"
    
    def _generate_sequential_chain(self, state, piece_sequence, chain_depth, placement_history, scenario_id):
        """
        Recursively generate sequential chain of piece placements
        
        Args:
            state: Current game state vector
            piece_sequence: List of piece indices to place
            chain_depth: Current depth in the chain (0-based)
            placement_history: List of previous placements for tracking
            scenario_id: Identifier for the current scenario
        Returns:
            List of terminal states generated from this chain
        """
        terminal_states = []
        
        # Base case: reached end of sequence
        if chain_depth >= len(piece_sequence):
            return terminal_states
        
        current_piece_idx = piece_sequence[chain_depth]
        current_piece_shape = self.shapes[current_piece_idx]
        current_piece_name = ['S','Z','I','O','J','L','T'][current_piece_idx]
        
        print(f"   🔗 Chain depth {chain_depth + 1}: Placing piece {current_piece_name}")
        
        # Get ALL valid placements for current piece
        valid_placements = self._get_all_valid_placements(state, current_piece_shape, current_piece_idx)
        
        if not valid_placements:
            print(f"     ⚠️  No valid placements for {current_piece_name} at depth {chain_depth + 1}")
            return terminal_states
        
        print(f"     📍 Found {len(valid_placements)} valid placements for {current_piece_name}")
        
        # For each valid placement of current piece
        placement_count = 0
        for placement_info in valid_placements:
            placement_count += 1
            
            # Extract placement details
            rotation, x_pos, y_pos = placement_info['placement']
            resulting_state = placement_info['resulting_state']
            
            # If this is the last piece in sequence, record as terminal state
            if chain_depth == len(piece_sequence) - 1:
                terminal_entry = {
                    'state': state,
                    'placement': (rotation, x_pos, y_pos),
                    'terminal_reward': placement_info['placement_value'],
                    'resulting_state': resulting_state,
                    'intrinsic_reward': 0.0,
                    'terminal_value': placement_info['placement_value'],
                    'novelty_score': 1.0,
                    'is_final_state': True,
                    'target_piece_type': current_piece_idx + 1,
                    'exploration_method': 'deterministic_sequential',
                    'chain_depth': chain_depth + 1,
                    'sequence_position': chain_depth + 1,
                    'piece_shape_name': current_piece_name,
                    'placement_history': placement_history + [(rotation, x_pos, y_pos, current_piece_name)],
                    'validated': True,
                    'is_chain_terminal': True,
                    'scenario_id': scenario_id
                }
                terminal_states.append(terminal_entry)
                
            else:
                # Continue chain: generate all terminal states of NEXT piece
                if chain_depth < len(piece_sequence) - 1:
                    next_piece_terminals = self._generate_sequential_chain(
                        state=resulting_state,
                        piece_sequence=piece_sequence,
                        chain_depth=chain_depth + 1,
                        placement_history=placement_history + [(rotation, x_pos, y_pos, current_piece_name)],
                        scenario_id=scenario_id
                    )
                    
                    # Add current placement info to each terminal state from next pieces
                    for next_terminal in next_piece_terminals:
                        # Create intermediate terminal state for current placement
                        intermediate_terminal = {
                            'state': state,
                            'placement': (rotation, x_pos, y_pos),
                            'terminal_reward': placement_info['placement_value'],
                            'resulting_state': resulting_state,
                            'intrinsic_reward': 0.0,
                            'terminal_value': placement_info['placement_value'],
                            'novelty_score': 1.0,
                            'is_final_state': False,  # Not final, leads to next piece
                            'target_piece_type': current_piece_idx + 1,
                            'exploration_method': 'deterministic_sequential',
                            'chain_depth': chain_depth + 1,
                            'sequence_position': chain_depth + 1,
                            'piece_shape_name': current_piece_name,
                            'placement_history': placement_history + [(rotation, x_pos, y_pos, current_piece_name)],
                            'validated': True,
                            'is_chain_terminal': False,
                            'leads_to_terminals': len(next_piece_terminals),
                            'next_piece_name': ['S','Z','I','O','J','L','T'][piece_sequence[chain_depth + 1]] if chain_depth + 1 < len(piece_sequence) else None,
                            'scenario_id': scenario_id
                        }
                        terminal_states.append(intermediate_terminal)
                    
                    # Add the final terminals from next pieces
                    terminal_states.extend(next_piece_terminals)
            
            # Limit explosion of states
            if len(terminal_states) > self.max_states_per_piece * (chain_depth + 1):
                print(f"     🛑 Limiting states at depth {chain_depth + 1} (reached {len(terminal_states)} states)")
                break
        
        print(f"     ✅ Generated {len(terminal_states)} terminal states from {placement_count} placements")
        return terminal_states
    
    def _get_all_valid_placements(self, state, piece_shape, piece_idx):
        """
        Get ALL valid placements for a piece in the current state
        FIXED: Proper rotation validation (max 4 rotations) and debugging
        
        Args:
            state: Current game state vector
            piece_shape: Piece shape definition
            piece_idx: Piece type index (0-6)
        Returns:
            List of valid placement dictionaries
        """
        valid_placements = []
        
        # Create temporary environment state for testing
        obs = self._state_vector_to_obs(state)
        
        # FIXED: Use proper rotation limit (max 4 rotations for any piece)
        num_rotations = min(4, len(piece_shape))  # Max 4 rotations
        piece_name = ['S','Z','I','O','J','L','T'][piece_idx]
        
        print(f"     🔍 Testing {piece_name} piece: {num_rotations} rotations × {self.board_width} positions = max {num_rotations * self.board_width} combinations")
        
        valid_count = 0
        total_tested = 0
        
        # Try all rotations and X positions
        for rotation in range(num_rotations):  # FIXED: Proper rotation limit
            for x_pos in range(self.board_width):  # All X positions (0-9)
                total_tested += 1
                
                # Check if this placement is valid
                if self._is_valid_terminal_placement(piece_shape, rotation, x_pos, obs):
                    # Find drop position
                    drop_y = self._find_drop_position(piece_shape, rotation, x_pos, obs)
                    
                    # Validate drop position is reasonable
                    if drop_y < 0 or drop_y >= 20:
                        continue  # Skip invalid drop positions
                    
                    # Generate resulting state after placement
                    resulting_state, placement_value = self._generate_state_after_placement(
                        state, piece_shape, rotation, x_pos, drop_y, piece_idx
                    )
                    
                    placement_info = {
                        'placement': (rotation, x_pos, drop_y),
                        'resulting_state': resulting_state,
                        'placement_value': placement_value,
                        'piece_positions': self._get_piece_positions(piece_shape, rotation, x_pos, drop_y)
                    }
                    
                    valid_placements.append(placement_info)
                    valid_count += 1
        
        print(f"     ✅ {piece_name}: {valid_count}/{total_tested} valid placements found")
        
        # Sanity check - typical pieces should have 10-40 valid placements
        if valid_count > 50:
            print(f"     ⚠️  WARNING: {valid_count} placements seems too high for {piece_name}")
        elif valid_count == 0:
            print(f"     ⚠️  WARNING: No valid placements found for {piece_name}")
        
        return valid_placements
    
    def _generate_state_after_placement(self, current_state, piece_shape, rotation, x_pos, y_pos, piece_idx):
        """
        Generate the state vector after placing a piece
        ENHANCED: Ensures ALL active block coordinates are properly represented
        
        Args:
            current_state: Current state vector
            piece_shape: Piece shape definition
            rotation: Rotation state
            x_pos: X position
            y_pos: Y position
            piece_idx: Piece type index
        Returns:
            tuple: (resulting_state_vector, placement_value)
        """
        resulting_state = current_state.copy()
        
        # Create piece to get ALL block positions - FIXED: no offset to x_pos
        piece = self.Piece(x_pos, y_pos, piece_shape)
        piece.rotation = rotation % len(piece_shape)
        
        # Get ALL coordinates of the piece (not just center)
        formatted_positions = self.convert_shape_format(piece)
        
        if not formatted_positions:
            print(f"     ⚠️  Warning: No formatted positions for piece at ({x_pos}, {y_pos}, rot={rotation})")
            return resulting_state, 0.0
        
        # COMPLETE BLOCK REPRESENTATION: Update current piece grid with ALL block coordinates
        current_piece_grid = resulting_state[:200].reshape(20, 10)
        
        # Clear any existing pieces first
        current_piece_grid.fill(0.0)
        
        # Mark ALL coordinates of the placed piece
        placed_blocks = 0
        for x, y in formatted_positions:
            if 0 <= x < 10 and 0 <= y < 20:
                current_piece_grid[y, x] = 1.0  # Mark this block position as occupied
                placed_blocks += 1
        
        resulting_state[:200] = current_piece_grid.flatten()
        
        # COMPLETE SPATIAL OCCUPANCY: Update empty grid to reflect ALL occupied spaces
        empty_grid = resulting_state[200:400].reshape(20, 10)
        
        # Mark ALL piece positions as not empty
        for x, y in formatted_positions:
            if 0 <= x < 10 and 0 <= y < 20:
                empty_grid[y, x] = 0.0  # Mark as occupied (not empty)
        
        resulting_state[200:400] = empty_grid.flatten()
        
        # Update metadata to reflect the placement (indices 407-409)
        resulting_state[407] = rotation / 4.0  # Normalized rotation
        resulting_state[408] = x_pos / 10.0    # Normalized x position  
        resulting_state[409] = y_pos / 20.0    # Normalized y position
        
        # CRITICAL FIX: Calculate placement value using ACTUAL board state for line clearing
        placement_value = self._calculate_placement_value_with_real_state(
            current_state, resulting_state, x_pos, y_pos, rotation, piece_idx, formatted_positions
        )
        
        # Validation: Ensure we placed the expected number of blocks
        if placed_blocks == 0:
            print(f"     ⚠️  Warning: No blocks placed for piece at ({x_pos}, {y_pos}, rot={rotation})")
        elif placed_blocks > 4:
            print(f"     ⚠️  Warning: Too many blocks placed ({placed_blocks}) for piece at ({x_pos}, {y_pos}, rot={rotation})")
        
        return resulting_state, placement_value
    
    def _calculate_placement_value_with_real_state(self, before_state, after_state, x_pos, y_pos, rotation, piece_type, formatted_positions):
        """
        FIXED: Calculate placement value using ACTUAL board states for line clearing
        This replaces the old mock state approach with real state transition analysis
        
        Args:
            before_state: State before placement
            after_state: State after placement
            x_pos, y_pos, rotation, piece_type: Placement parameters
            formatted_positions: Actual piece positions
        Returns:
            placement_value: Value including realistic line clearing opportunities
        """
        # CRITICAL: Use REAL state transition for line clearing calculation
        line_clear_value = self._calculate_deterministic_line_clearing_from_states(
            before_state, after_state, formatted_positions
        )
        
        # Base scoring factors (kept for diversity)
        center_bonus = 5.0 - abs(x_pos - 4.5)  # Prefer center positions
        height_bonus = max(0, 20 - y_pos)      # Prefer lower positions
        rotation_penalty = rotation * 0.5      # Slight penalty for rotation
        
        # Piece-specific bonuses
        piece_bonuses = [2.0, 1.5, 3.0, 1.0, 2.5, 2.5, 2.0]  # S, Z, I, O, J, L, T
        piece_bonus = piece_bonuses[piece_type] if piece_type < len(piece_bonuses) else 2.0
        
        # Compactness bonus (how tightly packed the piece positions are)
        if len(formatted_positions) > 1:
            min_x = min(pos[0] for pos in formatted_positions)
            max_x = max(pos[0] for pos in formatted_positions)
            min_y = min(pos[1] for pos in formatted_positions)
            max_y = max(pos[1] for pos in formatted_positions)
            compactness = 5.0 - (max_x - min_x) - (max_y - min_y)
        else:
            compactness = 5.0
        
        # ENHANCED: Combine factors with LINE CLEARING as primary objective
        base_value = center_bonus + height_bonus + piece_bonus + compactness - rotation_penalty
        
        # Add some deterministic variation
        variation_factor = ((x_pos + y_pos + rotation + piece_type) % 7) * 0.3
        
        # CRITICAL: Combine with REAL line clearing value (3x weight like RND)
        final_value = (
            line_clear_value * 3.0 +  # 🔥 LINE CLEARING VALUE (3x weight - most important!)
            base_value +              # Structural factors for diversity
            variation_factor          # Deterministic variation
        )
        
        # Clamp to expanded range for line clears
        return max(-100.0, min(200.0, final_value))
    
    def _calculate_deterministic_line_clearing_from_states(self, before_state, after_state, piece_positions):
        """
        CRITICAL FIX: Calculate line clearing value using actual before/after state transition
        This analyzes REAL board states to detect line clearing opportunities
        
        Args:
            before_state: State vector before placement
            after_state: State vector after placement  
            piece_positions: List of (x,y) positions where piece was placed
        Returns:
            line_clear_value: Realistic line clearing value based on actual board evolution
        """
        try:
            # Extract REAL board states
            before_empty_grid = before_state[200:400].reshape(20, 10)
            after_empty_grid = after_state[200:400].reshape(20, 10)
            
            # Convert to occupied cell representation (1 = occupied, 0 = empty)
            before_board = 1 - before_empty_grid  # Occupied cells before placement
            after_board = 1 - after_empty_grid    # Occupied cells after placement
            
            # Calculate lines that would be cleared by this placement
            lines_cleared = 0
            cleared_lines = []
            potential_lines = []
            
            # Check each row to see if placement creates line clears
            for row in range(20):
                before_filled = np.sum(before_board[row, :])
                after_filled = np.sum(after_board[row, :])
                
                # Line clearing detection
                if after_filled >= 10.0:  # Full line after placement
                    lines_cleared += 1
                    cleared_lines.append(row)
                elif after_filled >= 8.0:  # Near-complete line
                    potential_lines.append((row, after_filled))
            
            # Calculate line clearing rewards (same as real exploration)
            line_clear_reward = 0
            if lines_cleared > 0:
                # Use actual Tetris scoring
                line_clear_base = {1: 100, 2: 200, 3: 400, 4: 1600}
                line_clear_reward = line_clear_base.get(lines_cleared, lines_cleared * 400)
                
                # Add bonus for clearing multiple lines simultaneously
                if lines_cleared >= 2:
                    line_clear_reward *= 1.5
                
                print(f"     🔥 DETERMINISTIC LINE CLEAR! {lines_cleared} lines, reward: {line_clear_reward}")
            
            # Calculate line clearing POTENTIAL (near-complete lines)
            line_potential = 0
            for row, filled_count in potential_lines:
                potential_value = (filled_count - 7) * 10  # 10, 20, 30 for 8, 9, 10 filled
                line_potential += potential_value
            
            # Strategic improvement from placement
            strategic_improvement = self._calculate_deterministic_strategic_improvement(
                before_board, after_board, piece_positions
            )
            
            # Total line clearing value
            total_line_value = line_clear_reward + line_potential + strategic_improvement
            
            return total_line_value
            
        except Exception as e:
            print(f"Error in deterministic line clearing calculation: {e}")
            return 0.0  # Fallback
    
    def _calculate_deterministic_strategic_improvement(self, before_board, after_board, piece_positions):
        """
        Calculate strategic board improvements for deterministic exploration
        """
        try:
            # Calculate height changes
            before_height = self._calculate_board_height_det(before_board)
            after_height = self._calculate_board_height_det(after_board)
            height_improvement = max(0, before_height - after_height) * 5
            
            # Calculate hole changes
            before_holes = self._count_board_holes_det(before_board)
            after_holes = self._count_board_holes_det(after_board)
            hole_improvement = max(0, before_holes - after_holes) * 15
            
            # Well creation (good for I-pieces)
            well_value = self._calculate_well_creation_value(after_board, piece_positions)
            
            return height_improvement + hole_improvement + well_value
            
        except Exception as e:
            return 0.0
    
    def _calculate_board_height_det(self, board):
        """Calculate maximum height of board for deterministic exploration"""
        for row in range(20):
            if np.sum(board[row, :]) > 0:
                return 20 - row
        return 0
    
    def _count_board_holes_det(self, board):
        """Count holes in board for deterministic exploration"""
        holes = 0
        for col in range(10):
            filled_found = False
            for row in range(20):
                if board[row, col] > 0:
                    filled_found = True
                elif filled_found and board[row, col] == 0:
                    holes += 1
        return holes
    
    def _calculate_well_creation_value(self, board, piece_positions):
        """Calculate value of creating wells (good for future I-piece tetris)"""
        well_value = 0
        
        # Check if piece created any useful wells
        for x, y in piece_positions:
            if 0 <= x < 10 and 0 <= y < 20:
                # Check adjacent columns for height differences
                left_height = 0
                right_height = 0
                
                if x > 0:  # Left column
                    for row in range(20):
                        if board[row, x-1] > 0:
                            left_height = 20 - row
                            break
                
                if x < 9:  # Right column
                    for row in range(20):
                        if board[row, x+1] > 0:
                            right_height = 20 - row
                            break
                
                # Current column height
                current_height = 0
                for row in range(20):
                    if board[row, x] > 0:
                        current_height = 20 - row
                        break
                
                # Well bonus (height differences of 3+ are good)
                left_diff = max(0, left_height - current_height)
                right_diff = max(0, right_height - current_height)
                
                if left_diff >= 3 or right_diff >= 3:
                    well_value += min(left_diff, right_diff) * 3
        
        return well_value
    
    def _get_piece_positions(self, piece_shape, rotation, x_pos, y_pos):
        """Get list of (x,y) positions occupied by piece"""
        piece = self.Piece(x_pos + 2, y_pos, piece_shape)
        piece.rotation = rotation % len(piece_shape)
        return self.convert_shape_format(piece)
    
    def _state_vector_to_obs(self, state_vector):
        """
        Convert state vector back to observation format for validation
        
        Args:
            state_vector: 410-dimensional state vector
        Returns:
            obs: Observation dictionary
        """
        obs = {
            'current_piece_grid': state_vector[:200].reshape(20, 10),
            'empty_grid': state_vector[200:400].reshape(20, 10),
            'next_piece': state_vector[400:407],
            'current_rotation': state_vector[407] * 4.0,  # Denormalize
            'current_x': state_vector[408] * 10.0,       # Denormalize
            'current_y': state_vector[409] * 20.0        # Denormalize
        }
        return obs
    
    def _is_valid_terminal_placement(self, piece_shape, rotation, x_pos, obs):
        """
        Check if a piece placement results in a valid terminal state
        Uses actual Tetris validation mechanics
        FIXED: Simplified validation logic to be less restrictive
        
        Args:
            piece_shape: The shape definition from constants
            rotation: Rotation state (0-3)
            x_pos: X position on board (0-9)
            obs: Current observation
        Returns:
            bool: True if placement is valid and creates terminal state
        """
        try:
            # FIXED: Ensure rotation is within bounds
            max_rotations = len(piece_shape)
            if rotation >= max_rotations:
                return False
            
            # SIMPLIFIED: Just check if piece can fit at this x position with any rotation
            # Use spawn position Y=4 as starting point
            spawn_y = 4
            
            # Create piece at spawn position - FIXED: don't add offset to x_pos
            piece = self.Piece(x_pos, spawn_y, piece_shape)
            piece.rotation = rotation  # Use rotation directly, already validated
            
            # Create grid from current state
            grid = self._create_grid_from_obs(obs)
            
            # Check if piece fits at spawn position
            if not self.valid_space(piece, grid):
                # Try moving down a bit in case spawn position is blocked
                for try_y in range(spawn_y, spawn_y + 5):
                    piece.y = try_y
                    if self.valid_space(piece, grid):
                        break
                else:
                    return False  # Piece doesn't fit anywhere near spawn
            
            # Find actual drop position by simulating hard drop
            max_drops = 20  # Prevent infinite loops
            drops = 0
            
            while self.valid_space(piece, grid) and drops < max_drops:
                piece.y += 1
                drops += 1
            
            # Move back to last valid position
            piece.y -= 1
            
            # Validate that piece is properly landed (can't move further down)
            test_piece = self.Piece(piece.x, piece.y + 1, piece_shape)
            test_piece.rotation = rotation
            if self.valid_space(test_piece, grid):
                return False  # Piece would continue falling, not terminal
            
            # Ensure final position is within bounds
            formatted_positions = self.convert_shape_format(piece)
            if not formatted_positions:  # Empty positions
                return False
                
            for x, y in formatted_positions:
                if x < 0 or x >= 10 or y < 0 or y >= 20:
                    return False
            
            # This is a valid terminal placement
            return True
            
        except Exception as e:
            # Enhanced error reporting for debugging
            if rotation == 0 and x_pos == 0:  # Only print for first position to avoid spam
                print(f"     🐛 Validation error for first position: {e}")
            return False
    
    def _find_drop_position(self, piece_shape, rotation, x_pos, obs):
        """
        Find the Y position where a piece would land if hard dropped
        FIXED: Simplified and more robust drop position finding
        
        Args:
            piece_shape: The shape definition
            rotation: Rotation state
            x_pos: X position
            obs: Current observation
        Returns:
            int: Y position where piece lands
        """
        try:
            # FIXED: Ensure rotation is within bounds
            max_rotations = len(piece_shape)
            if rotation >= max_rotations:
                return 18  # Safe fallback
            
            # Start from spawn position - FIXED: don't add offset to x_pos
            spawn_y = 4
            piece = self.Piece(x_pos, spawn_y, piece_shape)
            piece.rotation = rotation  # Use rotation directly
            
            # Create grid
            grid = self._create_grid_from_obs(obs)
            
            # Find first valid position if spawn is blocked
            if not self.valid_space(piece, grid):
                for try_y in range(spawn_y, spawn_y + 5):
                    piece.y = try_y
                    if self.valid_space(piece, grid):
                        break
                else:
                    return 18  # Safe fallback if no valid position found
            
            # Drop piece until it can't move down
            max_drops = 20  # Prevent infinite loops
            drops = 0
            
            while self.valid_space(piece, grid) and drops < max_drops:
                piece.y += 1
                drops += 1
            
            # Return last valid position, clamped to valid range
            final_y = max(0, min(19, piece.y - 1))
            return final_y
            
        except Exception as e:
            print(f"     🐛 Drop position error: {e}")
            return 18  # Safe fallback near bottom
    
    def _create_grid_from_obs(self, obs):
        """
        Create a grid representation from observation for validation
        
        Args:
            obs: Environment observation
        Returns:
            Grid compatible with valid_space function
        """
        grid = [[(0, 0, 0) for _ in range(10)] for _ in range(20)]
        
        # Mark occupied spaces from empty_grid (inverse logic)
        empty_grid = obs['empty_grid']
        for y in range(20):
            for x in range(10):
                if empty_grid[y, x] == 0:  # Not empty
                    grid[y][x] = (128, 128, 128)  # Mark as occupied
        
        return grid
    
    def _hash_terminal_state(self, terminal_state):
        """Generate hash for terminal state to check uniqueness"""
        # Use key components for hashing to detect true duplicates
        key_components = terminal_state[:200]  # Current piece grid
        key_components = np.append(key_components, terminal_state[200:400])  # Empty grid
        key_components = np.append(key_components, terminal_state[407:410])  # Metadata
        
        # Round to avoid floating point precision issues
        rounded = np.round(key_components * 100).astype(int)
        return hash(tuple(rounded))
    
    def _obs_to_state_vector(self, obs):
        """Convert observation to 410-dimensional state vector"""
        # Current piece grid (200) + empty grid (200) + next piece (7) + metadata (3) = 410
        current_piece_flat = obs['current_piece_grid'].flatten()
        empty_grid_flat = obs['empty_grid'].flatten()
        next_piece = obs['next_piece']
        metadata = np.array([
            obs['current_rotation'],
            obs['current_x'], 
            obs['current_y']
        ])
        
        state_vector = np.concatenate([
            current_piece_flat, 
            empty_grid_flat,
            next_piece,
            metadata
        ])
        
        return state_vector

class TrueRandomExplorer:
    """
    True random exploration for the beginning of training
    Provides completely unbiased exploration of the action space
    """
    def __init__(self, env, state_dim=None):
        self.env = env
        
        # Get centralized config
        self.config = TetrisConfig()
        self.state_dim = state_dim or self.config.STATE_DIM  # 410
        
        # Random exploration parameters
        self.max_episode_steps = 50
        self.placement_attempts_per_step = 3
        
    def collect_random_placement_data(self, num_episodes=100):
        """
        Collect placement data using completely random actions
        Args:
            num_episodes: Number of random episodes to run
        Returns:
            List of placement data from random exploration
        """
        print(f"🎲 Starting true random exploration: {num_episodes} episodes...")
        placement_data = []
        
        for episode in range(num_episodes):
            try:
                obs = self.env.reset()
                
                for step in range(self.max_episode_steps):
                    # Get current state
                    current_state = self._obs_to_state_vector(obs)
                    
                    # Generate random terminal placements
                    for attempt in range(self.placement_attempts_per_step):
                        # Completely random placement parameters
                        random_rotation = np.random.randint(0, 4)
                        random_x_pos = np.random.randint(0, 10)
                        random_y_pos = np.random.randint(0, 20)
                        random_piece_type = np.random.randint(1, 8)
                        
                        # Generate random terminal state
                        terminal_state, terminal_value = self._generate_random_terminal(
                            current_state, random_rotation, random_x_pos, random_y_pos, random_piece_type
                        )
                        
                        # Create terminal data
                        terminal_entry = {
                            'state': current_state,
                            'placement': (random_rotation, random_x_pos, random_y_pos),
                            'terminal_reward': terminal_value,
                            'resulting_state': terminal_state,
                            'intrinsic_reward': 0.0,  # No RND for random
                            'terminal_value': terminal_value,
                            'novelty_score': 0.5,  # Neutral novelty for random
                            'is_final_state': False,
                            'target_piece_type': random_piece_type,
                            'exploration_method': 'random'
                        }
                        
                        placement_data.append(terminal_entry)
                    
                    # Take random action to continue episode
                    random_action = np.zeros(8, dtype=np.int8)
                    random_action[np.random.randint(0, 8)] = 1
                    obs, reward, done, info = self.env.step(random_action)
                    
                    if done:
                        break
            
            except Exception as e:
                print(f"Error in random episode {episode+1}: {e}")
                continue
        
        print(f"✅ Random exploration completed: {len(placement_data)} random terminal states")
        return placement_data
    
    def _generate_random_terminal(self, current_state, rotation, x_pos, y_pos, piece_type):
        """
        Generate random terminal state with LINE CLEARING VALUE included
        ENHANCED: Now includes realistic line clearing potential instead of pure random
        """
        terminal_state = current_state.copy()
        
        # Apply random modifications to entire state (keep some randomness)
        random_noise = np.random.uniform(-0.3, 0.3, len(terminal_state))
        terminal_state = terminal_state + random_noise
        
        # Ensure metadata reflects the random placement
        terminal_state[407] = rotation / 4.0
        terminal_state[408] = x_pos / 10.0
        terminal_state[409] = y_pos / 20.0
        
        # ENHANCED: Calculate line clearing value even for random exploration
        line_clear_value = self._calculate_line_clearing_value_random(
            current_state, rotation, x_pos, piece_type, y_pos
        )
        
        # Random base value (keep some randomness for exploration diversity)
        random_base = np.random.uniform(-20, 20)
        
        # CRITICAL: Combine line clearing value with random component
        terminal_value = (
            line_clear_value * 2.0 +  # 🔥 LINE CLEARING VALUE (2x weight for random - still important!)
            random_base               # Random component for exploration diversity
        )
        
        # Clamp to expanded range
        terminal_value = max(-100.0, min(200.0, terminal_value))
        
        return terminal_state, terminal_value
    
    def _calculate_line_clearing_value_random(self, current_state, rotation, x_pos, piece_type, y_pos):
        """
        Calculate line clearing value for random exploration
        Simplified version focusing on line clearing potential
        """
        try:
            # Extract board state
            empty_grid = current_state[200:400].reshape(20, 10)
            board = 1 - empty_grid  # Invert empty grid to get occupied cells
            
            # Add some randomness to board state for exploration
            noise = np.random.uniform(-0.1, 0.1, board.shape)
            noisy_board = np.clip(board + noise, 0, 1)
            
            # Simulate placing the piece
            simulated_board = noisy_board.copy()
            
            # Get piece positions (simplified)
            piece_positions = self._get_approximate_piece_positions_random(rotation, x_pos, y_pos, piece_type)
            
            for row, col in piece_positions:
                if 0 <= row < 20 and 0 <= col < 10:
                    simulated_board[row, col] = 1
            
            # Calculate lines that would be cleared
            lines_cleared = 0
            for row in range(20):
                filled_ratio = np.sum(simulated_board[row, :]) / 10.0
                if filled_ratio >= 0.9:  # Relaxed threshold for random exploration
                    lines_cleared += 1
            
            # Calculate line clearing rewards
            line_clear_reward = 0
            if lines_cleared > 0:
                # Use actual Tetris scoring with random bonus
                line_clear_base = {1: 100, 2: 200, 3: 400, 4: 1600}
                line_clear_reward = line_clear_base.get(lines_cleared, lines_cleared * 400)
                
                # Add random bonus for exploration
                random_bonus = np.random.uniform(0.8, 1.2)
                line_clear_reward *= random_bonus
            
            # Calculate line clearing POTENTIAL with random exploration bias
            line_potential = 0
            for row in range(20):
                filled_ratio = np.sum(simulated_board[row, :]) / 10.0
                if filled_ratio >= 0.7:  # Lower threshold for random exploration
                    potential_value = filled_ratio * 30  # Scale by fill ratio
                    line_potential += potential_value
            
            return line_clear_reward + line_potential
            
        except Exception as e:
            return np.random.uniform(0, 50)  # Random fallback
    
    def _get_approximate_piece_positions_random(self, rotation, x_pos, y_pos, piece_type):
        """
        Get approximate piece positions for random exploration
        With added randomness for exploration diversity
        """
        positions = []
        
        # Add small random offset for exploration diversity
        offset_x = np.random.randint(-1, 2)  # -1, 0, or 1
        offset_y = np.random.randint(-1, 2)  # -1, 0, or 1
        
        # Basic piece footprints with random variation
        if piece_type == 1:  # I-piece
            if rotation % 2 == 0:  # Horizontal
                positions = [(y_pos + offset_y, x_pos + i + offset_x) for i in range(4)]
            else:  # Vertical
                positions = [(y_pos + i + offset_y, x_pos + offset_x) for i in range(4)]
        elif piece_type == 2:  # O-piece
            positions = [
                (y_pos + offset_y, x_pos + offset_x), 
                (y_pos + offset_y, x_pos + 1 + offset_x),
                (y_pos + 1 + offset_y, x_pos + offset_x), 
                (y_pos + 1 + offset_y, x_pos + 1 + offset_x)
            ]
        else:  # Other pieces - random footprint
            positions = [
                (y_pos + offset_y, x_pos + offset_x),
                (y_pos + offset_y, x_pos + 1 + offset_x),
                (y_pos + 1 + offset_y, x_pos + offset_x),
                (y_pos + 1 + offset_y, x_pos + 1 + offset_x)
            ]
        
        # Filter valid positions
        valid_positions = [(r, c) for r, c in positions if 0 <= r < 20 and 0 <= c < 10]
        return valid_positions
    
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