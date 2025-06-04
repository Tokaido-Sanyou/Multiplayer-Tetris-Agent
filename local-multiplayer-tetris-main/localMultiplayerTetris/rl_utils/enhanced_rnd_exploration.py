"""
Enhanced RND Exploration with Piece Presence Reward Tracking
UPDATED: Remove empty_grid from observations (410D → 210D)
Extends the base RND exploration to include piece presence rewards across all methods
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from ..config import TetrisConfig
    from .rnd_exploration import RNDExplorationActor, DeterministicTerminalExplorer, TrueRandomExplorer
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TetrisConfig
    from rnd_exploration import RNDExplorationActor, DeterministicTerminalExplorer, TrueRandomExplorer


class PiecePresenceTracker:
    """
    Tracks piece presence rewards across training episodes
    Implements decay schedule as specified in config
    """
    
    def __init__(self):
        self.config = TetrisConfig()
        self.episode_count = 0
        self.base_reward = self.config.RewardConfig.PIECE_PRESENCE_REWARD
        self.decay_steps = self.config.RewardConfig.PIECE_PRESENCE_DECAY_STEPS
        self.min_reward = self.config.RewardConfig.PIECE_PRESENCE_MIN
        
    def get_current_piece_presence_reward(self):
        """
        Get current piece presence reward based on training progress
        """
        if self.episode_count >= self.decay_steps:
            return self.min_reward
        
        # Linear decay from base_reward to min_reward over decay_steps
        decay_factor = 1.0 - (self.episode_count / self.decay_steps)
        current_reward = self.min_reward + (self.base_reward - self.min_reward) * decay_factor
        
        return max(current_reward, self.min_reward)
    
    def calculate_piece_presence_for_state(self, state_data):
        """
        Calculate piece presence reward for a given state
        Based on number of pieces on the board
        UPDATED: Handle 210D state vectors (no empty_grid)
        """
        # Handle both state vectors and exploration data format
        if isinstance(state_data, dict):
            # Extract state from exploration data
            state_vector = state_data.get('state', state_data.get('resulting_state', []))
        else:
            # Already a state vector
            state_vector = state_data
        
        # UPDATED: Handle both 410D (old) and 210D (new) formats
        if len(state_vector) == 410:
            # Old format: current_piece_grid(200) + empty_grid(200) + next_piece(7) + metadata(3)
            # Extract occupied grid from state vector (positions 0-199)
            occupied_grid = np.array(state_vector[0:200]).reshape(20, 10)
        elif len(state_vector) >= 200:
            # New format: current_piece_grid(200) + next_piece(7) + metadata(3) = 210D
            # Extract occupied grid from state vector (positions 0-199)
            occupied_grid = np.array(state_vector[0:200]).reshape(20, 10)
        else:
            return 0.0
        
        # Count number of occupied cells (pieces on board)
        piece_count = np.sum(occupied_grid)
        
        # Current piece presence reward per piece
        current_reward = self.get_current_piece_presence_reward()
        
        # Total piece presence reward
        total_piece_presence = piece_count * current_reward
        
        return total_piece_presence
    
    def increment_episode(self):
        """
        Increment episode counter for decay tracking
        """
        self.episode_count += 1
    
    def get_stats(self):
        """
        Get current piece presence tracking stats
        """
        return {
            'episode_count': self.episode_count,
            'current_reward_per_piece': self.get_current_piece_presence_reward(),
            'decay_progress': min(1.0, self.episode_count / self.decay_steps),
            'is_fully_decayed': self.episode_count >= self.decay_steps
        }


class EnhancedRNDExplorationActor(RNDExplorationActor):
    """
    Enhanced RND exploration with piece presence reward tracking
    UPDATED: Use 210D state vectors (no empty_grid)
    """
    
    def __init__(self, env, state_dim=None):
        # UPDATED: Default to 210D state
        corrected_state_dim = state_dim or 210
        super().__init__(env, corrected_state_dim)
        self.piece_presence_tracker = PiecePresenceTracker()
        
    def collect_placement_data(self, num_episodes=100):
        """
        Enhanced collect_placement_data with piece presence tracking
        UPDATED: Handle 210D state vectors + integrated lines cleared tracking
        """
        placement_data = []
        piece_presence_rewards = []
        total_lines_cleared = 0  # NEW: Track lines cleared
        
        print(f"Starting RND exploration: {num_episodes} episodes (5 per piece type)...")
        print(f"   • Lines cleared tracking: ENABLED")
        print(f"   • State vector size: 210D (no empty_grid)")
        
        # Track RND training progress
        total_rnd_loss = 0
        rnd_updates = 0
        
        # Process episodes per piece type for balanced exploration
        episodes_per_piece = num_episodes // 7
        for piece_type in range(7):
            for episode in range(episodes_per_piece):
                self.env.reset()
                
                # Set specific piece type for this episode
                self.env.next_piece = piece_type
                
                # Get current state (UPDATED: 210D state vector)
                obs = self.env._get_observation()
                current_state = self._obs_to_state_vector_210(obs)
                
                # Calculate piece presence reward for current state
                piece_presence_reward = self.piece_presence_tracker.calculate_piece_presence_for_state(current_state)
                
                # Calculate intrinsic reward using RND
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
                intrinsic_reward, prediction_error = self.rnd_exploration(state_tensor)
                intrinsic_reward_value = intrinsic_reward.item()
                
                # NEW: Track lines cleared before placement
                lines_before = getattr(self.env, 'lines_cleared', 0)
                
                # Generate terminal placement with enhanced rewards
                terminal_data = self._generate_terminal_placement_with_novelty(
                    current_state, intrinsic_reward_value, piece_type
                )
                
                if terminal_data:
                    # NEW: Track lines cleared after placement
                    lines_after = getattr(self.env, 'lines_cleared', 0)
                    lines_cleared_this_episode = lines_after - lines_before
                    total_lines_cleared += lines_cleared_this_episode
                    
                    # Add piece presence reward and lines cleared to terminal data
                    terminal_data['piece_presence_reward'] = piece_presence_reward
                    terminal_data['lines_cleared'] = lines_cleared_this_episode  # NEW: lines cleared tracking
                    placement_data.append(terminal_data)
                    piece_presence_rewards.append(piece_presence_reward)
                
                # Train RND predictor on current state
                if self.rnd_optimizer is not None:
                    rnd_loss = self.rnd_exploration.train_predictor(state_tensor, self.rnd_optimizer)
                    total_rnd_loss += rnd_loss
                    rnd_updates += 1
                
                # Increment episode counter for piece presence decay
                self.piece_presence_tracker.increment_episode()
        
        # Enhanced statistics with piece presence AND lines cleared
        if placement_data:
            rewards = [d['terminal_reward'] for d in placement_data]
            avg_piece_presence = np.mean(piece_presence_rewards) if piece_presence_rewards else 0
            avg_lines_per_episode = total_lines_cleared / len(placement_data) if placement_data else 0
            
            piece_presence_stats = self.piece_presence_tracker.get_stats()
            
            print(f"✅ RND Exploration completed:")
            print(f"   • Total terminal placements: {len(placement_data)}")
            print(f"   • Episodes per piece type: {episodes_per_piece}")
            print(f"   • Unique terminal states discovered: {len(self.terminal_states)}")
            print(f"   • New distinct states this batch: {len(self.terminal_states) - self.prev_unique_terminals}")
            print(f"   • Average terminal value: {np.mean(rewards):.2f}")
            print(f"   • Average piece presence reward: {avg_piece_presence:.2f}")
            print(f"   • Total lines cleared: {total_lines_cleared}")  # NEW: lines cleared output
            print(f"   • Average lines per episode: {avg_lines_per_episode:.3f}")  # NEW
            print(f"   • Piece presence decay: {piece_presence_stats['decay_progress']*100:.1f}%")
            print(f"   • Current reward per piece: {piece_presence_stats['current_reward_per_piece']:.3f}")
            print(f"   • State vector size: 210D (no empty_grid)")
            
            if rnd_updates > 0:
                avg_rnd_loss = total_rnd_loss / rnd_updates
                print(f"   • RND learning progress - Mean: {self.rnd_exploration.reward_mean:.4f}, Std: {self.rnd_exploration.reward_std:.4f}")
            
            self.prev_unique_terminals = len(self.terminal_states)
        
        return placement_data
    
    def _obs_to_state_vector_210(self, obs):
        """
        Convert observation to 210D state vector (CORRECTED - no empty_grid)
        """
        if isinstance(obs, dict):
            current_piece_grid = obs['current_piece_grid'].flatten()  # 200
            next_piece = obs['next_piece']  # 7
            
            # Metadata (3 values)
            current_rotation = np.array([obs['current_rotation']], dtype=np.float32)
            current_x = np.array([obs['current_x']], dtype=np.float32)
            current_y = np.array([obs['current_y']], dtype=np.float32)
            
            # Concatenate (NO empty_grid)
            state_vector = np.concatenate([
                current_piece_grid,  # 200
                next_piece,          # 7
                current_rotation,    # 1
                current_x,           # 1
                current_y            # 1
            ], dtype=np.float32)
            
            return state_vector  # Total: 210 dimensions
        else:
            return obs


class EnhancedDeterministicTerminalExplorer(DeterministicTerminalExplorer):
    """
    Enhanced deterministic exploration with piece presence reward tracking
    UPDATED: Use 210D state vectors (no empty_grid)
    """
    
    def __init__(self, env, state_dim=None):
        # UPDATED: Default to 210D state
        corrected_state_dim = state_dim or 210
        super().__init__(env, corrected_state_dim)
        self.piece_presence_tracker = PiecePresenceTracker()
        
    def generate_all_terminal_states(self, sequence_length=3):
        """
        Enhanced terminal state generation with piece presence tracking
        UPDATED: Handle 210D state vectors
        """
        print(f"🔗 Starting Deterministic Sequential Chain Exploration")
        print(f"   • Target sequence length: {sequence_length}")
        print(f"   • Enhanced with piece presence rewards")
        print(f"   • State vector size: 210D (no empty_grid)")
        
        all_placement_data = []
        
        # Generate diverse starting scenarios with piece presence
        line_clearing_scenarios = self._generate_line_clearing_scenarios()
        
        total_terminals_generated = 0
        
        for scenario_id, (scenario_name, starting_state) in enumerate(line_clearing_scenarios):
            print(f"   🎯 Scenario {scenario_id+1}/{len(line_clearing_scenarios)}: {scenario_name}")
            
            # UPDATED: Convert to 210D if needed
            corrected_starting_state = self._convert_state_to_210(starting_state)
            
            # Calculate piece presence for starting state
            piece_presence_reward = self.piece_presence_tracker.calculate_piece_presence_for_state(corrected_starting_state)
            
            # Generate piece sequence for this scenario
            piece_sequence = [np.random.randint(0, 7) for _ in range(sequence_length)]
            
            # Generate sequential chain from this starting state
            chain_data = self._generate_sequential_chain(
                corrected_starting_state, 
                piece_sequence, 
                chain_depth=0, 
                placement_history=[], 
                scenario_id=scenario_id
            )
            
            # Add piece presence rewards to chain data
            for data in chain_data:
                data['piece_presence_reward'] = piece_presence_reward
                # Increment for decay (deterministic uses less episodes)
                self.piece_presence_tracker.increment_episode()
            
            all_placement_data.extend(chain_data)
            total_terminals_generated += len(chain_data)
        
        # Enhanced statistics
        if all_placement_data:
            rewards = [d['terminal_reward'] for d in all_placement_data]
            piece_presence_rewards = [d.get('piece_presence_reward', 0) for d in all_placement_data]
            
            piece_presence_stats = self.piece_presence_tracker.get_stats()
            
            print(f"✅ Deterministic Exploration completed:")
            print(f"   • Total terminal states: {len(all_placement_data)}")
            print(f"   • Average terminal value: {np.mean(rewards):.2f}")
            print(f"   • Average piece presence reward: {np.mean(piece_presence_rewards):.2f}")
            print(f"   • Piece presence decay: {piece_presence_stats['decay_progress']*100:.1f}%")
            print(f"   • Value range: {np.min(rewards):.1f} to {np.max(rewards):.1f}")
            print(f"   • State vector size: 210D (no empty_grid)")
        
        return all_placement_data
    
    def _convert_state_to_210(self, state_data):
        """
        Convert state to 210D format by removing empty_grid if present
        """
        if len(state_data) == 410:
            # Old format: current_piece_grid(200) + empty_grid(200) + next_piece(7) + metadata(3)
            # New format: current_piece_grid(200) + next_piece(7) + metadata(3) = 210D
            current_piece_grid = state_data[:200]
            next_piece = state_data[400:407]  # Skip empty_grid (200:400)
            metadata = state_data[407:410]
            return np.concatenate([current_piece_grid, next_piece, metadata])
        elif len(state_data) == 210:
            # Already converted
            return state_data
        else:
            # Unknown format, return as-is
            return state_data


class EnhancedTrueRandomExplorer(TrueRandomExplorer):
    """
    Enhanced random exploration with piece presence reward tracking
    UPDATED: Use 210D state vectors (no empty_grid)
    """
    
    def __init__(self, env, state_dim=None):
        # UPDATED: Default to 210D state
        corrected_state_dim = state_dim or 210
        super().__init__(env, corrected_state_dim)
        self.piece_presence_tracker = PiecePresenceTracker()
        
    def collect_random_placement_data(self, num_episodes=100):
        """
        Enhanced random exploration with piece presence tracking
        UPDATED: Handle 210D state vectors
        """
        placement_data = []
        piece_presence_rewards = []
        
        print(f"🎲 Starting True Random Exploration: {num_episodes} episodes")
        print(f"   • State vector size: 210D (no empty_grid)")
        
        for episode in range(num_episodes):
            self.env.reset()
            
            # Random piece type
            piece_type = np.random.randint(0, 7)
            self.env.next_piece = piece_type
            
            # Get current state (UPDATED: 210D state vector)
            obs = self.env._get_observation()
            current_state = self._obs_to_state_vector_210(obs)
            
            # Calculate piece presence reward
            piece_presence_reward = self.piece_presence_tracker.calculate_piece_presence_for_state(current_state)
            
            # Random placement parameters
            rotation = np.random.randint(0, 4)
            x_pos = np.random.randint(0, 10)
            y_pos = np.random.randint(5, 20)  # Random y position
            
            # Generate terminal state
            terminal_data = self._generate_random_terminal(
                current_state, rotation, x_pos, y_pos, piece_type
            )
            
            if terminal_data:
                # Add piece presence reward
                terminal_data['piece_presence_reward'] = piece_presence_reward
                placement_data.append(terminal_data)
                piece_presence_rewards.append(piece_presence_reward)
            
            # Increment episode counter
            self.piece_presence_tracker.increment_episode()
        
        # Enhanced statistics
        if placement_data:
            rewards = [d['terminal_reward'] for d in placement_data]
            avg_piece_presence = np.mean(piece_presence_rewards) if piece_presence_rewards else 0
            
            piece_presence_stats = self.piece_presence_tracker.get_stats()
            
            print(f"✅ Random Exploration completed:")
            print(f"   • Total placements: {len(placement_data)}")
            print(f"   • Average terminal value: {np.mean(rewards):.2f}")
            print(f"   • Average piece presence reward: {avg_piece_presence:.2f}")
            print(f"   • Piece presence decay: {piece_presence_stats['decay_progress']*100:.1f}%")
            print(f"   • Value range: {np.min(rewards):.1f} to {np.max(rewards):.1f}")
            print(f"   • State vector size: 210D (no empty_grid)")
        
        return placement_data
    
    def _obs_to_state_vector_210(self, obs):
        """
        Convert observation to 210D state vector (CORRECTED - no empty_grid)
        """
        if isinstance(obs, dict):
            current_piece_grid = obs['current_piece_grid'].flatten()  # 200
            next_piece = obs['next_piece']  # 7
            
            # Metadata (3 values)
            current_rotation = np.array([obs['current_rotation']], dtype=np.float32)
            current_x = np.array([obs['current_x']], dtype=np.float32)
            current_y = np.array([obs['current_y']], dtype=np.float32)
            
            # Concatenate (NO empty_grid)
            state_vector = np.concatenate([
                current_piece_grid,  # 200
                next_piece,          # 7
                current_rotation,    # 1
                current_x,           # 1
                current_y            # 1
            ], dtype=np.float32)
            
            return state_vector  # Total: 210 dimensions
        else:
            return obs 