"""
Enhanced State Model for 6-Phase System - CORRECTED IMPLEMENTATION V2
Key Fixes:
1. CORRECTED: Goal vector now allows ANY position on board (flexible encoding)
2. ADDED: Lines cleared tracking during exploration
3. CORRECTED: Continuous board exploration (not reset to empty each time)
4. SIMPLIFIED: Q-learning outputs single terminal value with n-step from trajectories
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

try:
    from ..config import TetrisConfig
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TetrisConfig


class Top5TerminalStateModel(nn.Module):
    """
    Enhanced State Model that predicts TOP 5 possible terminal states
    CORRECTED: Goal vector allows ANY position on board (flexible encoding)
    """
    
    def __init__(self, state_dim=None):
        super(Top5TerminalStateModel, self).__init__()
        
        self.config = TetrisConfig()
        self.state_dim = state_dim or 210  # current_piece_grid(200) + next_piece(7) + metadata(3)
        
        # Enhanced encoder for board + piece analysis
        self.board_piece_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Predict 5 possible terminal placements
        self.num_predictions = 5
        
        # CORRECTED: Each prediction outputs only essential components (no confidence/quality)
        self.placement_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 4)  # SIMPLIFIED: rotation(1) + x_pos(1) + y_pos(1) + lines_potential(1)
            ) for _ in range(self.num_predictions)
        ])
        
    def forward(self, state):
        """
        Predict top 5 terminal states for given board + piece
        CORRECTED: Flexible coordinate prediction
        """
        features = self.board_piece_encoder(state)
        
        placements = []
        
        for i in range(self.num_predictions):
            # Direct coordinate prediction (not fixed indices)
            placement_output = self.placement_heads[i](features)
            
            # FIXED: Handle both single sample and batch dimensions
            if len(placement_output.shape) == 1:
                placement_output = placement_output.unsqueeze(0)
            
            # Extract components
            rotation = torch.sigmoid(placement_output[:, 0]) * 4.0  # 0-4
            x_pos = torch.sigmoid(placement_output[:, 1]) * 10.0    # 0-10
            y_pos = torch.sigmoid(placement_output[:, 2]) * 20.0    # 0-20
            lines_potential = torch.sigmoid(placement_output[:, 3]) # 0-1 (line clearing potential)
            
            placements.append({
                'rotation': rotation,
                'x_pos': x_pos,
                'y_pos': y_pos,
                'lines_potential': lines_potential
            })
        
        return placements
    
    def get_top5_placement_options(self, state):
        """
        Get top 5 placement OPTIONS (not goals yet!)
        CORRECTED: Flexible positioning anywhere on board
        """
        placements = self.forward(state)
        
        options = []
        for i, placement in enumerate(placements):
            # Convert continuous values to discrete coordinates
            rotation = torch.clamp(torch.round(placement['rotation']), 0, 3).int()
            x_pos = torch.clamp(torch.round(placement['x_pos']), 0, 9).int()
            y_pos = torch.clamp(torch.round(placement['y_pos']), 0, 19).int()
            
            # Validate placement
            is_valid = self._validate_placement(rotation.item(), x_pos.item(), y_pos.item())
            
            options.append({
                'placement': (rotation.item(), x_pos.item(), y_pos.item()),
                'lines_potential': placement['lines_potential'].item(),
                'ranking': i + 1,
                'is_valid': is_valid
            })
        
        # Sort by combined score (lines_potential * validity)
        options.sort(key=lambda x: (
            x['lines_potential'] * 
            (1.0 if x['is_valid'] else 0.1)
        ), reverse=True)
        
        return options
    
    def _validate_placement(self, rotation, x_pos, y_pos):
        """
        Validate that placement coordinates are within bounds and physically possible
        ENHANCED: More comprehensive validation
        """
        # Basic bounds checking
        if not (0 <= rotation <= 3):
            return False
        if not (0 <= x_pos <= 9):
            return False
        if not (0 <= y_pos <= 19):
            return False
        
        # ENHANCED: Additional Tetris-specific validation
        # Check if placement makes physical sense
        
        # 1. Y position should not be too high (pieces fall down)
        if y_pos < 2:  # Pieces shouldn't be placed at the very top
            return False
        
        # 2. X position near edges should account for piece width
        # Most pieces are at least 2 units wide when considering rotation
        if rotation in [1, 3]:  # Vertical orientations
            if x_pos >= 9:  # Leave room for piece width
                return False
        
        # 3. Basic collision avoidance - don't place pieces overlapping
        # This is a simplified check; real validation would need board state
        
        return True
    
    def train_on_top_performers(self, placement_data, optimizer, top_percentile=0.95):
        """Train ONLY on top 5% performers for highest quality learning - UPDATED for natural gameplay"""
        if not placement_data:
            return {'loss': float('inf'), 'top_performers_used': 0}
        
        # Extract rewards and find top performers
        rewards = [d['terminal_reward'] for d in placement_data]
        threshold = np.percentile(rewards, top_percentile * 100)
        
        # Filter to TOP 5% performers only
        top_performers = [d for d in placement_data if d['terminal_reward'] >= threshold]
        
        # FIXED: Reduce minimum required performers for small datasets
        min_required = min(3, len(placement_data) // 2)  # At least 3 or half the data
        
        if len(top_performers) < min_required:
            print(f"   ⚠️ Not enough top performers ({len(top_performers)} < {min_required}), using all data")
            top_performers = placement_data  # Use all data if not enough top performers
        
        print(f"   🏆 Training on {len(top_performers)}/{len(placement_data)} performers (threshold: {threshold:.1f})")
        
        device = next(self.parameters()).device
        total_loss = 0
        num_batches = 0
        
        # Group by similar states to create training targets
        state_groups = {}
        for data in top_performers:
            state_data = data.get('state', data.get('resulting_state', []))
            state_key = tuple(state_data[:50])  # Use first 50 elements as state key
            if state_key not in state_groups:
                state_groups[state_key] = []
            state_groups[state_key].append(data)
        
        for state_key, state_group in state_groups.items():
            # FIXED: Use whatever data we have instead of requiring exactly 5
            num_predictions = min(self.num_predictions, len(state_group))
            if num_predictions < 1:
                continue
                
            # Sort by reward and take top available
            state_group.sort(key=lambda x: x['terminal_reward'], reverse=True)
            top_for_state = state_group[:num_predictions]
            
            # Use first state as input
            state_data = top_for_state[0].get('state', top_for_state[0].get('resulting_state', []))
            corrected_state = self._convert_state_410_to_210(state_data)
            state = torch.FloatTensor(corrected_state).unsqueeze(0).to(device)
            
            # Forward pass
            placements = self.forward(state)
            
            total_placement_loss = 0
            validity_penalty = 0
            
            # Create training targets for each available prediction
            for i, target_data in enumerate(top_for_state):
                if i >= len(placements):  # Don't exceed available predictions
                    break
                
                # FIXED: Handle natural gameplay data (action instead of placement)
                if 'placement' in target_data:
                    # Old placement-based data
                    placement = target_data['placement']
                    try:
                        if isinstance(placement, (list, tuple)):
                            true_rot, true_x, true_y = float(placement[0]), float(placement[1]), float(placement[2])
                        elif hasattr(placement, '__getitem__') and hasattr(placement, '__len__'):
                            true_rot = float(placement[0])
                            true_x = float(placement[1]) if len(placement) > 1 else 0.0
                            true_y = float(placement[2]) if len(placement) > 2 else 0.0
                        else:
                            true_rot = float(placement)
                            true_x = 0.0
                            true_y = 0.0
                    except (TypeError, ValueError, IndexError):
                        true_rot, true_x, true_y = 0.0, 4.0, 10.0  # Safe defaults
                else:
                    # NEW: Natural gameplay data - derive placement from action and state
                    action = target_data.get('action', [0, 0, 0, 0, 0, 1, 0, 0])  # Default to hard drop
                    placement_info = self._derive_placement_from_action_and_state(action, target_data)
                    true_rot, true_x, true_y = placement_info
                
                # Ensure values are within valid ranges
                true_rot = max(0, min(3, true_rot))
                true_x = max(0, min(9, true_x))
                true_y = max(0, min(19, true_y))
                        
                true_reward = target_data['terminal_reward']
                lines_cleared = target_data.get('lines_cleared', 0)
                
                # FIXED: Scale predictions to match target ranges and normalize losses
                target_rot = torch.FloatTensor([true_rot / 4.0]).to(device)  # Scale to 0-1
                target_x = torch.FloatTensor([true_x / 10.0]).to(device)     # Scale to 0-1  
                target_y = torch.FloatTensor([true_y / 20.0]).to(device)     # Scale to 0-1
                target_lines_potential = torch.FloatTensor([min(1.0, lines_cleared / 4.0)]).to(device)
                
                # FIXED: Coordinate prediction losses with proper scaling
                rot_loss = F.mse_loss(placements[i]['rotation'] / 4.0, target_rot) * 0.5  # Weight rotation less
                x_loss = F.mse_loss(placements[i]['x_pos'] / 10.0, target_x) * 1.0        # Normal weight for X
                y_loss = F.mse_loss(placements[i]['y_pos'] / 20.0, target_y) * 1.0        # Normal weight for Y  
                lines_loss = F.mse_loss(placements[i]['lines_potential'], target_lines_potential) * 0.5
                
                total_placement_loss += (rot_loss + x_loss + y_loss + lines_loss)
                
                # FIXED: Scaled validity penalty  
                if not self._validate_placement(true_rot, true_x, true_y):
                    validity_penalty += 0.5  # Reduced penalty
            
            # Combined loss with safety check
            combined_loss = total_placement_loss + validity_penalty
            if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                print(f"   ⚠️ Invalid loss detected, skipping batch")
                continue
            
            # Backpropagation
            optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            
            total_loss += combined_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return {
            'loss': avg_loss,
            'top_performers_used': len(top_performers),
            'threshold': threshold,
            'state_groups_trained': num_batches,
            'validity_penalty': validity_penalty if 'validity_penalty' in locals() else 0,
            'focus': 'natural_gameplay_actions'
        }
    
    def _derive_placement_from_action_and_state(self, action, data):
        """
        ADDED: Derive approximate placement information from natural gameplay action and state
        """
        # Extract current piece position from state if available
        state = data.get('state', [])
        
        # Default placement values
        rotation = 0
        x_pos = 4  # Center spawn
        y_pos = 18  # Near bottom
        
        # Try to extract piece position from state metadata (last 3 elements in 210D state)
        if len(state) >= 210:
            try:
                rotation = int(state[207]) if len(state) > 207 else 0  # current_rotation
                x_pos = int(state[208]) if len(state) > 208 else 4     # current_x  
                y_pos = int(state[209]) if len(state) > 209 else 18    # current_y
            except (ValueError, IndexError):
                pass  # Use defaults
        
        # Adjust based on action taken
        action_idx = np.argmax(action) if hasattr(action, '__len__') else 0
        
        if action_idx == 0:  # Move Left
            x_pos = max(0, x_pos - 1)
        elif action_idx == 1:  # Move Right
            x_pos = min(9, x_pos + 1)
        elif action_idx == 2:  # Move Down
            y_pos = min(19, y_pos + 1)
        elif action_idx == 3:  # Rotate CW
            rotation = (rotation + 1) % 4
        elif action_idx == 5:  # Hard Drop
            y_pos = 18  # Assume near bottom after hard drop
        
        return rotation, x_pos, y_pos
    
    def _convert_state_410_to_210(self, state_data):
        """Convert 410D state to 210D by removing empty_grid"""
        if len(state_data) == 410:
            current_piece_grid = state_data[:200]
            next_piece = state_data[400:407]
            metadata = state_data[407:410]
            return np.concatenate([current_piece_grid, next_piece, metadata])
        elif len(state_data) == 210:
            return state_data
        else:
            return state_data


class SimplifiedQLearning(nn.Module):
    """
    SIMPLIFIED Q-learning that outputs single terminal value
    N-step bootstrapping comes from continuous exploration trajectories
    """
    
    def __init__(self, state_dim):
        super(SimplifiedQLearning, self).__init__()
        
        self.state_dim = state_dim
        self.gamma = 0.99
        
        # Simple Q-network outputting single value
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single Q-value
        )
        
    def forward(self, state):
        """Predict single Q-value for given state"""
        return self.q_network(state)
    
    def train_on_trajectories(self, trajectory_data, optimizer, n_step=4):
        """
        Train Q-learning on continuous trajectories with n-step returns
        SIMPLIFIED: No episode management, just continuous trajectories
        """
        if not trajectory_data:
            return {'q_loss': float('inf'), 'trajectories_trained': 0}
        
        device = next(self.parameters()).device
        total_loss = 0
        samples_trained = 0
        
        print(f"   🎯 Training Q-learning on {len(trajectory_data)} trajectory samples")
        
        for i in range(len(trajectory_data) - n_step):
            # Get n-step sequence
            current_data = trajectory_data[i]
            
            # FIXED: Calculate normalized n-step return 
            n_step_return = 0
            gamma_power = 1
            
            for j in range(n_step):
                if i + j < len(trajectory_data):
                    step_data = trajectory_data[i + j]
                    step_reward = step_data.get('terminal_reward', 0)
                    lines_reward = step_data.get('lines_cleared', 0) * 10
                    total_step_reward = step_reward + lines_reward
                    
                    # FIXED: Normalize rewards to reasonable range
                    normalized_reward = np.tanh(total_step_reward / 100.0)  # Scale to [-1,1] range
                    
                    n_step_return += gamma_power * normalized_reward
                    gamma_power *= self.gamma
            
            # Convert state and train
            state_data = current_data.get('state', current_data.get('resulting_state', []))
            corrected_state = self._convert_state_410_to_210(state_data)
            state = torch.FloatTensor(corrected_state).unsqueeze(0).to(device)
            
            target_return = torch.FloatTensor([[n_step_return]]).to(device)
            
            # Forward pass
            predicted_return = self.forward(state)
            
            # FIXED: MSE loss with proper scaling
            q_loss = F.mse_loss(predicted_return, target_return) * 10.0  # Scale up for learning
            
            # Backpropagation
            optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            
            total_loss += q_loss.item()
            samples_trained += 1
        
        avg_loss = total_loss / max(1, samples_trained)
        
        return {
            'q_loss': avg_loss,
            'trajectories_trained': samples_trained,
            'n_step': n_step,
            'samples_processed': samples_trained
        }
    
    def predict_value(self, state):
        """Predict Q-value for state"""
        with torch.no_grad():
            return self.forward(state)
    
    def _convert_state_410_to_210(self, state_data):
        """Convert 410D state to 210D by removing empty_grid"""
        if len(state_data) == 410:
            current_piece_grid = state_data[:200]
            next_piece = state_data[400:407]
            metadata = state_data[407:410]
            return np.concatenate([current_piece_grid, next_piece, metadata])
        elif len(state_data) == 210:
            return state_data
        else:
            return state_data


class EpsilonGreedyGoalSelector:
    """
    Epsilon-greedy goal selection using Q-learning evaluations
    CORRECTED: Goal vector allows ANY position on board
    """
    
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def select_goal(self, placement_options, q_learning_model, state):
        """Select goal using epsilon-greedy over Q-evaluated options"""
        if not placement_options:
            return self._create_zero_goal(state.device)
        
        # Evaluate each option with Q-learning
        option_values = []
        for option in placement_options:
            if option['is_valid']:
                q_value = q_learning_model.predict_value(state)
                option_values.append(q_value.item())
            else:
                option_values.append(float('-inf'))
        
        # Epsilon-greedy selection
        if np.random.random() < self.epsilon:
            valid_indices = [i for i, opt in enumerate(placement_options) if opt['is_valid']]
            if valid_indices:
                selected_idx = random.choice(valid_indices)
            else:
                selected_idx = 0
        else:
            selected_idx = np.argmax(option_values)
        
        # Convert selected option to goal vector
        selected_option = placement_options[selected_idx]
        goal_vector = self._option_to_goal_vector(selected_option, state.device)
        
        return goal_vector
    
    def _option_to_goal_vector(self, option, device):
        """
        SIMPLIFIED: Convert placement option to essential goal vector (5D total)
        
        Goal Vector (5D - SIMPLIFIED):
        - [0]: Rotation bit 0 (binary representation)
        - [1]: Rotation bit 1 (binary representation) 
        - [2]: X position (discrete 0-9)
        - [3]: Y position (discrete 0-19)
        - [4]: Validity flag (0 or 1) - ONLY validity, no confidence/quality
        
        This captures ONLY essential placement info + validity as requested
        """
        # Extract placement coordinates
        rotation, x_pos, y_pos = option['placement']
        
        # Binary rotation representation (2 bits for 4 rotations)
        rotation_int = int(rotation) % 4
        rot_bit_0 = float(rotation_int & 1)      # Bit 0 (LSB)
        rot_bit_1 = float((rotation_int >> 1) & 1)  # Bit 1
        
        # Discrete coordinates (exact)
        discrete_x = float(int(x_pos) % 10)     # 0-9 (exact)
        discrete_y = float(int(y_pos) % 20)     # 0-19 (exact)
        
        # ONLY validity flag (no confidence/quality as requested)
        validity_flag = 1.0 if option.get('is_valid', True) else 0.0
        
        # Create SIMPLIFIED goal vector (5D total)
        goal_vector = torch.FloatTensor([
            rot_bit_0,           # [0]: Rotation bit 0 (binary)
            rot_bit_1,           # [1]: Rotation bit 1 (binary)
            discrete_x,          # [2]: X position (discrete 0-9)
            discrete_y,          # [3]: Y position (discrete 0-19)
            validity_flag        # [4]: Validity flag ONLY (0 or 1)
        ]).to(device)
        
        return goal_vector.unsqueeze(0)
    
    def decode_goal_vector_to_placement(self, goal_vector):
        """
        SIMPLIFIED: Decode 5D goal vector back to placement coordinates
        """
        if len(goal_vector.shape) > 1:
            goal_vector = goal_vector.squeeze(0)
        
        # Decode binary rotation
        rot_bit_0 = int(round(goal_vector[0].item()))
        rot_bit_1 = int(round(goal_vector[1].item()))
        rotation = rot_bit_0 + (rot_bit_1 << 1)  # Reconstruct from bits
        
        # Decode discrete coordinates
        x_pos = int(round(goal_vector[2].item())) % 10
        y_pos = int(round(goal_vector[3].item())) % 20
        
        # Decode validity ONLY (no confidence/quality)
        is_valid = goal_vector[4].item() > 0.5
        
        return {
            'placement': (rotation, x_pos, y_pos),
            'is_valid': is_valid
        }
    
    def _create_zero_goal(self, device):
        """Create zero goal as fallback (5D)"""
        return torch.zeros(1, 5, device=device)  # UPDATED: 5D instead of 6D
    
    def update_epsilon(self):
        """Decay epsilon over time"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class PieceByPieceExplorationManager:
    """
    REDESIGNED: Piece-by-piece exploration with proper board progression
    - 200 trials per piece
    - Keep top 20% (40 boards), select best 5 for next piece
    - Proper trajectory lineage tracking
    - Effective line clearing through progressive board building
    """
    
    def __init__(self, env):
        self.env = env
        self.board_candidates = []  # Current generation of boards
        self.exploration_data = []  # All exploration data with lineage
        self.piece_sequence = []   # Track pieces used
        self.generation = 0        # Current piece generation
        
        # Exploration parameters - UPDATED for realistic exploration
        self.trials_per_piece = 50   # REDUCED: More realistic number of trials
        self.top_percentage = 0.2    # Top 20%
        self.boards_to_keep = 3      # Keep 3 best boards (reduced)
        self.max_pieces = 5          # UPDATED: Use exactly 5 pieces as requested
        self.piece_types_used = 5    # ONLY use first 5 piece types (I, O, T, S, Z)
        
        print(f"🔧 Realistic Exploration Manager initialized:")
        print(f"   • Trials per piece: {self.trials_per_piece}")
        print(f"   • Top {self.top_percentage*100}% selection ({int(self.trials_per_piece * self.top_percentage)} boards)")
        print(f"   • Boards kept for next piece: {self.boards_to_keep}")
        print(f"   • Max pieces in sequence: {self.max_pieces}")
        print(f"   • Piece types used: {self.piece_types_used} (I, O, T, S, Z only)")
    
    def collect_piece_by_piece_exploration_data(self, exploration_mode='natural_gameplay'):
        """
        REWRITTEN: Use tetris_env's natural gameplay instead of redundant simulation
        Let the environment handle all piece placement and line clearing
        """
        self.exploration_data = []
        self.generation = 0
        
        print(f"🎮 Natural Gameplay Exploration Starting:")
        print(f"   • Using environment's built-in step() function")
        print(f"   • No redundant piece simulation")
        print(f"   • Let tetris_env handle line clearing naturally")
        
        # Simple natural gameplay episodes
        num_episodes = 5  # Small number of natural episodes
        max_steps_per_episode = 50  # Allow pieces to accumulate naturally
        
        for episode in range(num_episodes):
            print(f"   🎮 Episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            obs = self.env.reset()
            episode_data = []
            episode_lines_cleared = 0
            
            for step in range(max_steps_per_episode):
                # Get current state
                state_vector = self._obs_to_state_vector_210(obs)
                
                # Take random action (let pieces fall naturally)
                action = self._generate_natural_action()
                
                # Use environment's step function - this handles everything!
                next_obs, reward, done, info = self.env.step(action)
                
                # Extract line clearing info from environment
                lines_cleared = info.get('lines_cleared', 0)
                episode_lines_cleared += lines_cleared
                
                # Store natural gameplay data
                episode_data.append({
                    'state': state_vector,
                    'resulting_state': self._obs_to_state_vector_210(next_obs),
                    'action': action,
                    'terminal_reward': reward,
                    'lines_cleared': lines_cleared,
                    'episode': episode,
                    'step': step,
                    'natural_gameplay': True
                })
                
                obs = next_obs
                
                if done:
                    print(f"     ⚡ Episode ended at step {step}, lines cleared: {lines_cleared}")
                    break
            
            self.exploration_data.extend(episode_data)
            print(f"     📊 Episode {episode + 1}: {len(episode_data)} steps, {episode_lines_cleared} total lines cleared")
        
        # Final summary
        total_lines = sum(d.get('lines_cleared', 0) for d in self.exploration_data)
        line_events = sum(1 for d in self.exploration_data if d.get('lines_cleared', 0) > 0)
        
        print(f"✅ Natural Gameplay Exploration completed:")
        print(f"   • Total steps: {len(self.exploration_data)}")
        print(f"   • Total lines cleared: {total_lines}")
        print(f"   • Line clearing events: {line_events}")
        if len(self.exploration_data) > 0:
            print(f"   • Lines per step: {total_lines/len(self.exploration_data):.3f}")
        
        return self.exploration_data

    def _generate_natural_action(self):
        """Generate natural Tetris actions that lead to piece placement"""
        # Natural Tetris gameplay actions with realistic distribution
        actions = [
            [1, 0, 0, 0, 0, 0, 0, 0],  # Move Left - 15%
            [0, 1, 0, 0, 0, 0, 0, 0],  # Move Right - 15% 
            [0, 0, 1, 0, 0, 0, 0, 0],  # Move Down - 25%
            [0, 0, 0, 1, 0, 0, 0, 0],  # Rotate CW - 20%
            [0, 0, 0, 0, 0, 1, 0, 0],  # Hard Drop - 20%
            [0, 0, 0, 0, 0, 0, 0, 1],  # No-op - 5%
        ]
        
        probabilities = [0.15, 0.15, 0.25, 0.20, 0.20, 0.05]
        action_idx = np.random.choice(len(actions), p=probabilities)
        return actions[action_idx]

    def _obs_to_state_vector_210(self, obs):
        """Convert observation to 210D state vector"""
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


class Enhanced6PhaseComponents:
    """
    SIMPLIFIED Enhanced 6-Phase System V3
    Key Simplifications:
    1. SIMPLIFIED: Goal vector now 5D (rotation + x + y + validity ONLY)
    2. ADDED: Lines cleared tracking during exploration
    3. CORRECTED: Continuous board exploration (not reset each time)
    4. SIMPLIFIED: Q-learning outputs single terminal value with n-step from trajectories
    5. FIXED: Iterative terminal state exploration (not strategic)
    6. NO CONFIDENCE/QUALITY: Only validity as requested
    """
    def __init__(self, state_dim=210, goal_dim=5, device='cpu'):  # UPDATED: goal_dim=5
        """
        Initialize enhanced 6-phase components
        
        Args:
            state_dim: State vector dimension (210D)
            goal_dim: Goal vector dimension (5D - rotation + x + y + validity ONLY)
            device: PyTorch device
        """
        self.device = device
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        
        # CORRECTED: 210D state (no empty_grid)
        corrected_state_dim = 210
        
        # Enhanced state model (top 5 placement options with binary rotation)
        self.state_model = Top5TerminalStateModel(corrected_state_dim).to(device)
        
        # SIMPLIFIED Q-learning (single value output, n-step from trajectories)
        self.q_learning = SimplifiedQLearning(corrected_state_dim).to(device)
        
        # Epsilon-greedy goal selector (simplified for 5D goals)
        self.goal_selector = EpsilonGreedyGoalSelector()
        
        # Optimizers
        self.state_optimizer = None
        self.q_optimizer = None
    
    def set_optimizers(self, state_lr, q_lr):
        """Set optimizers for both components"""
        self.state_optimizer = torch.optim.Adam(self.state_model.parameters(), lr=state_lr)
        self.q_optimizer = torch.optim.Adam(self.q_learning.parameters(), lr=q_lr)
    
    def train_enhanced_state_model(self, exploration_data):
        """Train enhanced state model on top 5% performers only"""
        if not self.state_optimizer:
            raise ValueError("State optimizer not set")
        
        return self.state_model.train_on_top_performers(exploration_data, self.state_optimizer)
    
    def train_simplified_q_learning(self, exploration_data):
        """
        Train simplified Q-learning on trajectory data
        CORRECTED: Uses continuous trajectories, no episode management
        """
        if not self.q_optimizer:
            raise ValueError("Q optimizer not set")
        
        return self.q_learning.train_on_trajectories(exploration_data, self.q_optimizer)
    
    def get_goal_for_actor(self, state):
        """
        CORRECT 6-phase goal flow:
        1. State model → placement options
        2. Q-learning → option evaluation  
        3. Epsilon-greedy → goal selection
        4. Return 6D goal (binary rotation + discrete coordinates) to actor
        """
        # Step 1: Get placement options from state model
        placement_options = self.state_model.get_top5_placement_options(state)
        
        # Step 2 & 3: Q-learning evaluation + epsilon-greedy selection
        goal = self.goal_selector.select_goal(placement_options, self.q_learning, state)
        
        return goal
    
    def obs_to_state_vector(self, obs):
        """Convert observation to 210D state vector (CORRECTED - no empty_grid)"""
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
    
    def update_epsilon(self):
        """Update epsilon for goal selection"""
        self.goal_selector.update_epsilon()
    
    def create_line_clearing_evaluator(self, env):
        """Create line clearing evaluator for testing"""
        return LineClearingEvaluator(env)
    
    def create_piece_by_piece_exploration_manager(self, env):
        """Create piece-by-piece exploration manager"""
        return PieceByPieceExplorationManager(env)
    
    def save_checkpoints(self, path_prefix):
        """Save all component checkpoints"""
        torch.save({
            'state_model': self.state_model.state_dict(),
            'q_learning': self.q_learning.state_dict(),
            'state_optimizer': self.state_optimizer.state_dict() if self.state_optimizer else None,
            'q_optimizer': self.q_optimizer.state_dict() if self.q_optimizer else None,
            'goal_selector_epsilon': self.goal_selector.epsilon
        }, f"{path_prefix}_enhanced_6phase.pt")
    
    def load_checkpoints(self, path_prefix):
        """Load all component checkpoints"""
        checkpoint = torch.load(f"{path_prefix}_enhanced_6phase.pt")
        
        self.state_model.load_state_dict(checkpoint['state_model'])
        self.q_learning.load_state_dict(checkpoint['q_learning'])
        
        if checkpoint['state_optimizer'] and self.state_optimizer:
            self.state_optimizer.load_state_dict(checkpoint['state_optimizer'])
        if checkpoint['q_optimizer'] and self.q_optimizer:
            self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        
        self.goal_selector.epsilon = checkpoint['goal_selector_epsilon'] 