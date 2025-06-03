"""
State transition model: predicts optimal piece placements from terminal rewards.
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

class StateModel(nn.Module):
    """
    State model that learns to predict optimal piece placements from state vectors
    Uses centralized configuration for all network dimensions
    """
    def __init__(self, state_dim=None):
        super(StateModel, self).__init__()
        
        # Get centralized config
        self.config = TetrisConfig()
        self.net_config = self.config.NetworkConfig.StateModel
        
        # Use centralized state dimension
        self.state_dim = state_dim or self.config.STATE_DIM  # 410
        
        # MLP encoder with dropout (using centralized config)
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.net_config.ENCODER_LAYERS[1]),
            nn.ReLU(),
            nn.Dropout(self.net_config.DROPOUT_RATE),
            nn.Linear(self.net_config.ENCODER_LAYERS[1], self.net_config.ENCODER_LAYERS[2]),
            nn.ReLU(),
            nn.Dropout(self.net_config.DROPOUT_RATE),
            nn.Linear(self.net_config.ENCODER_LAYERS[2], self.net_config.ENCODER_LAYERS[3]),
            nn.ReLU()
        )
        
        # Output heads (using centralized config)
        self.rotation_head = nn.Linear(self.net_config.ENCODER_LAYERS[3], self.net_config.ROTATION_CLASSES)
        self.x_position_head = nn.Linear(self.net_config.ENCODER_LAYERS[3], self.net_config.X_POSITION_CLASSES)
        self.y_position_head = nn.Linear(self.net_config.ENCODER_LAYERS[3], self.net_config.Y_POSITION_CLASSES)
        self.value_head = nn.Linear(self.net_config.ENCODER_LAYERS[3], self.net_config.VALUE_OUTPUT)

    def forward(self, state):
        """
        Args:
            state: Tensor of shape (batch_size, state_dim)
        Returns:
            rot_logits: (batch_size, num_rotations)
            x_logits: (batch_size, board_width)
            y_logits: (batch_size, board_height)
            value: (batch_size, 1) - predicted terminal reward
        """
        h = self.encoder(state)
        rot_logits = self.rotation_head(h)
        x_logits = self.x_position_head(h)
        y_logits = self.y_position_head(h)
        value = self.value_head(h)
        return rot_logits, x_logits, y_logits, value
    
    def get_placement_distribution(self, state):
        """
        Get probability distributions over placements
        Returns:
            rot_probs: (batch_size, num_rotations)
            x_probs: (batch_size, board_width)
            y_probs: (batch_size, board_height)
        """
        rot_logits, x_logits, y_logits, _ = self.forward(state)
        rot_probs = F.softmax(rot_logits, dim=1)
        x_probs = F.softmax(x_logits, dim=1)
        y_probs = F.softmax(y_logits, dim=1)
        return rot_probs, x_probs, y_probs
    
    def train_from_placements(self, placement_data, optimizer, num_epochs=10):
        """
        Train the model from exploration placement data with terminal rewards
        ENHANCED: Long-term potential reward calculation and stabilized training
        Args:
            placement_data: List of dicts with 'state', 'placement', 'terminal_reward'
            optimizer: Optimizer for training
            num_epochs: Number of training epochs
        Returns:
            Dictionary with detailed loss information
        """
        if not placement_data:
            return {}
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Get valid ranges from config
        max_rotation = self.net_config.ROTATION_CLASSES - 1  # 3 (0-3)
        max_x_pos = self.net_config.X_POSITION_CLASSES - 1   # 9 (0-9)
        max_y_pos = self.net_config.Y_POSITION_CLASSES - 1   # 19 (0-19)
        
        # ENHANCEMENT: Calculate long-term potential rewards
        enhanced_placement_data = self._calculate_long_term_potential_rewards(placement_data)
        
        total_losses = []
        rot_losses = []
        x_losses = []
        y_losses = []
        value_losses = []
        
        criterion = nn.CrossEntropyLoss()
        value_criterion = nn.MSELoss()
        
        # Validation counters
        clamped_rotations = 0
        clamped_x_positions = 0
        clamped_y_positions = 0
        
        # ENHANCEMENT: Learning rate scheduler for stability
        initial_lr = optimizer.param_groups[0]['lr']
        
        for epoch in range(num_epochs):
            epoch_total_loss = 0
            epoch_rot_loss = 0
            epoch_x_loss = 0
            epoch_y_loss = 0
            epoch_value_loss = 0
            
            # ENHANCEMENT: Adaptive learning rate
            lr_decay = 0.95 ** epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr * lr_decay
            
            np.random.shuffle(enhanced_placement_data)
            
            for data in enhanced_placement_data:
                state = torch.FloatTensor(data['state']).unsqueeze(0).to(device)
                
                # Handle both 2-tuple and 3-tuple placements for backward compatibility
                placement = data['placement']
                if len(placement) == 2:
                    rotation, x_pos = placement
                    y_pos = 10  # Default y position for backward compatibility
                elif len(placement) == 3:
                    rotation, x_pos, y_pos = placement
                else:
                    raise ValueError(f"Invalid placement format: {placement}")
                
                # ENHANCEMENT: Use long-term potential reward instead of just terminal reward
                long_term_reward = data.get('long_term_reward', data['terminal_reward'])
                
                # FIXED: Validate and clamp all targets to valid ranges
                original_rotation, original_x_pos, original_y_pos = rotation, x_pos, y_pos
                
                rotation = max(0, min(max_rotation, int(rotation)))
                x_pos = max(0, min(max_x_pos, int(x_pos)))
                y_pos = max(0, min(max_y_pos, int(y_pos)))
                
                # Track clamping for debugging
                if rotation != original_rotation:
                    clamped_rotations += 1
                if x_pos != original_x_pos:
                    clamped_x_positions += 1
                if y_pos != original_y_pos:
                    clamped_y_positions += 1
                
                # Forward pass
                rot_logits, x_logits, y_logits, value_pred = self.forward(state)
                
                # Calculate losses with validated targets
                rot_loss = criterion(rot_logits, torch.LongTensor([rotation]).to(device))
                x_loss = criterion(x_logits, torch.LongTensor([x_pos]).to(device))
                y_loss = criterion(y_logits, torch.LongTensor([y_pos]).to(device))
                value_loss = value_criterion(value_pred, torch.FloatTensor([[long_term_reward]]).to(device))
                
                # ENHANCEMENT: Improved reward weighting for stability
                reward_weight = self._calculate_stable_reward_weight(long_term_reward)
                
                # ENHANCEMENT: Balanced loss with regularization
                classification_loss = rot_loss + x_loss + y_loss
                total_loss = (
                    reward_weight * classification_loss + 
                    value_loss + 
                    0.01 * self._calculate_regularization_loss()  # L2 regularization
                )
                
                # Backpropagation with gradient clipping
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Accumulate losses for averaging
                epoch_total_loss += total_loss.item()
                epoch_rot_loss += rot_loss.item()
                epoch_x_loss += x_loss.item()
                epoch_y_loss += y_loss.item()
                epoch_value_loss += value_loss.item()
            
            # Average losses for this epoch
            num_samples = len(enhanced_placement_data)
            total_losses.append(epoch_total_loss / num_samples)
            rot_losses.append(epoch_rot_loss / num_samples)
            x_losses.append(epoch_x_loss / num_samples)
            y_losses.append(epoch_y_loss / num_samples)
            value_losses.append(epoch_value_loss / num_samples)
        
        # Restore original learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr
        
        # Report clamping statistics if any occurred
        if clamped_rotations > 0 or clamped_x_positions > 0 or clamped_y_positions > 0:
            print(f"   🔧 Target validation: clamped {clamped_rotations} rotations, {clamped_x_positions} x-positions, {clamped_y_positions} y-positions")
        
        return {
            'total_loss': total_losses[-1],  # Final epoch loss
            'rotation_loss': rot_losses[-1],
            'x_position_loss': x_losses[-1],
            'y_position_loss': y_losses[-1],
            'value_loss': value_losses[-1],
            'all_total_losses': total_losses,
            'all_rotation_losses': rot_losses,
            'all_x_position_losses': x_losses,
            'all_y_position_losses': y_losses,
            'all_value_losses': value_losses,
            'clamped_targets': {
                'rotations': clamped_rotations,
                'x_positions': clamped_x_positions,
                'y_positions': clamped_y_positions
            },
            'long_term_enhanced': True  # Flag to indicate enhancement is active
        }
    
    def _calculate_long_term_potential_rewards(self, placement_data):
        """
        ENHANCEMENT: Calculate long-term potential rewards for each placement
        Considers future potential and placement quality beyond immediate reward
        
        Args:
            placement_data: Original placement data
        Returns:
            Enhanced placement data with long-term rewards
        """
        enhanced_data = []
        
        for data in placement_data:
            original_reward = data['terminal_reward']
            
            # Extract placement information
            placement = data['placement']
            state = data['state']
            
            # Calculate placement quality metrics
            placement_quality = self._assess_placement_quality(state, placement)
            
            # Calculate long-term potential based on board state
            board_potential = self._assess_board_potential(state)
            
            # Combine immediate reward with long-term factors
            long_term_reward = (
                original_reward * 0.6 +              # 60% immediate reward
                placement_quality * 0.25 +           # 25% placement quality
                board_potential * 0.15               # 15% board potential
            )
            
            # Create enhanced data entry
            enhanced_entry = data.copy()
            enhanced_entry['long_term_reward'] = long_term_reward
            enhanced_entry['placement_quality'] = placement_quality
            enhanced_entry['board_potential'] = board_potential
            
            enhanced_data.append(enhanced_entry)
        
        return enhanced_data
    
    def _assess_placement_quality(self, state, placement):
        """
        Assess the quality of a placement based on structural factors
        
        Args:
            state: Game state (410D vector)
            placement: Placement tuple (rotation, x_pos, y_pos)
        Returns:
            quality_score: Float score for placement quality
        """
        try:
            # Extract board information
            current_piece_grid = state[:200].reshape(20, 10)
            empty_grid = state[200:400].reshape(20, 10)
            
            # Calculate structural metrics
            total_height = np.sum(empty_grid == 0, axis=0).max()  # Max column height
            holes = self._count_holes(empty_grid)
            bumpiness = self._calculate_bumpiness(empty_grid)
            
            # Placement position analysis
            if len(placement) >= 2:
                x_pos = placement[1]
                # Prefer central positions (reduce edge bias)
                centrality_bonus = 10 - abs(x_pos - 4.5)  # 4.5 is center of 0-9 range
            else:
                centrality_bonus = 0
            
            # Quality scoring (normalized to reasonable range)
            quality_score = (
                -total_height * 2.0 +     # Lower heights are better
                -holes * 15.0 +           # Fewer holes are better  
                -bumpiness * 3.0 +        # Smoother surface is better
                centrality_bonus * 2.0    # Central placements are better
            )
            
            return max(-50.0, min(50.0, quality_score))  # Clamp to reasonable range
            
        except Exception as e:
            print(f"Error in placement quality assessment: {e}")
            return 0.0  # Neutral quality
    
    def _assess_board_potential(self, state):
        """
        Assess the long-term potential of the current board state
        
        Args:
            state: Game state (410D vector)
        Returns:
            potential_score: Float score for future potential
        """
        try:
            # Extract board information
            empty_grid = state[200:400].reshape(20, 10)
            occupied_grid = 1 - empty_grid  # Invert to get occupied cells
            
            # Calculate board metrics
            line_completion_potential = self._calculate_line_completion_potential(occupied_grid)
            stability_score = self._calculate_board_stability(occupied_grid)
            space_efficiency = self._calculate_space_efficiency(occupied_grid)
            
            # Combine factors for overall potential
            potential_score = (
                line_completion_potential * 0.4 +    # 40% line completion potential
                stability_score * 0.35 +             # 35% board stability
                space_efficiency * 0.25              # 25% space efficiency
            )
            
            return max(-30.0, min(30.0, potential_score))  # Clamp to reasonable range
            
        except Exception as e:
            print(f"Error in board potential assessment: {e}")
            return 0.0  # Neutral potential
    
    def _count_holes(self, empty_grid):
        """Count holes in the board (empty cells with occupied cells above)"""
        holes = 0
        for col in range(empty_grid.shape[1]):
            column = empty_grid[:, col]
            found_block = False
            for row in range(empty_grid.shape[0]):
                if column[row] == 0:  # Occupied cell
                    found_block = True
                elif found_block and column[row] == 1:  # Empty cell below occupied
                    holes += 1
        return holes
    
    def _calculate_bumpiness(self, empty_grid):
        """Calculate surface bumpiness"""
        column_heights = []
        for col in range(empty_grid.shape[1]):
            height = 0
            for row in range(empty_grid.shape[0]):
                if empty_grid[row, col] == 0:  # First occupied cell
                    height = empty_grid.shape[0] - row
                    break
            column_heights.append(height)
        
        # Calculate bumpiness as sum of absolute differences
        bumpiness = 0
        for i in range(len(column_heights) - 1):
            bumpiness += abs(column_heights[i] - column_heights[i + 1])
        
        return bumpiness
    
    def _calculate_line_completion_potential(self, occupied_grid):
        """Calculate how close lines are to being completed"""
        potential = 0
        for row in range(occupied_grid.shape[0]):
            filled_cells = np.sum(occupied_grid[row, :])
            if filled_cells >= 7:  # Close to completion
                completion_ratio = filled_cells / 10.0
                potential += completion_ratio * 20.0  # Bonus for near-complete lines
        return potential
    
    def _calculate_board_stability(self, occupied_grid):
        """Calculate board stability based on support structure"""
        stability = 0
        height, width = occupied_grid.shape
        
        for row in range(height - 1):  # Skip bottom row
            for col in range(width):
                if occupied_grid[row, col] == 1:  # Occupied cell
                    # Check if supported by cell below
                    if occupied_grid[row + 1, col] == 1:
                        stability += 1
                    # Check horizontal support
                    horizontal_support = 0
                    if col > 0 and occupied_grid[row, col - 1] == 1:
                        horizontal_support += 1
                    if col < width - 1 and occupied_grid[row, col + 1] == 1:
                        horizontal_support += 1
                    stability += horizontal_support * 0.5
        
        return stability
    
    def _calculate_space_efficiency(self, occupied_grid):
        """Calculate how efficiently the space is being used"""
        total_cells = occupied_grid.shape[0] * occupied_grid.shape[1]
        occupied_cells = np.sum(occupied_grid)
        
        if occupied_cells == 0:
            return 0.0
        
        # Calculate compactness (how clustered the pieces are)
        compactness = 0
        height, width = occupied_grid.shape
        
        for row in range(height):
            for col in range(width):
                if occupied_grid[row, col] == 1:
                    neighbors = 0
                    # Count adjacent occupied cells
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if occupied_grid[nr, nc] == 1:
                                neighbors += 1
                    compactness += neighbors
        
        # Normalize by occupied cells
        efficiency = compactness / max(1, occupied_cells)
        return min(10.0, efficiency * 5.0)  # Scale to reasonable range
    
    def _calculate_stable_reward_weight(self, reward):
        """Calculate more stable reward weighting to prevent oscillations"""
        # Use sigmoid-like weighting instead of linear to reduce volatility
        normalized_reward = (reward + 100) / 200.0  # Normalize to [0, 1] range
        stable_weight = 0.2 + 0.6 / (1 + np.exp(-5 * (normalized_reward - 0.5)))  # Sigmoid centering
        return max(0.1, min(1.0, stable_weight))
    
    def _calculate_regularization_loss(self):
        """Calculate L2 regularization loss for network parameters"""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return l2_loss

    def get_optimal_placement(self, state):
        """
        Get the optimal placement directly from the model predictions
        Returns:
            optimal_placement: Dict with 'rotation', 'x_position', 'y_position', and 'confidence'
        """
        with torch.no_grad():
            rot_logits, x_logits, y_logits, value = self.forward(state)
            
            # Get the most likely placement
            optimal_rotation = torch.argmax(rot_logits, dim=1)
            optimal_x = torch.argmax(x_logits, dim=1)
            optimal_y = torch.argmax(y_logits, dim=1)
            
            # Calculate confidence scores (softmax probabilities)
            rot_probs = F.softmax(rot_logits, dim=1)
            x_probs = F.softmax(x_logits, dim=1)
            y_probs = F.softmax(y_logits, dim=1)
            
            # Get confidence for the optimal placement
            batch_indices = torch.arange(rot_logits.shape[0])
            rot_confidence = rot_probs[batch_indices, optimal_rotation]
            x_confidence = x_probs[batch_indices, optimal_x]
            y_confidence = y_probs[batch_indices, optimal_y]
            
            # Overall confidence is the product of individual confidences
            overall_confidence = rot_confidence * x_confidence * y_confidence
            
            return {
                'rotation': optimal_rotation,
                'x_position': optimal_x,
                'y_position': optimal_y,
                'value': value.squeeze(-1),
                'confidence': overall_confidence
            }

    def get_placement_goal_vector(self, state):
        """
        Get optimal placement as a vector that can be used as goal for the actor
        Returns:
            goal_vector: Tensor of shape (batch_size, goal_dim) encoding the optimal placement
        """
        optimal_placement = self.get_optimal_placement(state)
        
        batch_size = state.shape[0]
        device = state.device
        
        # Encode placement as a concatenated vector
        # [rotation_one_hot(4) + x_position_one_hot(10) + y_position_one_hot(20) + value(1) + confidence(1)]
        goal_dim = 4 + 10 + 20 + 1 + 1  # 36 total
        goal_vector = torch.zeros(batch_size, goal_dim, device=device)
        
        # One-hot encode rotation (indices 0-3)
        rot_indices = optimal_placement['rotation']
        goal_vector[torch.arange(batch_size), rot_indices] = 1.0
        
        # One-hot encode x position (indices 4-13)
        x_indices = optimal_placement['x_position'] + 4
        goal_vector[torch.arange(batch_size), x_indices] = 1.0
        
        # One-hot encode y position (indices 14-33)  
        y_indices = optimal_placement['y_position'] + 14
        goal_vector[torch.arange(batch_size), y_indices] = 1.0
        
        # Add value and confidence (indices 34-35)
        goal_vector[:, 34] = optimal_placement['value']
        goal_vector[:, 35] = optimal_placement['confidence']
        
        return goal_vector
