"""
Exploration Actor: systematically tries different placements for data collection
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
from copy import deepcopy

# Handle both direct execution and module import
try:
    from ..piece import Piece
    from ..piece_utils import valid_space, convert_shape_format
    from ..utils import create_grid, clear_rows
    from ..constants import shapes
except ImportError:
    # Direct execution - add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from piece import Piece
    from piece_utils import valid_space, convert_shape_format
    from utils import create_grid, clear_rows
    from constants import shapes

class ExplorationActor:
    """
    Exploration actor that systematically tries different piece placements
    to collect terminal state rewards for state model training
    """
    def __init__(self, env, exploration_strategy='systematic'):
        self.env = env
        self.exploration_strategy = exploration_strategy
        self.placement_cache = {}  # Cache valid placements for each piece type
        
    def get_valid_placements(self, piece_shape, grid):
        """
        Get all valid placements for a given piece shape on the current grid
        Returns list of (rotation, x_position) tuples
        """
        # Convert grid to numpy array if it's a list
        if isinstance(grid, list):
            grid = np.array(grid)
        
        cache_key = (piece_shape, tuple(grid.flatten()))
        if cache_key in self.placement_cache:
            cached_result = self.placement_cache[cache_key]
            print(f"(cached: {len(cached_result)})", end="")
            return cached_result
            
        valid_placements = []
        total_attempts = 40  # 4 rotations * 10 x positions
        attempts_made = 0
        
        # Try all rotations (0-3) and x positions (0-9)
        for rotation in range(4):
            for x_pos in range(10):
                attempts_made += 1
                if attempts_made % 10 == 0:  # Progress every 10 attempts
                    print(f"({attempts_made}/{total_attempts})", end="")
                
                # Simulate piece placement
                if self._is_valid_placement(piece_shape, rotation, x_pos, grid):
                    valid_placements.append((rotation, x_pos))
        
        self.placement_cache[cache_key] = valid_placements
        
        # Debug output (reduced)
        print(f"({len(valid_placements)}/{total_attempts})", end="")
            
        return valid_placements
    
    def _is_valid_placement(self, piece_shape, rotation, x_pos, grid):
        """
        Check if a piece placement is valid using simplified game mechanics
        """
        try:
            # Simplified validation - just check if the piece can fit at x_pos=x_pos, y=0
            # This is much faster than the full simulation
            
            # Basic bounds check
            if x_pos < 0 or x_pos >= 10:
                return False
                
            # Get the piece shape (0-indexed)
            if piece_shape < 1 or piece_shape > 7:
                return False
                
            shape = shapes[piece_shape - 1]  # piece_shape is 1-indexed
            
            # Get the rotated shape
            if rotation < 0 or rotation >= len(shape):
                return False
                
            piece_format = shape[rotation % len(shape)]
            
            # Check if piece can fit at the top (simplified check)
            for row_idx, row in enumerate(piece_format):
                for col_idx, cell in enumerate(row):
                    if cell == '0':  # This is a piece block
                        final_x = x_pos + col_idx - 2  # Adjust for piece center
                        final_y = row_idx
                        
                        # Check bounds
                        if final_x < 0 or final_x >= 10 or final_y >= 20:
                            return False
                            
                        # Check collision (only if in valid grid bounds)
                        # Fix the numpy array comparison issue
                        if final_y >= 0:
                            grid_cell = grid[final_y][final_x]
                            # Check if grid cell is not empty (not black)
                            if hasattr(grid_cell, '__len__') and len(grid_cell) == 3:
                                # Color tuple comparison
                                if grid_cell != (0, 0, 0):
                                    return False
                            elif grid_cell != 0:
                                # Simple value comparison  
                                return False
                            
            return True
            
        except Exception as e:
            # Debug: Print first few exceptions to see what's going wrong
            if not hasattr(self, '_debug_exception_count'):
                self._debug_exception_count = 0
            if self._debug_exception_count < 2:  # Reduced from 3 to 2
                print(f"[EXC:{e}]", end="")
                self._debug_exception_count += 1
            return False
    
    def collect_placement_data(self, num_episodes=100):
        """
        Collect data by trying different placements and recording ONLY terminal rewards
        State model is meant to output the final destination state, so we only report terminal states
        Returns list of (state, placement, terminal_reward, resulting_state) tuples
        """
        placement_data = []
        
        print(f"Starting placement data collection for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            print(f"Episode {episode+1}/{num_episodes}...")
            try:
                obs = self.env.reset()
                episode_placements = 0
                max_steps_per_episode = 20  # Increased to allow more complete episodes
                
                for step in range(max_steps_per_episode):
                    print(f"  Step {step+1}/{max_steps_per_episode}", end="")
                    try:
                        # Get current state
                        current_state = self._obs_to_state_vector(obs)
                        print(f" - got state", end="")
                        
                        # Extract next piece from one-hot encoding
                        if np.any(obs['next_piece']):
                            next_piece_idx = np.argmax(obs['next_piece'])
                        else:
                            next_piece_idx = 0
                        piece_shape = next_piece_idx + 1  # Convert to 1-indexed
                        print(f" - piece {piece_shape}", end="")
                        
                        # Simulate multiple terminal placements for this state
                        print(f" - simulating terminal placements", end="")
                        
                        # Generate 2-4 terminal placement simulations per step
                        num_placements = random.randint(2, 4)
                        for i in range(num_placements):
                            # Create a placement that leads to a terminal state
                            terminal_rotation = random.randint(0, 3)
                            terminal_x_pos = random.randint(0, 9)
                            
                            # Simulate the terminal state after this placement
                            terminal_state, terminal_reward = self._simulate_terminal_placement(
                                current_state, terminal_rotation, terminal_x_pos, piece_shape
                            )
                            
                            # Only record if we have a valid terminal state
                            if terminal_state is not None:
                                placement_data.append({
                                    'state': current_state,
                                    'placement': (terminal_rotation, terminal_x_pos),
                                    'terminal_reward': terminal_reward,
                                    'resulting_state': terminal_state  # This is the final terminal state
                                })
                                episode_placements += 1
                        
                        print(f" - added {num_placements} terminal placements", end="")
                        
                        # Take an action to continue the episode (but we only care about terminal states)
                        action_one_hot = np.zeros(8, dtype=np.int8)
                        action_one_hot[random.randint(0, 7)] = 1
                        print(f" - taking action", end="")
                        obs, reward, done, info = self.env.step(action_one_hot)
                        print(f" - done={done}")
                        
                        # If episode is done, we've reached a terminal state
                        if done:
                            print(f"    Episode terminated at step {step+1} - recording final terminal state")
                            # Record the final terminal state
                            final_state = self._obs_to_state_vector(obs)
                            placement_data.append({
                                'state': current_state,
                                'placement': (0, 0),  # Dummy placement since episode ended
                                'terminal_reward': reward,  # The actual terminal reward
                                'resulting_state': final_state
                            })
                            break
                            
                    except Exception as e:
                        print(f"    Error in step {step+1}: {e}")
                        break
                    
                print(f"  Episode {episode+1} completed with {episode_placements} terminal placements")
                
            except Exception as e:
                print(f"  Error in episode {episode+1}: {e}")
                continue
        
        print(f"Collection completed. Total terminal placements: {len(placement_data)}")
        return placement_data

    def _simulate_terminal_placement(self, current_state, rotation, x_pos, piece_shape):
        """
        Simulate a terminal placement and return the final terminal state and reward
        Args:
            current_state: Current game state vector
            rotation: Target rotation (0-3)
            x_pos: Target x position (0-9)
            piece_shape: Piece shape index
        Returns:
            (terminal_state, terminal_reward) or (None, 0) if invalid
        """
        try:
            # Create a mock terminal state based on the placement
            terminal_state = current_state.copy()
            
            # Simulate the effects of the placement on the state
            # This is a simplified simulation - in practice, you'd run the full game mechanics
            
            # Calculate terminal reward based on placement quality
            terminal_reward = self._evaluate_terminal_placement(rotation, x_pos, piece_shape)
            
            # Modify the state to reflect the terminal placement
            # Add some random noise to simulate state changes
            state_noise = np.random.randn(len(terminal_state)) * 0.01
            terminal_state = terminal_state + state_noise
            
            return terminal_state, terminal_reward
            
        except Exception as e:
            print(f"[SIM_ERR:{e}]", end="")
            return None, 0
    
    def _evaluate_terminal_placement(self, rotation, x_pos, piece_shape):
        """
        Evaluate the quality of a terminal placement
        Returns a terminal reward value
        """
        # Simple heuristic: prefer center positions and specific rotations
        center_bonus = -abs(x_pos - 4.5) * 5  # Prefer center positions
        rotation_bonus = -abs(rotation - 1) * 2  # Prefer rotation 1
        piece_bonus = piece_shape * 2  # Different pieces get different bonuses
        
        # Add some randomness to simulate game variability
        random_bonus = random.uniform(-10, 10)
        
        terminal_reward = center_bonus + rotation_bonus + piece_bonus + random_bonus
        
        # Clamp reward to reasonable range
        return max(-100, min(100, terminal_reward))
    
    def _obs_to_state_vector(self, obs):
        """Convert simplified observation dict to flattened state vector (NEW: 410-dimensional)"""
        # Flatten the simplified grids
        current_piece_flat = obs['current_piece_grid'].flatten()  # 20*10 = 200
        empty_grid_flat = obs['empty_grid'].flatten()  # 20*10 = 200
        
        # Get one-hot encoding and metadata
        next_piece = obs['next_piece']  # 7 values (removed hold piece)
        metadata = np.array([
            obs['current_rotation'],
            obs['current_x'], 
            obs['current_y']
        ])  # 3 values
        
        # Concatenate all components: 200 + 200 + 7 + 3 = 410 (removed piece_grids + hold_piece)
        return np.concatenate([
            current_piece_flat, 
            empty_grid_flat,
            next_piece,
            metadata
        ])
    
    def _simulate_placement(self, obs, rotation, x_pos):
        """
        Simulate a piece placement and return the resulting state vector
        """
        try:
            # For now, return the current state as a simplified implementation
            # This can be enhanced later with proper piece placement simulation
            return self._obs_to_state_vector(obs)
            
        except Exception as e:
            # If simulation fails, return original state
            return self._obs_to_state_vector(obs)
    
    def _evaluate_terminal_state(self, state):
        """
        Evaluate the quality of a terminal state after piece placement
        """
        # Extract grid components from the 1817-dimensional state vector
        # piece_grids: 0:1400, current_piece: 1400:1600, empty_grid: 1600:1800
        piece_grids = state[:1400].reshape(7, 20, 10)
        current_piece_grid = state[1400:1600].reshape(20, 10)
        
        # Combine all grids to get overall occupancy
        combined_grid = np.any(piece_grids, axis=0).astype(np.int8)
        combined_grid = combined_grid + current_piece_grid
        combined_grid = np.clip(combined_grid, 0, 1)
        
        # Calculate heuristic features
        col_heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if combined_grid[row][col] > 0:
                    height = 20 - row
                    break
            col_heights.append(height)
        
        # Calculate reward based on heuristics
        max_height = max(col_heights) if col_heights else 0
        holes = self._count_holes(combined_grid)
        bumpiness = sum(abs(col_heights[i] - col_heights[i+1]) for i in range(9))
        
        # Reward function (negative because we want to minimize these)
        reward = -0.5 * max_height - 10 * holes - 0.1 * bumpiness
        
        return reward
    
    def _count_holes(self, grid):
        """Count holes in the grid"""
        holes = 0
        for col in range(10):
            found_block = False
            for row in range(20):
                if grid[row][col] > 0:
                    found_block = True
                elif found_block and grid[row][col] == 0:
                    holes += 1
        return holes
