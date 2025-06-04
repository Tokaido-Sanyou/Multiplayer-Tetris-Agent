import gym
import numpy as np
from gym import spaces
import pygame
import random
from .game import Game
from .player import Player
from .block_pool import BlockPool
from .utils import create_grid, check_lost
from .piece_utils import valid_space, convert_shape_format
from .constants import shapes, shape_colors, s_width, s_height
import time
import logging
from typing import Dict, Tuple, List, Any

class DualAgentTetrisEnv(gym.Env):
    """
    Dual-Agent Tetris Environment for two AI agents playing simultaneously
    Now uses 8-action space consistently
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, headless: bool = False, max_steps: int = 10000, action_space_size: int = 8):
        super(DualAgentTetrisEnv, self).__init__()
        
        self.headless = headless
        self.max_steps = max_steps
        self.action_space_size = action_space_size
        
        # Initialize pygame
        if not self.headless:
            pygame.init()
            self.surface = pygame.display.set_mode((s_width, s_height))
            pygame.display.set_caption("Dual Agent Tetris")
        else:
            self.surface = pygame.Surface((s_width, s_height))
        
        # Action space: 8 actions (0-7)
        # 0-6: Movement/rotation actions, 7: Hold
        self.action_space = spaces.Discrete(action_space_size)
        
        # Observation space for each agent (extended with opponent info)
        self.observation_space = spaces.Dict({
            # Own game state
            'grid': spaces.Box(low=0, high=2, shape=(20, 10), dtype=np.int8),
            'next_piece': spaces.Box(low=0, high=7, shape=(), dtype=np.int8),
            'hold_piece': spaces.Box(low=0, high=7, shape=(), dtype=np.int8),
            'current_shape': spaces.Box(low=0, high=7, shape=(), dtype=np.int8),
            'current_rotation': spaces.Box(low=0, high=3, shape=(), dtype=np.int8),
            'current_x': spaces.Box(low=0, high=9, shape=(), dtype=np.int8),
            'current_y': spaces.Box(low=-4, high=19, shape=(), dtype=np.int8),
            'can_hold': spaces.Box(low=0, high=1, shape=(), dtype=np.int8),
            
            # Opponent game state (key opponent metadata)
            'opponent_grid': spaces.Box(low=0, high=2, shape=(20, 10), dtype=np.int8),
            'opponent_score': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            'score_diff': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'opponent_lines': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            'opponent_level': spaces.Box(low=1, high=20, shape=(), dtype=np.float32),
            'opponent_max_height': spaces.Box(low=0, high=20, shape=(), dtype=np.float32),
            'opponent_holes': spaces.Box(low=0, high=200, shape=(), dtype=np.float32),
            'opponent_danger': spaces.Box(low=0, high=1, shape=(), dtype=np.float32),  # 1 if opponent in danger
        })
        
        # Initialize game components
        self.game = None
        self.block_pool = None
        self.player1 = None
        self.player2 = None
        self.episode_steps = 0
        
        # Track episode statistics
        self.episode_stats = {
            'player1': {'score': 0, 'lines': 0, 'pieces': 0},
            'player2': {'score': 0, 'lines': 0, 'pieces': 0}
        }
        
        # Initialize pygame clock
        self.clock = pygame.time.Clock()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator."""
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def reset(self) -> Tuple[Dict, Dict]:
        """
        Reset environment and return initial observations for both agents
        Returns:
            Tuple of (player1_obs, player2_obs)
        """
        # Initialize shared block pool for consistent piece sequence
        self.block_pool = BlockPool()
        
        # Initialize players
        self.player1 = Player(self.block_pool, is_player_one=True)
        self.player2 = Player(self.block_pool, is_player_one=False)
        
        # Reset episode tracking
        self.episode_steps = 0
        self.episode_stats = {
            'player1': {'score': 0, 'lines': 0, 'pieces': 0},
            'player2': {'score': 0, 'lines': 0, 'pieces': 0}
        }
        
        # Get initial observations
        player1_obs = self._get_observation(self.player1, self.player2)
        player2_obs = self._get_observation(self.player2, self.player1)
        
        return player1_obs, player2_obs

    def step(self, actions: Tuple[int, int]) -> Tuple[Tuple[Dict, Dict], Tuple[float, float], Tuple[bool, bool], Dict]:
        """
        Execute one step for both agents
        Args:
            actions: Tuple of (player1_action, player2_action) - each in range 0-7
        Returns:
            Tuple of (observations, rewards, dones, info)
        """
        action1, action2 = actions
        self.episode_steps += 1
        
        # Validate actions are in 8-action space
        action1 = max(0, min(7, int(action1)))
        action2 = max(0, min(7, int(action2)))
        
        # Store previous scores for reward calculation
        prev_score1 = self.player1.score
        prev_score2 = self.player2.score
        
        # Execute actions for both players
        reward1, info1 = self._execute_action(self.player1, action1, player_id=1)
        reward2, info2 = self._execute_action(self.player2, action2, player_id=2)
        
        # Check for game over conditions
        done1 = check_lost(self.player1.locked_positions)
        done2 = check_lost(self.player2.locked_positions)
        
        # Episode ends if either player loses or max steps reached
        episode_done = done1 or done2 or self.episode_steps >= self.max_steps
        
        # Calculate competitive rewards
        score_diff1 = self.player1.score - prev_score1
        score_diff2 = self.player2.score - prev_score2
        
        # Add competitive component to rewards
        competitive_bonus = 0.1
        reward1 += competitive_bonus * (score_diff1 - score_diff2)
        reward2 += competitive_bonus * (score_diff2 - score_diff1)
        
        # Penalty for losing
        if done1 and not done2:
            reward1 -= 50
            reward2 += 20
        elif done2 and not done1:
            reward2 -= 50
            reward1 += 20
        
        # Get new observations
        obs1 = self._get_observation(self.player1, self.player2)
        obs2 = self._get_observation(self.player2, self.player1)
        
        # Combine info
        info = {
            'player1': info1,
            'player2': info2,
            'episode_steps': self.episode_steps,
            'game_over': episode_done,
            'winner': 1 if done2 and not done1 else (2 if done1 and not done2 else 0)
        }
        
        return (obs1, obs2), (reward1, reward2), (episode_done, episode_done), info

    def _execute_action(self, player: Player, action: int, player_id: int) -> Tuple[float, Dict]:
        """
        Execute 8-action space action for a specific player
        Args:
            player: Player object
            action: Action to execute (0-7)
            player_id: Player identifier (1 or 2)
        Returns:
            Tuple of (reward, info)
        """
        prev_score = player.score
        prev_lines = self.episode_stats[f'player{player_id}']['lines']
        
        # Map 8-action space to game actions
        if action == 7:  # Hold piece
            if player.can_hold:
                player.action_handler.hold_piece()
        else:  # Movement/rotation actions (0-6)
            # Simple mapping: action 0-6 represents different placement strategies
            # For now, map to rotation and column placement
            rotation = action % 4  # 0-3 rotations
            column = (action * 2) % 10  # Spread across columns
            
            # Try to place piece at specified rotation and column
            success = self._try_place_piece(player, rotation, column)
            if success:
                self.episode_stats[f'player{player_id}']['pieces'] += 1
        
        # Update player (handle line clearing, etc.)
        lines_cleared = player.update(0.5, 1)  # Use default fall speed and level
        
        # Calculate reward based on score increase and lines cleared
        score_increase = player.score - prev_score
        reward = score_increase * 0.1  # Scale score reward
        
        if lines_cleared > 0:
            # Bonus for clearing lines
            line_bonuses = {1: 10, 2: 25, 3: 50, 4: 100}
            reward += line_bonuses.get(lines_cleared, 0)
            self.episode_stats[f'player{player_id}']['lines'] += lines_cleared
        
        # Small time penalty to encourage faster play
        reward -= 0.01
        
        info = {
            'score': player.score,
            'lines_cleared': lines_cleared,
            'total_lines': self.episode_stats[f'player{player_id}']['lines'],
            'pieces_placed': self.episode_stats[f'player{player_id}']['pieces'],
            'valid_move': True,
            'action': action  # Log the action taken
        }
        
        return reward, info

    def _try_place_piece(self, player: Player, target_rotation: int, target_column: int) -> bool:
        """
        Try to place the current piece at specified rotation and column
        Args:
            player: Player object
            target_rotation: Target rotation (0-3)
            target_column: Target column (0-9)
        Returns:
            True if placement was successful
        """
        if not player.current_piece:
            return False
        
        # Create a copy of the current piece to manipulate
        piece = player.current_piece
        original_rotation = piece.rotation
        original_x = piece.x
        original_y = piece.y
        
        # Set target rotation
        while piece.rotation != target_rotation:
            piece.rotation = (piece.rotation + 1) % len(piece.shape)
        
        # Set target column (adjust for piece width)
        piece.x = target_column
        
        # Find valid y position (drop from top)
        grid = create_grid(player.locked_positions)
        piece.y = 0
        
        # Drop piece down until it can't move further
        while valid_space(piece, grid):
            piece.y += 1
        piece.y -= 1  # Move back to last valid position
        
        # Check if placement is valid
        if valid_space(piece, grid) and piece.y >= 0:
            # Lock the piece in place
            shape_pos = convert_shape_format(piece)
            for pos in shape_pos:
                player.locked_positions[pos] = piece.color
            
            # Get next piece
            player.current_block_index += 1
            player.block_pool.ensure_blocks_ahead(player.current_block_index)
            player.current_piece = player.next_pieces[0]
            
            from .utils import get_shape_from_index
            player.next_pieces = [get_shape_from_index(idx) for idx in 
                                  player.block_pool.get_next_blocks(player.current_block_index)]
            player.can_hold = True
            
            return True
        else:
            # Restore original piece state if placement failed
            piece.rotation = original_rotation
            piece.x = original_x
            piece.y = original_y
            return False

    def _calculate_opponent_metadata(self, player: Player) -> Dict[str, float]:
        """
        Calculate comprehensive opponent metadata for enhanced observation
        Args:
            player: Opponent player object
        Returns:
            Dictionary with opponent game state analysis
        """
        grid = create_grid(player.locked_positions)
        
        # Column heights analysis
        col_heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if grid[row][col] != (0, 0, 0):
                    height = 20 - row
                    break
            col_heights.append(height)
        
        max_height = max(col_heights) if col_heights else 0
        avg_height = np.mean(col_heights) if col_heights else 0
        height_variance = np.var(col_heights) if col_heights else 0
        
        # Holes analysis
        holes = 0
        for col in range(10):
            found_block = False
            for row in range(20):
                if grid[row][col] != (0, 0, 0):
                    found_block = True
                elif found_block and grid[row][col] == (0, 0, 0):
                    holes += 1
        
        # Danger analysis
        danger_threshold = 16  # Height above which player is in danger
        critical_threshold = 18  # Critical danger zone
        is_in_danger = max_height > danger_threshold
        is_critical = max_height > critical_threshold
        
        # Line opportunities
        full_lines = 0
        for row in range(20):
            if all(grid[row][col] != (0, 0, 0) for col in range(10)):
                full_lines += 1
        
        return {
            'max_height': float(max_height),
            'avg_height': float(avg_height),
            'height_variance': float(height_variance),
            'holes': float(holes),
            'danger': float(is_in_danger),
            'critical': float(is_critical),
            'full_lines': float(full_lines),
            'bumpiness': float(sum(abs(col_heights[i] - col_heights[i+1]) for i in range(9)))
        }

    def _get_observation(self, player: Player, opponent: Player) -> Dict:
        """
        Get observation for a specific player (now includes rich opponent metadata)
        Args:
            player: The player for whom to get observation
            opponent: The opponent player
        Returns:
            Observation dictionary with extended opponent information
        """
        # Get grid state for player
        grid = create_grid(player.locked_positions)
        grid_obs = np.zeros((20, 10), dtype=np.int8)
        for i in range(20):
            for j in range(10):
                if grid[i][j] != (0, 0, 0):
                    grid_obs[i][j] = 1
        
        # Overlay current falling piece as 2s
        if player.current_piece:
            for x, y in convert_shape_format(player.current_piece):
                if 0 <= y < 20 and 0 <= x < 10:
                    grid_obs[y][x] = 2
        
        # Get opponent grid
        opponent_grid = create_grid(opponent.locked_positions)
        opponent_grid_obs = np.zeros((20, 10), dtype=np.int8)
        for i in range(20):
            for j in range(10):
                if opponent_grid[i][j] != (0, 0, 0):
                    opponent_grid_obs[i][j] = 1
        
        if opponent.current_piece:
            for x, y in convert_shape_format(opponent.current_piece):
                if 0 <= y < 20 and 0 <= x < 10:
                    opponent_grid_obs[y][x] = 2
        
        # Get opponent metadata
        opponent_metadata = self._calculate_opponent_metadata(opponent)
        
        # Encode pieces as shape IDs
        next_idx = shapes.index(player.next_pieces[0].shape) + 1 if player.next_pieces else 0
        hold_idx = shapes.index(player.hold_piece.shape) + 1 if player.hold_piece else 0
        
        # Current piece metadata
        curr_shape = shapes.index(player.current_piece.shape) + 1 if player.current_piece else 0
        curr_rot = player.current_piece.rotation if player.current_piece else 0
        curr_x = player.current_piece.x if player.current_piece else 0
        curr_y = player.current_piece.y if player.current_piece else 0
        
        return {
            # Own state
            'grid': grid_obs,
            'next_piece': next_idx,
            'hold_piece': hold_idx,
            'current_shape': curr_shape,
            'current_rotation': curr_rot,
            'current_x': curr_x,
            'current_y': curr_y,
            'can_hold': int(player.can_hold),
            
            # Opponent state (rich metadata)
            'opponent_grid': opponent_grid_obs,
            'opponent_score': float(opponent.score),
            'score_diff': float(player.score - opponent.score),
            'opponent_lines': float(getattr(opponent, 'lines_cleared', 0)),
            'opponent_level': float(getattr(opponent, 'level', 1)),
            'opponent_max_height': opponent_metadata['max_height'],
            'opponent_holes': opponent_metadata['holes'],
            'opponent_danger': opponent_metadata['danger']
        }

    def render(self, mode='human'):
        """Render the dual-agent game"""
        if self.headless:
            return
        
        # Clear surface
        self.surface.fill((33, 29, 29))
        
        # Draw grids and game state
        from .utils import draw_window, draw_next_pieces, draw_hold_piece
        
        # Get grids
        p1_grid = create_grid(self.player1.locked_positions)
        p2_grid = create_grid(self.player2.locked_positions)
        
        # Draw window with both grids
        draw_window(self.surface, p1_grid, p2_grid,
                   self.player1.current_piece, self.player2.current_piece,
                   self.player1.score, self.player2.score,
                   1, 0.5, add=int(s_width/2))  # Level 1, fall speed 0.5
        
        # Draw next and hold pieces
        draw_next_pieces(self.player1.next_pieces, self.surface, 0)
        draw_next_pieces(self.player2.next_pieces, self.surface, 1)
        draw_hold_piece(self.player1.hold_piece, self.surface, 0)
        draw_hold_piece(self.player2.hold_piece, self.surface, 1)
        
        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        """Close the environment"""
        if not self.headless:
            pygame.quit()

    def get_winner(self) -> int:
        """
        Get the winner of the current game
        Returns:
            0 for tie, 1 for player1, 2 for player2
        """
        done1 = check_lost(self.player1.locked_positions)
        done2 = check_lost(self.player2.locked_positions)
        
        if done1 and done2:
            # Both lost, winner based on score
            return 1 if self.player1.score > self.player2.score else (2 if self.player2.score > self.player1.score else 0)
        elif done1:
            return 2
        elif done2:
            return 1
        else:
            # Game ongoing, no winner yet
            return 0 