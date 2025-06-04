import gym
import numpy as np
from gym import spaces
import pygame
import random
from .game import Game, Piece
from .utils import create_grid, check_lost, count_holes
from .piece_utils import valid_space, convert_shape_format
from .constants import shapes, shape_colors, s_width, s_height
import time
import logging

# Configure debug logging to file
logging.basicConfig(level=logging.DEBUG,
                    filename='tetris_debug.log',
                    filemode='w',
                    format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

class TetrisEnv(gym.Env):
    """
    Custom Tetris Environment that follows gym interface
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, single_player=False, headless=False):
        super(TetrisEnv, self).__init__()

        # account for training without GUI
        self.headless = headless
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
        
        # Action space:
        # 0-39 : placement index = rotation*10 + column
        # 40   : hold current piece
        self.action_space = spaces.Discrete(41)
        
        # Define observation space
        # Grid: 20x10 matrix (0 for empty, 1 for locked, 2 for current piece)
        # Next piece: scalar shape ID (1-7, 0 if none)
        # Hold piece: scalar shape ID (1-7, 0 if none)
        # Current piece: shape ID, rotation, x and y coordinates
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=2, shape=(20, 10), dtype=np.int8),
            'next_piece': spaces.Box(low=0, high=7, shape=(), dtype=np.int8),
            'hold_piece': spaces.Box(low=0, high=7, shape=(), dtype=np.int8),
            'current_shape': spaces.Box(low=0, high=7, shape=(), dtype=np.int8),
            'current_rotation': spaces.Box(low=0, high=3, shape=(), dtype=np.int8),
            'current_x': spaces.Box(low=0, high=9, shape=(), dtype=np.int8),
            'current_y': spaces.Box(low=-4, high=19, shape=(), dtype=np.int8),
            'can_hold': spaces.Box(low=0, high=1, shape=(), dtype=np.int8)
        })
        
        # Initialize rendering surface
        if not self.headless:
            # Create a display window for real-time visualization
            self.surface = pygame.display.set_mode((s_width, s_height))
            pygame.display.set_caption("Tetris RL")
        else:
            # Off-screen surface for headless mode
            self.surface = pygame.Surface((s_width, s_height))
        
        self.game = None  # Will be initialized in reset()
        self.player = None  # Will be set in reset()
        self.single_player = single_player
        
        # Initialize pygame clock
        self.clock = pygame.time.Clock()
        
        # Initialize episode tracking
        self.episode_steps = 0
        self.max_steps = 25000  # Maximum steps per episode
        self.gravity_interval = 5  # agent steps per gravity drop

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator."""
        # Note: Pygame itself doesn't have a global random seed affecting game logic directly in this setup.
        # Seeding numpy and random should cover piece generation and other stochastic elements if they use it.
        np.random.seed(seed)
        random.seed(seed)
        # If the action space needs seeding (e.g., for sampling random actions):
        # self.action_space.seed(seed) 
        return [seed]
        
    def _get_observation(self):
        """Convert game state to observation space format"""
        # Get grid state
        grid = create_grid(self.player.locked_positions)
        grid_obs = np.zeros((20, 10), dtype=np.int8)
        for i in range(20):
            for j in range(10):
                if grid[i][j] != (0, 0, 0):
                    grid_obs[i][j] = 1
        
        # Overlay current falling piece as 2s
        if self.player.current_piece:
            for x, y in convert_shape_format(self.player.current_piece):
                if 0 <= y < 20 and 0 <= x < 10:
                    grid_obs[y][x] = 2
        
        # Encode next and hold pieces as shape IDs (1-7, 0 if none)
        next_idx = shapes.index(self.player.next_pieces[0].shape) + 1 if self.player.next_pieces else 0
        hold_idx = shapes.index(self.player.hold_piece.shape) + 1 if self.player.hold_piece else 0
        
        # Current piece metadata
        curr_shape = shapes.index(self.player.current_piece.shape) + 1 if self.player.current_piece else 0
        curr_rot = self.player.current_piece.rotation if self.player.current_piece else 0
        curr_x = self.player.current_piece.x if self.player.current_piece else 0
        curr_y = self.player.current_piece.y if self.player.current_piece else 0
        return {
            'grid': grid_obs,
            'next_piece': next_idx,
            'hold_piece': hold_idx,
            'current_shape': curr_shape,
            'current_rotation': curr_rot,
            'current_x': curr_x,
            'current_y': curr_y,
            'can_hold': int(self.player.can_hold)
        }
    
    def _get_reward(self, lines_cleared, game_over, new_positions=None):
        # 1. Line-clear + time penalty
        bases = {1:100, 2:300, 3:700, 4:1500}
        reward = bases.get(lines_cleared, 0) * (self.game.level + 1)
        
        # 2. Game-over check
        if game_over:
            return reward - 500

        # 3. Compute features
        grid = create_grid(self.player.locked_positions)
        col_heights = [  # from bottom (row 19) up
            next((r for r in range(20) if grid[r][c] != (0, 0, 0)), 20)
            for c in range(10)
        ]
        max_height = max(col_heights)
        min_height = min(col_heights)
        
        # Calculate column distribution metrics
        mean_height = sum(20 - h for h in col_heights) / 10.0
        height_variance = sum((20 - h - mean_height) ** 2 for h in col_heights) / 10.0
        height_std_dev = height_variance ** 0.5
        
        # Calculate height differences between adjacent columns
        height_diffs = [abs(col_heights[i] - col_heights[i+1]) for i in range(9)]
        max_height_diff = max(height_diffs)
        total_height_diff = sum(height_diffs)
        
        # Calculate empty and full column penalties
        empty_columns = sum(1 for h in col_heights if h == 20)
        full_columns = sum(1 for h in col_heights if h <= 5)  # Columns that are very high
        
        # Calculate holes and bumpiness
        holes = sum(
            1 for c in range(10)
            for r in range(col_heights[c] + 1, 20)
            if grid[r][c] == (0, 0, 0)
        )
        
        # Height-based penalties (exponential with height)
        height_penalty = 2.0 * (1.2 ** max_height)  # Base height penalty
        height_penalty += 1.0 * (1.3 ** (max_height - min_height))  # Penalty for height difference
        
        # Strong penalties for uneven distribution
        distribution_penalty = 15.0 * height_std_dev  # Increased from 8.0
        distribution_penalty += 5.0 * max_height_diff  # Increased penalty for large gaps between columns
        distribution_penalty += 4.0 * (empty_columns ** 2)  # Increased quadratic penalty for empty columns
        distribution_penalty += 4.0 * (full_columns ** 2)  # Increased quadratic penalty for very high columns
        
        # Add penalty for extreme height differences between any columns
        for i in range(10):
            for j in range(i+1, 10):
                diff = abs(col_heights[i] - col_heights[j])
                if diff > 3:  # If height difference is more than 3 blocks
                    distribution_penalty += 2.0 * (diff - 3) ** 2  # Quadratic penalty for large differences
        
        curr = {
            "holes": holes,
            "max_h": max_height,
            "min_h": min_height,
            "height_diff": total_height_diff,
            "std_dev": height_std_dev,
            "empty_cols": empty_columns,
            "full_cols": full_columns,
            "mean_h": mean_height
        }
        
        # Apply penalties
        reward -= 3.0 * curr["holes"]  # Holes penalty
        reward -= height_penalty  # Height penalty
        reward -= distribution_penalty  # Distribution penalties
        reward -= 1.0 * curr["height_diff"]  # Increased penalty for adjacent height differences
        
        # Delta-based shaping
        prev = self.prev_features if hasattr(self, 'prev_features') else None
        if prev:
            # Reward improvements
            reward += 0.5 * (prev["holes"] - curr["holes"])  # Reducing holes
            reward += 1.0 * (prev["max_h"] - curr["max_h"])  # Reducing maximum height
            reward += 1.0 * (curr["min_h"] - prev["min_h"])  # Increasing minimum height
            reward += 1.0 * (prev["height_diff"] - curr["height_diff"])  # Increased reward for reducing height differences
            
            # Strong rewards for distribution improvements
            if curr["std_dev"] < prev["std_dev"]:
                reward += 6.0 * (prev["std_dev"] - curr["std_dev"])  # Doubled reward for improving distribution
            if curr["empty_cols"] < prev["empty_cols"]:
                reward += 4.0 * (prev["empty_cols"] - curr["empty_cols"])  # Doubled reward for using empty columns
            if curr["full_cols"] < prev["full_cols"]:
                reward += 4.0 * (prev["full_cols"] - curr["full_cols"])  # Doubled reward for reducing full columns
            
            # Extra bonus for reducing height while maintaining good distribution
            if curr["holes"] == 0 and curr["max_h"] < prev["max_h"]:
                if curr["std_dev"] < 2.0:  # Very even distribution
                    reward += 10.0 * (prev["max_h"] - curr["max_h"])  # Doubled bonus for good distribution
                elif curr["std_dev"] < 3.0:  # Moderately even distribution
                    reward += 5.0 * (prev["max_h"] - curr["max_h"])
        
        # Store current features for next step
        self.prev_features = curr
        
        # Position-based incentives for new piece placements
        if new_positions:
            # Get placement column information
            piece_cols = set(x for x, _ in new_positions)
            piece_min_col = min(piece_cols)
            piece_max_col = max(piece_cols)
            piece_width = piece_max_col - piece_min_col + 1
            
            # Calculate local height metrics for placement area
            local_heights = col_heights[piece_min_col:piece_max_col+1]
            local_max_height = max(local_heights) if local_heights else 20
            
            # Reward for using more columns
            col_usage_score = len(piece_cols) / 4.0  # Normalize by typical piece width
            reward += 3.0 * col_usage_score  # Increased reward for using multiple columns
            
            # Strong reward for using columns with below-average height
            below_avg_cols = sum(1 for c in piece_cols if col_heights[c] > mean_height)
            reward += 2.0 * below_avg_cols  # Doubled reward for using lower columns
            
            # Penalize high placements more severely
            if local_max_height < 5:  # If placing in top quarter
                reward -= 6.0 * (5 - local_max_height)  # Increased penalty for high placements
            
            # Reward for bridging gaps between columns
            if piece_width > 1:  # Only for pieces spanning multiple columns
                height_diff_before = sum(abs(col_heights[i] - col_heights[i+1]) 
                                       for i in range(piece_min_col, piece_max_col))
                reward += 2.0 * height_diff_before  # Doubled reward for bridging height differences
            
            # Add bonus for placing pieces that improve distribution
            if piece_width > 1:  # For pieces that span multiple columns
                # Calculate local height standard deviation before and after placement
                local_heights_before = [col_heights[i] for i in range(piece_min_col, piece_max_col+1)]
                mean_before = sum(local_heights_before) / len(local_heights_before)
                std_dev_before = (sum((h - mean_before) ** 2 for h in local_heights_before) / len(local_heights_before)) ** 0.5
                
                # If piece improves local distribution, give bonus
                if std_dev_before > 2.0:  # Only if there was significant unevenness
                    reward += 3.0 * std_dev_before  # Reward proportional to how uneven it was
        
        return reward

    
    def _adjacency_reward(self, positions):
        """Compute adjacency bonus: count neighboring locked blocks around the new piece, squared"""
        adj = 0
        # positions: list of (x,y) tuples for the locked piece
        for x, y in positions:
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                if (x+dx, y+dy) in self.player.locked_positions:
                    adj += 0.005
        return adj * adj

    def _count_holes(self):
        """Count number of holes in the grid"""
        holes = 0
        grid = create_grid(self.player.locked_positions)
        for col in range(10):
            found_block = False
            for row in range(20):
                if grid[row][col] != (0, 0, 0):
                    found_block = True
                elif found_block and grid[row][col] == (0, 0, 0):
                    holes += 1
        return holes
    
    def _ensure_single_player_mode(self):
        """Ensure player 2 is completely disabled in single player mode"""
        if self.single_player:
            # Disable player 2's piece
            self.game.player2.current_piece = None
            self.game.player2.next_pieces = []
            self.game.player2.hold_piece = None
            self.game.player2.locked_positions = {}
            
            # Disable player 2's score and level
            self.game.player2.score = 0
            self.game.player2.level = 1
            
            # Disable player 2's block pool
            self.game.player2.block_pool = None

    def step(self, action):
        """Execute one time step: place current piece at (rotation, x) then drop and lock"""
        self.episode_steps += 1
        # decode action
        if isinstance(action, (list, tuple, np.ndarray)):
            # Provided as (rot, col)
            rot, x = int(action[0]), int(action[1])
            idx = rot * 10 + x
        else:
            idx = int(action)
            if idx == 40:
                # HOLD ACTION -------------------------------------------
                self.player.action_handler.hold_piece()
                # small time penalty to discourage excessive holding
                reward = -0.01
                obs = self._get_observation()
                game_over = check_lost(self.player.locked_positions)
                terminated = game_over
                truncated = (self.episode_steps >= self.max_steps)
                info = {
                    'lines_cleared': 0,
                    'score': self.player.score,
                    'level': self.game.level,
                    'episode_steps': self.episode_steps,
                    'piece_placed': False,
                    'piece_held': True
                }
                return obs, reward, terminated, truncated, info
            rot, x = divmod(idx, 10)
        # Place column first
        piece = self.player.current_piece
        piece.x = x

        grid = create_grid(self.player.locked_positions)

        # Rotate toward desired orientation with at most 3 attempts
        if rot != piece.rotation:
            # Decide direction and number of quarter-turns (1-3)
            cw_steps = (rot - piece.rotation) % 4
            ccw_steps = (piece.rotation - rot) % 4
            if cw_steps <= ccw_steps:
                direction, steps = 1, cw_steps
            else:
                direction, steps = -1, ccw_steps

            for _ in range(steps):
                piece.rotate(direction, grid)  # uses SRS wall kicks
                grid = create_grid(self.player.locked_positions)  # refresh grid after each attempt

        # Simple boundary kick: if rotated piece hangs off left/right, shift into bounds
        positions = convert_shape_format(piece)
        xs = [p[0] for p in positions]
        shift = 0
        if min(xs) < 0:
            shift = -min(xs)
        elif max(xs) > 9:
            shift = 9 - max(xs)
        piece.x += shift

        # drop until collision (but first ensure spawn placement is valid)
        grid = create_grid(self.player.locked_positions)
        # invalid placement: apply penalty but don't terminate
        if not valid_space(piece, grid):
            obs = self._get_observation()
            return obs, -10.0, False, False, {'invalid_move': True}
        while valid_space(piece, grid):
            piece.y += 1
        piece.y -= 1
        # trigger lock on placed piece
        self.player.change_piece = True
        # lock piece and compute cleared lines
        prev_locked = set(self.player.locked_positions.keys())
        lines_cleared = self.player.update(self.game.fall_speed, self.game.level)
        new_positions = set(self.player.locked_positions.keys()) - prev_locked
        # episode termination/truncation
        game_over = check_lost(self.player.locked_positions)
        terminated = game_over
        truncated = (self.episode_steps >= self.max_steps)
        # observe and reward
        obs = self._get_observation()
        reward = self._get_reward(lines_cleared, game_over, new_positions)
        info = {
            'lines_cleared': lines_cleared,
            'score': self.player.score,
            'level': self.game.level,
            'episode_steps': self.episode_steps,
            'piece_placed': True
        }
        return obs, reward, terminated, truncated, info
    
    def reset(self):
        """Reset the environment to initial state"""
        # Initialize or reset game
        if self.game is None:
            self.game = Game(self.surface)
        else:
            # Reset game state without creating new surface
            self.game = Game(self.surface)
        
        self.player = self.game.player1
        
        # In single player mode, disable player 2
        if self.single_player:
            self.game.player2.current_piece = None
            self.game.player2.next_pieces = []
            self.game.player2.hold_piece = None
            self.game.player2.locked_positions = {}
        
        # Reset episode tracking and reward shaping state
        self.episode_steps = 0
        self.accum_reward = 0  # initialize accumulated reward component
        self.prev_features = None  # clear feature-history for shaping

        # Debug: log reset state
        logger.debug(f"RESET: Game reset. Episode steps set to {self.episode_steps}")
        logger.debug(f"RESET: Spawn piece: shape={self.player.current_piece.shape}, pos=({self.player.current_piece.x},{self.player.current_piece.y})")
        logger.debug(f"RESET: Locked positions count={len(self.player.locked_positions)}")
        # Get initial observation
        observation = self._get_observation()
        
        # For Gym v0.26+, reset must return (obs, info)
        info = {} # Empty info dict for now, can be populated if needed
        return observation, info
    
    def render(self, mode='human'):
        if self.headless:
            # Skip all rendering in headless mode
            return
        
        try:
            # Pump Pygame events to keep the window responsive
            pygame.event.pump()
            # Sync locked blocks into the game grid before drawing
            self.game.p1_grid = create_grid(self.player.locked_positions)
            self.game.p2_grid = create_grid(self.game.player2.locked_positions)
            """Render the game state"""
            if mode == 'human':
                self.game.draw()
                pygame.display.update()
                # Cap frame rate to 10 FPS (100 ms per step)
                self.clock.tick(10)
        except pygame.error:
            # If we get a pygame error, try to reinitialize the display
            if not self.headless:
                pygame.display.init()
                self.surface = pygame.display.set_mode((s_width, s_height))
                pygame.display.set_caption("Tetris RL")

    def close(self):
        """Clean up resources"""
        if self.game is not None:
            self.game = None
        if self.surface is not None:
            self.surface = None
        # Don't quit Pygame here, just clean up our resources
