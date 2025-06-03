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
        if not self.headless:
            pygame.init()
        else:
            # Training headless: disable debug logging to file
            logger.setLevel(logging.WARNING)
        
        # Action space: rotation (0-3) × column (0-9)
        self.action_space = spaces.MultiDiscrete([4, 10])
        
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
        })
        
        # Initialize game components

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
            'current_y': curr_y
        }
    
    def _get_reward(self, lines_cleared, game_over, new_positions=None):
        # 1. Line-clear + time penalty
        bases = {1:10, 2:20, 3:40, 4:80}
        reward = bases.get(lines_cleared, 0) * (self.game.level + 1)
        
        # reward += -0.01   # small time penalty

        # 2. Game-over check
        if game_over:
            return reward - 20

        # 3. Compute features
        grid = create_grid(self.player.locked_positions)
        col_heights = [  # from bottom (row 19) up
            next((r for r in range(20) if grid[r][c] != (0, 0, 0)), 20)
            for c in range(10)
        ]
        max_height = max(col_heights)  # Only consider the highest column
        curr = {
            "holes": sum(
                1 for c in range(10)
                for r in range(col_heights[c] + 1, 20)
                if grid[r][c] == (0, 0, 0)
            ),
            "max_h": max_height,  # Use max height instead of aggregate height
            "bump": sum(abs(col_heights[i] - col_heights[i + 1]) for i in range(9)),
        }
        
        # 4. Delta-based shaping
        prev = self.prev_features if hasattr(self, 'prev_features') else None
        if prev:
            reward += 0.02 * (prev["holes"] - curr["holes"])
            reward += 0.4 * (prev["max_h"] - curr["max_h"])  # Adjust for max height
            reward += 0.005 * (prev["bump"] - curr["bump"])

        # 5. Store for next step
        self.prev_features = curr
        
        # adjacency bonus for newly locked blocks
        if new_positions:
            reward += self._adjacency_reward(new_positions)
            # height-preservation bonus: no increase in max column height
            if prev and curr['max_h'] <= prev['max_h']:
                reward += 0.01
            # position bonus: reward if highest block of new piece is above previous max height
            highest_y = min(y for _, y in new_positions)
            if prev and highest_y < prev['max_h']:
                reward += 0.01 * (prev['max_h'] - highest_y)
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
        # decode placement action (supports int or sequence)
        if isinstance(action, (list, tuple, np.ndarray)):
            rot, x = int(action[0]), int(action[1])
        else:
            idx = int(action)
            rot, x = divmod(idx, 10)
        # set piece orientation and column
        piece = self.player.current_piece
        piece.rotation = rot
        piece.x = x
        # drop until collision (but first ensure spawn placement is valid)
        grid = create_grid(self.player.locked_positions)
        # invalid placement: end episode with penalty
        if not valid_space(piece, grid):
            obs = self._get_observation()
            return obs, -10.0, True, False, {'invalid_move': True}
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
    
    def close(self):
        """Clean up resources"""
        if self.game is not None:
            self.game = None
        if self.surface is not None:
            self.surface = None
        if not self.headless:
            pygame.quit()
