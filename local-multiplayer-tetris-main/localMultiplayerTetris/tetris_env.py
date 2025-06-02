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
        
        # Action space:
        # 0: Move Left
        # 1: Move Right
        # 2: Move Down
        # 3: Rotate Clockwise
        # 4: Rotate Counter-clockwise
        # 5: Hard Drop
        # 6: Hold Piece
        # 7: No-op
        self.action_space = spaces.Discrete(8)
        
        # Define observation space
        # Grid: 20x10 matrix (0 for empty, 1 for locked, 2 for current piece)
        # Next piece: scalar shape ID (1-7, 0 if none)
        # Hold piece: scalar shape ID (1-7, 0 if none)
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=2, shape=(20, 10), dtype=np.int8),
            'next_piece': spaces.Box(low=0, high=7, shape=(), dtype=np.int8),
            'hold_piece': spaces.Box(low=0, high=7, shape=(), dtype=np.int8)
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
        
        return {
            'grid': grid_obs,
            'next_piece': next_idx,
            'hold_piece': hold_idx
        }
    
    def _get_reward(self, lines_cleared, game_over):
        # 1. Line-clear + time penalty
        bases = {1:100, 2:200, 3:400, 4:1600}
        reward = bases.get(lines_cleared, 0) * (self.game.level + 1)
        # reward += -0.01   # small time penalty

        # 2. Game-over check
        if game_over:
            return reward - 200

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
            reward += 4.0 * (prev["holes"] - curr["holes"])
            reward += 10 * (prev["max_h"] - curr["max_h"])  # Adjust for max height
            reward += 1.0 * (prev["bump"] - curr["bump"])

        # 5. Store for next step
        self.prev_features = curr

        return reward

    
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
        """Execute one time step within the environment"""
        self.episode_steps += 1
        # Debug: log step start
        logger.debug(
            f"STEP {self.episode_steps}: Action={action}. Current piece pos=({self.player.current_piece.x},{self.player.current_piece.y}). Locked count={len(self.player.locked_positions)}"
        )

        piece_placed = False
        lines_cleared = 0

        # Map action to game action
        if action == 0:  # Move Left
            self.player.action_handler.move_left()
        elif action == 1:  # Move Right
            self.player.action_handler.move_right()
        elif action == 2:  # Soft Drop / Move Down
            prev_y = self.player.current_piece.y
            self.player.action_handler.move_down()
            # If unable to move down, lock piece
            if self.player.current_piece.y == prev_y:
                # Lock piece after soft drop
                self.player.change_piece = True
                lines_cleared = self.player.update(self.game.fall_speed, self.game.level)
                piece_placed = True
                # Reset spawn check for new piece
                self._spawn_checked = False
                # Immediate game-over if locked blocks above grid
                if check_lost(self.player.locked_positions):
                    obs = self._get_observation()
                    rw = self._get_reward(lines_cleared, True)
                    info = {'lines_cleared': lines_cleared, 'score': self.player.score,
                            'level': self.game.level, 'episode_steps': self.episode_steps,
                            'piece_placed': piece_placed}
                    return obs, rw, True, info
                # Debug: log lock and spawn
                logger.debug(
                    f"LOCK (soft drop): lines_cleared={lines_cleared}. Next piece shape={self.player.current_piece.shape}, pos=({self.player.current_piece.x},{self.player.current_piece.y})"
                )
        elif action == 3:  # Rotate Clockwise
            self.player.action_handler.rotate_cw()
        elif action == 4:  # Rotate Counter-clockwise
            self.player.action_handler.rotate_ccw()
        elif action == 5:  # Hard Drop
            self.player.action_handler.hard_drop()
            # Ensure at least one block on playfield before locking
            formatted = convert_shape_format(self.player.current_piece)
            if not any(0 <= y < 20 for _, y in formatted):
                logging.debug(f"Hard drop invalid placement: formatted_positions={formatted}, locked_positions={self.player.locked_positions}")
                observation = self._get_observation()
                reward = self._get_reward(0, True)
                info = {
                    'lines_cleared': 0,
                    'score': self.player.score,
                    'level': self.game.level,
                    'episode_steps': self.episode_steps,
                    'piece_placed': False,
                    'invalid_placement': True
                }
                return observation, reward, True, info
            lines_cleared = self.player.update(self.game.fall_speed, self.game.level)
            piece_placed = True
            # Reset spawn check for new piece
            self._spawn_checked = False
            # Debug: log lock and spawn (hard drop)
            logger.debug(
                f"LOCK (hard drop): lines_cleared={lines_cleared}. Next piece shape={self.player.current_piece.shape}, pos=({self.player.current_piece.x},{self.player.current_piece.y})"
            )
            # Immediate game-over if locked blocks above grid
            if check_lost(self.player.locked_positions):
                logging.debug("Game over after hard drop: locked above visible grid detected")
                obs = self._get_observation()
                rw = self._get_reward(lines_cleared, True)
                info = {'lines_cleared': lines_cleared, 'score': self.player.score,
                        'level': self.game.level, 'episode_steps': self.episode_steps,
                        'piece_placed': piece_placed}
                return obs, rw, True, info
        # Gravity: drop every gravity_interval agent steps (skip after hard drop)
        if action != 5 and self.episode_steps % self.gravity_interval == 0:
            prev_y = self.player.current_piece.y
            self.player.action_handler.move_down()
            if self.player.current_piece.y == prev_y:
                # Stuck: check spawn collision if still above grid
                formatted = convert_shape_format(self.player.current_piece)
                if all(y < 0 for _, y in formatted):
                    logging.debug(f"Gravity spawn collision detected: formatted_positions={formatted}, locked_positions={self.player.locked_positions}")
                    obs = self._get_observation()
                    rw = self._get_reward(0, True)
                    info = {'lines_cleared': 0,
                            'score': self.player.score,
                            'level': self.game.level,
                            'episode_steps': self.episode_steps,
                            'piece_placed': False,
                            'spawn_collision': True}
                    return obs, rw, True, info
                # Normal lock
                self.player.change_piece = True
                lines_cleared += self.player.update(self.game.fall_speed, self.game.level)
                piece_placed = True
                # Debug: log lock and spawn (gravity)
                logger.debug(
                    f"LOCK (gravity): lines_cleared={lines_cleared}. Next piece shape={self.player.current_piece.shape}, pos=({self.player.current_piece.x},{self.player.current_piece.y})"
                )
                # Check if any locked block is above grid
                if check_lost(self.player.locked_positions):
                    logging.debug("Game over after gravity lock: locked above visible grid detected")
                    obs = self._get_observation()
                    rw = self._get_reward(lines_cleared, True)
                    info = {'lines_cleared': lines_cleared, 'score': self.player.score,
                            'level': self.game.level, 'episode_steps': self.episode_steps,
                            'piece_placed': piece_placed}
                    return obs, rw, True, info
        
        # Ensure single player mode is maintained
        self._ensure_single_player_mode()
        
        game_over = check_lost(self.player.locked_positions)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._get_reward(lines_cleared, game_over)
        
        # Check if episode is done on game over or max steps reached
        done = game_over or (self.episode_steps >= self.max_steps)
        
        # Additional info
        info = {
            'lines_cleared': lines_cleared,
            'score': self.player.score,
            'level': self.game.level,
            'episode_steps': self.episode_steps,
            'piece_placed': piece_placed
        }
        
        #time.sleep(0.05)  # Each environment step takes 200ms
        
        return observation, reward, done, info
    
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
        
        return observation
    
    def render(self, mode='human'):
        if self.headless:
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
