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
        bases = {1:10, 2:20, 3:40, 4:80}
        reward = bases.get(lines_cleared, 0) * (self.game.level + 1)

        # 2. Game-over check
        if game_over:
            return reward - 20

        # 3. Compute features
        grid = create_grid(self.player.locked_positions)
        col_heights = [  # from bottom (row 19) up
            next((r for r in range(20) if grid[r][c]!=(0,0,0)), 20)
            for c in range(10)
        ]
        curr = {
            "holes": sum(
                1 for c in range(10)
                for r in range(col_heights[c]+1, 20)
                if grid[r][c]==(0,0,0)
            ),
            "agg_h": sum(col_heights),
            "bump": sum(abs(col_heights[i]-col_heights[i+1]) for i in range(9)),
        }

        # 4. Delta-based shaping
        prev = self.prev_features if hasattr(self, 'prev_features') else None
        if prev:
            reward += 0.02 * (prev["holes"]   - curr["holes"])
            reward += 0.4 * (prev["agg_h"]   - curr["agg_h"])
            reward += 0.005 * (prev["bump"]    - curr["bump"])

            # Height preservation bonus
            if curr["agg_h"] <= prev["agg_h"]:
                reward += 0.3  # Bonus for not increasing height

        # 5. Adjacent pieces bonus
        if hasattr(self, 'prev_grid'):
            # Find newly placed blocks
            new_blocks = set()
            for i in range(20):
                for j in range(10):
                    if grid[i][j] != (0,0,0) and self.prev_grid[i][j] == (0,0,0):
                        new_blocks.add((j,i))  # (x,y) coordinates
            
            # Count adjacent pieces for new blocks
            if new_blocks:
                adjacent_count = 0
                for x, y in new_blocks:
                    # Check all adjacent positions
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < 10 and 0 <= ny < 20 and 
                            grid[ny][nx] != (0,0,0) and 
                            (nx,ny) not in new_blocks):
                            adjacent_count += 1
                
                # Reward based on number of adjacent pieces
                reward += 0.1 * adjacent_count

                # Position bonus for placing pieces above previous max height
                if prev:
                    prev_max_height = 20 - min(prev["agg_h"] / 10, 20)  # Convert to actual height
                    new_max_height = min(y for _, y in new_blocks)
                    if new_max_height < prev_max_height:  # Remember: lower y means higher up
                        reward += 0.2  # Bonus for building upward

        # Store current grid for next step
        self.prev_grid = [row[:] for row in grid]  # Deep copy of grid

        # 6. Store features for next step
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
        piece_placed = False
        lines_cleared = 0

        # Detect immediate game over if newly spawned piece overlaps locked blocks
        if not getattr(self, '_spawn_checked', False):
            self._spawn_checked = True
            formatted = convert_shape_format(self.player.current_piece)
            for x, y in formatted:
                if y >= 0 and (x, y) in self.player.locked_positions:
                    observation = self._get_observation()
                    reward = self._get_reward(0, True)
                    info = {
                        'lines_cleared': 0,
                        'score': self.player.score,
                        'level': self.game.level,
                        'episode_steps': self.episode_steps,
                        'piece_placed': False,
                        'spawn_collision': True
                    }
                    return observation, reward, True, info

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
        elif action == 3:  # Rotate Clockwise
            self.player.action_handler.rotate_cw()
        elif action == 4:  # Rotate Counter-clockwise
            self.player.action_handler.rotate_ccw()
        elif action == 5:  # Hard Drop
            self.player.action_handler.hard_drop()
            # Ensure at least one block on playfield before locking
            formatted = convert_shape_format(self.player.current_piece)
            if not any(0 <= y < 20 for _, y in formatted):
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
            # Immediately lock piece to avoid extra gravity
            lines_cleared = self.player.update(self.game.fall_speed, self.game.level)
            piece_placed = True
        elif action == 6:  # Hold Piece
            self.player.action_handler.hold_piece()
        elif action == 7:  # No-op action
            pass

        # Gravity: drop every gravity_interval agent steps (skip after hard drop)
        if action != 5 and self.episode_steps % self.gravity_interval == 0:
            # Soft gravity drop via action handler
            prev_y = self.player.current_piece.y
            self.player.action_handler.move_down()
            # If unable to move down, lock piece
            if self.player.current_piece.y == prev_y:
                # Ensure at least one block on playfield before locking
                formatted = convert_shape_format(self.player.current_piece)
                if not any(0 <= y < 20 for _, y in formatted):
                    observation = self._get_observation()
                    reward = self._get_reward(lines_cleared, True)
                    info = {
                        'lines_cleared': lines_cleared,
                        'score': self.player.score,
                        'level': self.game.level,
                        'episode_steps': self.episode_steps,
                        'piece_placed': False,
                        'invalid_placement': True
                    }
                    return observation, reward, True, info
                self.player.change_piece = True
                lines_cleared += self.player.update(self.game.fall_speed, self.game.level)
                piece_placed = True
        
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
        # Initialize previous grid state
        self.prev_grid = create_grid({})  # Empty grid
        # Reset spawn collision flag
        self._spawn_checked = False
        
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
