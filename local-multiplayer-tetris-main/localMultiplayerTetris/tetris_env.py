import gym
import numpy as np
from gym import spaces
import pygame
import random
from game import Game, Piece
from utils import create_grid, valid_space, convert_shape_format, check_lost
from constants import shapes, shape_colors

class TetrisEnv(gym.Env):
    """
    Custom Tetris Environment that follows gym interface
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TetrisEnv, self).__init__()
        
        # Initialize pygame
        pygame.init()
        
        # Define action space (7 possible actions)
        # 0: Move Left
        # 1: Move Right
        # 2: Move Down
        # 3: Rotate Clockwise
        # 4: Rotate Counter-clockwise
        # 5: Hard Drop
        # 6: Hold Piece
        self.action_space = spaces.Discrete(7)
        
        # Define observation space
        # Grid: 20x10 matrix (0 for empty, 1-7 for different piece colors)
        # Current piece: 4x4 matrix (0 for empty, 1 for filled)
        # Next piece: 4x4 matrix (0 for empty, 1 for filled)
        # Hold piece: 4x4 matrix (0 for empty, 1 for filled)
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=7, shape=(20, 10), dtype=np.int8),
            'current_piece': spaces.Box(low=0, high=1, shape=(4, 4), dtype=np.int8),
            'next_piece': spaces.Box(low=0, high=1, shape=(4, 4), dtype=np.int8),
            'hold_piece': spaces.Box(low=0, high=1, shape=(4, 4), dtype=np.int8)
        })
        
        # Initialize game components
        self.surface = pygame.Surface((1400, 700))
        self.game = Game(self.surface)
        self.player = self.game.player1  # Use player1 for single-agent training
        
        # Initialize pygame clock
        self.clock = pygame.time.Clock()
        
        # Initialize episode tracking
        self.episode_steps = 0
        self.max_steps = 1000  # Maximum steps per episode
        
    def _get_observation(self):
        """Convert game state to observation space format"""
        # Get grid state
        grid = create_grid(self.player.locked_positions)
        grid_obs = np.zeros((20, 10), dtype=np.int8)
        for i in range(20):
            for j in range(10):
                if grid[i][j] != (0, 0, 0):
                    grid_obs[i][j] = shape_colors.index(grid[i][j]) + 1
        
        # Get current piece state
        current_piece = np.zeros((4, 4), dtype=np.int8)
        if self.player.current_piece:
            format = self.player.current_piece.shape[self.player.current_piece.rotation % len(self.player.current_piece.shape)]
            for i, line in enumerate(format):
                for j, column in enumerate(line):
                    if column == '0':
                        current_piece[i][j] = 1
        
        # Get next piece state
        next_piece = np.zeros((4, 4), dtype=np.int8)
        if self.player.next_pieces:
            format = self.player.next_pieces[0].shape[0]
            for i, line in enumerate(format):
                for j, column in enumerate(line):
                    if column == '0':
                        next_piece[i][j] = 1
        
        # Get hold piece state
        hold_piece = np.zeros((4, 4), dtype=np.int8)
        if self.player.hold_piece:
            format = self.player.hold_piece.shape[self.player.hold_piece.rotation % len(self.player.hold_piece.shape)]
            for i, line in enumerate(format):
                for j, column in enumerate(line):
                    if column == '0':
                        hold_piece[i][j] = 1
        
        return {
            'grid': grid_obs,
            'current_piece': current_piece,
            'next_piece': next_piece,
            'hold_piece': hold_piece
        }
    
    def _get_reward(self, lines_cleared, game_over):
        """Calculate reward based on game state"""
        reward = 0
        
        # Reward for clearing lines
        if lines_cleared == 1:
            reward += 100
        elif lines_cleared == 2:
            reward += 300
        elif lines_cleared == 3:
            reward += 500
        elif lines_cleared == 4:  # Tetris
            reward += 800
        
        # Penalty for game over
        if game_over:
            reward -= 1000
        
        # Small penalty for each step to encourage faster play
        reward -= 1
        
        # Penalty for high stack height
        max_height = max([y for x, y in self.player.locked_positions.keys()]) if self.player.locked_positions else 0
        reward -= max_height * 0.5
        
        # Penalty for holes
        holes = self._count_holes()
        reward -= holes * 10
        
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
    
    def step(self, action):
        """Execute one time step within the environment"""
        self.episode_steps += 1
        
        # Map action to game action
        if action == 0:  # Move Left
            self.player.action_handler.move_left()
        elif action == 1:  # Move Right
            self.player.action_handler.move_right()
        elif action == 2:  # Move Down
            self.player.action_handler.move_down()
        elif action == 3:  # Rotate Clockwise
            self.player.action_handler.rotate_cw()
        elif action == 4:  # Rotate Counter-clockwise
            self.player.action_handler.rotate_ccw()
        elif action == 5:  # Hard Drop
            self.player.action_handler.hard_drop()
        elif action == 6:  # Hold Piece
            self.player.action_handler.hold_piece()
        
        # Update game state
        lines_cleared = self.player.update(self.game.fall_speed, self.game.level)
        game_over = check_lost(self.player.locked_positions)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._get_reward(lines_cleared, game_over)
        
        # Check if episode is done
        done = game_over or self.episode_steps >= self.max_steps
        
        # Additional info
        info = {
            'lines_cleared': lines_cleared,
            'score': self.player.score,
            'level': self.game.level,
            'episode_steps': self.episode_steps
        }
        
        return observation, reward, done, info
    
    def reset(self):
        """Reset the environment to initial state"""
        # Reset game
        self.surface = pygame.Surface((1400, 700))
        self.game = Game(self.surface)
        self.player = self.game.player1
        
        # Reset episode tracking
        self.episode_steps = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def render(self, mode='human'):
        """Render the game state"""
        if mode == 'human':
            self.game.draw()
            pygame.display.update()
            self.clock.tick(60)
    
    def close(self):
        """Clean up resources"""
        pygame.quit() 