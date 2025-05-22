import gym
from gym import spaces
import numpy as np
import pygame
from game import Game, Player, BlockPool, create_grid

class TetrisEnv(gym.Env):
    """
    Tetris Environment that follows gym interface
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(TetrisEnv, self).__init__()
        
        # Initialize pygame
        pygame.init()
        
        # Create game surface
        self.surface = pygame.Surface((1400, 700))
        
        # Create game instance
        self.game = Game(self.surface)
        
        # Define action space (discrete)
        # Actions: 0=left, 1=right, 2=down, 3=rotate_cw, 4=rotate_ccw, 5=hard_drop, 6=hold
        self.action_space = spaces.Discrete(7)
        
        # Define observation space
        # Observation: grid (20x10) + current piece + next pieces
        self.observation_space = spaces.Box(
            low=0,
            high=7,  # 0=empty, 1-7=piece colors
            shape=(20, 10),
            dtype=np.uint8
        )
        
        # Initialize game state
        self.reset()

    def step(self, action):
        """
        Execute one time step within the environment
        """
        # Convert action to pygame event
        event = self._action_to_event(action)
        
        # Handle the action
        self.game.handle_input(event)
        
        # Update game state
        self.game.update()
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if game is done
        done = self.game.check_lost(self.game.player1.locked_positions)
        
        # Additional info
        info = {
            'score': self.game.player1.score,
            'level': self.game.level,
            'lines_cleared': self.game.player1.lines_cleared
        }
        
        return obs, reward, done, info

    def reset(self):
        """
        Reset the environment to initial state
        """
        # Reset game
        self.game = Game(self.surface)
        return self._get_observation()

    def render(self, mode='human'):
        """
        Render the environment
        """
        if mode == 'human':
            self.game.draw()
            pygame.display.update()
        elif mode == 'rgb_array':
            return pygame.surfarray.array3d(self.surface)

    def close(self):
        """
        Clean up resources
        """
        pygame.quit()

    def _action_to_event(self, action):
        """
        Convert gym action to pygame event
        """
        event = pygame.event.Event(pygame.KEYDOWN)
        if action == 0:  # left
            event.key = pygame.K_LEFT
        elif action == 1:  # right
            event.key = pygame.K_RIGHT
        elif action == 2:  # down
            event.key = pygame.K_DOWN
        elif action == 3:  # rotate clockwise
            event.key = pygame.K_UP
        elif action == 4:  # rotate counter-clockwise
            event.key = pygame.K_q
        elif action == 5:  # hard drop
            event.key = pygame.K_SPACE
        elif action == 6:  # hold
            event.key = pygame.K_c
        return event

    def _get_observation(self):
        """
        Get current game state as observation
        """
        # Get the grid using create_grid function
        grid = create_grid(self.game.player1.locked_positions)
        
        # Convert grid to numpy array with color indices
        obs = np.zeros((20, 10), dtype=np.uint8)
        for y in range(20):
            for x in range(10):
                color = grid[y][x]
                obs[y][x] = self._color_to_index(color)
        return obs

    def _calculate_reward(self):
        """
        Calculate reward based on game state
        """
        reward = 0
        # Reward for lines cleared
        lines_cleared = self.game.player1.lines_cleared
        if lines_cleared > 0:
            reward += lines_cleared * 100
        
        # Penalty for height
        max_height = max([y for x, y in self.game.player1.locked_positions.keys()]) if self.game.player1.locked_positions else 0
        reward -= max_height * 0.5
        
        # Penalty for holes
        holes = self._count_holes()
        reward -= holes * 10
        
        return reward

    def _count_holes(self):
        """
        Count number of holes in the grid
        """
        holes = 0
        grid = self._get_observation()
        for col in range(10):
            found_block = False
            for row in range(20):
                if grid[row][col] > 0:
                    found_block = True
                elif found_block and grid[row][col] == 0:
                    holes += 1
        return holes

    def _color_to_index(self, color):
        """
        Convert color tuple to piece index
        """
        colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), 
                 (255, 255, 0), (255, 165, 0), (0, 0, 255), (230,230,250)]
        try:
            return colors.index(color) + 1
        except ValueError:
            return 0 