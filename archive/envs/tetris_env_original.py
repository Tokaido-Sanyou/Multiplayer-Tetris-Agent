import gym
import numpy as np
from gym import spaces
import pygame
import random
import time
import logging
import os
import sys
import copy
from collections import deque
from typing import Dict, List, Tuple, Any, Optional, Union

# Handle both direct execution and module import
try:
    from .game.game import Game
    from .game.piece import Piece
    from .game.utils import create_grid, check_lost, count_holes
    from .game.piece_utils import valid_space, convert_shape_format
    from .game.constants import shapes, shape_colors, s_width, s_height
    from .game.player import Player
    from .game.block_pool import BlockPool
except ImportError:
    # Direct execution - imports without relative paths
    from game.game import Game
    from game.piece import Piece
    from game.utils import create_grid, check_lost, count_holes
    from game.piece_utils import valid_space, convert_shape_format
    from game.constants import shapes, shape_colors, s_width, s_height
    from game.player import Player
    from game.block_pool import BlockPool

# Configure debug logging to file
logging.basicConfig(level=logging.DEBUG,
                    filename='tetris_debug.log',
                    filemode='w',
                    format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

class TetrisTrajectory:
    """Manages trajectory data for tree-based learning"""
    
    def __init__(self, trajectory_id: str):
        self.trajectory_id = trajectory_id
        self.states = []
        self.actions = []
        self.rewards = []
        self.infos = []
        self.parent_trajectory = None
        self.child_trajectories = []
        self.branch_point = -1
        
    def add_step(self, state, action, reward, info):
        """Add a step to the trajectory"""
        self.states.append(copy.deepcopy(state))
        self.actions.append(copy.deepcopy(action))
        self.rewards.append(reward)
        self.infos.append(copy.deepcopy(info))
        
    def branch_from(self, parent_trajectory, branch_point: int):
        """Create a branch from another trajectory at a specific point"""
        self.parent_trajectory = parent_trajectory
        self.branch_point = branch_point
        parent_trajectory.child_trajectories.append(self)
        
        # Copy states up to branch point
        if branch_point < len(parent_trajectory.states):
            self.states = copy.deepcopy(parent_trajectory.states[:branch_point + 1])
            self.actions = copy.deepcopy(parent_trajectory.actions[:branch_point])
            self.rewards = copy.deepcopy(parent_trajectory.rewards[:branch_point])
            self.infos = copy.deepcopy(parent_trajectory.infos[:branch_point])

class BoardState:
    """Represents a complete board state that can be saved and restored"""
    
    def __init__(self, player_state: Dict, game_state: Dict):
        self.player_state = copy.deepcopy(player_state)
        self.game_state = copy.deepcopy(game_state)
        self.timestamp = time.time()
        
    @classmethod
    def from_env(cls, env):
        """Create a board state from current environment"""
        player_states = []
        for player in env.players:
            player_state = {
                'locked_positions': copy.deepcopy(player.locked_positions),
                'current_piece': copy.deepcopy(player.current_piece),
                'next_pieces': copy.deepcopy(player.next_pieces),
                'hold_piece': copy.deepcopy(player.hold_piece),
                'can_hold': player.can_hold,
                'score': player.score,
                'current_block_index': player.current_block_index,
                'change_piece': player.change_piece
            }
            player_states.append(player_state)
        
        game_state = {
            'level': env.game.level,
            'fall_speed': env.game.fall_speed,
            'episode_steps': env.episode_steps,
            'num_agents': env.num_agents
        }
        
        return cls(player_states, game_state)
    
    def restore_to_env(self, env):
        """Restore this board state to an environment"""
        # Restore player states
        for i, player_state in enumerate(self.player_state):
            if i < len(env.players):
                player = env.players[i]
                player.locked_positions = copy.deepcopy(player_state['locked_positions'])
                player.current_piece = copy.deepcopy(player_state['current_piece'])
                player.next_pieces = copy.deepcopy(player_state['next_pieces'])
                player.hold_piece = copy.deepcopy(player_state['hold_piece'])
                player.can_hold = player_state['can_hold']
                player.score = player_state['score']
                player.current_block_index = player_state['current_block_index']
                player.change_piece = player_state['change_piece']
        
        # Restore game state
        env.game.level = self.game_state['level']
        env.game.fall_speed = self.game_state['fall_speed']
        env.episode_steps = self.game_state['episode_steps']
        
        # Ensure correct number of agents
        if 'num_agents' in self.game_state:
            env.num_agents = self.game_state['num_agents']

class TetrisEnv(gym.Env):
    """
    Enhanced Custom Tetris Environment for Multi-Agent ML Training
    Supports both single and multi-agent modes with advanced trajectory tracking
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents: int = 1, headless: bool = False, 
                 step_mode: str = 'action', enable_trajectory_tracking: bool = True):
        super(TetrisEnv, self).__init__()
        
        # Configuration
        self.num_agents = num_agents
        self.headless = headless
        self.step_mode = step_mode  # 'action' or 'block_placed'
        self.enable_trajectory_tracking = enable_trajectory_tracking
        
        # Initialize pygame
        if not self.headless:
            pygame.init()
        
        # Action space - now supports both single and multi-agent
        single_action_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.int8)
        if self.num_agents == 1:
            self.action_space = single_action_space
        else:
            self.action_space = spaces.Dict({
                f'agent_{i}': single_action_space for i in range(self.num_agents)
            })
        
        # Observation space - enhanced for multi-agent
        single_obs_space = spaces.Dict({
            'piece_grids': spaces.Box(low=0, high=1, shape=(7, 20, 10), dtype=np.int8),
            'current_piece_grid': spaces.Box(low=0, high=1, shape=(20, 10), dtype=np.int8),
            'empty_grid': spaces.Box(low=0, high=1, shape=(20, 10), dtype=np.int8),
            'next_piece': spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8),
            'hold_piece': spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8),
            'current_rotation': spaces.Box(low=0, high=3, shape=(), dtype=np.int8),
            'current_x': spaces.Box(low=0, high=9, shape=(), dtype=np.int8),
            'current_y': spaces.Box(low=0, high=19, shape=(), dtype=np.int8),
            'opponent_grid': spaces.Box(low=0, high=1, shape=(20, 10), dtype=np.int8)  # For multi-agent
        })
        
        if self.num_agents == 1:
            self.observation_space = single_obs_space
        else:
            self.observation_space = spaces.Dict({
                f'agent_{i}': single_obs_space for i in range(self.num_agents)
            })
        
        # Initialize rendering surface
        if not self.headless:
            self.surface = pygame.display.set_mode((s_width, s_height))
            pygame.display.set_caption("Tetris ML Training")
        else:
            self.surface = pygame.Surface((s_width, s_height))
            
        # Game components
        self.game = None
        self.players = []
        self.clock = pygame.time.Clock()
        
        # Training components
        self.episode_steps = 0
        self.max_steps = 25000
        self.gravity_interval = 5
        
        # Trajectory tracking
        self.trajectories = {}
        self.current_trajectory_id = None
        self.saved_board_states = {}
        
        # Reward components
        self.prev_features = [None] * self.num_agents
        
    def switch_mode(self, num_agents: int = None, step_mode: str = None):
        """Switch between single/multi-agent modes and step modes"""
        if num_agents is not None:
            self.num_agents = num_agents
            # Recreate action and observation spaces
            single_action_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.int8)
            if self.num_agents == 1:
                self.action_space = single_action_space
            else:
                self.action_space = spaces.Dict({
                    f'agent_{i}': single_action_space for i in range(self.num_agents)
                })
                
        if step_mode is not None:
            self.step_mode = step_mode
            
        # Reset environment to apply changes
        return self.reset()
    
    def save_board_state(self, state_id: str):
        """Save current board state for later restoration"""
        self.saved_board_states[state_id] = BoardState.from_env(self)
        
    def restore_board_state(self, state_id: str):
        """Restore a previously saved board state"""
        if state_id not in self.saved_board_states:
            raise ValueError(f"Board state '{state_id}' not found")
        
        board_state = self.saved_board_states[state_id]
        board_state.restore_to_env(self)
        return self._get_observation()
        
    def start_trajectory(self, trajectory_id: str, parent_id: str = None, branch_point: int = -1):
        """Start a new trajectory for tree-based learning"""
        if not self.enable_trajectory_tracking:
            return
            
        trajectory = TetrisTrajectory(trajectory_id)
        
        if parent_id and parent_id in self.trajectories:
            trajectory.branch_from(self.trajectories[parent_id], branch_point)
            
        self.trajectories[trajectory_id] = trajectory
        self.current_trajectory_id = trajectory_id
        
    def get_trajectory(self, trajectory_id: str) -> Optional[TetrisTrajectory]:
        """Get a trajectory by ID"""
        return self.trajectories.get(trajectory_id)
        
    def _get_observation(self):
        """Get current observation(s) - supports both single and multi-agent"""
        if self.num_agents == 1:
            return self._get_single_agent_observation(0)
        else:
            return {
                f'agent_{i}': self._get_single_agent_observation(i) 
                for i in range(self.num_agents)
            }
    
    def _get_single_agent_observation(self, agent_idx: int):
        """Get observation for a single agent"""
        player = self.players[agent_idx] if agent_idx < len(self.players) else self.players[0]
        
        # Initialize grids
        piece_grids = np.zeros((7, 20, 10), dtype=np.int8)
        current_piece_grid = np.zeros((20, 10), dtype=np.int8)
        empty_grid = np.ones((20, 10), dtype=np.int8)
        
        # Process locked positions by piece type
        for (x, y), color in player.locked_positions.items():
            if 0 <= x < 10 and 0 <= y < 20:
                empty_grid[y][x] = 0
                # Map color to piece type (simplified)
                piece_type = 0
                for i, shape_color in enumerate(shape_colors):
                    if color == shape_color:
                        piece_type = i
                        break
                piece_grids[piece_type][y][x] = 1
        
        # Add current falling piece
        if player.current_piece:
            for x, y in convert_shape_format(player.current_piece):
                if 0 <= y < 20 and 0 <= x < 10:
                    current_piece_grid[y][x] = 1
                    empty_grid[y][x] = 0
        
        # One-hot encoding for next and hold pieces
        next_piece_onehot = np.zeros(7, dtype=np.int8)
        hold_piece_onehot = np.zeros(7, dtype=np.int8)
        
        if player.next_pieces and len(player.next_pieces) > 0:
            next_shape_idx = shapes.index(player.next_pieces[0].shape)
            next_piece_onehot[next_shape_idx] = 1
            
        if player.hold_piece:
            hold_shape_idx = shapes.index(player.hold_piece.shape)
            hold_piece_onehot[hold_shape_idx] = 1
        
        # Opponent grid for multi-agent
        opponent_grid = np.zeros((20, 10), dtype=np.int8)
        if self.num_agents > 1 and len(self.players) > 1:
            opponent_idx = (agent_idx + 1) % len(self.players)
            opponent = self.players[opponent_idx]
            for (x, y), _ in opponent.locked_positions.items():
                if 0 <= x < 10 and 0 <= y < 20:
                    opponent_grid[y][x] = 1
        
        return {
            'piece_grids': piece_grids,
            'current_piece_grid': current_piece_grid,
            'empty_grid': empty_grid,
            'next_piece': next_piece_onehot,
            'hold_piece': hold_piece_onehot,
            'current_rotation': player.current_piece.rotation if player.current_piece else 0,
            'current_x': max(0, min(9, player.current_piece.x)) if player.current_piece else 0,
            'current_y': max(0, min(19, player.current_piece.y)) if player.current_piece else 0,
            'opponent_grid': opponent_grid
        }
    
    def _get_reward(self, agent_idx: int, lines_cleared: int, game_over: bool):
        """Enhanced reward function with improved features"""
        player = self.players[agent_idx]
        
        # Base line clear rewards
        line_rewards = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        reward = line_rewards.get(lines_cleared, 0) * (self.game.level + 1)
        
        # Game over penalty
        if game_over:
            return reward - 100
        
        # Calculate board features
        grid = create_grid(player.locked_positions)
        col_heights = []
        for c in range(10):
            height = 0
            for r in range(20):
                if grid[r][c] != (0, 0, 0):
                    height = 20 - r
                    break
            col_heights.append(height)
        
        # Calculate features
        aggregate_height = sum(col_heights)
        max_height = max(col_heights)
        holes = self._count_holes(grid)
        bumpiness = sum(abs(col_heights[i] - col_heights[i + 1]) for i in range(9))
        
        # Lines cleared on top of full rows
        wells = self._count_wells(grid)
        
        curr_features = {
            "lines": lines_cleared,
            "aggregate_height": aggregate_height,
            "max_height": max_height,
            "holes": holes,
            "bumpiness": bumpiness,
            "wells": wells
        }
        
        # Delta-based reward shaping
        prev = self.prev_features[agent_idx]
        if prev:
            # Reward improvements
            reward += 10 * (prev["lines"])  # Lines cleared
            reward += -0.5 * (curr_features["aggregate_height"] - prev["aggregate_height"])
            reward += -0.5 * (curr_features["holes"] - prev["holes"])
            reward += -0.5 * (curr_features["bumpiness"] - prev["bumpiness"])
            reward += 2.0 * (prev["max_height"] - curr_features["max_height"])  # Height reduction
            reward += -0.3 * (curr_features["wells"] - prev["wells"])  # Penalize creating wells
        
        # Store current features for next step
        self.prev_features[agent_idx] = curr_features
        
        return reward
    
    def _count_holes(self, grid):
        """Count holes in the grid"""
        holes = 0
        for col in range(10):
            found_block = False
            for row in range(20):
                if grid[row][col] != (0, 0, 0):
                    found_block = True
                elif found_block and grid[row][col] == (0, 0, 0):
                    holes += 1
        return holes
    
    def _count_wells(self, grid):
        """Count wells (consecutive empty cells surrounded by blocks)"""
        wells = 0
        for col in range(10):
            well_depth = 0
            for row in range(19, -1, -1):
                if grid[row][col] == (0, 0, 0):
                    # Check if surrounded by blocks or borders
                    left_blocked = (col == 0 or grid[row][col-1] != (0, 0, 0))
                    right_blocked = (col == 9 or grid[row][col+1] != (0, 0, 0))
                    
                    if left_blocked and right_blocked:
                        well_depth += 1
                    else:
                        wells += well_depth * (well_depth + 1) // 2  # Quadratic penalty
                        well_depth = 0
                else:
                    wells += well_depth * (well_depth + 1) // 2
                    well_depth = 0
            wells += well_depth * (well_depth + 1) // 2
        return wells

    def step(self, action):
        """Execute one time step - supports both single and multi-agent"""
        self.episode_steps += 1
        
        if self.num_agents == 1:
            return self._step_single_agent(action)
        else:
            return self._step_multi_agent(action)
    
    def _step_single_agent(self, action):
        """Handle single agent step"""
        player = self.players[0]
        action_idx = self._action_one_hot_to_scalar(action)
        
        piece_placed = False
        lines_cleared = 0
        
        # Execute action based on step mode
        if self.step_mode == 'action':
            # Step by individual action
            piece_placed, lines_cleared = self._execute_action(player, action_idx)
            
            # Apply gravity if needed
            if action_idx != 5 and self.episode_steps % self.gravity_interval == 0:
                gravity_placed, gravity_lines = self._apply_gravity(player)
                piece_placed = piece_placed or gravity_placed
                lines_cleared += gravity_lines
                
        elif self.step_mode == 'block_placed':
            # Step until block is placed
            piece_placed, lines_cleared = self._execute_until_placement(player, action_idx)
        
        # Check game over
        game_over = check_lost(player.locked_positions)
        
        # Get observation and reward
        observation = self._get_observation()
        reward = self._get_reward(0, lines_cleared, game_over)
        
        # Check if episode is done
        done = game_over or (self.episode_steps >= self.max_steps)
        
        # Store trajectory data
        if self.enable_trajectory_tracking and self.current_trajectory_id:
            trajectory = self.trajectories[self.current_trajectory_id]
            trajectory.add_step(observation, action, reward, {
                'lines_cleared': lines_cleared,
                'piece_placed': piece_placed,
                'game_over': game_over
            })
        
        info = {
            'lines_cleared': lines_cleared,
            'score': player.score,
            'level': self.game.level,
            'episode_steps': self.episode_steps,
            'piece_placed': piece_placed,
            'game_over': game_over
        }
        
        return observation, reward, done, info
    
    def _step_multi_agent(self, actions):
        """Handle multi-agent step"""
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        # Execute actions for all agents
        for i in range(self.num_agents):
            if i < len(self.players):
                agent_key = f'agent_{i}'
                action = actions.get(agent_key, np.zeros(8))
                action_idx = self._action_one_hot_to_scalar(action)
                
                player = self.players[i]
                piece_placed, lines_cleared = self._execute_action(player, action_idx)
                
                # Apply gravity
                if action_idx != 5 and self.episode_steps % self.gravity_interval == 0:
                    gravity_placed, gravity_lines = self._apply_gravity(player)
                    piece_placed = piece_placed or gravity_placed
                    lines_cleared += gravity_lines
                
                # Check game over for this agent
                game_over = check_lost(player.locked_positions)
                
                # Get observation and reward for this agent
                observations[agent_key] = self._get_single_agent_observation(i)
                rewards[agent_key] = self._get_reward(i, lines_cleared, game_over)
                dones[agent_key] = game_over or (self.episode_steps >= self.max_steps)
                
                infos[agent_key] = {
                    'lines_cleared': lines_cleared,
                    'score': player.score,
                    'level': self.game.level,
                    'episode_steps': self.episode_steps,
                    'piece_placed': piece_placed,
                    'game_over': game_over
                }
        
        # Global done condition
        done = any(dones.values()) or (self.episode_steps >= self.max_steps)
        
        return observations, rewards, done, infos
    
    def _execute_action(self, player, action_idx):
        """Execute a single action for a player"""
        piece_placed = False
        lines_cleared = 0
        
        # Map action to game action
        if action_idx == 0:  # Move Left
            player.action_handler.move_left()
        elif action_idx == 1:  # Move Right
            player.action_handler.move_right()
        elif action_idx == 2:  # Soft Drop / Move Down
            prev_y = player.current_piece.y if player.current_piece else 0
            player.action_handler.move_down()
            if player.current_piece and player.current_piece.y == prev_y:
                player.change_piece = True
                lines_cleared = player.update(self.game.fall_speed, self.game.level)
                piece_placed = True
        elif action_idx == 3:  # Rotate Clockwise
            player.action_handler.rotate_cw()
        elif action_idx == 4:  # Rotate Counter-clockwise
            player.action_handler.rotate_ccw()
        elif action_idx == 5:  # Hard Drop
            player.action_handler.hard_drop()
            lines_cleared = player.update(self.game.fall_speed, self.game.level)
            piece_placed = True
        elif action_idx == 6:  # Hold Piece
            player.action_handler.hold_piece()
        elif action_idx == 7:  # No-op
            pass
            
        return piece_placed, lines_cleared
    
    def _apply_gravity(self, player):
        """Apply gravity to a player's piece"""
        if not player.current_piece:
            return False, 0
            
        prev_y = player.current_piece.y
        player.action_handler.move_down()
        
        if player.current_piece.y == prev_y:
            player.change_piece = True
            lines_cleared = player.update(self.game.fall_speed, self.game.level)
            return True, lines_cleared
        
        return False, 0
    
    def _execute_until_placement(self, player, action_idx):
        """Execute actions until a piece is placed"""
        piece_placed = False
        total_lines_cleared = 0
        
        # First execute the requested action
        placed, lines = self._execute_action(player, action_idx)
        piece_placed = piece_placed or placed
        total_lines_cleared += lines
        
        # Continue with gravity until piece is placed
        while not piece_placed and player.current_piece:
            placed, lines = self._apply_gravity(player)
            piece_placed = piece_placed or placed
            total_lines_cleared += lines
            
        return piece_placed, total_lines_cleared
    
    def reset(self):
        """Reset the environment to initial state"""
        # Initialize or reset game
        if self.game is None:
            self.game = Game(self.surface, auto_start=False)
        else:
            self.game = Game(self.surface, auto_start=False)
        
        # Set up players based on number of agents
        self.players = []
        if self.num_agents == 1:
            self.players = [self.game.player1]
            # Disable player 2
            self.game.player2.current_piece = None
            self.game.player2.next_pieces = []
            self.game.player2.hold_piece = None
            self.game.player2.locked_positions = {}
        else:
            self.players = [self.game.player1, self.game.player2]
            # Add more players if needed (for future extension)
            for i in range(2, self.num_agents):
                # Create additional players (placeholder for future)
                additional_player = Player(self.game.block_pool, is_player_one=(i % 2 == 0))
                self.players.append(additional_player)
        
        # Reset episode tracking and reward shaping state
        self.episode_steps = 0
        self.prev_features = [None] * self.num_agents
        
        # Reset trajectory tracking
        if self.enable_trajectory_tracking:
            self.trajectories.clear()
            self.current_trajectory_id = None
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def render(self, mode='human'):
        """Render the game state"""
        if self.headless:
            return
        
        # Pump Pygame events to keep the window responsive
        pygame.event.pump()
        
        # Sync locked blocks into the game grid before drawing
        if len(self.players) > 0:
            self.game.p1_grid = create_grid(self.players[0].locked_positions)
        if len(self.players) > 1:
            self.game.p2_grid = create_grid(self.players[1].locked_positions)
        else:
            self.game.p2_grid = create_grid({})
        
        if mode == 'human':
            self.game.draw()
            pygame.display.update()
            # Cap frame rate to 10 FPS
            self.clock.tick(10)
    
    def close(self):
        """Clean up resources"""
        if self.game is not None:
            self.game = None
        if self.surface is not None:
            self.surface = None
        if not self.headless:
            pygame.quit()

    def _action_one_hot_to_scalar(self, action_one_hot):
        """Convert one-hot action vector to scalar action index"""
        return np.argmax(action_one_hot)
    
    def _action_scalar_to_one_hot(self, action_scalar):
        """Convert scalar action index to one-hot vector"""
        one_hot = np.zeros(8, dtype=np.int8)
        one_hot[action_scalar] = 1
        return one_hot