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
    from .game.constants import shapes, s_width, s_height  # Removed shape_colors
    from .game.player import Player
    from .game.block_pool import BlockPool
except ImportError:
    # Direct execution - imports without relative paths
    from game.game import Game
    from game.piece import Piece
    from game.utils import create_grid, check_lost, count_holes
    from game.piece_utils import valid_space, convert_shape_format
    from game.constants import shapes, s_width, s_height  # Removed shape_colors
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
    Uses binary tuple structures for observations and actions
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
        
        # Action space - binary tuple structure (8 actions as tuple of 0s and 1s)
        single_action_space = spaces.Tuple([spaces.Discrete(2) for _ in range(8)])
        if self.num_agents == 1:
            self.action_space = single_action_space
        else:
            self.action_space = spaces.Dict({
                f'agent_{i}': single_action_space for i in range(self.num_agents)
            })
        
        # Observation space - binary tuple structures (no colors, only binary occupancy)
        # Flattened binary representation: board (200) + piece info (7+7+4+4+4) + opponent (200)
        obs_size = 200 + 7 + 7 + 4 + 4 + 4 + 200  # Total: 426 binary values
        single_obs_space = spaces.Tuple([spaces.Discrete(2) for _ in range(obs_size)])
        
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
        
        # Store previous features for delta reward calculation
        self.prev_features = {}
        
        # Meta-learning variables for RL2, DREAM, etc.
        self.meta_episode_history = []
        self.episode_buffer = deque(maxlen=100)  # For meta-learning
        self.task_context = {}
        self.adaptation_buffer = deque(maxlen=50)  # For quick adaptation
        
        # DQN-specific variables
        self.experience_buffer = deque(maxlen=100000)
        self.priority_weights = deque(maxlen=100000)
        
        # Dream-specific variables
        self.dream_states = []
        self.world_model_data = deque(maxlen=50000)

    def switch_mode(self, num_agents: int = None, step_mode: str = None):
        """Switch between single/multi-agent modes and step modes dynamically"""
        if num_agents is not None:
            self.num_agents = num_agents
            
            # Update action and observation spaces
            single_action_space = spaces.Tuple([spaces.Discrete(2) for _ in range(8)])
            obs_size = 426  # Binary observation size
            single_obs_space = spaces.Tuple([spaces.Discrete(2) for _ in range(obs_size)])
            
            if self.num_agents == 1:
                self.action_space = single_action_space
                self.observation_space = single_obs_space
            else:
                self.action_space = spaces.Dict({f'agent_{i}': single_action_space for i in range(self.num_agents)})
                self.observation_space = spaces.Dict({f'agent_{i}': single_obs_space for i in range(self.num_agents)})
        
        if step_mode is not None:
            self.step_mode = step_mode

    def save_board_state(self, state_id: str):
        """Save current board state for later restoration"""
        self.saved_states[state_id] = BoardState.from_env(self)

    def restore_board_state(self, state_id: str):
        """Restore a previously saved board state"""
        if state_id in self.saved_states:
            self.saved_states[state_id].restore_to_env(self)
        else:
            raise ValueError(f"No saved state with id '{state_id}'")

    def start_trajectory(self, trajectory_id: str, parent_id: str = None, branch_point: int = -1):
        """Start a new trajectory for tree-based learning"""
        trajectory = TetrisTrajectory(trajectory_id)
        
        if parent_id and parent_id in self.trajectories:
            trajectory.branch_from(self.trajectories[parent_id], branch_point)
        
        self.trajectories[trajectory_id] = trajectory
        self.current_trajectory_id = trajectory_id

    def get_trajectory(self, trajectory_id: str) -> Optional[TetrisTrajectory]:
        """Get trajectory by ID"""
        return self.trajectories.get(trajectory_id)

    def _get_observation(self):
        """Get current observation - handles both single and multi-agent"""
        if self.num_agents == 1:
            return self._get_single_agent_observation(0)
        else:
            observations = {}
            for i in range(self.num_agents):
                observations[f'agent_{i}'] = self._get_single_agent_observation(i)
            return observations

    def _get_single_agent_observation(self, agent_idx: int):
        """Get observation for a single agent as binary tuple (no colors)"""
        if agent_idx >= len(self.players):
            # Return empty observation as binary tuple
            return tuple([0] * 426)
        
        player = self.players[agent_idx]
        observation_bits = []
        
        # 1. Board state (20x10 = 200 bits) - only occupancy, no colors
        grid = create_grid(player.locked_positions)
        for row in range(20):
            for col in range(10):
                # 1 if occupied, 0 if empty (ignoring color)
                observation_bits.append(1 if grid[row][col] != (0, 0, 0) else 0)
        
        # 2. Next piece one-hot (7 bits)
        next_piece_bits = [0] * 7
        if player.next_pieces and len(player.next_pieces) > 0:
            next_shape = player.next_pieces[0].shape
            if next_shape in shapes:
                next_piece_bits[shapes.index(next_shape)] = 1
        observation_bits.extend(next_piece_bits)
        
        # 3. Hold piece one-hot (7 bits)
        hold_piece_bits = [0] * 7
        if player.hold_piece:
            hold_shape = player.hold_piece.shape
            if hold_shape in shapes:
                hold_piece_bits[shapes.index(hold_shape)] = 1
        observation_bits.extend(hold_piece_bits)
        
        # 4. Current piece rotation (4 bits - binary representation of 0-3)
        current_rotation = player.current_piece.rotation if player.current_piece else 0
        rotation_bits = [(current_rotation >> i) & 1 for i in range(4)]
        observation_bits.extend(rotation_bits)
        
        # 5. Current piece X position (4 bits - binary representation of 0-9)
        current_x = player.current_piece.x if player.current_piece else 0
        x_bits = [(current_x >> i) & 1 for i in range(4)]
        observation_bits.extend(x_bits)
        
        # 6. Current piece Y position (4 bits - binary representation of 0-19, but limited to 4 bits)
        current_y = player.current_piece.y if player.current_piece else 0
        y_bits = [(current_y >> i) & 1 for i in range(4)]
        observation_bits.extend(y_bits)
        
        # 7. Opponent grid (20x10 = 200 bits) - for multi-agent, only occupancy
        if self.num_agents > 1 and len(self.players) > 1:
            # Use the other player's grid
            opponent_idx = 1 - agent_idx if agent_idx < 2 else 0
            if opponent_idx < len(self.players):
                opponent_grid = create_grid(self.players[opponent_idx].locked_positions)
                for row in range(20):
                    for col in range(10):
                        observation_bits.append(1 if opponent_grid[row][col] != (0, 0, 0) else 0)
            else:
                observation_bits.extend([0] * 200)
        else:
            observation_bits.extend([0] * 200)
        
        return tuple(observation_bits)

    def _get_reward(self, agent_idx: int, lines_cleared: int, game_over: bool):
        """Enhanced reward function with updated line rewards and removed max height penalty and wells"""
        player = self.players[agent_idx]
        
        # Updated line clear rewards: 1:3, 2:5, 3:8, 4:12
        line_rewards = {0: 0, 1: 3, 2: 5, 3: 8, 4: 12}
        reward = line_rewards.get(lines_cleared, 0) * (self.game.level + 1)
        
        # Game over penalty
        if game_over:
            return reward - 100
        
        # Calculate board features (color-independent)
        grid = create_grid(player.locked_positions)
        col_heights = []
        for c in range(10):
            height = 0
            for r in range(20):
                if grid[r][c] != (0, 0, 0):  # Any non-empty cell
                    height = 20 - r
                    break
            col_heights.append(height)
        
        # Calculate features (removed max_height and wells)
        aggregate_height = sum(col_heights)
        holes = self._count_holes(grid)
        bumpiness = sum(abs(col_heights[i] - col_heights[i + 1]) for i in range(9))
        
        curr_features = {
            "lines": lines_cleared,
            "aggregate_height": aggregate_height,
            "holes": holes,
            "bumpiness": bumpiness
        }
        
        # Delta-based reward shaping (removed max_height and wells components)
        prev = self.prev_features.get(agent_idx)
        if prev:
            # Reward improvements
            reward += 10 * (prev["lines"])  # Lines cleared
            reward += -0.5 * (curr_features["aggregate_height"] - prev["aggregate_height"])
            reward += -0.5 * (curr_features["holes"] - prev["holes"])
            reward += -0.5 * (curr_features["bumpiness"] - prev["bumpiness"])
        
        # Store current features for next step
        self.prev_features[agent_idx] = curr_features
        
        return reward
    
    def _count_holes(self, grid):
        """Count holes in the grid (color-independent)"""
        holes = 0
        for col in range(10):
            found_block = False
            for row in range(20):
                if grid[row][col] != (0, 0, 0):  # Any non-empty cell
                    found_block = True
                elif found_block and grid[row][col] == (0, 0, 0):
                    holes += 1
        return holes

    # === DQN Support Functions ===
    def add_experience(self, state, action, reward, next_state, done, priority=None):
        """Add experience to replay buffer for DQN training"""
        experience = {
            'state': copy.deepcopy(state),
            'action': copy.deepcopy(action),
            'reward': reward,
            'next_state': copy.deepcopy(next_state),
            'done': done,
            'timestamp': time.time()
        }
        self.experience_buffer.append(experience)
        
        # Add priority for prioritized experience replay
        if priority is not None:
            self.priority_weights.append(priority)
        else:
            self.priority_weights.append(1.0)
    
    def sample_experience_batch(self, batch_size: int, prioritized: bool = False):
        """Sample a batch of experiences for training"""
        if len(self.experience_buffer) < batch_size:
            return None
        
        if prioritized and len(self.priority_weights) >= len(self.experience_buffer):
            # Prioritized sampling
            weights = np.array(list(self.priority_weights)[-len(self.experience_buffer):])
            probs = weights / np.sum(weights)
            indices = np.random.choice(len(self.experience_buffer), batch_size, p=probs)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        
        batch = [self.experience_buffer[i] for i in indices]
        return batch, indices if prioritized else batch
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for prioritized experience replay"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priority_weights):
                self.priority_weights[idx] = priority

    # === DREAM Support Functions ===
    def add_world_model_data(self, state, action, next_state, reward, done):
        """Add data for world model training in DREAM algorithm"""
        model_data = {
            'state': copy.deepcopy(state),
            'action': copy.deepcopy(action),
            'next_state': copy.deepcopy(next_state),
            'reward': reward,
            'done': done,
            'timestamp': time.time()
        }
        self.world_model_data.append(model_data)
    
    def sample_world_model_batch(self, batch_size: int):
        """Sample batch for world model training"""
        if len(self.world_model_data) < batch_size:
            return None
        
        indices = np.random.choice(len(self.world_model_data), batch_size, replace=False)
        return [self.world_model_data[i] for i in indices]
    
    def generate_dream_experience(self, initial_state, num_steps: int = 10):
        """Generate imagined experience using world model (placeholder for DREAM)"""
        # This would interface with your trained world model
        dream_trajectory = []
        current_state = copy.deepcopy(initial_state)
        
        for step in range(num_steps):
            # This is a placeholder - you would use your actual world model here
            random_action = tuple([random.randint(0, 1) for _ in range(8)])
            
            # Placeholder next state (in practice, use world model prediction)
            next_state = copy.deepcopy(current_state)
            reward = np.random.normal(0, 1)  # Placeholder reward
            done = False
            
            dream_trajectory.append({
                'state': current_state,
                'action': random_action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            current_state = next_state
            if done:
                break
        
        self.dream_states.extend(dream_trajectory)
        return dream_trajectory

    # === RL2 and Meta-Learning Support Functions ===
    def add_episode_to_meta_history(self, episode_data: Dict):
        """Add completed episode to meta-learning history"""
        episode_summary = {
            'total_reward': sum(episode_data.get('rewards', [])),
            'episode_length': len(episode_data.get('rewards', [])),
            'final_score': episode_data.get('final_score', 0),
            'lines_cleared': episode_data.get('total_lines_cleared', 0),
            'task_context': copy.deepcopy(self.task_context),
            'timestamp': time.time()
        }
        self.meta_episode_history.append(episode_summary)
    
    def get_meta_context(self, num_episodes: int = 10):
        """Get meta-learning context from recent episodes"""
        if len(self.meta_episode_history) < num_episodes:
            num_episodes = len(self.meta_episode_history)
        
        recent_episodes = self.meta_episode_history[-num_episodes:]
        
        # Create context vector
        context = {
            'avg_reward': np.mean([ep['total_reward'] for ep in recent_episodes]),
            'avg_length': np.mean([ep['episode_length'] for ep in recent_episodes]),
            'avg_score': np.mean([ep['final_score'] for ep in recent_episodes]),
            'avg_lines': np.mean([ep['lines_cleared'] for ep in recent_episodes]),
            'episode_count': len(recent_episodes)
        }
        
        return context
    
    def set_task_context(self, context: Dict):
        """Set task context for meta-learning (e.g., difficulty, game variant)"""
        self.task_context = copy.deepcopy(context)
    
    def add_adaptation_data(self, state, action, reward, adaptation_signal):
        """Add data for quick adaptation in meta-learning"""
        adaptation_data = {
            'state': copy.deepcopy(state),
            'action': copy.deepcopy(action),
            'reward': reward,
            'adaptation_signal': adaptation_signal,
            'timestamp': time.time()
        }
        self.adaptation_buffer.append(adaptation_data)
    
    def get_adaptation_batch(self, batch_size: int = None):
        """Get recent adaptation data for quick learning"""
        if batch_size is None:
            return list(self.adaptation_buffer)
        
        if len(self.adaptation_buffer) < batch_size:
            return list(self.adaptation_buffer)
        
        return list(self.adaptation_buffer)[-batch_size:]

    # === General Support Functions ===
    def get_state_features(self, agent_idx: int = 0):
        """Extract engineered features from current state (color-independent)"""
        if agent_idx >= len(self.players):
            return {}
        
        player = self.players[agent_idx]
        grid = create_grid(player.locked_positions)
        
        # Calculate various features
        col_heights = []
        for c in range(10):
            height = 0
            for r in range(20):
                if grid[r][c] != (0, 0, 0):  # Any non-empty cell
                    height = 20 - r
                    break
            col_heights.append(height)
        
        features = {
            'aggregate_height': sum(col_heights),
            'max_height': max(col_heights),
            'min_height': min(col_heights),
            'holes': self._count_holes(grid),
            'bumpiness': sum(abs(col_heights[i] - col_heights[i + 1]) for i in range(9)),
            'height_variance': np.var(col_heights),
            'filled_cells': sum(1 for row in grid for cell in row if cell != (0, 0, 0)),
            'empty_cells': sum(1 for row in grid for cell in row if cell == (0, 0, 0)),
            'current_piece_type': shapes.index(player.current_piece.shape) if player.current_piece and player.current_piece.shape in shapes else -1,
            'next_piece_type': shapes.index(player.next_pieces[0].shape) if player.next_pieces and len(player.next_pieces) > 0 and player.next_pieces[0].shape in shapes else -1,
            'can_hold': player.can_hold,
            'score': player.score,
            'level': self.game.level
        }
        
        return features
    
    def reset_buffers(self):
        """Reset all learning buffers"""
        self.experience_buffer.clear()
        self.priority_weights.clear()
        self.world_model_data.clear()
        self.adaptation_buffer.clear()
        self.dream_states.clear()
    
    def get_info_dict(self, agent_idx: int = 0):
        """Get comprehensive info dictionary for learning algorithms"""
        base_info = {
            'episode_steps': self.episode_steps,
            'max_steps': self.max_steps,
            'step_mode': self.step_mode,
            'num_agents': self.num_agents
        }
        
        if agent_idx < len(self.players):
            player = self.players[agent_idx]
            base_info.update({
                'score': player.score,
                'level': self.game.level,
                'can_hold': player.can_hold,
                'pieces_placed': getattr(player, 'pieces_placed', 0)
            })
            
            # Add state features
            base_info.update(self.get_state_features(agent_idx))
        
        return base_info

    def step(self, action):
        """Execute one time step - supports both single and multi-agent with binary tuple actions"""
        self.episode_steps += 1
        
        if self.num_agents == 1:
            return self._step_single_agent(action)
        else:
            return self._step_multi_agent(action)
    
    def _step_single_agent(self, action):
        """Handle single agent step with binary tuple action"""
        player = self.players[0]
        action_idx = self._action_tuple_to_scalar(action)
        
        piece_placed = False
        lines_cleared = 0
        
        # Store previous state for learning algorithms
        prev_state = self._get_observation()
        
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
        
        # Add experience for DQN training
        self.add_experience(prev_state, action, reward, observation, done)
        
        # Add world model data for DREAM
        self.add_world_model_data(prev_state, action, observation, reward, done)
        
        # Store trajectory data
        if self.enable_trajectory_tracking and self.current_trajectory_id:
            trajectory = self.trajectories[self.current_trajectory_id]
            trajectory.add_step(observation, action, reward, {
                'lines_cleared': lines_cleared,
                'piece_placed': piece_placed,
                'game_over': game_over
            })
        
        info = self.get_info_dict(0)
        info.update({
            'lines_cleared': lines_cleared,
            'piece_placed': piece_placed,
            'game_over': game_over
        })
        
        return observation, reward, done, info
    
    def _step_multi_agent(self, actions):
        """Handle multi-agent step with binary tuple actions"""
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        # Store previous states for learning algorithms
        prev_states = {}
        for i in range(self.num_agents):
            if i < len(self.players):
                prev_states[f'agent_{i}'] = self._get_single_agent_observation(i)
        
        # Execute actions for all agents
        for i in range(self.num_agents):
            if i < len(self.players):
                agent_key = f'agent_{i}'
                action = actions.get(agent_key, tuple([0] * 8))
                action_idx = self._action_tuple_to_scalar(action)
                
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
                
                # Add experience for DQN training
                if agent_key in prev_states:
                    self.add_experience(
                        prev_states[agent_key], 
                        action, 
                        rewards[agent_key], 
                        observations[agent_key], 
                        dones[agent_key]
                    )
                
                infos[agent_key] = self.get_info_dict(i)
                infos[agent_key].update({
                    'lines_cleared': lines_cleared,
                    'piece_placed': piece_placed,
                    'game_over': game_over
                })
        
        # Global done condition
        done = any(dones.values()) or (self.episode_steps >= self.max_steps)
        
        return observations, rewards, done, infos
    
    def _execute_action(self, player, action_idx):
        """Execute a single action for a player"""
        piece_placed = False
        lines_cleared = 0
        
        try:
            # Import action handler
            try:
                from .game.action_handler import ActionHandler
            except ImportError:
                from game.action_handler import ActionHandler
            
            action_handler = ActionHandler(player)
            
            # Execute action
            if action_idx == 0:  # Move left
                action_handler.move_left()
            elif action_idx == 1:  # Move right
                action_handler.move_right()
            elif action_idx == 2:  # Move down (soft drop)
                result = action_handler.move_down()
                if not result:  # Piece couldn't move down, it's placed
                    piece_placed = True
                    lines_cleared = player.update(self.game.fall_speed, self.game.level)
            elif action_idx == 3:  # Rotate clockwise
                action_handler.rotate_cw()
            elif action_idx == 4:  # Rotate counter-clockwise
                action_handler.rotate_ccw()
            elif action_idx == 5:  # Hard drop
                action_handler.hard_drop()
                piece_placed = True
                lines_cleared = player.update(self.game.fall_speed, self.game.level)
            elif action_idx == 6:  # Hold piece
                action_handler.hold_piece()
            # action_idx == 7 is no-op
                
        except Exception as e:
            logger.error(f"Error executing action {action_idx}: {e}")
            # Continue without action
            
        return piece_placed, lines_cleared

    def _apply_gravity(self, player):
        """Apply gravity to a player's current piece"""
        piece_placed = False
        lines_cleared = 0
        
        try:
            from .game.action_handler import ActionHandler
        except ImportError:
            from game.action_handler import ActionHandler
        
        action_handler = ActionHandler(player)
        result = action_handler.move_down()
        if not result:  # Piece couldn't move down, it's placed
            piece_placed = True
            lines_cleared = player.update(self.game.fall_speed, self.game.level)
        else:
            piece_placed = False
            lines_cleared = 0
        
        return piece_placed, lines_cleared

    def _execute_until_placement(self, player, action_idx):
        """Execute actions until a piece is placed"""
        total_lines_cleared = 0
        
        # Execute the initial action
        piece_placed, lines_cleared = self._execute_action(player, action_idx)
        total_lines_cleared += lines_cleared
        
        # Continue applying gravity until piece is placed
        max_iterations = 50  # Safety limit
        iterations = 0
        
        while not piece_placed and iterations < max_iterations:
            piece_placed, lines_cleared = self._apply_gravity(player)
            total_lines_cleared += lines_cleared
            iterations += 1
            
            # Small delay to prevent infinite loops
            if not self.headless:
                pygame.time.wait(10)
        
        return piece_placed, total_lines_cleared

    def reset(self):
        """Reset the environment for a new episode"""
        # Initialize saved states
        self.saved_states = {}
        
        # Initialize the game
        self.game = Game(self.surface)
        self.players = []
        
        # Create shared block pool for multi-agent to ensure same block sequence
        try:
            from .game.block_pool import BlockPool
        except ImportError:
            from game.block_pool import BlockPool
        
        shared_block_pool = BlockPool()
        
        # Create players with shared block pool
        for i in range(self.num_agents):
            player = Player(shared_block_pool)
            self.players.append(player)
        
        # Reset training variables
        self.episode_steps = 0
        self.prev_features = {}
        
        # Reset trajectory tracking
        if self.enable_trajectory_tracking:
            self.trajectories.clear()
            self.current_trajectory_id = None
        
        # Reset episode buffer data for meta-learning
        self.episode_buffer.clear()
        
        # Don't reset experience buffer and world model data (persistent across episodes)
        # self.experience_buffer.clear()  # Keep for experience replay
        # self.world_model_data.clear()   # Keep for world model training
        
        return self._get_observation()

    def render(self, mode='human'):
        """Render the environment"""
        if self.headless:
            return
        
        try:
            # Clear surface
            self.surface.fill((0, 0, 0))
            
            # Render each player's game
            for i, player in enumerate(self.players):
                offset_x = i * (s_width // max(self.num_agents, 2))
                self.game.draw_window(self.surface, player, offset_x)
            
            # Update display
            pygame.display.update()
            self.clock.tick(60)
            
        except Exception as e:
            logger.error(f"Error during rendering: {e}")

    def close(self):
        """Close the environment"""
        if not self.headless:
            pygame.quit()

    def _action_tuple_to_scalar(self, action_tuple):
        """Convert binary tuple action to scalar"""
        # Find the first 1 in the tuple, or return 7 (no-op) if no 1s
        for i, bit in enumerate(action_tuple):
            if bit == 1:
                return i
        return 7  # No-op if no action selected

    def _action_scalar_to_tuple(self, action_scalar):
        """Convert scalar action to binary tuple"""
        action_tuple = [0] * 8
        if 0 <= action_scalar < 8:
            action_tuple[action_scalar] = 1
        return tuple(action_tuple)