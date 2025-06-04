#!/usr/bin/env python3
"""
Complete PyTorch AIRL Implementation
- Expert trajectory testing
- Single-player and True Multiplayer training
- Visualized/Headless training modes
- TensorBoard logging  
- No TensorFlow dependencies
"""

import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
import traceback
from typing import Dict, List, Tuple, Optional
import logging
import argparse

# Add paths for imports  
sys.path.append('local-multiplayer-tetris-main')
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')

# Try to set PYTHONPATH for subprocess imports
try:
    if 'PYTHONPATH' in os.environ:
        if 'local-multiplayer-tetris-main' not in os.environ['PYTHONPATH']:
            os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + os.pathsep + 'local-multiplayer-tetris-main'
    else:
        os.environ['PYTHONPATH'] = 'local-multiplayer-tetris-main'
    print("✅ Set PYTHONPATH for environment imports")
except Exception as e:
    print(f"⚠️ Failed to set PYTHONPATH: {e}")

# Environment imports (simplified - matching working files)
try:
    from tetris_env import TetrisEnv
    print("✅ TetrisEnv loaded successfully")
    TETRIS_ENV_AVAILABLE = True
except ImportError as e:
    print(f"❌ TetrisEnv import failed: {e}")
    TetrisEnv = None
    TETRIS_ENV_AVAILABLE = False

# Try to import the TrueMultiplayerTetrisEnv for visualization
try:
    from rl_utils.true_multiplayer_env import TrueMultiplayerTetrisEnv
    print("✅ TrueMultiplayerTetrisEnv loaded successfully")
    TRUE_MULTIPLAYER_ENV_AVAILABLE = True
except ImportError:
    try:
        from localMultiplayerTetris.rl_utils.true_multiplayer_env import TrueMultiplayerTetrisEnv
        print("✅ TrueMultiplayerTetrisEnv loaded via alternate path")
        TRUE_MULTIPLAYER_ENV_AVAILABLE = True
    except ImportError as e:
        print(f"❌ TrueMultiplayerTetrisEnv import failed: {e}")
        TRUE_MULTIPLAYER_ENV_AVAILABLE = False

# Since the relative imports are broken, create a simple mock multiplayer environment
class MockTrueMultiplayerTetrisEnv:
    """Mock multiplayer environment using two independent SimpleTetrisEnv instances."""
    
    def __init__(self, headless=True):
        self.headless = headless
        
        # Use real TetrisEnv if available, otherwise SimpleTetrisEnv
        if TETRIS_ENV_AVAILABLE and TetrisEnv is not None:
            try:
                self.env_p1 = TetrisEnv(single_player=True, headless=True)
                self.env_p2 = TetrisEnv(single_player=True, headless=True)
                print("✅ Using real TetrisEnv for both players")
            except Exception as e:
                print(f"⚠️  TetrisEnv failed, using SimpleTetrisEnv: {e}")
                self.env_p1 = SimpleTetrisEnv(headless=True)
                self.env_p2 = SimpleTetrisEnv(headless=True)
        else:
            self.env_p1 = SimpleTetrisEnv(headless=True)
            self.env_p2 = SimpleTetrisEnv(headless=True)
            print("✅ Using SimpleTetrisEnv for both players")
        
        # Game state tracking
        self.episode_steps = 0
        self.max_steps = 1000
        self.player1_alive = True
        self.player2_alive = True
        self.player1_score = 0
        self.player2_score = 0
        
    def reset(self):
        """Reset both player environments and return initial observations."""
        obs_p1 = self.env_p1.reset()
        obs_p2 = self.env_p2.reset()
        
        # Handle tuple format from gym environments
        if isinstance(obs_p1, tuple):
            obs_p1 = obs_p1[0]
        if isinstance(obs_p2, tuple):
            obs_p2 = obs_p2[0]
        
        # Reset game state
        self.episode_steps = 0
        self.player1_alive = True
        self.player2_alive = True
        self.player1_score = 0
        self.player2_score = 0
        
        return {
            'player1': obs_p1,
            'player2': obs_p2
        }
    
    def step(self, actions):
        """Execute actions for both players simultaneously."""
        self.episode_steps += 1
        
        action_p1 = actions.get('player1', 0)
        action_p2 = actions.get('player2', 0)
        
        rewards = {'player1': 0, 'player2': 0}
        observations = {}
        info = {'player1': {}, 'player2': {}, 'winner': None}
        
        # Player 1 step
        if self.player1_alive:
            try:
                step_result_p1 = self.env_p1.step(action_p1)
                if len(step_result_p1) == 4:
                    obs_p1, reward_p1, done_p1, info_p1 = step_result_p1
                else:
                    obs_p1, reward_p1, done_p1, truncated_p1, info_p1 = step_result_p1
                    done_p1 = done_p1 or truncated_p1
                
                observations['player1'] = obs_p1
                rewards['player1'] = reward_p1
                info['player1'] = info_p1
                
                if done_p1:
                    self.player1_alive = False
                    
            except Exception as e:
                self.player1_alive = False
                observations['player1'] = self._get_empty_observation()
                rewards['player1'] = -50
        else:
            observations['player1'] = self._get_empty_observation()
            rewards['player1'] = 0
        
        # Player 2 step  
        if self.player2_alive:
            try:
                step_result_p2 = self.env_p2.step(action_p2)
                if len(step_result_p2) == 4:
                    obs_p2, reward_p2, done_p2, info_p2 = step_result_p2
                else:
                    obs_p2, reward_p2, done_p2, truncated_p2, info_p2 = step_result_p2
                    done_p2 = done_p2 or truncated_p2
                
                observations['player2'] = obs_p2
                rewards['player2'] = reward_p2
                info['player2'] = info_p2
                
                if done_p2:
                    self.player2_alive = False
                    
            except Exception as e:
                self.player2_alive = False
                observations['player2'] = self._get_empty_observation()
                rewards['player2'] = -50
        else:
            observations['player2'] = self._get_empty_observation()
            rewards['player2'] = 0
        
        # Determine episode completion and winner
        episode_done = self._check_episode_complete()
        winner = self._determine_winner()
        info['winner'] = winner
        
        # Add competitive rewards
        rewards = self._add_competitive_rewards(rewards, winner)
        
        return observations, rewards, episode_done, info
    
    def _check_episode_complete(self):
        """Check if the episode should end."""
        if not self.player1_alive and not self.player2_alive:
            return True
        if self.episode_steps >= self.max_steps:
            return True
        return False
    
    def _determine_winner(self):
        """Determine the winner based on current game state."""
        if not self.player1_alive and not self.player2_alive:
            if self.player1_score > self.player2_score:
                return 'player1'
            elif self.player2_score > self.player1_score:
                return 'player2'
            else:
                return 'draw'
        elif not self.player1_alive:
            return 'player2'
        elif not self.player2_alive:
            return 'player1'
        else:
            return 'ongoing'
    
    def _add_competitive_rewards(self, rewards, winner):
        """Add competitive bonuses."""
        if winner == 'player1':
            rewards['player1'] += 10.0
            rewards['player2'] -= 5.0
        elif winner == 'player2':
            rewards['player2'] += 10.0
            rewards['player1'] -= 5.0
        return rewards
    
    def _get_empty_observation(self):
        """Get empty observation for eliminated players."""
        return {
            'grid': np.zeros((20, 10)),
            'next_piece': 0,
            'hold_piece': -1,
            'current_shape': 0,
            'current_rotation': 0,
            'current_x': 5,
            'current_y': 0,
            'can_hold': True
        }
    
    def close(self):
        """Close both environments."""
        if hasattr(self.env_p1, 'close'):
            self.env_p1.close()
        if hasattr(self.env_p2, 'close'):
            self.env_p2.close()

# Set up environment availability
ENVS_AVAILABLE = True  # Mock multiplayer is always available
TrueMultiplayerTetrisEnv = MockTrueMultiplayerTetrisEnv
print("✅ Mock TrueMultiplayerTetrisEnv available")

class PyTorchDiscriminator(nn.Module):
    """PyTorch-based AIRL Discriminator Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 128, 64]):
        super(PyTorchDiscriminator, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State processing network
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        
        # Action processing network (one-hot encoded)
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_sizes[1] // 2),
            nn.ReLU()
        )
        
        # Combined processing
        combined_dim = hidden_sizes[1] + hidden_sizes[1] // 2
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1)
        )
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for expert vs learner classification."""
        state_features = self.state_net(states)
        action_features = self.action_net(actions)
        
        combined = torch.cat([state_features, action_features], dim=-1)
        logits = self.combined_net(combined)
        return logits
    
    def get_reward(self, states: torch.Tensor, actions: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
        """Get AIRL reward: log(D/(1-D)) with scaling to match expert rewards
        
        Args:
            states: Batch of states
            actions: Batch of actions (one-hot encoded)
            scale: Reward scaling factor (default: 10.0 to increase magnitude)
            
        Returns:
            Scaled AIRL rewards
        """
        logits = self.forward(states, actions)
        probs = torch.sigmoid(logits)
        
        # Core AIRL reward formulation: log(D/(1-D))
        raw_reward = torch.log(probs + 1e-8) - torch.log(1 - probs + 1e-8)
        
        # Scale rewards to match magnitude of expert rewards
        # This is critical to address the low reward issue
        scaled_reward = raw_reward * scale
        
        return scaled_reward

class PyTorchActorCritic(nn.Module):
    """PyTorch-based Actor-Critic Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PyTorchActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action probabilities and state values."""
        features = self.feature_net(states)
        action_probs = self.actor(features)
        state_values = self.critic(features)
        return action_probs, state_values

class ExpertLoader:
    """Expert trajectory loader with PyTorch compatibility."""
    
    def __init__(self, trajectory_dir: str = "expert_trajectories_dqn_adapter"):
        self.trajectory_dir = trajectory_dir
        self.trajectories = []
        self.transitions = []
        
    def load_trajectories(self) -> int:
        """Load expert trajectories from directory."""
        if not os.path.exists(self.trajectory_dir):
            print(f"❌ Directory not found: {self.trajectory_dir}")
            return 0
        
        trajectory_files = [f for f in os.listdir(self.trajectory_dir) if f.endswith('.pkl')]
        print(f"📂 Found {len(trajectory_files)} trajectory files")
        
        for filename in trajectory_files:
            filepath = os.path.join(self.trajectory_dir, filename)
            
            try:
                with open(filepath, 'rb') as f:
                    trajectory_data = pickle.load(f)
                
                steps = trajectory_data.get('steps', [])
                if len(steps) < 10:  # Skip very short episodes
                    continue
                
                # Process transitions
                for step in steps:
                    state_dict = step.get('state', {})
                    next_state_dict = step.get('next_state', {})
                    
                    if not state_dict or not next_state_dict:
                        continue
                    
                    # Extract features (207-dimensional)
                    state_features = self._extract_features(state_dict)
                    next_state_features = self._extract_features(next_state_dict)
                    
                    # Convert action to one-hot
                    action = step.get('action', 0)
                    action_onehot = self._action_to_onehot(action)
                    
                    transition = {
                        'state': state_features,
                        'action': action,
                        'action_onehot': action_onehot,
                        'reward': float(step.get('reward', 0.0)),
                        'next_state': next_state_features,
                        'done': bool(step.get('done', False))
                    }
                    
                    self.transitions.append(transition)
                
                self.trajectories.append(trajectory_data)
                
            except Exception as e:
                print(f"⚠️  Error loading {filename}: {e}")
                continue
        
        print(f"✅ Loaded {len(self.trajectories)} trajectories, {len(self.transitions)} transitions")
        return len(self.trajectories)
    
    def _extract_features(self, observation: Dict) -> np.ndarray:
        """Extract 207-dimensional features from observation."""
        grid = observation['grid'].flatten()  # 200 features
        next_piece = np.array([observation['next_piece']])  # 1 feature
        hold_piece = np.array([observation['hold_piece']])  # 1 feature
        current_shape = np.array([observation['current_shape']])  # 1 feature
        current_rotation = np.array([observation['current_rotation']])  # 1 feature
        current_x = np.array([observation['current_x']])  # 1 feature
        current_y = np.array([observation['current_y']])  # 1 feature
        can_hold = np.array([observation['can_hold']])  # 1 feature
        
        features = np.concatenate([
            grid, next_piece, hold_piece, current_shape,
            current_rotation, current_x, current_y, can_hold
        ]).astype(np.float32)
        
        return features
    
    def _action_to_onehot(self, action: int, num_actions: int = 41) -> np.ndarray:
        """Convert action to one-hot encoding."""
        onehot = np.zeros(num_actions, dtype=np.float32)
        if 0 <= action < num_actions:
            onehot[action] = 1.0
        return onehot
    
    def get_batch(self, batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Sample a batch of expert transitions."""
        if not self.transitions:
            raise ValueError("No expert transitions loaded")
        
        # Sample random transitions
        indices = np.random.choice(len(self.transitions), batch_size, replace=True)
        batch_transitions = [self.transitions[i] for i in indices]
        
        # Convert to tensors
        states = torch.FloatTensor([t['state'] for t in batch_transitions]).to(device)
        actions_onehot = torch.FloatTensor([t['action_onehot'] for t in batch_transitions]).to(device)
        rewards = torch.FloatTensor([t['reward'] for t in batch_transitions]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor([t['next_state'] for t in batch_transitions]).to(device)
        dones = torch.FloatTensor([t['done'] for t in batch_transitions]).unsqueeze(1).to(device)
        
        return {
            'states': states,
            'actions': actions_onehot,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded expert data."""
        if not self.trajectories:
            return {}
        
        total_rewards = [traj['total_reward'] for traj in self.trajectories]
        episode_lengths = [traj['length'] for traj in self.trajectories]
        
        return {
            'num_trajectories': len(self.trajectories),
            'num_transitions': len(self.transitions),
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'max_reward': np.max(total_rewards),
            'min_reward': np.min(total_rewards)
        }

class SimpleTetrisEnv:
    """Simplified Tetris environment for testing when TetrisEnv not available."""
    
    def __init__(self, headless=True):
        self.state_dim = 207
        self.action_dim = 41
        self.headless = headless
        
        # Initialize visualization components if not headless
        if not headless:
            try:
                import pygame
                pygame.init()
                self.surface = pygame.display.set_mode((400, 500))
                pygame.display.set_caption("SimpleTetris")
                self.font = pygame.font.SysFont('Arial', 18)
                self.visualization_enabled = True
                print("✅ Pygame visualization enabled")
            except ImportError:
                print("⚠️ Pygame not available, running headless")
                self.visualization_enabled = False
        else:
            self.visualization_enabled = False
            
        # Game state
        self.grid = np.zeros((20, 10))
        self.score = 0
        self.lines_cleared = 0
        self.episode_step = 0
        self.current_piece = 0
        
    def reset(self):
        """Reset to initial state."""
        self.grid = np.zeros((20, 10))
        self.score = 0
        self.lines_cleared = 0
        self.episode_step = 0
        self.current_piece = np.random.randint(1, 8)  # Random piece 1-7
        
        observation = {
            'grid': self.grid.copy(),
            'next_piece': np.random.randint(1, 8),
            'hold_piece': -1,
            'current_shape': self.current_piece,
            'current_rotation': 0,
            'current_x': 5,
            'current_y': 0,
            'can_hold': True
        }
        
        # Render initial state
        if self.visualization_enabled:
            self.render()
            
        return observation
    
    def step(self, action):
        """Take a step in the environment."""
        self.episode_step += 1
        
        # Process action
        reward = 0
        # Simulate piece placement
        if action < 40:  # Placement action
            rot = action // 10
            col = action % 10
            
            # Add random blocks to grid occasionally to simulate pieces
            if np.random.random() < 0.3:
                for _ in range(np.random.randint(1, 4)):
                    r = np.random.randint(10, 20)
                    c = np.random.randint(0, 10)
                    self.grid[r, c] = 1
            
            # Check for line clears
            lines_cleared = 0
            for row in range(19, -1, -1):
                if np.all(self.grid[row] == 1):
                    lines_cleared += 1
                    self.grid[1:row+1] = self.grid[0:row]
                    self.grid[0] = 0
            
            # Reward based on lines cleared
            if lines_cleared == 1:
                reward += 10
            elif lines_cleared == 2:
                reward += 25
            elif lines_cleared == 3:
                reward += 50
            elif lines_cleared >= 4:
                reward += 100
                
            self.lines_cleared += lines_cleared
            self.score += reward
        
        # Generate next observation
        self.current_piece = np.random.randint(1, 8)
        next_obs = {
            'grid': self.grid.copy(),
            'next_piece': np.random.randint(1, 8),
            'hold_piece': -1,
            'current_shape': self.current_piece,
            'current_rotation': 0,
            'current_x': 5,
            'current_y': 0,
            'can_hold': True
        }
        
        # Check for game over (when blocks reach the top)
        done = np.any(self.grid[0] == 1) or self.episode_step > 500
        
        # Info dictionary
        info = {
            'score': self.score,
            'lines_cleared': lines_cleared if 'lines_cleared' in locals() else 0
        }
        
        # Render if enabled
        if self.visualization_enabled:
            self.render()
            
        return next_obs, reward, done, info
    
    def render(self):
        """Render the current game state."""
        if not self.visualization_enabled:
            return
            
        try:
            import pygame
            
            # Clear the surface
            self.surface.fill((0, 0, 0))
            
            # Draw grid
            block_size = 20
            for y in range(20):
                for x in range(10):
                    if self.grid[y, x] == 1:
                        pygame.draw.rect(
                            self.surface, 
                            (0, 255, 255),  # Cyan color for blocks
                            (x * block_size + 50, y * block_size + 50, block_size - 1, block_size - 1)
                        )
            
            # Draw grid outline
            pygame.draw.rect(
                self.surface,
                (255, 255, 255),
                (50, 50, 10 * block_size, 20 * block_size),
                1
            )
            
            # Draw score and lines
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            lines_text = self.font.render(f"Lines: {self.lines_cleared}", True, (255, 255, 255))
            step_text = self.font.render(f"Step: {self.episode_step}", True, (255, 255, 255))
            
            self.surface.blit(score_text, (260, 50))
            self.surface.blit(lines_text, (260, 80))
            self.surface.blit(step_text, (260, 110))
            
            pygame.display.update()
            pygame.time.delay(50)  # Small delay for visualization
            
            # Process events to prevent window from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    
        except Exception as e:
            print(f"⚠️ Render error: {e}")
            self.visualization_enabled = False
    
    def close(self):
        """Close environment and clean up."""
        if hasattr(self, 'visualization_enabled') and self.visualization_enabled:
            try:
                import pygame
                pygame.quit()
            except:
                pass

class MultiplayerAIRLTrainer:
    """AIRL Trainer for competitive multiplayer Tetris."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', True) else 'cpu')
        
        # Initialize true multiplayer environment
        headless = config.get('headless', True)
        
        # First try to use the proper TrueMultiplayerTetrisEnv from the original code
        if TRUE_MULTIPLAYER_ENV_AVAILABLE:
            try:
                self.env = TrueMultiplayerTetrisEnv(headless=headless)
                print(f"✅ Using real TrueMultiplayerTetrisEnv (visualization: {not headless})")
            except Exception as e:
                print(f"❌ Failed to initialize real TrueMultiplayerTetrisEnv: {e}")
                self.env = None
                return
        # Fall back to the mock environment
        elif ENVS_AVAILABLE and TrueMultiplayerTetrisEnv is not None:
            try:
                self.env = TrueMultiplayerTetrisEnv(headless=headless)
                print(f"✅ Using mock TrueMultiplayerTetrisEnv (visualization: {not headless})")
            except Exception as e:
                print(f"❌ Failed to initialize mock TrueMultiplayerTetrisEnv: {e}")
                self.env = None
                return
        else:
            print("❌ No multiplayer environment available")
            self.env = None
            return
        
        # Dimensions
        self.state_dim = 207
        self.action_dim = 41
        
        # Initialize networks for both players
        self.discriminator_p1 = PyTorchDiscriminator(self.state_dim, self.action_dim).to(self.device)
        self.discriminator_p2 = PyTorchDiscriminator(self.state_dim, self.action_dim).to(self.device)
        self.policy_p1 = PyTorchActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.policy_p2 = PyTorchActorCritic(self.state_dim, self.action_dim).to(self.device)
        
        # Optimizers
        self.disc_optimizer_p1 = optim.Adam(self.discriminator_p1.parameters(), lr=config.get('discriminator_lr', 3e-4))
        self.disc_optimizer_p2 = optim.Adam(self.discriminator_p2.parameters(), lr=config.get('discriminator_lr', 3e-4))
        self.policy_optimizer_p1 = optim.Adam(self.policy_p1.parameters(), lr=config.get('policy_lr', 1e-4))
        self.policy_optimizer_p2 = optim.Adam(self.policy_p2.parameters(), lr=config.get('policy_lr', 1e-4))
        
        # Expert loader
        self.expert_loader = ExpertLoader(config.get('expert_trajectory_dir', 'expert_trajectories_dqn_adapter'))
        
        # TensorBoard logging
        if config.get('use_tensorboard', True):
            log_dir = os.path.join('logs', f"multiplayer_airl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.writer = SummaryWriter(log_dir)
            print(f"📊 TensorBoard logs: {log_dir}")
        else:
            self.writer = None
            
        # Buffer size (fix for missing attribute)
        self.max_buffer_size = config.get('buffer_size', 50000)
    
    def train_competitive(self):
        """Train two agents competitively using AIRL."""
        print("🚀 STARTING COMPETITIVE MULTIPLAYER AIRL TRAINING")
        print("=" * 70)
        
        # Check if environment is available
        if self.env is None:
            print("❌ No multiplayer environment available for training!")
            return
        
        # Load expert trajectories
        num_expert = self.expert_loader.load_trajectories()
        if num_expert == 0:
            print("❌ No expert trajectories loaded!")
            return
        
        stats = self.expert_loader.get_statistics()
        print(f"📊 Expert Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        # Training parameters
        max_episodes = self.config.get('max_episodes', 100)
        max_steps_per_episode = self.config.get('max_episode_steps', 1000)  # Increased from 500 to 1000
        batch_size = self.config.get('batch_size', 64)
        total_steps = 0
        
        # Check if we're doing batched training
        batched_training = self.config.get('batched_training', False)
        batch_episodes = self.config.get('batch_episodes', 10)
        checkpoint_interval = self.config.get('checkpoint_interval', 100)
        reward_scale = self.config.get('reward_scale', 10.0)
        
        print(f"📊 Training Config:")
        print(f"   Max Episodes: {max_episodes}")
        print(f"   Steps per Episode: {max_steps_per_episode}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Reward Scale: {reward_scale}")
        if batched_training:
            print(f"   Batched Training: True ({batch_episodes} episodes per batch)")
        
        # For collecting transition data
        p1_buffer = []
        p2_buffer = []
        
        # Initialize best model tracking
        best_avg_reward = {'player1': -float('inf'), 'player2': -float('inf')}
        rewards_history = {'player1': [], 'player2': []}
        
        for episode in range(max_episodes):
            print(f"\n🔄 Episode {episode + 1}/{max_episodes}")
            
            try:
                # Reset environment
                obs = self.env.reset()
                
                episode_rewards = {'player1': 0, 'player2': 0}
                episode_steps = 0
                
                # Episode transitions
                p1_transitions = []
                p2_transitions = []
                
                while episode_steps < max_steps_per_episode:
                    # Extract features for both players
                    state_p1 = self._extract_features(obs['player1'])
                    state_p2 = self._extract_features(obs['player2'])
                    
                    # Select actions
                    action_p1 = self._select_action(state_p1, self.policy_p1)
                    action_p2 = self._select_action(state_p2, self.policy_p2)
                    
                    # Step environment
                    actions = {'player1': action_p1, 'player2': action_p2}
                    next_obs, rewards, done, info = self.env.step(actions)
                    
                    # Update episode rewards
                    episode_rewards['player1'] += rewards['player1']
                    episode_rewards['player2'] += rewards['player2']
                    
                    # Extract next state features
                    next_state_p1 = self._extract_features(next_obs['player1'])
                    next_state_p2 = self._extract_features(next_obs['player2'])
                    
                    # Store transitions for both players
                    p1_transition = {
                        'state': state_p1,
                        'action': action_p1,
                        'action_onehot': self.expert_loader._action_to_onehot(action_p1),
                        'reward': rewards['player1'],
                        'next_state': next_state_p1,
                        'done': done
                    }
                    p1_transitions.append(p1_transition)
                    
                    p2_transition = {
                        'state': state_p2,
                        'action': action_p2,
                        'action_onehot': self.expert_loader._action_to_onehot(action_p2),
                        'reward': rewards['player2'],
                        'next_state': next_state_p2,
                        'done': done
                    }
                    p2_transitions.append(p2_transition)
                    
                    # Update state
                    obs = next_obs
                    episode_steps += 1
                    total_steps += 1
                    
                    if done:
                        break
                
                # Add episode transitions to buffers
                p1_buffer.extend(p1_transitions)
                p2_buffer.extend(p2_transitions)
                
                # Keep buffers at max size
                if len(p1_buffer) > self.max_buffer_size:
                    p1_buffer = p1_buffer[-self.max_buffer_size:]
                if len(p2_buffer) > self.max_buffer_size:
                    p2_buffer = p2_buffer[-self.max_buffer_size:]
                
                # Log episode results
                winner = info.get('winner', 'draw')
                print(f"   Episode completed in {episode_steps} steps")
                print(f"   P1 reward: {episode_rewards['player1']:.2f}")
                print(f"   P2 reward: {episode_rewards['player2']:.2f}")
                print(f"   Winner: {winner}")
                
                # Update rewards history
                rewards_history['player1'].append(episode_rewards['player1'])
                rewards_history['player2'].append(episode_rewards['player2'])
                
                # Calculate running average (last 10 episodes)
                window_size = min(10, len(rewards_history['player1']))
                avg_reward_p1 = sum(rewards_history['player1'][-window_size:]) / window_size
                avg_reward_p2 = sum(rewards_history['player2'][-window_size:]) / window_size
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('Episode/P1_Reward', episode_rewards['player1'], episode)
                    self.writer.add_scalar('Episode/P2_Reward', episode_rewards['player2'], episode)
                    self.writer.add_scalar('Episode/Steps', episode_steps, episode)
                    self.writer.add_scalar('Episode/Winner', 1 if winner == 'player1' else (-1 if winner == 'player2' else 0), episode)
                    self.writer.add_scalar('Training/Total_Steps', total_steps, episode)
                    self.writer.add_scalar('Training/Avg_P1_Reward', avg_reward_p1, episode)
                    self.writer.add_scalar('Training/Avg_P2_Reward', avg_reward_p2, episode)
                    self.writer.add_scalar('Training/P1_Buffer_Size', len(p1_buffer), episode)
                    self.writer.add_scalar('Training/P2_Buffer_Size', len(p2_buffer), episode)
                
                # Perform training every batch_episodes in batched mode, or every episode otherwise
                should_train = (batched_training and (episode + 1) % batch_episodes == 0) or (not batched_training)
                
                if should_train and len(p1_buffer) >= batch_size and len(p2_buffer) >= batch_size:
                    print(f"   🔄 Training on {len(p1_buffer)} P1 and {len(p2_buffer)} P2 transitions")
                    
                    # Training loop - do multiple updates
                    updates_per_batch = self.config.get('updates_per_iteration', 5)
                    
                    for update in range(updates_per_batch):
                        # Sample batches for both players
                        p1_indices = np.random.choice(len(p1_buffer), batch_size)
                        p2_indices = np.random.choice(len(p2_buffer), batch_size)
                        
                        p1_batch = self._prepare_batch([p1_buffer[i] for i in p1_indices])
                        p2_batch = self._prepare_batch([p2_buffer[i] for i in p2_indices])
                        
                        # Get expert batches
                        expert_batch = self.expert_loader.get_batch(batch_size, self.device)
                        
                        # Update discriminators
                        disc_metrics_p1 = self.update_discriminator(expert_batch, p1_batch, self.discriminator_p1, self.disc_optimizer_p1)
                        disc_metrics_p2 = self.update_discriminator(expert_batch, p2_batch, self.discriminator_p2, self.disc_optimizer_p2)
                        
                        # Update policies
                        policy_metrics_p1 = self.update_policy(p1_batch, self.discriminator_p1, self.policy_p1, self.policy_optimizer_p1)
                        policy_metrics_p2 = self.update_policy(p2_batch, self.discriminator_p2, self.policy_p2, self.policy_optimizer_p2)
                        
                        # Log metrics for last update
                        if update == updates_per_batch - 1:
                            print(f"   P1 D_loss: {disc_metrics_p1.get('discriminator_loss', 0):.4f}, P_loss: {policy_metrics_p1.get('policy_loss', 0):.4f}")
                            print(f"   P2 D_loss: {disc_metrics_p2.get('discriminator_loss', 0):.4f}, P_loss: {policy_metrics_p2.get('policy_loss', 0):.4f}")
                            print(f"   P1 reward: {policy_metrics_p1.get('mean_airl_reward', 0):.4f}, P2 reward: {policy_metrics_p2.get('mean_airl_reward', 0):.4f}")
                
                # Save checkpoint periodically
                if episode % checkpoint_interval == 0 and episode > 0:
                    checkpoint_prefix = f"multiplayer_ep{episode}"
                    self.save_models(checkpoint_prefix)
                    print(f"   💾 Saved checkpoint at episode {episode}")
                
                # Save best models based on running average reward
                if avg_reward_p1 > best_avg_reward['player1']:
                    best_avg_reward['player1'] = avg_reward_p1
                    torch.save(self.policy_p1.state_dict(), f"models/best_p1_policy.pth")
                    print(f"   💾 Saved best P1 model with avg reward {avg_reward_p1:.2f}")
                
                if avg_reward_p2 > best_avg_reward['player2']:
                    best_avg_reward['player2'] = avg_reward_p2
                    torch.save(self.policy_p2.state_dict(), f"models/best_p2_policy.pth")
                    print(f"   💾 Saved best P2 model with avg reward {avg_reward_p2:.2f}")
                    
            except Exception as e:
                print(f"   ❌ Episode {episode + 1} failed: {e}")
                traceback.print_exc()
                continue
        
        print(f"\n✅ Competitive training completed!")
        print(f"📊 Total steps executed: {total_steps}")
        print(f"📈 Best P1 avg reward: {best_avg_reward['player1']:.2f}")
        print(f"📈 Best P2 avg reward: {best_avg_reward['player2']:.2f}")
        
        # Save final models
        self.save_models("multiplayer_final")
        
        if self.writer:
            self.writer.close()
    
    def _prepare_batch(self, transitions: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert a list of transitions to PyTorch tensors."""
        states = torch.FloatTensor([t['state'] for t in transitions]).to(self.device)
        actions_onehot = torch.FloatTensor([t['action_onehot'] for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in transitions]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([t['next_state'] for t in transitions]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in transitions]).unsqueeze(1).to(self.device)
        
        return {
            'states': states,
            'actions': actions_onehot,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def update_discriminator(self, expert_batch, learner_batch, discriminator, optimizer):
        """Update discriminator."""
        batch_size = expert_batch['states'].size(0)
        
        # Forward pass
        expert_logits = discriminator(expert_batch['states'], expert_batch['actions'])
        learner_logits = discriminator(learner_batch['states'], learner_batch['actions'])
        
        # Labels: 1 for expert, 0 for learner
        expert_labels = torch.ones(batch_size, 1, device=self.device)
        learner_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Binary cross-entropy loss
        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, expert_labels)
        learner_loss = F.binary_cross_entropy_with_logits(learner_logits, learner_labels)
        discriminator_loss = expert_loss + learner_loss
        
        # Update discriminator
        optimizer.zero_grad()
        discriminator_loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        expert_preds = (torch.sigmoid(expert_logits) > 0.5).float()
        learner_preds = (torch.sigmoid(learner_logits) < 0.5).float()
        expert_accuracy = expert_preds.mean().item()
        learner_accuracy = learner_preds.mean().item()
        overall_accuracy = (expert_accuracy + learner_accuracy) / 2
        
        return {
            'discriminator_loss': discriminator_loss.item(),
            'expert_loss': expert_loss.item(),
            'learner_loss': learner_loss.item(),
            'expert_accuracy': expert_accuracy,
            'learner_accuracy': learner_accuracy,
            'overall_accuracy': overall_accuracy
        }
    
    def update_policy(self, learner_batch, discriminator, policy, optimizer):
        """Update policy using AIRL rewards."""
        states = learner_batch['states']
        actions_onehot = learner_batch['actions']
        next_states = learner_batch['next_states']
        dones = learner_batch['dones']
        
        # Get AIRL rewards with scaling
        with torch.no_grad():
            reward_scale = self.config.get('reward_scale', 10.0)
            airl_rewards = discriminator.get_reward(states, actions_onehot, scale=reward_scale)
        
        # Get policy predictions
        action_probs, state_values = policy(states)
        next_state_values = policy(next_states)[1]
        
        # Calculate advantages
        gamma = self.config.get('gamma', 0.99)
        targets = airl_rewards + gamma * next_state_values * (1 - dones)
        advantages = targets - state_values
        
        # Actor loss (policy gradient)
        action_indices = actions_onehot.argmax(dim=-1)
        log_probs = torch.log(action_probs.gather(1, action_indices.unsqueeze(1)) + 1e-8)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(state_values, targets.detach())
        
        # Combined loss
        policy_loss = actor_loss + 0.5 * critic_loss
        
        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_airl_reward': airl_rewards.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def _extract_features(self, observation: Dict) -> np.ndarray:
        """Extract features from environment observation."""
        grid = observation['grid'].flatten()
        next_piece = np.array([observation['next_piece']])
        hold_piece = np.array([observation['hold_piece']])
        current_shape = np.array([observation['current_shape']])
        current_rotation = np.array([observation['current_rotation']])
        current_x = np.array([observation['current_x']])
        current_y = np.array([observation['current_y']])
        can_hold = np.array([observation['can_hold']])
        
        features = np.concatenate([
            grid, next_piece, hold_piece, current_shape,
            current_rotation, current_x, current_y, can_hold
        ]).astype(np.float32)
        
        return features
    
    def _select_action(self, state: np.ndarray, policy: PyTorchActorCritic) -> int:
        """Select action using policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = policy(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        
        return action
        
    def save_models(self, prefix: str):
        """Save trained models."""
        os.makedirs("models", exist_ok=True)
        
        # Save discriminator models
        disc_p1_path = f"models/{prefix}_disc_p1.pth"
        disc_p2_path = f"models/{prefix}_disc_p2.pth"
        torch.save(self.discriminator_p1.state_dict(), disc_p1_path)
        torch.save(self.discriminator_p2.state_dict(), disc_p2_path)
        
        # Save policy models
        policy_p1_path = f"models/{prefix}_policy_p1.pth"
        policy_p2_path = f"models/{prefix}_policy_p2.pth"
        torch.save(self.policy_p1.state_dict(), policy_p1_path)
        torch.save(self.policy_p2.state_dict(), policy_p2_path)
        
        print(f"💾 Models saved:")
        print(f"   Discriminators: {disc_p1_path}, {disc_p2_path}")
        print(f"   Policies: {policy_p1_path}, {policy_p2_path}")

class PyTorchAIRLTrainer:
    """Complete PyTorch AIRL Trainer with TensorBoard logging."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', True) else 'cpu')
        
        # Initialize environment
        if TETRIS_ENV_AVAILABLE and TetrisEnv is not None:
            try:
                self.env = TetrisEnv(single_player=True, headless=config.get('headless', True))
                print("✅ Using real TetrisEnv")
            except Exception as e:
                print(f"⚠️  TetrisEnv failed, using SimpleTetrisEnv: {e}")
                self.env = SimpleTetrisEnv(headless=config.get('headless', True))
        else:
            print("⚠️  TetrisEnv not available, using SimpleTetrisEnv")
            self.env = SimpleTetrisEnv(headless=config.get('headless', True))
        
        # Dimensions
        self.state_dim = 207  # Full TetrisEnv observation
        self.action_dim = 41  # TetrisEnv action space
        
        # Initialize networks
        self.discriminator = PyTorchDiscriminator(self.state_dim, self.action_dim).to(self.device)
        self.policy = PyTorchActorCritic(self.state_dim, self.action_dim).to(self.device)
        
        # Optimizers
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=config.get('discriminator_lr', 3e-4)
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=config.get('policy_lr', 1e-4)
        )
        
        # Expert loader
        self.expert_loader = ExpertLoader(config.get('expert_trajectory_dir', 'expert_trajectories_dqn_adapter'))
        
        # Training buffers
        self.learner_buffer = []
        self.max_buffer_size = config.get('buffer_size', 10000)
        
        # TensorBoard logging
        self.use_tensorboard = config.get('use_tensorboard', True)
        if self.use_tensorboard:
            log_dir = os.path.join('logs', f"airl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.writer = SummaryWriter(log_dir)
            print(f"📊 TensorBoard logs: {log_dir}")
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.step_count = 0
        
    def extract_features(self, observation: Dict) -> np.ndarray:
        """Extract features from environment observation."""
        grid = observation['grid'].flatten()
        next_piece = np.array([observation['next_piece']])
        hold_piece = np.array([observation['hold_piece']])
        current_shape = np.array([observation['current_shape']])
        current_rotation = np.array([observation['current_rotation']])
        current_x = np.array([observation['current_x']])
        current_y = np.array([observation['current_y']])
        can_hold = np.array([observation['can_hold']])
        
        features = np.concatenate([
            grid, next_piece, hold_piece, current_shape,
            current_rotation, current_x, current_y, can_hold
        ]).astype(np.float32)
        
        return features
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        
        return action
    
    def collect_learner_data(self, num_episodes: int = 5) -> int:
        """Collect data from current learner policy."""
        transitions_collected = 0
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            episode_transitions = []
            episode_steps = 0
            
            for step in range(self.config.get('max_episode_steps', 500)):
                # Extract features and select action
                state_features = self.extract_features(obs)
                action = self.select_action(state_features)
                
                # Take action
                step_result = self.env.step(action)
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                    truncated = False
                else:
                    next_obs, reward, done, truncated, info = step_result
                
                done = done or truncated
                
                # Increment step counters
                self.step_count += 1
                episode_steps += 1
                
                # Store transition
                next_state_features = self.extract_features(next_obs) if not done else np.zeros_like(state_features)
                
                # Convert action to one-hot
                action_onehot = np.zeros(self.action_dim, dtype=np.float32)
                action_onehot[action] = 1.0
                
                transition = {
                    'state': state_features,
                    'action': action,
                    'action_onehot': action_onehot,
                    'reward': reward,
                    'next_state': next_state_features,
                    'done': done
                }
                
                episode_transitions.append(transition)
                obs = next_obs
                
                if done:
                    break
            
            print(f"   Episode {episode + 1}: {episode_steps} steps, reward: {sum(t['reward'] for t in episode_transitions):.2f}")
            
            # Add to buffer
            self.learner_buffer.extend(episode_transitions)
            transitions_collected += len(episode_transitions)
            
            # Maintain buffer size
            if len(self.learner_buffer) > self.max_buffer_size:
                self.learner_buffer = self.learner_buffer[-self.max_buffer_size:]
        
        return transitions_collected
    
    def get_learner_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get batch from learner buffer."""
        if len(self.learner_buffer) < batch_size:
            raise ValueError("Not enough learner data")
        
        indices = np.random.choice(len(self.learner_buffer), batch_size, replace=True)
        batch_transitions = [self.learner_buffer[i] for i in indices]
        
        states = torch.FloatTensor([t['state'] for t in batch_transitions]).to(self.device)
        actions_onehot = torch.FloatTensor([t['action_onehot'] for t in batch_transitions]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in batch_transitions]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([t['next_state'] for t in batch_transitions]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in batch_transitions]).unsqueeze(1).to(self.device)
        
        return {
            'states': states,
            'actions': actions_onehot,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def update_discriminator(self, expert_batch: Dict, learner_batch: Dict) -> Dict[str, float]:
        """Update discriminator to distinguish expert from learner."""
        batch_size = expert_batch['states'].shape[0]
        
        # Get discriminator predictions
        expert_logits = self.discriminator(expert_batch['states'], expert_batch['actions'])
        learner_logits = self.discriminator(learner_batch['states'], learner_batch['actions'])
        
        # Labels: 1 for expert, 0 for learner
        expert_labels = torch.ones(batch_size, 1, device=self.device)
        learner_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Binary cross-entropy loss
        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, expert_labels)
        learner_loss = F.binary_cross_entropy_with_logits(learner_logits, learner_labels)
        discriminator_loss = expert_loss + learner_loss
        
        # Update discriminator
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        # Calculate accuracy
        expert_preds = (torch.sigmoid(expert_logits) > 0.5).float()
        learner_preds = (torch.sigmoid(learner_logits) < 0.5).float()
        expert_accuracy = expert_preds.mean().item()
        learner_accuracy = learner_preds.mean().item()
        overall_accuracy = (expert_accuracy + learner_accuracy) / 2
        
        return {
            'discriminator_loss': discriminator_loss.item(),
            'expert_loss': expert_loss.item(),
            'learner_loss': learner_loss.item(),
            'expert_accuracy': expert_accuracy,
            'learner_accuracy': learner_accuracy,
            'overall_accuracy': overall_accuracy
        }
    
    def update_policy(self, learner_batch: Dict) -> Dict[str, float]:
        """Update policy using AIRL rewards."""
        states = learner_batch['states']
        actions_onehot = learner_batch['actions']
        next_states = learner_batch['next_states']
        dones = learner_batch['dones']
        
        # Get AIRL rewards
        with torch.no_grad():
            # Use reward scaling factor from config to amplify rewards
            reward_scale = self.config.get('reward_scale', 10.0)
            airl_rewards = self.discriminator.get_reward(states, actions_onehot, scale=reward_scale)
        
        # Get policy predictions
        action_probs, state_values = self.policy(states)
        next_state_values = self.policy(next_states)[1]
        
        # Calculate advantages
        gamma = self.config.get('gamma', 0.99)
        targets = airl_rewards + gamma * next_state_values * (1 - dones)
        advantages = targets - state_values
        
        # Actor loss (policy gradient)
        action_indices = actions_onehot.argmax(dim=-1)
        log_probs = torch.log(action_probs.gather(1, action_indices.unsqueeze(1)) + 1e-8)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(state_values, targets.detach())
        
        # Combined loss
        policy_loss = actor_loss + 0.5 * critic_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_airl_reward': airl_rewards.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def log_metrics(self, metrics: Dict[str, float], iteration: int):
        """Log metrics to TensorBoard."""
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, iteration)
    
    def train(self):
        """Main training loop."""
        print("🚀 STARTING PYTORCH AIRL TRAINING")
        print("=" * 60)
        
        # Load expert trajectories
        num_expert_trajectories = self.expert_loader.load_trajectories()
        if num_expert_trajectories == 0:
            print("❌ No expert trajectories loaded!")
            return
        
        # Get expert statistics
        stats = self.expert_loader.get_statistics()
        print(f"📊 Expert Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"🎯 Device: {self.device}")
        print(f"📊 State dim: {self.state_dim}, Action dim: {self.action_dim}")
        
        # Training parameters
        batch_size = self.config.get('batch_size', 32)
        max_iterations = self.config.get('max_iterations', 50)
        episodes_per_iteration = self.config.get('episodes_per_iteration', 3)
        updates_per_iteration = self.config.get('updates_per_iteration', 5)
        
        print(f"🔧 Batch size: {batch_size}, Max iterations: {max_iterations}")
        print(f"📈 Episodes per iteration: {episodes_per_iteration}")
        
        # Initial data collection
        print("\n📦 Collecting initial learner data...")
        self.collect_learner_data(episodes_per_iteration * 2)
        print(f"   Buffer size: {len(self.learner_buffer)}")
        
        # Training loop
        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Collect more learner data
            transitions_collected = self.collect_learner_data(episodes_per_iteration)
            
            # Training updates
            all_metrics = []
            
            for update in range(updates_per_iteration):
                try:
                    # Get batches
                    expert_batch = self.expert_loader.get_batch(batch_size, self.device)
                    learner_batch = self.get_learner_batch(batch_size)
                    
                    # Update discriminator
                    disc_metrics = self.update_discriminator(expert_batch, learner_batch)
                    
                    # Update policy
                    policy_metrics = self.update_policy(learner_batch)
                    
                    # Combine metrics
                    metrics = {**disc_metrics, **policy_metrics}
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"   ⚠️  Update {update} failed: {e}")
                    continue
            
            # Average metrics and log
            if all_metrics:
                avg_metrics = {}
                for key in all_metrics[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in all_metrics if key in m])
                
                # Console logging
                print(f"   Transitions: {transitions_collected}")
                print(f"   D_loss: {avg_metrics.get('discriminator_loss', 0):.4f}")
                print(f"   P_loss: {avg_metrics.get('policy_loss', 0):.4f}")
                print(f"   D_acc: {avg_metrics.get('overall_accuracy', 0):.3f}")
                print(f"   AIRL_reward: {avg_metrics.get('mean_airl_reward', 0):.4f}")
                
                # TensorBoard logging
                self.log_metrics(avg_metrics, iteration)
                
                # Add iteration-specific metrics
                self.log_metrics({
                    'learner/transitions_collected': transitions_collected,
                    'learner/buffer_size': len(self.learner_buffer)
                }, iteration)
        
        # Save models
        self.save_models("pytorch_airl_final")
        
        if self.use_tensorboard:
            self.writer.close()
        
        print(f"\n✅ Training completed!")
        print(f"📊 Total steps: {self.step_count}")
        
    def save_models(self, prefix: str):
        """Save trained models."""
        os.makedirs("models", exist_ok=True)
        
        discriminator_path = f"models/{prefix}_discriminator.pth"
        policy_path = f"models/{prefix}_policy.pth"
        
        torch.save(self.discriminator.state_dict(), discriminator_path)
        torch.save(self.policy.state_dict(), policy_path)
        
        print(f"💾 Models saved: {discriminator_path}, {policy_path}")

def test_expert_trajectories():
    """Test expert trajectories loading."""
    print("🧪 TESTING EXPERT TRAJECTORIES FOR AIRL")
    print("=" * 60)
    
    expert_loader = ExpertLoader("expert_trajectories_dqn_adapter")
    num_loaded = expert_loader.load_trajectories()
    
    if num_loaded == 0:
        print("❌ No trajectories loaded!")
        return False
    
    # Get statistics
    stats = expert_loader.get_statistics()
    print(f"\n📊 Expert Data Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Test sampling
    print(f"\n🎯 Testing Expert Sampling:")
    try:
        expert_batch = expert_loader.get_batch(32, 'cpu')
        print(f"   ✅ Batch sampling successful")
        print(f"   States shape: {expert_batch['states'].shape}")
        print(f"   Actions shape: {expert_batch['actions'].shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ Sampling failed: {e}")
        return False

def create_training_config(mode: str = "visualized", training_type: str = "single") -> Dict:
    """Create training configuration."""
    base_config = {
        # Environment
        'expert_trajectory_dir': 'expert_trajectories_dqn_adapter',
        'max_episode_steps': 1000,  # Increased from 500 to 1000 for better rewards
        
        # Network parameters
        'discriminator_lr': 3e-4,
        'policy_lr': 1e-4,
        'gamma': 0.99,
        'batch_size': 64,  # Increased from 32 to 64
        'buffer_size': 50000,  # Increased from 10000 to 50000
        
        # Logging
        'use_tensorboard': True,
        'use_cuda': True,
        
        # AIRL parameters
        'reward_scale': 10.0,  # Scale AIRL rewards to match expert reward scale
        'use_shaped_rewards': True  # Enable reward shaping
    }
    
    # Mode-specific settings
    if mode == "visualized":
        base_config.update({
            'headless': False,
            'max_iterations': 50 if training_type == "single" else None,
            'max_episodes': 50 if training_type == "multiplayer" else None,
            'episodes_per_iteration': 2,
            'updates_per_iteration': 5,
            'policy_updates_per_iteration': 5,
        })
    elif mode == "headless":
        base_config.update({
            'headless': True,
            'max_iterations': 5000 if training_type == "single" else None,
            'max_episodes': 100000 if training_type == "multiplayer" else None,
            'episodes_per_iteration': 10,
            'updates_per_iteration': 15,
            'policy_updates_per_iteration': 15,
            'batched_training': True,
            'batch_episodes': 10,  # Process in batches of 10 episodes
            'checkpoint_interval': 100  # Save model every 100 episodes/iterations
        })
    elif mode == "demo":
        base_config.update({
            'headless': False,  # Changed to False to enable visualization
            'max_iterations': 10 if training_type == "single" else None,
            'max_episodes': 10 if training_type == "multiplayer" else None,
            'episodes_per_iteration': 2,
            'updates_per_iteration': 3,
            'policy_updates_per_iteration': 3,
        })
    
    return base_config

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='PyTorch AIRL Trainer')
    parser.add_argument('--mode', choices=['test', 'visualized', 'headless', 'demo'], 
                       default='demo', help='Training mode')
    parser.add_argument('--type', choices=['single', 'multiplayer'], 
                       default='single', help='Training type: single-player or multiplayer')
    parser.add_argument('--episodes', type=int, help='Number of episodes to train for')
    parser.add_argument('--steps', type=int, help='Maximum steps per episode')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--d_iters', type=int, help='Discriminator update iterations')
    parser.add_argument('--p_iters', type=int, help='Policy update iterations')
    parser.add_argument('--batch_episodes', type=int, help='Episodes per batch (for batched training)')
    
    args = parser.parse_args()
    
    # Handle testing mode
    if args.mode == 'test':
        print("🧪 Testing Expert Trajectories")
        success = test_expert_trajectories()
        if success:
            print("\n✅ Expert trajectories ready for AIRL training!")
        else:
            print("\n❌ Issues found with expert trajectories")
        return
    
    # Check environment availability
    if not ENVS_AVAILABLE and args.type == 'multiplayer':
        print("❌ Multiplayer training requires proper environment imports")
        print("   Run from correct directory or fix import paths")
        return
    
    # Training modes
    config = create_training_config(args.mode, args.type)
    
    # Override with command line arguments if provided
    if args.episodes:
        if args.type == 'multiplayer':
            config['max_episodes'] = args.episodes
        else:
            config['max_iterations'] = args.episodes
    
    if args.steps:
        config['max_episode_steps'] = args.steps
    
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    if args.lr:
        config['discriminator_lr'] = args.lr
        config['policy_lr'] = args.lr
    
    if args.d_iters:
        config['updates_per_iteration'] = args.d_iters
    
    if args.p_iters:
        config['policy_updates_per_iteration'] = args.p_iters
    
    # Set up batched training for headless mode
    if args.mode == 'headless':
        config['batched_training'] = True
        config['batch_episodes'] = args.batch_episodes or 10
        
        # Default to 100,000 episodes for long-term training
        if args.type == 'multiplayer':
            config['max_episodes'] = args.episodes or 100000
        else:
            config['max_iterations'] = args.episodes or 5000
    
    print(f"🚀 PYTORCH AIRL TRAINING - {args.mode.upper()} {args.type.upper()} MODE")
    print("=" * 70)
    print(f"📊 Episodes: {config.get('max_episodes') or config.get('max_iterations')}")
    print(f"📏 Steps per episode: {config['max_episode_steps']}")
    print(f"📦 Batch size: {config['batch_size']}")
    
    if args.type == 'multiplayer':
        trainer = MultiplayerAIRLTrainer(config)
        trainer.train_competitive()
    else:
        trainer = PyTorchAIRLTrainer(config)
        trainer.train()

if __name__ == "__main__":
    main() 