#!/usr/bin/env python3
"""
Self-contained DQN Training Module for Tetris

This module contains everything needed for DQN training without dependency issues.
It's designed to be completely modular and self-contained.

Features:
- TensorBoard logging with comprehensive metrics
- Automatic checkpointing every 1000 episodes
- Parallel environment support
- All dependencies contained within this module

Usage:
    python dqn_training_module.py --mode vectorized --num_episodes 10000 --num_envs 8
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from datetime import datetime
from collections import deque
import random
import gym
from gym import spaces
import pygame
import time
from torch.utils.tensorboard import SummaryWriter
from gym.vector import SyncVectorEnv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dqn_training.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================================================
# TETRIS GAME COMPONENTS (Self-contained)
# =====================================================================================

# Shape definitions
shapes = [
    [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']],

    [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']],

    [['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '.0...',
      '.00..',
      '.0...',
      '.....']],

    [['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '.0...',
      '.0...',
      '.00..',
      '.....']],

    [['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....'],
     ['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '.00..',
      '.0...',
      '.0...',
      '.....']],

    [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']],

    [['.....',
      '.....',
      '.0000',
      '.....',
      '.....'],
     ['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....']]
]

shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]

class Piece:
    def __init__(self, column, row, shape):
        self.x = column
        self.y = row
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0

def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]
    
    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))
    
    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)
    
    return positions

def valid_space(shape, grid):
    accepted_positions = [[(j, i) for j in range(10) if grid[i][j] == (0,0,0)] for i in range(20)]
    accepted_positions = [j for sub in accepted_positions for j in sub]
    formatted = convert_shape_format(shape)
    
    for pos in formatted:
        if pos not in accepted_positions:
            if pos[1] > -1:
                return False
    return True

def create_grid(locked_positions={}):
    grid = [[(0,0,0) for x in range(10)] for x in range(20)]
    
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j,i) in locked_positions:
                c = locked_positions[(j,i)]
                grid[i][j] = c
    return grid

def get_shape():
    return Piece(5, 0, random.choice(shapes))

def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False

# =====================================================================================
# DQN NEURAL NETWORK
# =====================================================================================

class DQN(nn.Module):
    """Ultra-Compact Deep Q-Network for Tetris with Tucking Support
    
    Features:
    - Ultra-minimal conv layers: 1->4 channels max
    - Support for tucking actions (x, y, rotation combinations)
    - Action space: 10 (x) × 20 (y) × 4 (rotations) + 1 (hold) = 801 actions
    - Compact metadata processing
    """
    def __init__(self):
        super(DQN, self).__init__()
        
        # Ultra-compact grid processing (1->4 channels max)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        
        # Global average pooling to reduce spatial dimensions dramatically
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Compact metadata processing (7 values -> 8 features)
        self.fc_meta = nn.Linear(7, 8)
        
        # Combined processing - ultra minimal
        self.fc1 = nn.Linear(4 + 8, 32)  # conv features + metadata
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 801)  # 10*20*4 + 1 = 801 actions (tucking + hold)
        
        # Minimal dropout
        self.dropout = nn.Dropout(0.05)
        
        # Initialize weights
        self._init_weights()
        
        # Print parameter count
        self._print_params()
    
    def _init_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _print_params(self):
        """Print parameter count for verification"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔥 Ultra-Compact DQN Network: {total_params:,} total parameters ({trainable_params:,} trainable)")
        print(f"   ✅ Minimal conv channels: 1→4 (as requested)")
        print(f"   🎯 Action space: 10×20×4 + 1 = 801 actions (tucking support)")
        
        # Detailed breakdown
        conv_params = sum(p.numel() for n, p in self.named_parameters() if 'conv' in n)
        fc_params = sum(p.numel() for n, p in self.named_parameters() if 'fc' in n)
        print(f"   📊 Parameter breakdown:")
        print(f"      - Conv layers: {conv_params:,} parameters")
        print(f"      - FC layers: {fc_params:,} parameters")
        
        # Layer-by-layer breakdown
        print(f"   🔍 Layer details:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"      - {name}: {param.numel():,} parameters {list(param.shape)}")
        
        # Memory estimate
        param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        print(f"   💾 Estimated model memory: {param_memory_mb:.2f} MB")
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Split input into grid and metadata
        grid = x[:, :200].view(batch_size, 1, 20, 10)
        metadata = x[:, 200:]
        
        # Process grid with ultra-compact CNN (1->4 channels)
        grid = F.relu(self.conv1(grid))  # 1->4 channels
        grid = F.relu(self.conv2(grid))  # 4->4 channels
        
        # Global average pooling to get fixed-size representation
        grid = self.global_pool(grid)  # (batch_size, 4, 1, 1)
        grid = grid.view(batch_size, -1)  # (batch_size, 4)
        
        # Process compact metadata  
        metadata = F.relu(self.fc_meta(metadata))  # (batch_size, 8)
        
        # Combine features
        combined = torch.cat([grid, metadata], dim=1)  # (batch_size, 12)
        
        # Final processing through ultra-compact MLP
        x = F.relu(self.fc1(combined))  # 12 -> 32
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # 32 -> 64
        x = self.dropout(x)
        q_values = self.fc3(x)  # 64 -> 801 actions
        
        return q_values

# =====================================================================================
# REPLAY BUFFER
# =====================================================================================

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        state = state.astype(np.float32)
        next_state = next_state.astype(np.float32)
        self.buffer.append((state, action, reward, next_state, done))
    
    def push_batch(self, states, actions, rewards, next_states, dones):
        """Add batch of experiences to buffer"""
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.push(state, action, reward, next_state, done)
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# =====================================================================================
# TETRIS ENVIRONMENT
# =====================================================================================

class TetrisEnv(gym.Env):
    """Self-contained Tetris Environment with Tucking Support
    
    Action Space:
    - Actions 0-799: Placement actions (x, y, rotation combinations)
      - x: 0-9 (10 columns)
      - y: 0-19 (20 rows) 
      - rotation: 0-3 (4 rotations)
      - action = x + y*10 + rotation*200
    - Action 800: Hold action
    
    This allows pieces to be "tucked" into any valid position, not just dropped vertically.
    """
    def __init__(self, single_player=True, headless=True):
        super(TetrisEnv, self).__init__()
        
        self.headless = headless
        if not self.headless:
            pygame.init()
            self.surface = pygame.display.set_mode((400, 800))
            pygame.display.set_caption("Tetris RL with Tucking")
        else:
            self.surface = None
            
        # Updated action space: 10*20*4 + 1 = 801 actions (tucking + hold)
        self.action_space = spaces.Discrete(801)
        
        # Observation space
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
        
        self.reset()
    
    def _decode_action(self, action):
        """Decode action into (x, y, rotation) or 'hold'
        
        Args:
            action: Integer action from 0-800
            
        Returns:
            If action < 800: (x, y, rotation) tuple
            If action == 800: 'hold'
        """
        if action == 800:
            return 'hold'
        
        # Decode placement action
        rotation = action // 200
        remaining = action % 200
        y = remaining // 10
        x = remaining % 10
        
        return (x, y, rotation)
    
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]
    
    def reset(self):
        self.locked_positions = {}
        self.grid = create_grid(self.locked_positions)
        self.current_piece = get_shape()
        self.next_pieces = [get_shape() for _ in range(3)]
        self.hold_piece = None
        self.can_hold = True
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        self.fall_time = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Create grid observation
        grid = create_grid(self.locked_positions)
        grid_obs = np.zeros((20, 10), dtype=np.int8)
        for i in range(20):
            for j in range(10):
                if grid[i][j] != (0, 0, 0):
                    grid_obs[i][j] = 1
        
        # Overlay current falling piece
        if self.current_piece:
            for x, y in convert_shape_format(self.current_piece):
                if 0 <= y < 20 and 0 <= x < 10:
                    grid_obs[y][x] = 2
        
        # Piece information
        next_idx = shapes.index(self.next_pieces[0].shape) + 1 if self.next_pieces else 0
        hold_idx = shapes.index(self.hold_piece.shape) + 1 if self.hold_piece else 0
        curr_shape = shapes.index(self.current_piece.shape) + 1 if self.current_piece else 0
        curr_rot = self.current_piece.rotation if self.current_piece else 0
        curr_x = self.current_piece.x if self.current_piece else 0
        curr_y = self.current_piece.y if self.current_piece else 0
        
        return {
            'grid': grid_obs,
            'next_piece': next_idx,
            'hold_piece': hold_idx,
            'current_shape': curr_shape,
            'current_rotation': curr_rot,
            'current_x': curr_x,
            'current_y': curr_y,
            'can_hold': int(self.can_hold)
        }
    
    def step(self, action):
        decoded = self._decode_action(action)
        
        if decoded == 'hold':  # Hold action
            if self.can_hold:
                if self.hold_piece is None:
                    self.hold_piece = self.current_piece
                    self.current_piece = self.next_pieces.pop(0)
                    self.next_pieces.append(get_shape())
                else:
                    self.current_piece, self.hold_piece = self.hold_piece, self.current_piece
                    self.current_piece.x = 5
                    self.current_piece.y = 0
                self.can_hold = False
            reward = -0.01  # Small penalty for hold action
        else:
            # Tucking placement action
            x, y, rotation = decoded
            
            # Create test piece at specified position
            test_piece = Piece(x, y, self.current_piece.shape)
            test_piece.rotation = rotation
            
            # Check if placement is valid
            if valid_space(test_piece, self.grid):
                # Place the piece at the specified location
                piece_positions = convert_shape_format(test_piece)
                
                # Only place if all positions are above ground level
                valid_placement = all(pos[1] >= 0 for pos in piece_positions)
                
                if valid_placement:
                    # Place the piece
                    for pos in piece_positions:
                        self.locked_positions[pos] = test_piece.color
                    
                    # Get new piece
                    self.current_piece = self.next_pieces.pop(0)
                    self.next_pieces.append(get_shape())
                    self.can_hold = True
                    
                    # Check for line clears
                    lines_cleared = self._clear_lines()
                    self.lines_cleared_total += lines_cleared
                    
                    # Calculate reward with tucking bonus
                    reward = self._calculate_reward(lines_cleared, piece_positions, y)
                else:
                    reward = -5  # Invalid placement (above grid)
            else:
                reward = -10  # Invalid placement (collision)
        
        # Check if game is over
        done = check_lost(self.locked_positions)
        if done:
            reward -= 20
        
        obs = self._get_observation()
        info = {
            'score': self.score,
            'lines_cleared': self.lines_cleared_total,
            'level': self.level
        }
        
        return obs, reward, done, False, info
    
    def _clear_lines(self):
        lines_cleared = 0
        for y in range(19, -1, -1):
            if all((x, y) in self.locked_positions for x in range(10)):
                lines_cleared += 1
                # Remove the line
                for x in range(10):
                    del self.locked_positions[(x, y)]
                # Move everything down
                for key in sorted(list(self.locked_positions), key=lambda x: x[1]):
                    x, y_pos = key
                    if y_pos < y:
                        self.locked_positions[(x, y_pos + 1)] = self.locked_positions.pop(key)
        
        # Update score
        if lines_cleared > 0:
            points = {1: 100, 2: 300, 3: 500, 4: 800}
            self.score += points.get(lines_cleared, 0) * (self.level + 1)
        
        return lines_cleared
    
    def _calculate_reward(self, lines_cleared, new_positions, y):
        """Calculate reward with tucking bonus
        
        Args:
            lines_cleared: Number of lines cleared
            new_positions: Positions where piece was placed
            y: Y position where piece was placed (for tucking bonus)
        """
        # Line clear rewards
        reward = 0
        if lines_cleared > 0:
            reward += {1: 10, 2: 25, 3: 50, 4: 100}[lines_cleared] * (self.level + 1)
        
        # Height and hole penalties
        grid = create_grid(self.locked_positions)
        heights = [next((r for r in range(20) if grid[r][c] != (0, 0, 0)), 20) for c in range(10)]
        max_height = min(heights)
        holes = sum(1 for c in range(10) 
                   for r in range(heights[c] + 1, 20) 
                   if grid[r][c] == (0, 0, 0))
        
        reward += 0.1 * (20 - max_height)  # Reward for keeping pieces low
        reward -= 0.5 * holes  # Penalty for holes
        
        # Tucking bonus - reward for placing pieces lower in the grid
        tucking_bonus = 0.05 * y  # Small bonus for placing lower
        reward += tucking_bonus
        
        return reward + 0.1  # Small step reward
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        if not self.headless:
            pygame.quit()

# =====================================================================================
# DQN AGENT
# =====================================================================================

class DQNAgent:
    """Enhanced DQN Agent with logging and parallel support"""
    def __init__(self, 
                 learning_rate=1e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 target_update=10,
                 buffer_size=100000,
                 batch_size=64,
                 device=None,
                 log_dir='logs/dqn_tensorboard',
                 checkpoint_dir='checkpoints',
                 save_interval=1000):
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")
        
        # Networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_size)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.total_losses = []
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Logging and checkpointing
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"{self.log_dir}/dqn_run_{timestamp}")
        
        print(f"DQN Agent initialized with TensorBoard logging to: {self.writer.log_dir}")
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy (801 action space)"""
        if random.random() < self.epsilon:
            return random.randint(0, 800)  # Updated for 801 actions (0-800)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
    
    def select_actions_batch(self, states):
        """Select actions for batch of states (801 action space)"""
        if isinstance(states, np.ndarray):
            states = torch.FloatTensor(states).to(self.device)
        
        batch_size = states.size(0)
        actions = []
        random_mask = torch.rand(batch_size) < self.epsilon
        
        with torch.no_grad():
            q_values = self.policy_net(states)
            greedy_actions = q_values.max(1)[1].cpu().numpy()
        
        for i in range(batch_size):
            if random_mask[i]:
                actions.append(random.randint(0, 800))  # Updated for 801 actions (0-800)
            else:
                actions.append(greedy_actions[i])
        
        return actions
    
    def train_step(self):
        """Perform one training step"""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None
            
        states, actions, rewards, next_states, dones = batch
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double Q-learning
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.writer.add_scalar('Training/Target_Network_Updates', self.steps_done // self.target_update, self.steps_done)
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Log training metrics
        self.total_losses.append(loss.item())
        if self.steps_done % 100 == 0:
            self.writer.add_scalar('Training/Loss', loss.item(), self.steps_done)
            self.writer.add_scalar('Training/Epsilon', self.epsilon, self.steps_done)
            self.writer.add_scalar('Training/Buffer_Size', len(self.memory), self.steps_done)
        
        return loss.item()
    
    def train_batch_update(self, num_updates=1):
        """Perform multiple training steps"""
        if len(self.memory) < self.batch_size:
            return None
        
        losses = []
        for _ in range(num_updates):
            loss = self.train_step()
            if loss is not None:
                losses.append(loss)
        
        return np.mean(losses) if losses else None
    
    def log_episode(self, episode_reward, episode_length, episode_info=None):
        """Log episode metrics and handle checkpointing"""
        self.episodes_done += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Log to TensorBoard
        self.writer.add_scalar('Episode/Reward', episode_reward, self.episodes_done)
        self.writer.add_scalar('Episode/Length', episode_length, self.episodes_done)
        
        if episode_info:
            if 'score' in episode_info:
                self.writer.add_scalar('Episode/Score', episode_info['score'], self.episodes_done)
            if 'lines_cleared' in episode_info:
                self.writer.add_scalar('Episode/Lines_Cleared', episode_info['lines_cleared'], self.episodes_done)
        
        # Log running averages
        if self.episodes_done % 10 == 0:
            avg_reward_10 = np.mean(self.episode_rewards[-10:])
            avg_length_10 = np.mean(self.episode_lengths[-10:])
            self.writer.add_scalar('Episode/Average_Reward_10', avg_reward_10, self.episodes_done)
            self.writer.add_scalar('Episode/Average_Length_10', avg_length_10, self.episodes_done)
        
        if self.episodes_done % 100 == 0:
            avg_reward_100 = np.mean(self.episode_rewards[-100:])
            avg_length_100 = np.mean(self.episode_lengths[-100:])
            self.writer.add_scalar('Episode/Average_Reward_100', avg_reward_100, self.episodes_done)
            self.writer.add_scalar('Episode/Average_Length_100', avg_length_100, self.episodes_done)
            
            print(f"Episode {self.episodes_done}: Avg Reward (100): {avg_reward_100:.2f}, "
                  f"Avg Length (100): {avg_length_100:.2f}, Epsilon: {self.epsilon:.4f}")
        
        # Save checkpoint
        if self.episodes_done % self.save_interval == 0:
            self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"dqn_checkpoint_episode_{self.episodes_done}.pt")
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'total_losses': self.total_losses
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "dqn_latest.pt")
        torch.save(checkpoint, latest_path)
        
        self.writer.add_text("Checkpoints", f"Saved checkpoint at episode {self.episodes_done}: {checkpoint_path}")
    
    def save(self, path):
        """Save model"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'total_losses': self.total_losses
        }
        torch.save(checkpoint, path)
        print(f"Model saved: {path}")
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        
        if 'episodes_done' in checkpoint:
            self.episodes_done = checkpoint['episodes_done']
        if 'episode_rewards' in checkpoint:
            self.episode_rewards = checkpoint['episode_rewards']
        if 'episode_lengths' in checkpoint:
            self.episode_lengths = checkpoint['episode_lengths']
        if 'total_losses' in checkpoint:
            self.total_losses = checkpoint['total_losses']
        
        print(f"Model loaded: {path}")
        print(f"Resumed at episode {self.episodes_done}, step {self.steps_done}, epsilon {self.epsilon:.4f}")
    
    def close(self):
        """Close TensorBoard writer"""
        if hasattr(self, 'writer'):
            self.writer.close()
            print("TensorBoard writer closed")

# =====================================================================================
# TRAINING FUNCTIONS
# =====================================================================================

def preprocess_obs_dict(obs_dict):
    """Convert observation dictionary to state vector (207,)"""
    grid_flat = obs_dict['grid'].flatten().astype(np.float32)
    metadata = np.array([
        obs_dict['next_piece'],
        obs_dict['hold_piece'],
        obs_dict['current_shape'],
        obs_dict['current_rotation'],
        obs_dict['current_x'],
        obs_dict['current_y'],
        obs_dict['can_hold']
    ]).astype(np.float32)
    
    return np.concatenate([grid_flat, metadata])

def preprocess_obs_batch(batched_obs_dict):
    """Convert batched observations to batch of state vectors"""
    batch_size = batched_obs_dict['grid'].shape[0]
    states = []
    
    for i in range(batch_size):
        obs_dict = {key: value[i] for key, value in batched_obs_dict.items()}
        state = preprocess_obs_dict(obs_dict)
        states.append(state)
    
    return np.array(states)

def make_env(env_id, seed, headless=True):
    """Factory for creating TetrisEnv instances"""
    def _init():
        env = TetrisEnv(single_player=True, headless=headless)
        env.seed(seed + env_id)
        return env
    return _init

def train_single_env(agent, num_episodes=10000, eval_interval=1000, checkpoint_path=None):
    """Train agent on single environment"""
    logger.info(f"Starting single environment training for {num_episodes} episodes")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    env = TetrisEnv(single_player=True, headless=True)
    
    for episode in range(agent.episodes_done, agent.episodes_done + num_episodes):
        obs_dict, info = env.reset()
        state = preprocess_obs_dict(obs_dict)
        
        episode_reward = 0
        episode_length = 0
        episode_info = {'score': 0, 'lines_cleared': 0, 'level': 1}
        
        done = False
        while not done:
            action = agent.select_action(state)
            next_obs_dict, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            next_state = preprocess_obs_dict(next_obs_dict)
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            episode_info['score'] = step_info.get('score', episode_info['score'])
            episode_info['lines_cleared'] = step_info.get('lines_cleared', episode_info['lines_cleared'])
            episode_info['level'] = step_info.get('level', episode_info['level'])
        
        agent.log_episode(episode_reward, episode_length, episode_info)
        
        if (episode + 1) % eval_interval == 0:
            logger.info(f"Episode {episode + 1} completed")
    
    env.close()
    logger.info("Training completed")

def train_vectorized_env(agent, num_envs=4, num_episodes=10000, eval_interval=1000, 
                        update_frequency=4, checkpoint_path=None):
    """Train agent using vectorized environments"""
    logger.info(f"Starting vectorized training with {num_envs} environments for {num_episodes} episodes")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    envs = SyncVectorEnv([make_env(i, seed=i + 1000, headless=True) for i in range(num_envs)])
    
    # Handle vectorized environment reset return format
    reset_result = envs.reset()
    if isinstance(reset_result, tuple):
        obs_dicts, _ = reset_result  # (observations, infos)
    else:
        obs_dicts = reset_result
    
    states = preprocess_obs_batch(obs_dicts)
    
    episode_rewards = [0] * num_envs
    episode_lengths = [0] * num_envs
    episode_infos = [{'score': 0, 'lines_cleared': 0, 'level': 1} for _ in range(num_envs)]
    completed_episodes = agent.episodes_done
    step_count = 0
    
    while completed_episodes < num_episodes:
        actions = agent.select_actions_batch(states)
        
        # Handle vectorized environment step return format
        step_result = envs.step(actions)
        if len(step_result) == 5:
            next_obs_dicts, rewards, terminated, truncated, infos = step_result
            dones = terminated | truncated  # Combine terminated and truncated
        else:
            next_obs_dicts, rewards, dones, infos = step_result
        
        next_states = preprocess_obs_batch(next_obs_dicts)
        
        for i in range(num_envs):
            agent.memory.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
            
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1
            
            # Handle info access safely - infos might be tuple, list, or dict
            try:
                if infos and hasattr(infos, '__getitem__'):
                    info_item = infos[i] if i < len(infos) else {}
                    if info_item:
                        episode_infos[i]['score'] = info_item.get('score', episode_infos[i]['score'])
                        episode_infos[i]['lines_cleared'] = info_item.get('lines_cleared', episode_infos[i]['lines_cleared'])
                        episode_infos[i]['level'] = info_item.get('level', episode_infos[i]['level'])
            except (KeyError, IndexError, TypeError):
                pass  # Skip if info access fails
            
            if dones[i]:
                agent.log_episode(episode_rewards[i], episode_lengths[i], episode_infos[i])
                completed_episodes += 1
                
                episode_rewards[i] = 0
                episode_lengths[i] = 0
                episode_infos[i] = {'score': 0, 'lines_cleared': 0, 'level': 1}
                
                if completed_episodes % eval_interval == 0:
                    logger.info(f"Episode {completed_episodes} completed")
        
        step_count += 1
        if step_count % update_frequency == 0:
            agent.train_batch_update(num_updates=update_frequency)
        
        states = next_states
    
    envs.close()
    logger.info("Vectorized training completed")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Self-contained DQN Training for Tetris')
    
    # Training configuration
    parser.add_argument('--mode', choices=['single', 'vectorized'], default='vectorized',
                       help='Training mode')
    parser.add_argument('--num_episodes', type=int, default=10000,
                       help='Number of episodes to train')
    parser.add_argument('--num_envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--eval_interval', type=int, default=1000,
                       help='Episodes between evaluations')
    parser.add_argument('--update_frequency', type=int, default=4,
                       help='Steps between agent updates')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint to resume from')
    
    # DQN hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--target_update', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=1000)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    agent = DQNAgent(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=device,
        save_interval=args.save_interval
    )
    
    try:
        if args.mode == 'single':
            train_single_env(
                agent=agent,
                num_episodes=args.num_episodes,
                eval_interval=args.eval_interval,
                checkpoint_path=args.checkpoint
            )
        else:
            train_vectorized_env(
                agent=agent,
                num_envs=args.num_envs,
                num_episodes=args.num_episodes,
                eval_interval=args.eval_interval,
                update_frequency=args.update_frequency,
                checkpoint_path=args.checkpoint
            )
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        final_save_path = f"checkpoints/dqn_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        agent.save(final_save_path)
        agent.close()
        logger.info(f"Final model saved to: {final_save_path}")

if __name__ == "__main__":
    main() 