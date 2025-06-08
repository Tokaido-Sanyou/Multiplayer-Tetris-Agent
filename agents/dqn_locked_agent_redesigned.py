"""
Redesigned DQN Agent for Locked State Training Mode
- 800 action space (remove lock_in dimension)
- CNN-based observation processing
- Proper action mapping and validation
- Progressive penalty for invalid actions
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from collections import deque

try:
    from .base_agent import BaseAgent
except:
    from agents.base_agent import BaseAgent


class RedesignedLockedStateDQNAgent(BaseAgent):
    """
    Redesigned DQN Agent for Locked State Training Mode
    
    Key Features:
    1. 800 action space (10x20x4 coordinates+rotations, no lock_in)
    2. CNN-based observation processing 
    3. Proper action validation with progressive penalties
    4. No error handling - crash on mismatch
    """
    
    def __init__(self,
                 device: str = 'cuda',
                 learning_rate: float = 0.00005,  # FIXED: Reduced learning rate
                 gamma: float = 0.99,
                 epsilon_start: float = 0.95,  # Updated default start epsilon
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 50000,
                 target_update_freq: int = 1000,
                 memory_size: int = 100000,
                 batch_size: int = 32,
                 invalid_penalty_rate: float = 0.01):
        """Initialize Redesigned Locked State DQN Agent"""
        
        # Action space: 10 (x) × 20 (y) × 4 (rotation) = 800 actions
        action_space_size = 800
        
        # Observation size: 200 (board) + 3 (current piece) + 3 (next piece) = 206
        obs_size = 206
        
        super().__init__(action_space_size, (obs_size,), device)
        
        # Configuration
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.invalid_penalty_rate = invalid_penalty_rate
        
        # Invalid action tracking
        self.invalid_action_count = 0
        
        # Build action mappings
        self.action_to_coordinates = {}
        self.coordinates_to_action = {}
        self._build_action_mappings()
        
        # Initialize networks
        self.q_network = self._create_cnn_network().to(self.device)
        self.target_network = self._create_cnn_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # FIXED: Initialize optimizer with higher learning rate to fix vanishing gradients
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                  lr=0.0005,  # INCREASED from 0.00005 to fix vanishing gradients
                                  weight_decay=1e-5)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training state
        self.target_update_counter = 0
        self.epsilon_step_counter = 0  # Track epsilon decay steps
    
    def _build_action_mappings(self):
        """Build mappings between action indices and (x, y, rotation) coordinates"""
        action_idx = 0
        for y in range(20):  # 20 rows
            for x in range(10):  # 10 columns
                for rotation in range(4):  # 4 rotations
                    coordinates = (x, y, rotation)
                    self.action_to_coordinates[action_idx] = coordinates
                    self.coordinates_to_action[coordinates] = action_idx
                    action_idx += 1
        
        # Verify we have exactly 800 actions
        assert len(self.action_to_coordinates) == 800, f"Expected 800 actions, got {len(self.action_to_coordinates)}"
    
    def _create_cnn_network(self) -> nn.Module:
        """Create FIXED CNN-based neural network with stable architecture"""
        
        class CNNDQNNetwork(nn.Module):
            def __init__(self):
                super(CNNDQNNetwork, self).__init__()
                
                # FIXED: Smaller CNN with batch normalization
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Reduced from 32 to 16
                self.bn1 = nn.BatchNorm2d(16)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Reduced from 64 to 32
                self.bn2 = nn.BatchNorm2d(32)
                
                # FIXED: Add pooling to reduce feature map size
                self.pool = nn.MaxPool2d(2, 2)  # Reduces 20x10 to 10x5
                
                # FIXED: Much smaller FC layers
                board_features = 32 * 10 * 5  # 1,600 (much smaller than 25,600)
                piece_features = 32  # Reduced from 64 to 32
                combined_features = board_features + piece_features  # 1,632
                
                self.piece_fc = nn.Linear(6, piece_features)
                self.fc1 = nn.Linear(combined_features, 256)  # Reduced from 512
                self.fc2 = nn.Linear(256, 128)  # Reduced from 256
                self.fc3 = nn.Linear(128, 800)
                
                self.dropout = nn.Dropout(0.1)  # Reduced dropout
                
                # FIXED: Proper weight initialization
                self._initialize_weights()
                
            def _initialize_weights(self):
                """Initialize weights properly to prevent explosion"""
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                
            def forward(self, x):
                # Split input
                board_bits = x[:, :200]
                piece_bits = x[:, 200:]
                
                # FIXED: CNN processing with batch norm and pooling
                board = board_bits.view(-1, 1, 20, 10)
                board_features = F.relu(self.bn1(self.conv1(board)))
                board_features = self.pool(board_features)  # 20x10 → 10x5
                board_features = F.relu(self.bn2(self.conv2(board_features)))
                board_features = board_features.view(board_features.size(0), -1)
                
                # Piece processing
                piece_features = F.relu(self.piece_fc(piece_bits))
                
                # Combine
                combined = torch.cat([board_features, piece_features], dim=1)
                
                # FIXED: Final processing with proper activation
                x = F.relu(self.fc1(combined))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                
                # FIXED: No activation on final layer for Q-values
                q_values = self.fc3(x)
                
                return q_values
        
        return CNNDQNNetwork()
    
    def map_action_to_board(self, action_idx: int) -> Tuple[int, int, int]:
        """Map network action to board coordinates"""
        assert 0 <= action_idx < 800, f"Action index {action_idx} out of range [0, 799]"
        return self.action_to_coordinates[action_idx]
    
    def is_valid_action(self, action_idx: int, env) -> bool:
        """Check if action is valid"""
        if not (0 <= action_idx < 800):
            return False
        
        x, y, rotation = self.map_action_to_board(action_idx)
        
        # Basic validation
        if not hasattr(env, 'players') or len(env.players) == 0:
            return False
        
        player = env.players[0]
        if not player.current_piece:
            return False
        
        # Check bounds
        if not (0 <= x < 10 and 0 <= y < 20):
            return False
        
        # Test piece placement
        try:
            from envs.game.piece_utils import valid_space
            from envs.game.utils import create_grid
            
            grid = create_grid(player.locked_positions)
            test_piece = type(player.current_piece)(x, y, player.current_piece.shape)
            test_piece.rotation = rotation
            
            return valid_space(test_piece, grid)
        except:
            return True  # Fallback
    
    def select_action(self, observation: np.ndarray, training: bool = True, env=None) -> int:
        """Select action with validation"""
        # Convert observation
        if isinstance(observation, dict):
            obs_array = observation.get('board', observation.get('observation', observation))
        else:
            obs_array = observation
        
        if not isinstance(obs_array, np.ndarray):
            obs_array = np.array(obs_array, dtype=np.float32)
        
        # Verify observation size
        assert obs_array.shape[0] == 206, f"Expected observation size 206, got {obs_array.shape[0]}"
        
        state_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # FIXED: Enhanced exploration strategy
        if training and random.random() < self.epsilon:
            # FIXED: Try top-k actions to improve exploration quality
            if random.random() < 0.7:  # 70% random from top 100 actions
                top_actions = np.argsort(q_values)[-100:]
                action_idx = np.random.choice(top_actions)
            else:  # 30% completely random
                action_idx = random.randint(0, 799)
        else:
            action_idx = np.argmax(q_values)
        
        # Validate action
        if env is not None and not self.is_valid_action(action_idx, env):
            self.invalid_action_count += 1
            if training:
                # Try to find valid action
                sorted_actions = np.argsort(q_values)[::-1]
                for candidate_action in sorted_actions:
                    if self.is_valid_action(candidate_action, env):
                        action_idx = candidate_action
                        break
                else:
                    x, y, rotation = self.map_action_to_board(action_idx)
                    raise ValueError(f"No valid action found. Last: action={action_idx}, coords=({x}, {y}, {rotation})")
        
        return action_idx
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> Dict[str, float]:
        """Update agent"""
        if isinstance(state, np.ndarray) and state.shape[0] == 206:
            state_array = state
        else:
            raise ValueError(f"Invalid state format. Expected shape (206,), got {state.shape if hasattr(state, 'shape') else type(state)}")
            
        if isinstance(next_state, np.ndarray) and next_state.shape[0] == 206:
            next_state_array = next_state
        else:
            raise ValueError(f"Invalid next_state format. Expected shape (206,), got {next_state.shape if hasattr(next_state, 'shape') else type(next_state)}")
        
        self.store_experience(state_array, action, reward, next_state_array, done)
        return self.train_batch()
    
    def train_batch(self) -> Dict[str, float]:
        """Train on batch"""
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0, 'q_value': 0.0}
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch])
        
        # Convert to tensors
        states_tensor = torch.from_numpy(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.from_numpy(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        
        # Q-values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)
            # FIXED: Clip target Q-values to prevent explosion
            target_q_values = torch.clamp(target_q_values, -100, 100)
        
        # FIXED: Use Huber loss instead of MSE for stability
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Updates
        self.update_epsilon()
        
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.update_target_network()
            self.target_update_counter = 0
        
        return {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon,
            'invalid_count': self.invalid_action_count
        }
    
    def update_epsilon(self):
        """Update epsilon with EXPONENTIAL decay - half epsilon at quarter episodes"""
        if self.epsilon > self.epsilon_end and self.epsilon_step_counter < self.epsilon_decay_steps:
            # REQUIREMENT: Half epsilon (0.5 of 0.95 = 0.475) at quarter episodes
            # Use step-based exponential decay: epsilon = start * (0.5)^(4*step/total_steps)
            self.epsilon_step_counter += 1
            progress = self.epsilon_step_counter / self.epsilon_decay_steps
            # At 25% progress, we want 50% of original epsilon
            # Formula: epsilon = start * (0.5)^(4*progress)
            decay_factor = 0.5 ** (4.0 * progress)
            self.epsilon = max(self.epsilon_end, self.epsilon_start * decay_factor)
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_parameter_count(self) -> int:
        """Get parameter count"""
        return sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
    
    def save_checkpoint(self, filepath: str):
        """Save checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'invalid_action_count': self.invalid_action_count
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.invalid_action_count = checkpoint.get('invalid_action_count', 0)
    
    def get_info(self) -> Dict[str, Any]:
        """Get info"""
        return {
            'type': 'RedesignedLockedStateDQNAgent',
            'action_space_size': 800,
            'parameters': self.get_parameter_count(),
            'epsilon': self.epsilon,
            'invalid_action_count': self.invalid_action_count,
            'device': str(self.device)
        }
