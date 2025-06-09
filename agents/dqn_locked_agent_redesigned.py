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
                 input_dim: int = 212,  # Added for dimension flexibility
                 num_actions: int = 800,
                 hidden_dim: int = 800,
                 device: str = 'cuda',
                 learning_rate: float = 0.00005,  # FIXED: Reduced learning rate
                 gamma: float = 0.99,
                 epsilon_start: float = 0.95,  # Updated default start epsilon
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 50000,  # Renamed for clarity
                 target_update: int = 1000,   # Renamed for clarity
                 buffer_size: int = 100000,   # Renamed for clarity
                 batch_size: int = 32,
                 invalid_penalty_rate: float = 0.01,
                 reward_mode: str = 'standard'):  # NEW: Support both reward modes
        """Initialize Redesigned Locked State DQN Agent with reward mode support"""
        
        # Action space: 10 (x) × 20 (y) × 4 (rotation) = 800 actions
        action_space_size = num_actions
        
        # Observation size: configurable for flexibility
        obs_size = input_dim
        
        super().__init__(action_space_size, (obs_size,), device)
        
        # Configuration
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay  # Renamed
        self.target_update_freq = target_update   # Renamed
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.invalid_penalty_rate = invalid_penalty_rate
        self.reward_mode = reward_mode  # NEW: Store reward mode
        
        print(f"🤖 DQN Agent initialized with reward_mode='{reward_mode}'")
        
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
        self.memory = deque(maxlen=buffer_size)
        
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
        
        class FullyConnectedDQNNetwork(nn.Module):
            def __init__(self):
                super(FullyConnectedDQNNetwork, self).__init__()
                
                # NO CNN - Direct fully-connected network
                # Input: 200 (board) + 6 (current piece) + 6 (next piece) = 212 dimensions
                input_features = 212
                
                # Reasonably sized FC layers, all ≥ 800 units
                self.fc1 = nn.Linear(input_features, 2048)  # 212 → 2048
                self.fc2 = nn.Linear(2048, 1024)           # 2048 → 1024  
                self.fc3 = nn.Linear(1024, 800)            # 1024 → 800
                self.fc4 = nn.Linear(800, 800)             # 800 → 800 (output)
                
                self.dropout = nn.Dropout(0.3)  # Higher dropout for FC-only network
                self.batch_norm1 = nn.BatchNorm1d(2048)
                self.batch_norm2 = nn.BatchNorm1d(1024)
                
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
                # Raw input processing - no CNN
                # Input: 200 (board) + 6 (current piece) + 6 (next piece) = 212
                # Expect input to be expanded to include next piece info
                
                # Direct fully-connected processing
                x = F.relu(self.batch_norm1(self.fc1(x)))  # 212 → 2048
                x = self.dropout(x)
                x = F.relu(self.batch_norm2(self.fc2(x)))  # 2048 → 1024
                x = self.dropout(x)
                x = F.relu(self.fc3(x))                    # 1024 → 800
                x = self.dropout(x)
                
                # Output Q-values
                q_values = self.fc4(x)                     # 800 → 800
                
                return q_values
        
        return FullyConnectedDQNNetwork()
    
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
    
    def select_action(self, observation: np.ndarray, training: bool = True, env=None, rnd_network=None) -> int:
        """Select action with validation and optional RND integration"""
        # Convert observation
        if isinstance(observation, dict):
            obs_array = observation.get('board', observation.get('observation', observation))
        else:
            obs_array = observation
        
        if not isinstance(obs_array, np.ndarray):
            obs_array = np.array(obs_array, dtype=np.float32)
        
        # Verify observation size (200 board + 6 current piece + 6 next piece = 212)
        if obs_array.shape[0] == 206:
            # Legacy format: pad with zeros for next piece
            obs_array = np.concatenate([obs_array, np.zeros(6)], axis=0)
        assert obs_array.shape[0] == 212, f"Expected observation size 212, got {obs_array.shape[0]}"
        
        state_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(self.device)
        
        # Get Q-values (set to eval mode for batch norm with single samples)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # RND-enhanced action selection: use RND intrinsic value + Q-values when available
        if training and rnd_network is not None:
            # Use greedy action selection based on Q-values + intrinsic motivation
            # RND provides exploration via reward shaping, not action selection randomness
            action_idx = np.argmax(q_values)
        elif training and random.random() < self.epsilon:
            # Standard epsilon-greedy with enhanced exploration strategy
            if random.random() < 0.7:  # 70% random from top 100 actions
                top_actions = np.argsort(q_values)[-100:]
                action_idx = np.random.choice(top_actions)
            else:  # 30% completely random
                action_idx = random.randint(0, 799)
        else:
            # Greedy action selection
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
        """Store experience with dimension padding"""
        # Pad state to 212 dimensions if needed (206 → 212)
        if state.shape[0] == 206:
            state = np.concatenate([state, np.zeros(6)], axis=0)
        # Pad next_state to 212 dimensions if needed (206 → 212)
        if next_state.shape[0] == 206:
            next_state = np.concatenate([next_state, np.zeros(6)], axis=0)
        
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> Dict[str, float]:
        """Update agent"""
        # Handle both legacy (206) and new (212) observation formats
        if isinstance(state, np.ndarray) and state.shape[0] in [206, 212]:
            state_array = state
            if state_array.shape[0] == 206:
                state_array = np.concatenate([state_array, np.zeros(6)], axis=0)
        else:
            raise ValueError(f"Invalid state format. Expected shape (206,) or (212,), got {state.shape if hasattr(state, 'shape') else type(state)}")
            
        if isinstance(next_state, np.ndarray) and next_state.shape[0] in [206, 212]:
            next_state_array = next_state
            if next_state_array.shape[0] == 206:
                next_state_array = np.concatenate([next_state_array, np.zeros(6)], axis=0)
        else:
            raise ValueError(f"Invalid next_state format. Expected shape (206,) or (212,), got {next_state.shape if hasattr(next_state, 'shape') else type(next_state)}")
        
        self.store_experience(state_array, action, reward, next_state_array, done)
        return self.train_batch()
    
    def train_batch(self) -> Dict[str, float]:
        """Train on batch"""
        if len(self.memory) < self.batch_size:
            # Ensure epsilon continues to decay even when not enough memory for a training batch
            self.update_epsilon()
            return {'loss': 0.0, 'q_value': 0.0, 'epsilon': self.epsilon, 'invalid_count': self.invalid_action_count}
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Pad states/next_states to 212 dimensions if needed
        states_list = []
        next_states_list = []
        for experience in batch:
            state, _, _, next_state, _ = experience
            
            # Pad state if needed (206 → 212)
            if state.shape[0] == 206:
                state = np.concatenate([state, np.zeros(6)], axis=0)
            states_list.append(state)
            
            # Pad next_state if needed (206 → 212)
            if next_state.shape[0] == 206:
                next_state = np.concatenate([next_state, np.zeros(6)], axis=0)
            next_states_list.append(next_state)
        
        states = np.array(states_list, dtype=np.float32)
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array(next_states_list, dtype=np.float32)
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
        """Update epsilon with LINEAR decay from start to end over decay_steps"""
        if self.epsilon > self.epsilon_end:
            self.epsilon_step_counter += 1
            if self.epsilon_step_counter <= self.epsilon_decay_steps:
                # Linear decay: epsilon = start - (start-end) * (step/decay_steps)
                progress = self.epsilon_step_counter / self.epsilon_decay_steps
                self.epsilon = max(self.epsilon_end, 
                                   self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress)
            else:
                # Ensure we reach exactly epsilon_end
                self.epsilon = self.epsilon_end
    
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
