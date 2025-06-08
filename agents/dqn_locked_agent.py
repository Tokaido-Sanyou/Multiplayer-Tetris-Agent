"""
Enhanced DQN Agent for Locked State Training Mode
Implements standardized state/action representation for Tetris locked position training
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List
from collections import deque

from .base_agent import BaseAgent


class LockedStateDQNAgent(BaseAgent):
    """
    Enhanced DQN Agent for Locked State Training Mode
    
    Features:
    - Standardized state representation: observation + current selection
    - Standardized action representation: (x, y, rotation, lock_in) bitwise
    - Locked position training support
    - GPU acceleration throughout
    """
    
    def __init__(self,
                 device: str = 'cuda',
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 50000,
                 target_update_freq: int = 1000,
                 memory_size: int = 100000,
                 batch_size: int = 32,
                 model_config: Dict[str, Any] = None):
        """
        Initialize Locked State DQN Agent
        
        Args:
            device: Device to run on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            target_update_freq: Frequency to update target network
            memory_size: Size of experience replay buffer
            batch_size: Batch size for training
            model_config: Configuration for neural network
        """
        
        # Enhanced state representation:
        # Original observation (425) + current selection (4x10x4=160) = 585
        enhanced_obs_size = 425 + 160  # 585 total
        
        # Enhanced action representation:
        # 200 coordinates (20x10) × 4 rotations × 2 lock_in states = 1600 total actions
        action_space_size = 1600  # 200 * 4 * 2
        
        super().__init__(action_space_size, (enhanced_obs_size,), device)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        
        # Enhanced model configuration for locked state training (OPTIMIZED)
        default_model_config = {
            'input_size': enhanced_obs_size,
            'output_size': action_space_size,
            'hidden_layers': [256, 128],  # Optimized to ~389K parameters (< 1M limit)
            'activation_type': 'relu',
            'use_dropout': True,
            'dropout_rate': 0.2,  # Increased dropout for better regularization
            'use_batch_norm': False  # Disable batch norm to avoid single sample issues
        }
        self.model_config = {**default_model_config, **(model_config or {})}
        
        # Initialize networks with enhanced architecture
        self.q_network = self._create_enhanced_network().to(self.device)
        self.target_network = self._create_enhanced_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer with better settings for locked state training
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                  lr=self.learning_rate, 
                                  weight_decay=1e-5)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training state
        self.target_update_counter = 0
        self.current_selection = np.zeros((4, 10, 4), dtype=np.float32)  # 4 rotations x 10 positions x 4 pieces
        
        # Action space mappings
        self.action_to_components = {}
        self.components_to_action = {}
        self._build_action_mappings()
        
    def _create_enhanced_network(self) -> nn.Module:
        """Create enhanced neural network for locked state training"""
        layers = []
        input_size = self.model_config['input_size']
        
        # Input layer
        for i, hidden_size in enumerate(self.model_config['hidden_layers']):
            layers.append(nn.Linear(input_size, hidden_size))
            
            if self.model_config.get('use_batch_norm', False):
                layers.append(nn.BatchNorm1d(hidden_size))
                
            if self.model_config['activation_type'] == 'relu':
                layers.append(nn.ReLU())
            elif self.model_config['activation_type'] == 'tanh':
                layers.append(nn.Tanh())
                
            if self.model_config.get('use_dropout', False):
                layers.append(nn.Dropout(self.model_config['dropout_rate']))
                
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, self.model_config['output_size']))
        
        return nn.Sequential(*layers)
    
    def _build_action_mappings(self):
        """Build mappings between action indices and (x, y, rotation, lock_in) components"""
        action_idx = 0
        for y in range(20):  # 20 rows
            for x in range(10):  # 10 columns
                for rotation in range(4):  # 4 rotations
                    for lock_in in range(2):  # 2 lock_in states
                        components = (x, y, rotation, lock_in)
                        self.action_to_components[action_idx] = components
                        self.components_to_action[components] = action_idx
                        action_idx += 1
    
    def encode_state_with_selection(self, observation: np.ndarray) -> np.ndarray:
        """
        Encode enhanced state: original observation + current selection
        
        Args:
            observation: Original environment observation (425,)
            
        Returns:
            Enhanced state representation (585,)
        """
        # Flatten current selection (4x10x4 -> 160)
        selection_flat = self.current_selection.flatten()
        
        # Concatenate observation with current selection
        enhanced_state = np.concatenate([observation, selection_flat])
        
        return enhanced_state.astype(np.float32)
    
    def decode_action_components(self, action_idx: int) -> Tuple[int, int, int, int]:
        """
        Decode action index to (x, y, rotation, lock_in) components
        
        Args:
            action_idx: Action index (0-1599)
            
        Returns:
            Tuple of (x, y, rotation, lock_in)
        """
        return self.action_to_components.get(action_idx, (0, 0, 0, 0))
    
    def encode_action_components(self, x: int, y: int, rotation: int, lock_in: int) -> int:
        """
        Encode (x, y, rotation, lock_in) components to action index
        
        Args:
            x: X coordinate (0-9)
            y: Y coordinate (0-19)
            rotation: Rotation (0-3)
            lock_in: Lock in flag (0-1)
            
        Returns:
            Action index (0-1599)
        """
        return self.components_to_action.get((x, y, rotation, lock_in), 0)
    
    def update_selection(self, x: int, y: int, rotation: int, piece_type: int):
        """
        Update current selection state
        
        Args:
            x: X coordinate
            y: Y coordinate  
            rotation: Rotation
            piece_type: Type of piece (0-3)
        """
        # Clear previous selection
        self.current_selection.fill(0.0)
        
        # Set current selection
        if 0 <= rotation < 4 and 0 <= x < 10 and 0 <= piece_type < 4:
            self.current_selection[rotation, x, piece_type] = 1.0
    
    def select_action(self, observation: Any, training: bool = True, valid_actions: List[int] = None) -> int:
        """
        Select action using epsilon-greedy policy with action masking
        
        Args:
            observation: Environment observation
            training: Whether in training mode
            valid_actions: List of valid action indices (for masking)
            
        Returns:
            Selected action index
        """
        # Encode enhanced state
        enhanced_state = self.encode_state_with_selection(observation)
        
        if training and random.random() < self.epsilon:
            # Random action (exploration) from valid actions
            if valid_actions and len(valid_actions) > 0:
                action_idx = random.choice(valid_actions)
            else:
                action_idx = random.randint(0, self.action_space_size - 1)
        else:
            # Greedy action (exploitation) with action masking
            with torch.no_grad():
                state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).squeeze(0)
                
                # Apply action masking if provided
                if valid_actions and len(valid_actions) > 0:
                    # Create mask
                    mask = torch.full_like(q_values, float('-inf'))
                    mask[valid_actions] = 0
                    masked_q_values = q_values + mask
                    action_idx = masked_q_values.argmax().item()
                else:
                    action_idx = q_values.argmax().item()
        
        # Decode action components
        x, y, rotation, lock_in = self.decode_action_components(action_idx)
        
        # Update selection state if not locking in
        if not lock_in:
            self.update_selection(x, y, rotation, 0)  # Assume piece type 0 for now
        
        self.step()
        
        return action_idx
    
    def select_action_with_info(self, observation: Any, training: bool = True, valid_actions: List[int] = None) -> Dict[str, Any]:
        """
        Select action and return detailed info (for training and debugging)
        
        Args:
            observation: Environment observation
            training: Whether in training mode
            valid_actions: List of valid action indices (for masking)
            
        Returns:
            Dictionary containing action info
        """
        # Encode enhanced state
        enhanced_state = self.encode_state_with_selection(observation)
        
        if training and random.random() < self.epsilon:
            # Random action (exploration) from valid actions
            if valid_actions and len(valid_actions) > 0:
                action_idx = random.choice(valid_actions)
            else:
                action_idx = random.randint(0, self.action_space_size - 1)
        else:
            # Greedy action (exploitation) with action masking
            with torch.no_grad():
                state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).squeeze(0)
                
                # Apply action masking if provided
                if valid_actions and len(valid_actions) > 0:
                    # Create mask
                    mask = torch.full_like(q_values, float('-inf'))
                    mask[valid_actions] = 0
                    masked_q_values = q_values + mask
                    action_idx = masked_q_values.argmax().item()
                else:
                    action_idx = q_values.argmax().item()
        
        # Decode action components
        x, y, rotation, lock_in = self.decode_action_components(action_idx)
        
        # Update selection state if not locking in
        if not lock_in:
            self.update_selection(x, y, rotation, 0)  # Assume piece type 0 for now
        
        self.step()
        
        return {
            'action_idx': action_idx,
            'x': x,
            'y': y,
            'rotation': rotation,
            'lock_in': lock_in,
            'enhanced_state': enhanced_state
        }
    
    def store_experience(self, state: np.ndarray, action_info: Dict, reward: float, 
                        next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state (enhanced)
            action_info: Action information from select_action
            reward: Reward received
            next_state: Next state (enhanced)
            done: Whether episode is done
        """
        experience = (state, action_info['action_idx'], reward, next_state, done)
        self.memory.append(experience)
    
    def update_epsilon(self) -> None:
        """Update epsilon for exploration decay"""
        if self.step_count < self.epsilon_decay_steps:
            decay_progress = self.step_count / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress
        else:
            self.epsilon = self.epsilon_end
    
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> Dict[str, float]:
        """
        Update agent with single experience (required by BaseAgent)
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary of training metrics
        """
        # Convert states to enhanced format if needed
        if isinstance(state, np.ndarray) and state.shape[0] == 425:
            enhanced_state = self.encode_state_with_selection(state)
        else:
            enhanced_state = state
            
        if isinstance(next_state, np.ndarray) and next_state.shape[0] == 425:
            next_enhanced_state = self.encode_state_with_selection(next_state)
        else:
            next_enhanced_state = next_state
        
        # Store experience
        experience = (enhanced_state, action, reward, next_enhanced_state, done)
        self.memory.append(experience)
        
        # Train on batch
        return self.train_batch()
    
    def train_batch(self) -> Dict[str, float]:
        """
        Train the network on a batch of experiences
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0, 'q_value': 0.0, 'epsilon': self.epsilon}
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors (optimize by converting to numpy arrays first)
        states_array = np.array(states, dtype=np.float32)
        states_tensor = torch.from_numpy(states_array).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_array = np.array(next_states, dtype=np.float32)
        next_states_tensor = torch.from_numpy(next_states_array).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states_tensor)
        current_q_value = current_q_values.gather(1, actions_tensor.unsqueeze(1))
        
        # Next Q-values (double DQN)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards_tensor + (self.gamma * max_next_q_values * ~dones_tensor)
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q_value.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.update_target_network()
            self.target_update_counter = 0
        
        # Update epsilon
        self.update_epsilon()
        
        return {
            'loss': loss.item(),
            'q_value': current_q_value.mean().item(),
            'epsilon': self.epsilon
        }
    
    def update_target_network(self) -> None:
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'current_selection': self.current_selection,
            'model_config': self.model_config
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.current_selection = checkpoint['current_selection']
    
    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
        """Get Q-values for given observation"""
        enhanced_state = self.encode_state_with_selection(observation)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
        return q_values.cpu().numpy().flatten()
    
    def set_training_mode(self, training: bool = True) -> None:
        """Set training mode"""
        self.q_network.train(training)
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'type': 'LockedStateDQNAgent',
            'parameters': sum(p.numel() for p in self.q_network.parameters() if p.requires_grad),
            'architecture': self.model_config['hidden_layers'],
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'memory_size': len(self.memory),
            'device': self.device,
            'current_selection_active': np.sum(self.current_selection) > 0
        } 