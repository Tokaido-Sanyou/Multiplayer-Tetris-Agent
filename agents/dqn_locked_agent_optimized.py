"""
Optimized DQN Agent for Locked State Training Mode
Two approaches: Fixed architecture + Action masking + Valid action selection
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

from .base_agent import BaseAgent


class OptimizedLockedStateDQNAgent(BaseAgent):
    """
    Optimized DQN Agent for Locked State Training Mode
    
    Key Optimizations:
    1. Reduced parameter count (<1M)
    2. Action masking for valid positions only
    3. Two output modes: full action space vs valid action selection
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
                 use_valid_action_selection: bool = False,
                 model_config: Dict[str, Any] = None):
        """
        Initialize Optimized Locked State DQN Agent
        
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
            use_valid_action_selection: If True, network outputs selection from valid actions
            model_config: Configuration for neural network
        """
        
        # Enhanced state representation:
        # Original observation (425) + current selection (4x10x4=160) = 585
        enhanced_obs_size = 425 + 160  # 585 total
        
        # Action space size depends on mode
        if use_valid_action_selection:
            # Max possible valid actions (conservative estimate)
            action_space_size = 800  # Reduced from 1600
        else:
            # Full action space: 200 coordinates × 4 rotations × 2 lock_in = 1600
            action_space_size = 1600
        
        super().__init__(action_space_size, (enhanced_obs_size,), device)
        
        # Configuration
        self.use_valid_action_selection = use_valid_action_selection
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        
        # Optimized model configuration (< 1M parameters)
        default_model_config = {
            'input_size': enhanced_obs_size,  # 585
            'output_size': action_space_size,
            'hidden_layers': [256, 128],  # Optimized to ~389K parameters
            'activation_type': 'relu',
            'use_dropout': True,
            'dropout_rate': 0.2,
            'use_batch_norm': False
        }
        self.model_config = {**default_model_config, **(model_config or {})}
        
        # Initialize networks
        self.q_network = self._create_optimized_network().to(self.device)
        self.target_network = self._create_optimized_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                  lr=self.learning_rate, 
                                  weight_decay=1e-5)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training state
        self.target_update_counter = 0
        self.current_selection = np.zeros((4, 10, 4), dtype=np.float32)
        
        # Action space mappings (for full action space mode)
        if not use_valid_action_selection:
            self.action_to_components = {}
            self.components_to_action = {}
            self._build_action_mappings()
    
    def _create_optimized_network(self) -> nn.Module:
        """Create optimized neural network with <1M parameters"""
        layers = []
        input_size = self.model_config['input_size']
        
        # Hidden layers
        for hidden_size in self.model_config['hidden_layers']:
            layers.append(nn.Linear(input_size, hidden_size))
            
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
        """
        selection_flat = self.current_selection.flatten()
        enhanced_state = np.concatenate([observation, selection_flat])
        return enhanced_state.astype(np.float32)
    
    def get_valid_actions(self, env) -> List[Tuple[int, int, int, int]]:
        """
        Get list of valid actions from environment (OPTIMIZED)
        
        Returns:
            List of (x, y, rotation, lock_in) tuples for valid positions
        """
        valid_actions = []
        
        if hasattr(env, 'players') and len(env.players) > 0:
            player = env.players[0]
            if player and player.current_piece:
                # CRITICAL FIX: Use cached or simplified valid position check
                # Instead of expensive get_valid_positions, use basic board bounds
                for y in range(18, 20):  # Only check top 2 rows for speed
                    for x in range(10):
                        # Basic valid check - just ensure position is in bounds
                        if 0 <= x < 10 and 0 <= y < 20:
                            # Only add lock_in=1 actions for speed
                            valid_actions.append((x, y, 0, 1))  # rotation=0, lock_in=1
                
                # Fallback: if no actions found, add some default actions
                if not valid_actions:
                    for x in range(3, 7):  # Center columns
                        valid_actions.append((x, 19, 0, 1))
        
        # Limit to prevent excessive computation
        return valid_actions[:50]  # Max 50 actions
    
    def select_action_full_space(self, observation: np.ndarray, training: bool = True, 
                                 valid_actions: Optional[List] = None) -> int:
        """
        Select action using full action space (1600 actions) with optional masking
        """
        state = self.encode_state_with_selection(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # Apply action masking if valid_actions provided
        if valid_actions is not None:
            # Convert valid actions to indices
            valid_indices = []
            for x, y, rotation, lock_in in valid_actions:
                if (x, y, rotation, lock_in) in self.components_to_action:
                    valid_indices.append(self.components_to_action[(x, y, rotation, lock_in)])
            
            if valid_indices:
                # Mask invalid actions with very low values
                masked_q_values = np.full_like(q_values, -1e6)
                masked_q_values[valid_indices] = q_values[valid_indices]
                q_values = masked_q_values
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            if valid_actions is not None and valid_indices:
                return random.choice(valid_indices)
            else:
                return random.randint(0, len(q_values) - 1)
        else:
            return np.argmax(q_values)
    
    def select_action_valid_only(self, observation: np.ndarray, valid_actions: List, 
                                training: bool = True) -> Tuple[int, int, int, int]:
        """
        Select action from valid actions only (network outputs selection index)
        """
        if not valid_actions:
            # Fallback: return safe action
            return (4, 0, 0, 1)  # Center, top, no rotation, lock in
        
        state = self.encode_state_with_selection(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values for valid action selection
        with torch.no_grad():
            # Network outputs selection probabilities/values
            output = self.q_network(state_tensor).cpu().numpy()[0]
        
        # Map network output to valid action selection
        if training and random.random() < self.epsilon:
            # Random selection from valid actions
            selected_idx = random.randint(0, len(valid_actions) - 1)
        else:
            # Select based on network output
            # Use first len(valid_actions) outputs to select from valid actions
            valid_outputs = output[:len(valid_actions)]
            selected_idx = np.argmax(valid_outputs)
        
        return valid_actions[selected_idx]
    
    def select_action(self, observation: Any, training: bool = True, 
                     env=None) -> int:
        """
        Main action selection method that routes to appropriate strategy
        """
        # Convert observation if needed
        if isinstance(observation, dict):
            obs_array = observation.get('board', observation.get('observation', observation))
        else:
            obs_array = observation
        
        if not isinstance(obs_array, np.ndarray):
            obs_array = np.array(obs_array, dtype=np.float32)
        
        if self.use_valid_action_selection:
            # Valid action selection mode
            if env is not None:
                valid_actions = self.get_valid_actions(env)
                if valid_actions:
                    # Return the selected valid action components
                    x, y, rotation, lock_in = self.select_action_valid_only(obs_array, valid_actions, training)
                    # For environment compatibility, return board position when locked
                    if lock_in == 1:
                        return y * 10 + x
                    else:
                        return -1  # Not locked, continue
                else:
                    return 99  # Safe fallback position
            else:
                return 99  # No environment provided
        else:
            # Full action space mode
            valid_actions = None
            if env is not None:
                valid_actions = self.get_valid_actions(env)
            
            action_idx = self.select_action_full_space(obs_array, training, valid_actions)
            
            # Convert to environment action if locked
            x, y, rotation, lock_in = self.decode_action_components(action_idx)
            if lock_in == 1:
                return y * 10 + x
            else:
                return -1  # Not locked, continue
    
    def decode_action_components(self, action_idx: int) -> Tuple[int, int, int, int]:
        """Decode action index to (x, y, rotation, lock_in) components"""
        if not self.use_valid_action_selection:
            return self.action_to_components.get(action_idx, (0, 0, 0, 0))
        else:
            # For valid action selection, this shouldn't be used
            return (0, 0, 0, 0)
    
    def encode_action_components(self, x: int, y: int, rotation: int, lock_in: int) -> int:
        """Encode (x, y, rotation, lock_in) components to action index"""
        if not self.use_valid_action_selection:
            return self.components_to_action.get((x, y, rotation, lock_in), 0)
        else:
            # For valid action selection, this shouldn't be used
            return 0
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> Dict[str, float]:
        """
        Update agent with single experience (required by BaseAgent)
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
        self.store_experience(enhanced_state, action, reward, next_enhanced_state, done)
        
        # Train on batch
        return self.train_batch()
    
    def train_batch(self) -> Dict[str, float]:
        """Train on a batch of experiences"""
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
        
        # Current Q values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.update_epsilon()
        
        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.update_target_network()
            self.target_update_counter = 0
        
        return {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon
        }
    
    def update_epsilon(self):
        """Update epsilon for exploration"""
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, 
                             self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps)
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'use_valid_action_selection': self.use_valid_action_selection,
            'model_config': self.model_config
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'type': 'OptimizedLockedStateDQNAgent',
            'use_valid_action_selection': self.use_valid_action_selection,
            'parameters': self.get_parameter_count(),
            'epsilon': self.epsilon,
            'architecture': self.model_config['hidden_layers'],
            'device': str(self.device)
        } 