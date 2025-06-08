"""
Action-Level DQN Agent for Hierarchical Tetris Control
Learns 8 basic Tetris actions to reach target locked states
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque

from .base_agent import BaseAgent


class ActionDQNNetwork(nn.Module):
    """
    Neural network for Action-level DQN
    Maps current state + target position to action values
    """
    
    def __init__(self, 
                 observation_size: int = 425,
                 target_position_size: int = 4,  # x, y, rotation, lock_in
                 action_space_size: int = 8,
                 hidden_sizes: list = [256, 128]):
        super().__init__()
        
        # Combined input: observation + target position
        input_size = observation_size + target_position_size
        
        # Build network layers
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, action_space_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, observation: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining observation and target position
        
        Args:
            observation: Current game state [batch_size, 425]
            target_position: Target position [batch_size, 4] (x, y, rotation, lock_in)
            
        Returns:
            Q-values for 8 actions [batch_size, 8]
        """
        # Concatenate inputs
        combined_input = torch.cat([observation, target_position], dim=1)
        return self.network(combined_input)


class ActionDQNAgent(BaseAgent):
    """
    Action-Level DQN Agent for Hierarchical Control
    
    Learns to execute the 8 basic Tetris actions to reach target positions
    determined by a higher-level locked state DQN.
    """
    
    def __init__(self,
                 device: str = 'cuda',
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,  # Lower gamma for immediate rewards
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 1000,
                 target_update_freq: int = 100,
                 memory_size: int = 10000,
                 batch_size: int = 32):
        """
        Initialize Action DQN Agent
        
        Args:
            device: Device to run on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            gamma: Discount factor (lower for immediate rewards)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            target_update_freq: Frequency to update target network
            memory_size: Size of experience replay buffer
            batch_size: Batch size for training
        """
        
        # 8 basic Tetris actions
        action_space_size = 8
        observation_space_shape = (425,)  # Tetris observation size
        
        super().__init__(action_space_size, observation_space_shape, device)
        
        # Configuration
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        
        # Initialize networks
        self.q_network = ActionDQNNetwork().to(self.device)
        self.target_network = ActionDQNNetwork().to(self.device)
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
        self.current_target_position = None
        
        # Action names for debugging
        self.action_names = [
            "Move Left", "Move Right", "Move Down", "Rotate CW", 
            "Rotate CCW", "Hard Drop", "Hold", "No-op"
        ]
    
    def set_target_position(self, x: int, y: int, rotation: int, lock_in: int):
        """
        Set the target position that actions should work towards
        
        Args:
            x: Target x coordinate
            y: Target y coordinate  
            rotation: Target rotation
            lock_in: Whether to lock the piece (0 or 1)
        """
        self.current_target_position = np.array([x, y, rotation, lock_in], dtype=np.float32)
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy with target position
        
        Args:
            observation: Current game observation
            training: Whether in training mode
            
        Returns:
            Selected action index (0-7)
        """
        if self.current_target_position is None:
            raise ValueError("Target position must be set before selecting actions")
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        
        # Greedy action selection
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            target_tensor = torch.FloatTensor(self.current_target_position).unsqueeze(0).to(self.device)
            
            q_values = self.q_network(obs_tensor, target_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def store_experience(self, 
                        state: np.ndarray, 
                        action: int, 
                        reward: float,
                        next_state: np.ndarray, 
                        done: bool,
                        target_position: np.ndarray):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received  
            next_state: Next state observation
            done: Whether episode is done
            target_position: Target position used for this experience
        """
        experience = (state, action, reward, next_state, done, target_position.copy())
        self.memory.append(experience)
    
    def train_batch(self) -> Dict[str, float]:
        """
        Train on a batch of experiences from replay buffer
        
        Returns:
            Training metrics dictionary
        """
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0, 'q_value': 0.0}
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, target_positions = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.BoolTensor(dones).to(self.device)
        target_batch = torch.FloatTensor(np.array(target_positions)).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(state_batch, target_batch)
        current_q_value = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # Next Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch, target_batch)
            max_next_q_value = next_q_values.max(1)[0]
            target_q_value = reward_batch + (self.gamma * max_next_q_value * ~done_batch)
        
        # Compute loss
        loss = F.mse_loss(current_q_value.squeeze(), target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
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
    
    def compute_reward(self, 
                      current_piece_pos: Tuple[int, int, int], 
                      target_pos: Tuple[int, int, int],
                      action: int,
                      piece_placed: bool,
                      lines_cleared: int) -> float:
        """
        Compute reward for action taken towards target position
        
        Args:
            current_piece_pos: Current piece (x, y, rotation)
            target_pos: Target position (x, y, rotation) 
            action: Action taken (0-7)
            piece_placed: Whether piece was placed this step
            lines_cleared: Number of lines cleared
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Distance reward - closer to target is better
        current_x, current_y, current_rot = current_piece_pos
        target_x, target_y, target_rot = target_pos
        
        # Manhattan distance to target position
        pos_distance = abs(current_x - target_x) + abs(current_y - target_y)
        rot_distance = min(abs(current_rot - target_rot), 4 - abs(current_rot - target_rot))
        
        # Reward for getting closer (negative distance)
        reward -= pos_distance * 0.1
        reward -= rot_distance * 0.05
        
        # Strong reward for reaching exact target
        if current_x == target_x and current_y == target_y and current_rot == target_rot:
            reward += 10.0
        
        # Reward for clearing lines
        reward += lines_cleared * 5.0
        
        # Small penalty for each step to encourage efficiency  
        reward -= 0.01
        
        # Penalty for invalid/ineffective actions
        if action == 7:  # No-op
            reward -= 0.1
        
        return reward
    
    def update(self, 
               state: np.ndarray, 
               action: int, 
               reward: float,
               next_state: np.ndarray, 
               done: bool,
               target_position: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Update agent with single experience (implements abstract method)
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received  
            next_state: Next state observation
            done: Whether episode is done
            target_position: Target position (optional, uses current if None)
            
        Returns:
            Training metrics dictionary
        """
        # Use current target if not provided
        if target_position is None:
            if self.current_target_position is None:
                return {'loss': 0.0, 'q_value': 0.0}
            target_position = self.current_target_position
        
        # Store experience
        self.store_experience(state, action, reward, next_state, done, target_position)
        
        # Train if enough experiences
        if len(self.memory) >= self.batch_size:
            return self.train_batch()
        else:
            return {'loss': 0.0, 'q_value': 0.0}
    
    def update_epsilon(self):
        """Update epsilon for exploration decay"""
        if self.step_count < self.epsilon_decay_steps:
            decay_progress = self.step_count / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress
        else:
            self.epsilon = self.epsilon_end
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
    
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'target_update_counter': self.target_update_counter
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.target_update_counter = checkpoint['target_update_counter']
            return True
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'type': 'ActionDQN',
            'parameters': self.get_parameter_count(),
            'architecture': 'Observation+Target -> FC(256) -> FC(128) -> Actions(8)',
            'device': str(self.device),
            'action_space': self.action_space_size,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'target_position': self.current_target_position.tolist() if self.current_target_position is not None else None
        } 