"""
DQN Agent Implementation
Deep Q-Network agent for Tetris environment
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
from collections import deque

from .base_agent import BaseAgent
from models.tetris_cnn import TetrisCNN


class DQNAgent(BaseAgent):
    """
    Deep Q-Network Agent for Tetris
    
    Implements DQN with experience replay and target networks
    """
    
    def __init__(self,
                 action_space_size: int = 8,
                 observation_space_shape: Tuple = (1, 20, 10),
                 device: str = 'cuda',
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 50000,
                 target_update_freq: int = 1000,
                 model_config: Dict[str, Any] = None):
        """
        Initialize DQN Agent
        
        Args:
            action_space_size: Number of possible actions
            observation_space_shape: Shape of observation space
            device: Device to run on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            target_update_freq: Frequency to update target network
            model_config: Configuration for neural network
        """
        super().__init__(action_space_size, observation_space_shape, device)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        
        # Model configuration
        default_model_config = {
            'output_size': action_space_size,
            'activation_type': 'identity',
            'use_dropout': True,
            'dropout_rate': 0.1
        }
        self.model_config = {**default_model_config, **(model_config or {})}
        
        # Initialize networks
        self.q_network = TetrisCNN(**self.model_config).to(self.device)
        self.target_network = TetrisCNN(**self.model_config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Training state
        self.target_update_counter = 0
        
    def select_action(self, observation: Any, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            observation: Environment observation
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randint(0, self.action_space_size - 1)
        
        # Greedy action (exploitation)
        with torch.no_grad():
            state_tensor = self.preprocess_observation(observation)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
            
        self.step()
        return action
    
    def update_epsilon(self) -> None:
        """Update epsilon for exploration decay"""
        if self.step_count < self.epsilon_decay_steps:
            decay_progress = self.step_count / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress
        else:
            self.epsilon = self.epsilon_end
    
    def update(self, 
               state: Any,
               action: int,
               reward: float,
               next_state: Any,
               done: bool) -> Dict[str, float]:
        """
        Update agent with single experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        state_tensor = self.preprocess_observation(state)
        next_state_tensor = self.preprocess_observation(next_state)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.BoolTensor([done]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(state_tensor)
        current_q_value = current_q_values.gather(1, action_tensor.unsqueeze(1))
        
        # Next Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor)
            max_next_q_value = next_q_values.max(1)[0]
            target_q_value = reward_tensor + (self.gamma * max_next_q_value * ~done_tensor)
        
        # Compute loss
        loss = F.mse_loss(current_q_value.squeeze(), target_q_value.unsqueeze(0))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
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
    
    def batch_update(self,
                     batch_states: list,
                     batch_actions: list,
                     batch_rewards: list,
                     batch_next_states: list,
                     batch_dones: list,
                     importance_weights: list = None) -> Dict[str, float]:
        """
        Update agent with batch of experiences
        
        Args:
            batch_states: List of states
            batch_actions: List of actions
            batch_rewards: List of rewards
            batch_next_states: List of next states
            batch_dones: List of done flags
            importance_weights: Importance sampling weights for prioritized replay
            
        Returns:
            Dictionary of training metrics
        """
        batch_size = len(batch_states)
        
        # Convert to tensors
        state_batch = torch.cat([self.preprocess_observation(s) for s in batch_states])
        next_state_batch = torch.cat([self.preprocess_observation(s) for s in batch_next_states])
        action_batch = torch.LongTensor(batch_actions).to(self.device)
        reward_batch = torch.FloatTensor(batch_rewards).to(self.device)
        done_batch = torch.BoolTensor(batch_dones).to(self.device)
        
        if importance_weights is not None:
            weight_batch = torch.FloatTensor(importance_weights).to(self.device)
        else:
            weight_batch = torch.ones(batch_size).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # Next Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = reward_batch + (self.gamma * max_next_q_values * ~done_batch)
        
        # Compute weighted loss
        td_errors = current_q_values.squeeze() - target_q_values
        loss = (weight_batch * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.update_target_network()
            self.target_update_counter = 0
        
        # Update epsilon
        self.update_epsilon()
        
        # Return TD errors for prioritized replay
        td_errors_abs = td_errors.abs().detach().cpu().numpy()
        
        return {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon,
            'td_errors': td_errors_abs
        }
    
    def update_target_network(self) -> None:
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save agent checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'model_config': self.model_config,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay_steps': self.epsilon_decay_steps,
                'target_update_freq': self.target_update_freq
            }
        }
        
        # Create directory if filepath contains a directory
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load agent checkpoint
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        
        # Load hyperparameters if available
        if 'hyperparameters' in checkpoint:
            hyperparams = checkpoint['hyperparameters']
            self.learning_rate = hyperparams['learning_rate']
            self.gamma = hyperparams['gamma']
            self.epsilon_start = hyperparams['epsilon_start']
            self.epsilon_end = hyperparams['epsilon_end']
            self.epsilon_decay_steps = hyperparams['epsilon_decay_steps']
            self.target_update_freq = hyperparams['target_update_freq']
    
    def get_q_values(self, observation: Any) -> np.ndarray:
        """
        Get Q-values for given observation
        
        Args:
            observation: Environment observation
            
        Returns:
            Q-values for all actions
        """
        with torch.no_grad():
            state_tensor = self.preprocess_observation(observation)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def set_training_mode(self, training: bool = True) -> None:
        """
        Set agent training mode
        
        Args:
            training: Whether to enable training mode
        """
        super().set_training_mode(training)
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information
        
        Returns:
            Dictionary containing agent state information
        """
        info = super().get_info()
        info.update({
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'target_update_freq': self.target_update_freq,
            'target_update_counter': self.target_update_counter,
            'model_config': self.model_config
        })
        return info 