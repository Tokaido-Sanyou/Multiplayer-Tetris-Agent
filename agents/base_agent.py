"""
Base Agent Class
Defines the interface for all RL agents in the Tetris environment
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents
    
    Defines the standard interface that all agents must implement
    """
    
    def __init__(self, 
                 action_space_size: int = 8,
                 observation_space_shape: Tuple = None,
                 device: str = 'cpu'):
        """
        Initialize base agent
        
        Args:
            action_space_size: Number of possible actions
            observation_space_shape: Shape of observation space
            device: Device to run agent on ('cpu' or 'cuda')
        """
        self.action_space_size = action_space_size
        self.observation_space_shape = observation_space_shape
        self.device = torch.device(device)
        
        # Training state
        self.training_mode = True
        self.episode_count = 0
        self.step_count = 0
        
    @abstractmethod
    def select_action(self, observation: Any, training: bool = True) -> int:
        """
        Select action given observation
        
        Args:
            observation: Environment observation
            training: Whether agent is in training mode
            
        Returns:
            Selected action (integer)
        """
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """
        Update agent parameters
        
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save agent checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load agent checkpoint
        
        Args:
            filepath: Path to load checkpoint from
        """
        pass
    
    def set_training_mode(self, training: bool = True) -> None:
        """
        Set agent training mode
        
        Args:
            training: Whether to enable training mode
        """
        self.training_mode = training
    
    def set_eval_mode(self) -> None:
        """Set agent to evaluation mode"""
        self.set_training_mode(False)
    
    def preprocess_observation(self, observation: Any) -> torch.Tensor:
        """
        Preprocess observation for neural network input
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            Preprocessed observation tensor
        """
        if isinstance(observation, (list, tuple)):
            # Binary tuple observation - convert to numpy array
            obs_array = np.array(observation, dtype=np.float32)
        else:
            obs_array = observation
        
        # Reshape to board format (20x10) + piece info
        if obs_array.size >= 200:
            board = obs_array[:200].reshape(20, 10)
            # Add batch and channel dimensions
            board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)  # [1, 1, 20, 10]
            return board_tensor.to(self.device)
        else:
            # Fallback for smaller observations
            obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0)
            return obs_tensor.to(self.device)
    
    def convert_action_to_tuple(self, action_idx: int) -> Tuple[int, ...]:
        """
        Convert scalar action to tuple format for environment
        
        Args:
            action_idx: Action index (0-7)
            
        Returns:
            Action tuple
        """
        action_tuple = [0] * self.action_space_size
        if 0 <= action_idx < self.action_space_size:
            action_tuple[action_idx] = 1
        return tuple(action_tuple)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information
        
        Returns:
            Dictionary containing agent state information
        """
        return {
            'agent_type': self.__class__.__name__,
            'action_space_size': self.action_space_size,
            'observation_space_shape': self.observation_space_shape,
            'device': str(self.device),
            'training_mode': self.training_mode,
            'episode_count': self.episode_count,
            'step_count': self.step_count
        }
    
    def reset_episode(self) -> None:
        """Reset agent state for new episode"""
        self.episode_count += 1
    
    def step(self) -> None:
        """Increment step counter"""
        self.step_count += 1 