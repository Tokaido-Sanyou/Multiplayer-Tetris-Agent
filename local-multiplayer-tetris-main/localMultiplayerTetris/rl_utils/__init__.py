"""
Reinforcement Learning utilities for Tetris

Core DQN implementation with TensorBoard logging and parallel training support.
"""

# Only import modules that don't have circular dependencies
from .dqn_new import DQNAgent, ReplayBuffer, DQN

__all__ = [
    'DQNAgent',
    'ReplayBuffer', 
    'DQN'
]

# Version info
__version__ = '1.0.0' 