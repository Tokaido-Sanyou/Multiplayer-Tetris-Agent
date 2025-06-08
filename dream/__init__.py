"""
DREAM (Dreamer) - Model-Based Reinforcement Learning for Tetris

This package implements DREAM algorithm for the Tetris environment, 
providing state-of-the-art model-based RL capabilities with improved 
sample efficiency and long-term planning.
"""

from dream.agents.dream_agent import DREAMAgent
from dream.algorithms.dream_trainer import DREAMTrainer
from dream.configs.dream_config import DREAMConfig

__version__ = "1.0.0"
__author__ = "Tetris DREAM Team"

__all__ = [
    "DREAMAgent",
    "DREAMTrainer", 
    "DREAMConfig"
] 