# Reinforcement Learning Utilities
# Import the core components

# from .single_player_train import train_single_player  # Commented out due to import issues
# from .hierarchical_agent import HierarchicalAgent
# from .mcts_agent import MCTSAgent
# from .replay_buffer import ReplayBuffer
# from .state_model import StateModel
# from .reward_model import RewardModel
# from .state_explorer import StateExplorer

# Handle both direct execution and module import
try:
    # from .single_player_train import train_single_player  # Commented out due to import issues
    # from .hierarchical_agent import HierarchicalAgent
    # from .mcts_agent import MCTSAgent
    # from .replay_buffer import ReplayBuffer
    # from .state_model import StateModel
    # from .reward_model import RewardModel
    # from .state_explorer import StateExplorer
    pass
except ImportError:
    # Direct execution - imports without relative paths
    # from single_player_train import train_single_player  # Commented out due to import issues
    # from hierarchical_agent import HierarchicalAgent
    # from mcts_agent import MCTSAgent
    # from replay_buffer import ReplayBuffer
    # from state_model import StateModel
    # from reward_model import RewardModel
    # from state_explorer import StateExplorer
    pass

"""
Reinforcement Learning utilities for Tetris

This package contains various RL algorithms and utilities:
- State models for learning piece placement policies
- Actor-critic agents for policy optimization
- Exploration strategies for data collection  
- Reward models for value estimation
- Replay buffers for experience storage
"""

# 
__all__ = [
    # 'train_single_player',
    # 'HierarchicalAgent',
    # 'MCTSAgent',
    # 'StateModel',
    # 'RewardModel',
    # 'StateExplorer',
    # 'ReplayBuffer'
]