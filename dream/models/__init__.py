"""
DREAM Neural Network Models

This module contains the neural network architectures for DREAM:
- World Model (RSSM): Learns environment dynamics
- Actor-Critic: Policy and value networks
- Observation Encoder/Decoder: State representation
"""

from dream.models.world_model import WorldModel
from dream.models.actor_critic import ActorCritic
from dream.models.observation_model import TetrisEncoder, TetrisDecoder

__all__ = [
    "WorldModel",
    "ActorCritic",
    "TetrisEncoder",
    "TetrisDecoder"
] 