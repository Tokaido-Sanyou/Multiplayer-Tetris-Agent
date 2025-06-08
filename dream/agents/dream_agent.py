"""
DREAM Agent

Main agent class that wraps DREAM functionality for easy deployment
and evaluation.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple

from dream.models.world_model import WorldModel
from dream.models.actor_critic import ActorCritic
from dream.configs.dream_config import DREAMConfig
from agents.base_agent import BaseAgent


class DREAMAgent(BaseAgent):
    """
    DREAM Agent for Tetris
    
    Combines world model and actor-critic for action selection.
    Can be used for training or evaluation.
    """
    
    def __init__(self,
                 config: DREAMConfig,
                 load_checkpoint: Optional[str] = None):
        """
        Initialize DREAM agent
        
        Args:
            config: DREAM configuration
            load_checkpoint: Path to checkpoint to load
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.world_model = WorldModel(**config.world_model).to(self.device)
        self.actor_critic = ActorCritic(**config.actor_critic).to(self.device)
        
        # Load checkpoint if provided
        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)
        
        self.eval()
    
    def select_action(self, observation: Any, training: bool = True) -> int:
        """
        Select action given observation
        
        Args:
            observation: Environment observation
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        # Process observation
        obs_tensor = self._process_observation(observation)
        
        with torch.no_grad():
            # Get action from actor-critic
            action, log_prob, value = self.actor_critic.get_action_and_value(
                obs_tensor.unsqueeze(0).to(self.device),
                deterministic=not training
            )
            
            action = action.squeeze(0)
            
            # Convert to appropriate format
            if self.config.action_mode == 'direct':
                # Convert binary actions to single action index
                action = self.convert_action_to_tuple(action.cpu().numpy())
            else:
                # Already discrete action
                action = action.cpu().item()
        
        return action
    
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """
        Update is handled by DREAMTrainer, not individual agent
        """
        return {}
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint"""
        checkpoint = {
            'world_model_state_dict': self.world_model.state_dict(),
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'config': self.config.get_full_config()
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.world_model.load_state_dict(checkpoint['world_model_state_dict'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        
        print(f"✅ Loaded DREAM agent from {filepath}")
    
    def preprocess_observation(self, observation: Any) -> torch.Tensor:
        """Preprocess observation for the agent"""
        return self._process_observation(observation)
    
    def convert_action_to_tuple(self, action_array: np.ndarray) -> int:
        """Convert action array to tuple format"""
        if self.config.action_mode == 'direct':
            # Convert binary action array to action index
            # This is a simplified conversion - in practice you'd want
            # a proper mapping from binary actions to game actions
            return int(np.argmax(action_array))
        else:
            # Already a single action
            return int(action_array)
    
    def _process_observation(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """Process observation to tensor"""
        # Convert observation dict to tensors
        processed_obs = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                processed_obs[key] = torch.from_numpy(value).float()
            else:
                processed_obs[key] = torch.tensor(value, dtype=torch.float32)
        
        # Use world model encoder to get state representation
        with torch.no_grad():
            state = self.world_model.encode_observation(
                {k: v.unsqueeze(0) for k, v in processed_obs.items()}
            ).squeeze(0)
        
        return state
    
    def eval(self):
        """Set agent to evaluation mode"""
        self.world_model.eval()
        self.actor_critic.eval()
    
    def train(self):
        """Set agent to training mode"""
        self.world_model.train()
        self.actor_critic.train()
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_type': 'DREAM',
            'action_mode': self.config.action_mode,
            'device': str(self.device),
            'world_model_params': sum(p.numel() for p in self.world_model.parameters()),
            'actor_critic_params': sum(p.numel() for p in self.actor_critic.parameters())
        } 