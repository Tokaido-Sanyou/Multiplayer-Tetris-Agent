"""
World Model for DREAM Algorithm

Simple world model that works directly with tuple observations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any


class WorldModel(nn.Module):
    """Simple world model for tuple observations"""
    
    def __init__(self,
                 observation_dim: int = 212,
                 action_dim: int = 8,
                 hidden_dim: int = 256,
                 state_dim: int = 128,
                 **kwargs):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # RNN for temporal modeling
        self.rnn = nn.GRU(state_dim + action_dim, state_dim, batch_first=True)
        
        # Prediction heads
        self.reward_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.continue_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.obs_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_dim)
        )
    
    def get_initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Get initial hidden state"""
        return {
            'h': torch.zeros(1, batch_size, self.state_dim, device=device)
        }
    
    def encode_observation(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations to state representation"""
        return self.obs_encoder(observations)
    
    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through world model
        
        Args:
            observations: [batch, seq_len, obs_dim]
            actions: [batch, seq_len]
        """
        batch_size, seq_len = observations.shape[:2]
        device = observations.device
        
        # Encode observations
        encoded_obs = self.obs_encoder(observations)  # [batch, seq_len, state_dim]
        
        # One-hot encode actions
        actions_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()  # [batch, seq_len, action_dim]
        
        # Combine encoded observations and actions
        rnn_input = torch.cat([encoded_obs, actions_one_hot], dim=-1)  # [batch, seq_len, state_dim + action_dim]
        
        # Get initial state
        initial_state = self.get_initial_state(batch_size, device)
        h0 = initial_state['h']  # [1, batch, state_dim]
        
        # Run through RNN
        rnn_output, _ = self.rnn(rnn_input, h0)  # [batch, seq_len, state_dim]
        
        # Predict rewards and continues
        predicted_rewards = self.reward_head(rnn_output).squeeze(-1)  # [batch, seq_len]
        predicted_continues = torch.sigmoid(self.continue_head(rnn_output)).squeeze(-1)  # [batch, seq_len]
        
        # Predict next observations
        predicted_observations = self.obs_decoder(rnn_output)  # [batch, seq_len, obs_dim]
        
        # Simple KL loss (regularization)
        kl_loss = torch.zeros(batch_size, seq_len, device=device)
        
        return {
            'predicted_rewards': predicted_rewards,
            'predicted_continues': predicted_continues,
            'predicted_observations': predicted_observations,
            'kl_loss': kl_loss,
            'states': rnn_output
        }
    
    def imagine(self, initial_state: Dict[str, torch.Tensor], actions: torch.Tensor) -> Dict[str, Any]:
        """
        Imagine trajectory using world model
        
        Args:
            initial_state: Initial hidden state
            actions: [batch, seq_len]
        """
        batch_size, seq_len = actions.shape
        device = actions.device
        
        # One-hot encode actions
        actions_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
        
        # Initialize
        h = initial_state['h']  # [1, batch, state_dim]
        predicted_rewards = []
        predicted_continues = []
        predicted_observations = []
        states = []
        
        for t in range(seq_len):
            # Get current action
            action_t = actions_one_hot[:, t:t+1]  # [batch, 1, action_dim]
            
            # Dummy observation encoding (use zeros for imagination)
            obs_encoding = torch.zeros(batch_size, 1, self.state_dim, device=device)
            
            # Combine with action
            rnn_input = torch.cat([obs_encoding, action_t], dim=-1)
            
            # Step through RNN
            rnn_output, h = self.rnn(rnn_input, h)
            
            # Predict
            reward = self.reward_head(rnn_output).squeeze(-1)
            continue_prob = torch.sigmoid(self.continue_head(rnn_output)).squeeze(-1)
            obs = self.obs_decoder(rnn_output)
            
            predicted_rewards.append(reward)
            predicted_continues.append(continue_prob)
            predicted_observations.append(obs)
            states.append(rnn_output.squeeze(1))
        
        return {
            'predicted_rewards': torch.cat(predicted_rewards, dim=1),
            'predicted_continues': torch.cat(predicted_continues, dim=1),
            'predicted_observations': torch.cat(predicted_observations, dim=1),
            'final_state': {'h': h}
        } 