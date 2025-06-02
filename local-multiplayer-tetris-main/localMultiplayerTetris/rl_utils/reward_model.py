"""
RewardModel: estimates immediate placement reward given current state and action.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, state_dim=206, action_dim=8, hidden_dim=256):
        super(RewardModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # MLP to predict scalar reward
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """
        Args:
            state: Tensor of shape (batch_size, state_dim)
            action: LongTensor of shape (batch_size, ) containing action indices
        Returns:
            reward_pred: Tensor of shape (batch_size,) predicted immediate reward
        """
        # One-hot encode actions
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float().to(state.device)
        x = torch.cat([state, action_onehot], dim=1)
        reward_pred = self.model(x).squeeze(-1)
        return reward_pred
