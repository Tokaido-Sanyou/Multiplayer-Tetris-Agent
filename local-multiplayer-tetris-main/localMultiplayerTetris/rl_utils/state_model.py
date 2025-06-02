"""
State transition model: predicts next state given current state and action.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class StateModel(nn.Module):
    """
    Predicts ideal placement (rotation and x-position) for the current piece given board state.
    Input: flattened state vector (grid + piece metadata)
    Output: rotation logits and x-position logits
    """
    def __init__(self, state_dim=206, hidden_dim=256, num_rotations=4, board_width=10):
        super(StateModel, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_rotations = num_rotations
        self.board_width = board_width
        # MLP encoder
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Predict placement parameters
        self.rot_out = nn.Linear(hidden_dim, num_rotations)
        self.x_out = nn.Linear(hidden_dim, board_width)
        # Predict vertical landing row (y-position)
        self.y_out = nn.Linear(hidden_dim, board_width)  # board height logits

    def forward(self, state):
        """
        Args:
            state: Tensor of shape (batch_size, state_dim)
        Returns:
            rot_logits: (batch_size, num_rotations), x_logits: (batch_size, board_width),
            y_logits: (batch_size, board_width)  # vertical position logits
        """
        h = self.fc(state)
        rot_logits = self.rot_out(h)
        x_logits = self.x_out(h)
        y_logits = self.y_out(h)
        return rot_logits, x_logits, y_logits
