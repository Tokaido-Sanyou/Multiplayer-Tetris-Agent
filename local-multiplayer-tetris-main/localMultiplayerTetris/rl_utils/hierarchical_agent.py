"""
HierarchicalAgent: two-level planner using learned StateModel
High-level: predict target (rot, x, y)
Low-level: emit primitive actions to reach target
"""
import torch

class HierarchicalAgent:
    def __init__(self, state_model, device=None):
        """
        Args:
            state_model: StateModel instance
            device: torch device
        """
        self.model = state_model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.current_target = None  # (rot, x, y)

    def select_action(self, state_vec):
        """
        Given flattened state_206 vector, returns an action (0-7)
        until the predicted landing (rot,x,y) is reached or fails.
        """
        # state_vec: numpy array or list
        state = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        # Predict target if none
        if self.current_target is None:
            with torch.no_grad():
                rot_logits, x_logits, y_logits = self.model(state)
            rot = int(rot_logits.argmax(dim=1).item())
            x = int(x_logits.argmax(dim=1).item())
            y = int(y_logits.argmax(dim=1).item())
            self.current_target = (rot, x, y)

        # Extract current piece metadata from state_vec
        cur_rot = int(state_vec[201])
        cur_x = int(state_vec[202])
        cur_y = int(state_vec[203])
        target_rot, target_x, target_y = self.current_target
        # Primitive actions:
        # 0: left, 1: right, 2: down, 3: rotate cw, 4: rotate ccw, 5: hard drop, 6: hold, 7: no-op
        # First handle rotation
        if cur_rot != target_rot:
            return 3  # rotate clockwise until matching (could also rotate ccw)
        # Then horizontal
        if cur_x < target_x:
            return 1  # move right
        if cur_x > target_x:
            return 0  # move left
        # Then vertical: if not yet at target_y
        if cur_y < target_y:
            return 2  # soft drop
        # If at target position, lock with hard drop
        # Reset target afterwards
        self.current_target = None
        return 5

    def reset(self):
        """Clear any existing goal between episodes"""
        self.current_target = None
