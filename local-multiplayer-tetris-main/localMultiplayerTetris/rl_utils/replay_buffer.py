import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    """
    Prioritized Experience Replay Buffer for Tetris DQN
    
    State Structure (from tetris_env.py):
    - Grid: 20x10 matrix (0 for empty, 1 for locked, 2 for current piece)
    - Next piece: scalar ID (1-7, 0 if none)
    - Hold piece: scalar ID (1-7, 0 if none)
    - Current piece: shape ID, rotation, x and y coordinates
    - Can hold: binary flag
    Total input dimension: 207
    
    Action Space (from tetris_env.py):
    - 0: Move Left
    - 1: Move Right
    - 2: Move Down
    - 3: Rotate Clockwise
    - 4: Rotate Counter-clockwise
    - 5: Hard Drop
    - 6: Hold Piece
    - 7: No-op
    
    Related Files:
    - tetris_env.py: Defines action space and state structure
    - action_handler.py: Implements action mechanics
    - game.py: Contains game state and piece movement logic
    """
    def __init__(self, capacity):
        """
        Initialize replay buffer with given capacity
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_increment = 0.001
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done, info=None):
        """
        Add a transition to the buffer
        Args:
            state: Either a dictionary containing state components or a preprocessed array
            action: Integer (0-40) representing the action taken
            reward: Float reward value
            next_state: Same format as state
            done: Boolean indicating episode end
            info: Dictionary containing additional information
        """
        # Convert states to tensors
        state_tensor = self._state_to_tensor(state)
        next_state_tensor = self._state_to_tensor(next_state)
        
        # Extract only necessary info
        if info is not None:
            info = {
                'lines_cleared': info.get('lines_cleared', 0),
                'score': info.get('score', 0),
                'level': info.get('level', 1),
                'episode_steps': info.get('episode_steps', 0)
            }
        
        # Store transition with max priority
        self.buffer.append((state_tensor, action, reward, next_state_tensor, done, info))
        self.priorities.append(self.max_priority)
    
    def _state_to_tensor(self, state):
        """
        Convert state to tensor
        Args:
            state: Either a dictionary containing state components or a preprocessed array
        Returns:
            Tensor of shape (207,) detached from computation graph
        """
        if isinstance(state, dict):
            # Convert dictionary state to tensor
            grid = torch.FloatTensor(state['grid'].astype(np.float32).flatten()).detach()
            next_piece = torch.FloatTensor([float(state.get('next_piece', 0))]).detach()
            hold_piece = torch.FloatTensor([float(state.get('hold_piece', 0))]).detach()
            curr_shape = torch.FloatTensor([float(state.get('current_shape', 0))]).detach()
            curr_rot = torch.FloatTensor([float(state.get('current_rotation', 0))]).detach()
            curr_x = torch.FloatTensor([float(state.get('current_x', 0))]).detach()
            curr_y = torch.FloatTensor([float(state.get('current_y', 0))]).detach()
            can_hold = torch.FloatTensor([float(state.get('can_hold', 1))]).detach()
            
            # Validate grid shape
            if grid.shape[0] != 200:
                raise ValueError(f"Grid should have 200 values when flattened, got {grid.shape[0]}")
            
            # Concatenate and validate final shape
            state_tensor = torch.cat([grid, next_piece, hold_piece, curr_shape, curr_rot, curr_x, curr_y, can_hold])
            if state_tensor.shape[0] != 207:
                raise ValueError(f"State tensor should have 207 values, got {state_tensor.shape[0]}")
            return state_tensor
        else:
            # State is already preprocessed array
            if not isinstance(state, (np.ndarray, torch.Tensor)):
                raise ValueError(f"Expected numpy array or torch tensor, got {type(state)}")
            
            # Convert to float32 numpy array first if needed
            if isinstance(state, np.ndarray):
                state = state.astype(np.float32)
            
            # Convert to tensor
            state = torch.FloatTensor(state).detach()
            
            # Ensure state has correct shape
            if state.shape[0] != 207:
                raise ValueError(f"State array should have 207 values, got {state.shape[0]}")
            
            return state.float()
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions using prioritized sampling
        Args:
            batch_size: Number of transitions to sample
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, info, indices, weights)
            where:
            - states: Tensor of shape (batch_size, 207)
            - actions: Tensor of shape (batch_size,)
            - rewards: Tensor of shape (batch_size,)
            - next_states: Tensor of shape (batch_size, 207)
            - dones: Tensor of shape (batch_size,)
            - info: List of info dictionaries
            - indices: Array of sampled indices
            - weights: Tensor of importance sampling weights
        """
        if len(self.buffer) < batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get transitions
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones, info = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, info, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions
        Args:
            indices: Indices of transitions to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)