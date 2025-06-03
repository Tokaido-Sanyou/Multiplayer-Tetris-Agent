import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    """
    Prioritized Experience Replay Buffer for Tetris DQN with CUDA Support
    ENHANCEMENT: Optimized for CUDA when available for better performance
    
    State Structure (from tetris_env.py):
    - Grid: 20x10 matrix (0 for empty, 1-7 for different piece colors)
    - Current piece: 4x4 matrix (0 for empty, 1 for filled)
    - Next piece: 4x4 matrix (0 for empty, 1 for filled)
    - Hold piece: 4x4 matrix (0 for empty, 1 for filled)
    
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
    def __init__(self, capacity, device=None):
        """
        Initialize replay buffer with given capacity and device
        Args:
            capacity: Maximum number of transitions to store
            device: PyTorch device (cuda/cpu) for tensor operations
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_increment = 0.001
        self.max_priority = 1.0
        
        # CUDA optimization: Set device for all tensor operations
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"📱 ReplayBuffer initialized on device: {self.device}")
    
    def push(self, state, action, reward, next_state, done, info=None):
        """
        Add a transition to the buffer
        Args:
            state: Dictionary containing:
                - grid: 20x10 matrix of pieces, 1 for locked piece and 2 for current piece
                - next piece (0-7)
                - hold piece (0-7)
            action: One-hot vector (8-dimensional) representing the action taken
            reward: Float reward value
            next_state: Dictionary with same structure as state
            done: Boolean indicating episode end
            info: Dictionary containing:
                - lines_cleared: Number of lines cleared
                - score: Current game score
                - level: Current game level
                - episode_steps: Number of steps taken
        """
        # Convert state dictionaries to tensors ON CORRECT DEVICE
        state_tensor = self._state_to_tensor(state)
        next_state_tensor = self._state_to_tensor(next_state)
        
        # Convert one-hot action to scalar for efficient storage
        action_scalar = np.argmax(action) if hasattr(action, '__len__') else action
        
        # Extract only necessary info
        if info is not None:
            info = {
                'lines_cleared': info.get('lines_cleared', 0),
                'score': info.get('score', 0),
                'level': info.get('level', 1),
                'episode_steps': info.get('episode_steps', 0)
            }
        
        # Store transition with max priority
        self.buffer.append((state_tensor, action_scalar, reward, next_state_tensor, done, info))
        self.priorities.append(self.max_priority)
    
    def _state_to_tensor(self, state):
        """
        Convert state dictionary to tensor ON CORRECT DEVICE
        ENHANCEMENT: Creates tensors directly on CUDA for optimal performance
        NEW: Simplified state structure (410 dimensions):
        - current_piece_grid: 20x10 (200 values)  
        - empty_grid: 20x10 (200 values)
        - next_piece: 7 values (one-hot)
        - metadata: 3 values (rotation, x, y)
        Total: 410 values (reduced from 1817)
        """
        # Flatten grid components and create on correct device
        current_piece_flat = torch.tensor(state['current_piece_grid'].flatten(), dtype=torch.float32, device=self.device)
        empty_grid_flat = torch.tensor(state['empty_grid'].flatten(), dtype=torch.float32, device=self.device)
        
        # Get one-hot encoding and metadata on correct device
        next_piece = torch.tensor(state['next_piece'], dtype=torch.float32, device=self.device)
        metadata = torch.tensor([
            state['current_rotation'],
            state['current_x'],
            state['current_y']
        ], dtype=torch.float32, device=self.device)
        
        return torch.cat([
            current_piece_flat,  # 200 values
            empty_grid_flat,     # 200 values
            next_piece,          # 7 values  
            metadata             # 3 values
        ])  # Total: 410 values
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions using prioritized sampling
        ENHANCEMENT: Returns tensors already on correct device
        Args:
            batch_size: Number of transitions to sample
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, info, indices, weights)
            where:
            - states: Tensor of shape (batch_size, 410) ON DEVICE
            - actions: Tensor of shape (batch_size,) ON DEVICE
            - rewards: Tensor of shape (batch_size,) ON DEVICE
            - next_states: Tensor of shape (batch_size, 410) ON DEVICE
            - dones: Tensor of shape (batch_size,) ON DEVICE
            - info: List of info dictionaries
            - indices: Array of sampled indices
            - weights: Tensor of importance sampling weights ON DEVICE
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
        
        # Convert to tensors ON CORRECT DEVICE
        states = torch.stack(states)  # Already on device from _state_to_tensor
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states)  # Already on device from _state_to_tensor
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
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