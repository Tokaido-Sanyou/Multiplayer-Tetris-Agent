import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    """
    Prioritized Experience Replay Buffer for Tetris DQN
    
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
            state: Dictionary containing:
                - grid: 20x10 matrix of piece colors
                - current_piece: 4x4 matrix of current piece
                - next_piece: 4x4 matrix of next piece
                - hold_piece: 4x4 matrix of hold piece
            action: Integer (0-6) representing the action taken
            reward: Float reward value
            next_state: Dictionary with same structure as state
            done: Boolean indicating episode end
            info: Dictionary containing:
                - lines_cleared: Number of lines cleared
                - score: Current game score
                - level: Current game level
                - episode_steps: Number of steps taken
        """
        # Convert state dictionaries to tensors
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
        Convert state dictionary to tensor
        State structure defined in tetris_env.py:
        - Grid: 20x10 matrix (200 values)
        - Next piece: scalar shape ID (0-7)
        - Hold piece: scalar shape ID (0-7)
        Total: 202 values
        """
        grid = torch.FloatTensor(state['grid'].flatten())
        # Encode next and hold pieces as scalars
        next_piece = torch.FloatTensor([state['next_piece']])
        hold_piece = torch.FloatTensor([state['hold_piece']])
        return torch.cat([grid, next_piece, hold_piece])
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions using prioritized sampling
        Args:
            batch_size: Number of transitions to sample
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, info, indices, weights)
            where:
            - states: Tensor of shape (batch_size, 248)
            - actions: Tensor of shape (batch_size,)
            - rewards: Tensor of shape (batch_size,)
            - next_states: Tensor of shape (batch_size, 248)
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