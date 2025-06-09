"""
Redesigned Movement Agent Implementation
DQN agent for movement actions in hierarchical Tetris system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
from collections import deque

from .base_agent import BaseAgent

class MovementNetwork(nn.Module):
    """Neural network for movement agent"""
    
    def __init__(self, input_dim=1012, num_actions=8, hidden_dim=512):
        super().__init__()
        
        # Process combined state: board(200) + current(6) + next(6) + locked_q(800) = 1012
        self.fc1 = nn.Linear(input_dim, hidden_dim)      # 1012 → 512
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)  # 512 → 256
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4) # 256 → 128
        self.fc4 = nn.Linear(hidden_dim//4, num_actions) # 128 → 8
        
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim//2)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class RedesignedMovementAgent(BaseAgent):
    """
    Movement Agent for hierarchical DQN system
    Takes combined state (board + pieces + locked Q-values) and outputs movement actions
    """
    
    def __init__(self,
                 input_dim: int = 1012,
                 num_actions: int = 8,
                 device: str = 'cuda',
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 50000,
                 buffer_size: int = 100000,
                 batch_size: int = 32,
                 target_update_freq: int = 1000,
                 reward_mode: str = 'standard'):
        
        # Initialize base agent
        super().__init__(action_space_size=num_actions, 
                        observation_space_shape=(input_dim,), 
                        device=device)
        
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.reward_mode = reward_mode
        
        # Build networks
        self.q_network = MovementNetwork(input_dim, num_actions).to(self.device)
        self.target_network = MovementNetwork(input_dim, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=buffer_size)
        self.target_update_counter = 0
        
        print(f"   Movement Agent: {input_dim} → {num_actions}")
        print(f"   Parameters: {self.get_parameter_count():,}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select movement action based on combined state"""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        
        assert state.shape[0] == self.input_dim, f"Expected {self.input_dim} dims, got {state.shape[0]}"
        
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        
        # Get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """Update the movement network"""
        # Store experience
        self.store_experience(state, action, reward, next_state, done)
        
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0}
        
        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        self.q_network.train()
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (Double DQN)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, 
                             self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay)
        
        return {'loss': loss.item()}
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.q_network.parameters())
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for given state"""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().numpy().flatten()
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'target_update_counter': self.target_update_counter
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.target_update_counter = checkpoint['target_update_counter'] 