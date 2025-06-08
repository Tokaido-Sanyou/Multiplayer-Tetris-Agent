"""
Replay Buffer for DREAM Algorithm

Stores experience sequences with consistent tuple-based observations.
"""

import torch
import numpy as np
import random
from typing import Dict, List, Any, Optional


class ReplayBuffer:
    """Experience replay buffer with consistent data format"""
    
    def __init__(self, 
                 capacity: int = 100000,
                 sequence_length: int = 50,
                 device: str = 'cuda'):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.device = device
        
        # Storage
        self.trajectories: List[Dict[str, List[Any]]] = []
        self.current_size = 0
        
        # Statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
    
    def add_trajectory(self, trajectory: Dict[str, List[Any]]) -> None:
        """Add a trajectory to the buffer"""
        # Remove oldest trajectory if at capacity
        if len(self.trajectories) >= self.capacity:
            self.trajectories.pop(0)
        
        # Store trajectory directly (no conversion)
        self.trajectories.append(trajectory)
        self.current_size = min(self.current_size + len(trajectory['rewards']), self.capacity)
        
        # Update statistics
        episode_reward = sum(trajectory['rewards'])
        episode_length = len(trajectory['rewards'])
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Keep only recent statistics
        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]
            self.episode_lengths = self.episode_lengths[-1000:]
    
    def sample_sequences(self, batch_size: int, sequence_length: Optional[int] = None) -> Dict[str, List[Any]]:
        """Sample sequences from the buffer"""
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        sequences = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        for _ in range(batch_size):
            # Sample a random trajectory
            trajectory = random.choice(self.trajectories)
            traj_length = len(trajectory['rewards'])
            
            if traj_length >= sequence_length:
                # Sample a random subsequence
                start_idx = random.randint(0, traj_length - sequence_length)
                end_idx = start_idx + sequence_length
            else:
                # Use entire trajectory and pad
                start_idx = 0
                end_idx = traj_length
                # For short trajectories, repeat the last step
                padding_needed = sequence_length - traj_length
            
            # Extract sequence
            seq_observations = trajectory['observations'][start_idx:end_idx]
            seq_actions = trajectory['actions'][start_idx:end_idx]
            seq_rewards = trajectory['rewards'][start_idx:end_idx]
            seq_dones = trajectory['dones'][start_idx:end_idx]
            
            # Pad if necessary
            if traj_length < sequence_length:
                padding_needed = sequence_length - traj_length
                # Repeat last values
                last_obs = seq_observations[-1]
                last_action = seq_actions[-1]
                last_reward = seq_rewards[-1]
                
                seq_observations.extend([last_obs] * padding_needed)
                seq_actions.extend([last_action] * padding_needed)
                seq_rewards.extend([0.0] * padding_needed)  # Zero reward for padding
                seq_dones.extend([True] * padding_needed)   # Mark as done
            
            # Add to batch
            sequences['observations'].append(seq_observations)
            sequences['actions'].append(seq_actions)
            sequences['rewards'].append(seq_rewards)
            sequences['dones'].append(seq_dones)
        
        return sequences
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics"""
        if not self.episode_rewards:
            return {
                'mean_episode_reward': 0.0,
                'std_episode_reward': 0.0,
                'mean_episode_length': 0.0,
                'num_episodes': 0,
                'buffer_size': 0
            }
        
        return {
            'mean_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'max_episode_reward': np.max(self.episode_rewards),
            'min_episode_reward': np.min(self.episode_rewards),
            'num_episodes': len(self.episode_rewards),
            'buffer_size': self.current_size
        }
    
    def __len__(self) -> int:
        """Return number of transitions in buffer"""
        return self.current_size
    
    def clear(self) -> None:
        """Clear the buffer"""
        self.trajectories.clear()
        self.current_size = 0
        self.episode_rewards.clear()
        self.episode_lengths.clear()


class ImaginationBuffer(ReplayBuffer):
    """Buffer for imagined trajectories"""
    
    def __init__(self, 
                 capacity: int = 50000,
                 sequence_length: int = 15,
                 device: str = 'cuda'):
        super().__init__(capacity, sequence_length, device)
        
        # Track imagination quality metrics
        self.world_model_losses = []
    
    def add_imagined_trajectory(self, 
                              trajectory: Dict[str, List[Any]], 
                              world_model_loss: float = 0.0) -> None:
        """Add an imagined trajectory with quality metrics"""
        self.add_trajectory(trajectory)
        self.world_model_losses.append(world_model_loss)
        
        # Keep only recent losses
        if len(self.world_model_losses) > 1000:
            self.world_model_losses = self.world_model_losses[-1000:] 