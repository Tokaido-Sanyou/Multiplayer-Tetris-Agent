import pickle
import os
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict

# Import from promoted modules
from dqn_adapter import board_props, enumerate_next_states
from tetris_env import TetrisEnv

class ExpertTrajectoryLoader:
    """
    Loads and processes expert trajectories for AIRL training.
    
    This class handles:
    1. Loading expert trajectories from pickle files
    2. Converting DQN actions to TetrisEnv actions using dqn_adapter
    3. State feature extraction and normalization
    4. Creating batches for training
    """
    
    def __init__(self, 
                 trajectory_dir: str = "expert_trajectories",
                 max_trajectories: int = None,
                 min_episode_length: int = 10,
                 max_hold_percentage: float = 20.0,
                 state_feature_extractor=None):
        """
        Initialize expert trajectory loader.
        
        Args:
            trajectory_dir: Directory containing expert trajectory pickle files
            max_trajectories: Maximum number of trajectories to load
            min_episode_length: Minimum episode length to include
            max_hold_percentage: Maximum percentage of HOLD actions to allow
            state_feature_extractor: Function to extract features from state observations
        """
        self.trajectory_dir = trajectory_dir
        self.max_trajectories = max_trajectories
        self.min_episode_length = min_episode_length
        self.max_hold_percentage = max_hold_percentage
        self.state_feature_extractor = state_feature_extractor or self._default_feature_extractor
        
        self.trajectories = []
        self.transitions = []
        
        self.logger = logging.getLogger(__name__)
        
    def _default_feature_extractor(self, observation: Dict) -> np.ndarray:
        """
        Default feature extraction from TetrisEnv observation.
        
        Converts the complex observation dict to a flat feature vector.
        """
        grid = observation['grid'].flatten()  # 20*10 = 200 features
        next_piece = np.array([observation['next_piece']])  # 1 feature
        hold_piece = np.array([observation['hold_piece']])  # 1 feature
        current_shape = np.array([observation['current_shape']])  # 1 feature
        current_rotation = np.array([observation['current_rotation']])  # 1 feature
        current_x = np.array([observation['current_x']])  # 1 feature
        current_y = np.array([observation['current_y']])  # 1 feature
        can_hold = np.array([observation['can_hold']])  # 1 feature
        
        # Concatenate all features: 200 + 7 = 207 total features
        features = np.concatenate([
            grid, next_piece, hold_piece, current_shape, 
            current_rotation, current_x, current_y, can_hold
        ]).astype(np.float32)
        
        return features
    
    def _action_to_onehot(self, action: int, num_actions: int = 41) -> np.ndarray:
        """Convert action index to one-hot encoding."""
        onehot = np.zeros(num_actions, dtype=np.float32)
        if 0 <= action < num_actions:
            onehot[action] = 1.0
        return onehot
    
    def _is_valid_trajectory(self, trajectory_data: Dict) -> bool:
        """
        Check if trajectory meets quality criteria.
        
        Args:
            trajectory_data: Dictionary containing trajectory steps
            
        Returns:
            True if trajectory is valid for training
        """
        steps = trajectory_data.get('steps', [])
        
        # Check minimum length
        if len(steps) < self.min_episode_length:
            return False
        
        # Check HOLD action percentage
        actions = [step.get('action', -1) for step in steps]
        hold_count = sum(1 for a in actions if a == 40)  # Action 40 is HOLD
        hold_percentage = (hold_count / len(actions)) * 100 if actions else 0
        
        if hold_percentage > self.max_hold_percentage:
            return False
        
        return True
    
    def load_trajectories(self) -> int:
        """
        Load expert trajectories from pickle files.
        
        Returns:
            Number of valid trajectories loaded
        """
        if not os.path.exists(self.trajectory_dir):
            self.logger.error(f"Trajectory directory not found: {self.trajectory_dir}")
            return 0
        
        trajectory_files = sorted([
            f for f in os.listdir(self.trajectory_dir) 
            if f.endswith('.pkl')
        ])
        
        if self.max_trajectories:
            trajectory_files = trajectory_files[:self.max_trajectories]
        
        valid_count = 0
        total_transitions = 0
        
        for filename in trajectory_files:
            filepath = os.path.join(self.trajectory_dir, filename)
            
            try:
                with open(filepath, 'rb') as f:
                    trajectory_data = pickle.load(f)
                
                if not self._is_valid_trajectory(trajectory_data):
                    self.logger.debug(f"Skipping invalid trajectory: {filename}")
                    continue
                
                # Process trajectory steps
                steps = trajectory_data['steps']
                processed_transitions = []
                
                for i, step in enumerate(steps):
                    # Extract state features
                    state_obs = step.get('state', {})
                    next_state_obs = step.get('next_state', {})
                    
                    # Skip if observations are incomplete
                    if not state_obs or not next_state_obs:
                        continue
                    
                    state_features = self.state_feature_extractor(state_obs)
                    next_state_features = self.state_feature_extractor(next_state_obs)
                    
                    # Convert action to one-hot
                    action = step.get('action', -1)
                    action_onehot = self._action_to_onehot(action)
                    
                    transition = {
                        'state': state_features,
                        'action': action,
                        'action_onehot': action_onehot,
                        'reward': float(step.get('reward', 0.0)),
                        'next_state': next_state_features,
                        'done': bool(step.get('done', False))
                    }
                    
                    processed_transitions.append(transition)
                
                if processed_transitions:
                    self.trajectories.append({
                        'filename': filename,
                        'transitions': processed_transitions
                    })
                    self.transitions.extend(processed_transitions)
                    total_transitions += len(processed_transitions)
                    valid_count += 1
                    
                    self.logger.info(f"Loaded trajectory {filename}: {len(processed_transitions)} transitions")
                
            except Exception as e:
                self.logger.error(f"Error loading trajectory {filename}: {e}")
        
        self.logger.info(f"Loaded {valid_count} valid trajectories with {total_transitions} total transitions")
        return valid_count
    
    def get_batch(self, batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Sample a batch of expert transitions.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to put tensors on
            
        Returns:
            Dictionary of batched tensors
        """
        if not self.transitions:
            raise ValueError("No expert transitions loaded. Call load_trajectories() first.")
        
        # Sample random transitions
        indices = np.random.choice(len(self.transitions), batch_size, replace=True)
        batch_transitions = [self.transitions[i] for i in indices]
        
        # Stack into tensors
        states = torch.FloatTensor([t['state'] for t in batch_transitions]).to(device)
        actions_onehot = torch.FloatTensor([t['action_onehot'] for t in batch_transitions]).to(device)
        rewards = torch.FloatTensor([t['reward'] for t in batch_transitions]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor([t['next_state'] for t in batch_transitions]).to(device)
        dones = torch.FloatTensor([t['done'] for t in batch_transitions]).unsqueeze(1).to(device)
        
        return {
            'states': states,
            'actions': actions_onehot,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def get_state_action_pairs(self, num_pairs: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get state-action pairs for discriminator training.
        
        Args:
            num_pairs: Number of state-action pairs to return
            device: Device to put tensors on
            
        Returns:
            Tuple of (states, actions) tensors
        """
        if not self.transitions:
            raise ValueError("No expert transitions loaded. Call load_trajectories() first.")
        
        # Sample random transitions
        indices = np.random.choice(len(self.transitions), num_pairs, replace=True)
        
        states = torch.FloatTensor([self.transitions[i]['state'] for i in indices]).to(device)
        actions = torch.FloatTensor([self.transitions[i]['action_onehot'] for i in indices]).to(device)
        
        return states, actions
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about loaded expert data.
        
        Returns:
            Dictionary with statistics
        """
        if not self.transitions:
            return {}
        
        rewards = [t['reward'] for t in self.transitions]
        actions = [t['action'] for t in self.transitions]
        episode_lengths = [len(traj['transitions']) for traj in self.trajectories]
        
        stats = {
            'num_trajectories': len(self.trajectories),
            'num_transitions': len(self.transitions),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'action_distribution': dict(zip(*np.unique(actions, return_counts=True)))
        }
        
        return stats
    
    def normalize_states(self, mean: Optional[np.ndarray] = None, 
                        std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize state features and return normalization parameters.
        
        Args:
            mean: Pre-computed mean (if None, compute from data)
            std: Pre-computed std (if None, compute from data)
            
        Returns:
            Tuple of (mean, std) used for normalization
        """
        if not self.transitions:
            raise ValueError("No transitions loaded")
        
        states = np.array([t['state'] for t in self.transitions])
        
        if mean is None:
            mean = np.mean(states, axis=0)
        if std is None:
            std = np.std(states, axis=0)
            std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
        
        # Apply normalization
        for transition in self.transitions:
            transition['state'] = (transition['state'] - mean) / std
            transition['next_state'] = (transition['next_state'] - mean) / std
        
        self.logger.info("Applied state normalization to expert transitions")
        return mean, std 