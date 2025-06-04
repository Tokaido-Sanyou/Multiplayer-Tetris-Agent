import os
import pickle
import numpy as np
import torch
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any

def convert_action_41_to_8(action: int) -> int:
    """
    Convert 41-action space to 8-action space
    
    41-action space:
    - 0-39: placement index = rotation*10 + column  
    - 40: hold piece
    
    8-action space:
    - 0-6: directional/rotational actions
    - 7: hold piece
    """
    if action == 40:
        return 7  # Hold action
    else:
        # Map placement actions to 0-6 based on some strategy
        # Simple mapping: use modulo to compress 40 placements to 7 actions
        return action % 7

class TrajectoryCollector:
    """
    Collects and saves agent trajectories for AIRL training.
    Trajectories are stored as sequences of (state, action, reward, done) tuples.
    Supports both 8-action and 41-action spaces with automatic conversion.
    """
    
    def __init__(self, save_dir: str = "trajectories", max_traj_length: int = 5000, 
                 convert_actions: bool = False, target_action_space: int = 41):
        """
        Initialize trajectory collector
        Args:
            save_dir: Directory to save trajectory files
            max_traj_length: Maximum length of a single trajectory
            convert_actions: Whether to convert actions from 41 to 8 space
            target_action_space: Target action space size (8 or 41)
        """
        self.save_dir = save_dir
        self.max_traj_length = max_traj_length
        self.convert_actions = convert_actions
        self.target_action_space = target_action_space
        os.makedirs(save_dir, exist_ok=True)
        
        # Current trajectory being collected
        self.current_trajectory = []
        self.episode_count = 0
        
        # Statistics
        self.trajectory_stats = {
            'total_trajectories': 0,
            'avg_length': 0,
            'avg_reward': 0,
            'total_steps': 0,
            'action_space': target_action_space
        }
    
    def add_step(self, state: Dict, action: int, reward: float, done: bool, info: Dict = None):
        """
        Add a step to the current trajectory
        Args:
            state: Environment state dictionary
            action: Action taken (will be converted if needed)
            reward: Reward received
            done: Whether episode ended
            info: Additional info dictionary
        """
        # Convert action if needed
        if self.convert_actions and self.target_action_space == 8:
            if action > 7:  # Assume it's from 41-action space
                action = convert_action_41_to_8(action)
        
        # Validate action is in correct range
        if not (0 <= action < self.target_action_space):
            print(f"Warning: Action {action} out of range for {self.target_action_space}-action space")
            action = min(max(0, action), self.target_action_space - 1)
        
        # Convert state to serializable format
        serialized_state = self._serialize_state(state)
        
        step = {
            'state': serialized_state,
            'action': action,
            'reward': reward,
            'done': done,
            'info': info or {}
        }
        
        self.current_trajectory.append(step)
        
        # Save trajectory if episode ends or max length reached
        if done or len(self.current_trajectory) >= self.max_traj_length:
            self._save_current_trajectory()
    
    def _serialize_state(self, state: Dict) -> Dict:
        """Convert state dictionary to serializable format"""
        serialized = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                serialized[key] = value.cpu().numpy().tolist()
            else:
                serialized[key] = value
        return serialized
    
    def _save_current_trajectory(self):
        """Save the current trajectory to disk"""
        if not self.current_trajectory:
            return
        
        # Calculate trajectory statistics
        total_reward = sum(step['reward'] for step in self.current_trajectory)
        trajectory_length = len(self.current_trajectory)
        
        # Validate actions in trajectory
        actions = [step['action'] for step in self.current_trajectory]
        action_range = f"{min(actions)}-{max(actions)}"
        
        # Create trajectory data structure
        trajectory_data = {
            'episode_id': self.episode_count,
            'steps': self.current_trajectory,
            'total_reward': total_reward,
            'length': trajectory_length,
            'action_space': self.target_action_space,
            'action_range': action_range,
            'timestamp': np.datetime64('now')
        }
        
        # Save to file
        filename = f"trajectory_ep{self.episode_count:06d}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(trajectory_data, f)
        
        # Update statistics
        self._update_stats(total_reward, trajectory_length)
        
        # Reset for next trajectory
        self.current_trajectory = []
        self.episode_count += 1
        
        print(f"Saved trajectory {self.episode_count} (length: {trajectory_length}, reward: {total_reward:.2f}, actions: {action_range})")
    
    def _update_stats(self, reward: float, length: int):
        """Update trajectory statistics"""
        self.trajectory_stats['total_trajectories'] += 1
        self.trajectory_stats['total_steps'] += length
        
        # Running average
        n = self.trajectory_stats['total_trajectories']
        self.trajectory_stats['avg_reward'] = (
            (self.trajectory_stats['avg_reward'] * (n-1) + reward) / n
        )
        self.trajectory_stats['avg_length'] = (
            (self.trajectory_stats['avg_length'] * (n-1) + length) / n
        )
    
    def load_trajectories(self, num_trajectories: int = None) -> List[Dict]:
        """
        Load trajectories from disk
        Args:
            num_trajectories: Number of trajectories to load (None for all)
        Returns:
            List of trajectory dictionaries
        """
        trajectory_files = sorted([
            f for f in os.listdir(self.save_dir) 
            if f.startswith('trajectory_') and f.endswith('.pkl')
        ])
        
        if num_trajectories:
            trajectory_files = trajectory_files[:num_trajectories]
        
        trajectories = []
        for filename in trajectory_files:
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath, 'rb') as f:
                traj = pickle.load(f)
                # Convert old format if needed
                if 'action_space' not in traj:
                    traj['action_space'] = 41  # Assume old format
                trajectories.append(traj)
        
        return trajectories
    
    def get_stats(self) -> Dict:
        """Get trajectory collection statistics"""
        return self.trajectory_stats.copy()
    
    def convert_to_airl_format(self, trajectories: List[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert trajectories to AIRL training format
        Args:
            trajectories: List of trajectory dicts (loads from disk if None)
        Returns:
            Tuple of (states, actions, rewards) as numpy arrays
        """
        if trajectories is None:
            trajectories = self.load_trajectories()
        
        all_states = []
        all_actions = []
        all_rewards = []
        
        for traj in trajectories:
            for step in traj['steps']:
                # Convert state back to numpy
                state_dict = step['state']
                state_vector = self._state_dict_to_vector(state_dict)
                
                action = step['action']
                
                # Convert action if needed
                if self.convert_actions and self.target_action_space == 8:
                    # Check if action is from old 41-space
                    if action > 7:
                        action = convert_action_41_to_8(action)
                
                all_states.append(state_vector)
                all_actions.append(action)
                all_rewards.append(step['reward'])
        
        return np.array(all_states), np.array(all_actions), np.array(all_rewards)
    
    def _state_dict_to_vector(self, state_dict: Dict) -> np.ndarray:
        """Convert state dictionary to flat vector (same as preprocess_state)"""
        grid = np.array(state_dict['grid']).flatten()  # 200 elements
        
        # Scalar features: next_piece, hold_piece, current_shape, rotation, x, y, can_hold
        scalars = np.array([
            state_dict.get('next_piece', 0),
            state_dict.get('hold_piece', 0),
            state_dict.get('current_shape', 0),
            state_dict.get('current_rotation', 0),
            state_dict.get('current_x', 0),
            state_dict.get('current_y', 0),
            state_dict.get('can_hold', 1)
        ])
        
        return np.concatenate([grid, scalars])  # 207 elements total

class ExpertTrajectoryCollector(TrajectoryCollector):
    """Extended trajectory collector that captures expert demonstrations"""
    
    def __init__(self, save_dir: str = "expert_trajectories", max_traj_length: int = 5000,
                 convert_actions: bool = False, target_action_space: int = 41):
        super().__init__(save_dir, max_traj_length, convert_actions, target_action_space)
        self.expert_metadata = {
            'collection_method': 'checkpoint_replay',
            'model_path': None,
            'performance_metrics': {},
            'action_space': target_action_space,
            'action_conversion': convert_actions
        }
    
    def set_expert_metadata(self, model_path: str, performance_metrics: Dict):
        """Set metadata about the expert that generated trajectories"""
        self.expert_metadata['model_path'] = model_path
        self.expert_metadata['performance_metrics'] = performance_metrics
    
    def save_expert_dataset(self, min_reward_threshold: float = None):
        """
        Save all trajectories as a single expert dataset
        Args:
            min_reward_threshold: Only include trajectories above this reward
        """
        trajectories = self.load_trajectories()
        
        if min_reward_threshold is not None:
            trajectories = [t for t in trajectories if t['total_reward'] >= min_reward_threshold]
        
        states, actions, rewards = self.convert_to_airl_format(trajectories)
        
        # Validate action space
        if len(actions) > 0:
            action_min, action_max = actions.min(), actions.max()
            print(f"Action range in dataset: {action_min}-{action_max}")
            if action_max >= self.target_action_space:
                print(f"Warning: Actions exceed target space {self.target_action_space}")
        
        expert_dataset = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'metadata': self.expert_metadata,
            'num_trajectories': len(trajectories),
            'total_steps': len(states),
            'avg_reward': np.mean([t['total_reward'] for t in trajectories]),
            'action_space': self.target_action_space,
            'timestamp': np.datetime64('now')
        }
        
        filepath = os.path.join(self.save_dir, 'expert_dataset.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(expert_dataset, f)
        
        print(f"Saved expert dataset: {len(trajectories)} trajectories, {len(states)} steps")
        print(f"Action space: {self.target_action_space}, range: {action_min}-{action_max}")
        return filepath 