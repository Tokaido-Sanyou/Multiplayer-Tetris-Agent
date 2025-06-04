import numpy as np
import torch
from typing import List, Dict, Any, Optional
from ..tetris_env import TetrisEnv
from collections import defaultdict

class Path:
    """Container for trajectory data"""
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.length = len(states)

class TrajectorySampler:
    """
    Samples trajectories from the Tetris environment using a given policy
    Compatible with both DQN and Actor-Critic policies
    """
    def __init__(self, 
                 env: TetrisEnv,
                 max_path_length: int = 5000,
                 render: bool = False,
                 render_delay: Optional[float] = None):
        """
        Initialize trajectory sampler
        
        Args:
            env: Tetris environment instance
            max_path_length: Maximum steps per trajectory
            render: Whether to render the environment
            render_delay: Optional delay between renders (seconds)
        """
        self.env = env
        self.max_path_length = max_path_length
        self.render = render
        self.render_delay = render_delay
        
    def sample_trajectory(self, policy, deterministic: bool = False) -> Path:
        """
        Sample a single trajectory using the given policy
        
        Args:
            policy: Policy object with select_action method
            deterministic: Whether to sample deterministically
            
        Returns:
            Path object containing trajectory data
        """
        state = self.env.reset()[0]  # Reset returns (obs, info) in gym 0.26.0
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        steps = 0
        
        while True:
            if self.render:
                self.env.render()
                if self.render_delay:
                    import time
                    time.sleep(self.render_delay)
            
            # Store current state
            states.append(state)
            
            # Get action from policy
            action = policy.select_action(state, deterministic=deterministic)
            actions.append(action)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store step data
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            steps += 1
            
            if done or steps >= self.max_path_length:
                break
                
        return Path(states, actions, rewards, next_states, dones)
    
    def sample_trajectories(self, 
                          policy,
                          num_trajectories: int,
                          deterministic: bool = False) -> List[Path]:
        """
        Sample multiple trajectories
        
        Args:
            policy: Policy object with select_action method
            num_trajectories: Number of trajectories to sample
            deterministic: Whether to sample deterministically
            
        Returns:
            List of Path objects
        """
        trajectories = []
        for _ in range(num_trajectories):
            traj = self.sample_trajectory(policy, deterministic)
            trajectories.append(traj)
        return trajectories
    
    def convert_trajectories_to_transitions(self, trajectories: List[Path]) -> Dict[str, np.ndarray]:
        """
        Convert trajectories to transition format for training
        
        Args:
            trajectories: List of Path objects
            
        Returns:
            Dictionary with keys 'states', 'actions', 'rewards', 'next_states', 'dones'
            containing numpy arrays of transitions
        """
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        
        for traj in trajectories:
            all_states.extend(traj.states)
            all_actions.extend(traj.actions)
            all_rewards.extend(traj.rewards)
            all_next_states.extend(traj.next_states)
            all_dones.extend(traj.dones)
            
        return {
            'states': np.array(all_states),
            'actions': np.array(all_actions),
            'rewards': np.array(all_rewards),
            'next_states': np.array(all_next_states),
            'dones': np.array(all_dones)
        } 