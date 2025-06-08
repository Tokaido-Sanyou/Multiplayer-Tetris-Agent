#!/usr/bin/env python3
"""
Actor-Locked Hierarchical System with Hindsight Experience Replay (HER)

Architecture:
1. Locked Model: Selects piece positions (current DQN)
2. Actor Model: Gets multiple trials per locked state, optimizes placement
3. Hindsight Relabelling: Rewards exact goal matching
4. Retry Mechanism: Configurable attempts per state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import copy

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent
from base_agent import BaseAgent

class MovementActorNetwork(nn.Module):
    """
    Movement Actor Network for executing piece movements
    
    Input: Board state + current piece position + target placement position
    Output: Movement actions (left, right, down, rotate, drop, etc.)
    """
    
    def __init__(self, input_dim: int = 212, movement_action_dim: int = 8):
        super(MovementActorNetwork, self).__init__()
        
        # Input: 206 (board) + 3 (current position) + 3 (target position) = 212
        # Output: 8 movement actions (left, right, down, rotate_cw, rotate_ccw, soft_drop, hard_drop, no_op)
        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)  # LayerNorm works with any batch size
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, movement_action_dim)  # 8 movement actions
        
        self.dropout = nn.Dropout(0.1)
        
        # Movement action mapping
        self.movement_actions = {
            0: "MOVE_LEFT",
            1: "MOVE_RIGHT", 
            2: "MOVE_DOWN",
            3: "ROTATE_CW",
            4: "ROTATE_CCW",
            5: "SOFT_DROP",
            6: "HARD_DROP",
            7: "NO_OP"
        }
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):  # FIXED: LayerNorm instead of BatchNorm1d
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.ln1(self.fc1(x)))  # FIXED: LayerNorm instead of BatchNorm1d
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))  # FIXED: LayerNorm instead of BatchNorm1d
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        action_probs = F.softmax(self.fc4(x), dim=-1)
        return action_probs

class HindsightExperienceBuffer:
    """
    Experience buffer with Hindsight Experience Replay (HER)
    
    Stores experiences and applies proper HER with random future goals:
    - Collects trajectory of locked model placements
    - For relabeling, uses random selection of future goals from trajectory
    - Trains movement model to reach various target placements
    """
    
    def __init__(self, capacity: int = 50000, her_ratio: float = 0.4):
        self.capacity = capacity
        self.her_ratio = her_ratio  # Fraction of experiences to relabel
        self.buffer = deque(maxlen=capacity)
        self.goal_trajectory = deque(maxlen=10000)  # Store goal trajectory for random sampling
        
    def store(self, experience: Dict[str, Any]):
        """Store experience"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch with proper HER using random future goals"""
        if len(self.buffer) < batch_size:
            return []
        
        # Sample experiences
        experiences = random.sample(self.buffer, batch_size)
        
        # Apply HER relabelling with random future goals
        her_count = int(batch_size * self.her_ratio)
        for i in range(her_count):
            exp = experiences[i]
            
            # FIXED HER: Use random future goal instead of achieved goal
            if len(self.goal_trajectory) > 0:
                # Randomly select a future goal from the trajectory
                random_future_goal = random.choice(list(self.goal_trajectory))
                
                # Create hindsight experience with random future goal
                hindsight_exp = copy.deepcopy(exp)
                hindsight_exp['desired_goal'] = random_future_goal
                hindsight_exp['reward'] = self._compute_hindsight_reward(
                    exp.get('achieved_goal', exp.get('desired_goal')), 
                    random_future_goal
                )
                experiences[i] = hindsight_exp
        
        return experiences
    
    def _compute_hindsight_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """Compute reward for hindsight relabelling"""
        # Exact goal matching reward
        if np.array_equal(achieved_goal, desired_goal):
            return 100.0  # High reward for exact match
        else:
            # Distance-based penalty
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return -distance * 10.0
    
    def __len__(self):
        return len(self.buffer)

class ActorLockedSystem(BaseAgent):
    """
    Hierarchical Actor-Locked System
    
    1. Locked Model: Provides initial piece placement suggestions
    2. Actor Model: Refines placements with multiple trials
    3. HER: Learns from achieved goals via hindsight relabelling
    """
    
    def __init__(self,
                 device: str = 'cuda',
                 actor_trials: int = 10,
                 actor_learning_rate: float = 0.0001,
                 locked_model_path: Optional[str] = None):
        
        super().__init__(action_space_size=800, observation_space_shape=(206,), device=device)
        
        self.actor_trials = actor_trials
        self.device = torch.device(device)
        
        # Initialize Locked Model (pre-trained DQN)
        self.locked_model = RedesignedLockedStateDQNAgent(device=device)
        if locked_model_path:
            self.locked_model.load_checkpoint(locked_model_path)
            print(f"Loaded locked model from: {locked_model_path}")
        
        # Initialize Movement Actor Model (8 movement actions)
        self.actor_network = MovementActorNetwork().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=actor_learning_rate)
        
        # Hindsight Experience Replay
        self.her_buffer = HindsightExperienceBuffer()
        
        # Training state
        self.training_mode = True
        self.episode_count = 0
        self.actor_success_rate = 0.0
        
        print(f"Actor-Locked System initialized:")
        print(f"   Device: {self.device}")
        print(f"   Actor trials per state: {self.actor_trials}")
        print(f"   Locked model parameters: {self.locked_model.get_parameter_count():,}")
        print(f"   Actor model parameters: {sum(p.numel() for p in self.actor_network.parameters()):,}")
    
    def _encode_goal(self, x: int, y: int, rotation: int) -> np.ndarray:
        """Encode goal as 3D vector"""
        return np.array([x / 9.0, y / 19.0, rotation / 3.0], dtype=np.float32)
    
    def _decode_goal(self, goal: np.ndarray) -> Tuple[int, int, int]:
        """Decode goal from 3D vector"""
        x = int(goal[0] * 9)
        y = int(goal[1] * 19)
        rotation = int(goal[2] * 3)
        return x, y, rotation
    
    def _create_actor_input(self, board_state: np.ndarray, locked_suggestion: np.ndarray, target_goal: np.ndarray) -> torch.Tensor:
        """Create input for actor network"""
        # Combine: board (206) + locked suggestion (3) + target goal (3) = 212
        actor_input = np.concatenate([board_state, locked_suggestion, target_goal])
        return torch.FloatTensor(actor_input).unsqueeze(0).to(self.device)
    
    def select_action(self, observation: np.ndarray, training: bool = True, env=None) -> int:
        """
        Option A: Sequential Movement Execution
        
        1. Locked model selects target position (x, y, rotation)
        2. Actor model generates movement sequence to reach target
        3. Execute movement sequence step by step
        4. Return final position action for piece placement
        """
        if not isinstance(observation, np.ndarray) or observation.shape[0] != 206:
            raise ValueError(f"Expected observation shape (206,), got {observation.shape if hasattr(observation, 'shape') else type(observation)}")
        
        # Step 1: Get locked model target position
        locked_action = self.locked_model.select_action(observation, training=False, env=env)
        locked_coords = self.locked_model.map_action_to_board(locked_action)
        target_position = self._encode_goal(*locked_coords)
        
        # Store target in goal trajectory for HER
        if training:
            self.her_buffer.goal_trajectory.append(target_position)
        
        # Step 2: In evaluation mode, use locked model directly
        if not training or not self.training_mode:
            return locked_action
        
        # Step 3: Actor generates movement sequence to reach target
        if env is not None:
            final_action = self._execute_movement_sequence(observation, target_position, env)
            return final_action
        else:
            # No environment available, fall back to locked model
            return locked_action
    
    def _evaluate_action(self, action: int, target_goal: np.ndarray, env, observation: np.ndarray) -> float:
        """
        Evaluate action by simulating its outcome
        Note: This is a simplified evaluation - in practice you'd want more sophisticated simulation
        """
        try:
            # Get action coordinates
            action_coords = self.locked_model.map_action_to_board(action)
            achieved_goal = self._encode_goal(*action_coords)
            
            # Exact goal matching reward
            goal_distance = np.linalg.norm(achieved_goal - target_goal)
            if goal_distance < 0.1:  # Close enough threshold
                return 100.0
            else:
                return -goal_distance * 50.0
                
        except Exception:
            return -100.0  # Invalid action penalty
    
    def _store_actor_experience(self, observation: np.ndarray, locked_suggestion: np.ndarray, trial_results: List[Dict]):
        """Store actor experiences in HER buffer"""
        for result in trial_results:
            action_coords = self.locked_model.map_action_to_board(result['action'])
            achieved_goal = self._encode_goal(*action_coords)
            
            experience = {
                'observation': observation,
                'locked_suggestion': locked_suggestion,
                'action': result['action'],
                'desired_goal': result['target_goal'],
                'achieved_goal': achieved_goal,
                'reward': result['reward'],
                'action_probs': result['action_probs']
            }
            
            self.her_buffer.store(experience)
    
    def train_actor(self, batch_size: int = 32) -> Dict[str, float]:
        """Train actor network using HER experiences"""
        if len(self.her_buffer) < batch_size:
            return {'actor_loss': 0.0}
        
        # Sample batch with HER
        experiences = self.her_buffer.sample(batch_size)
        if not experiences:
            return {'actor_loss': 0.0}
        
        # Prepare batch tensors
        observations = torch.FloatTensor([exp['observation'] for exp in experiences]).to(self.device)
        locked_suggestions = torch.FloatTensor([exp['locked_suggestion'] for exp in experiences]).to(self.device)
        desired_goals = torch.FloatTensor([exp['desired_goal'] for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences]).to(self.device)
        
        # Create actor inputs
        actor_inputs = torch.cat([observations, locked_suggestions, desired_goals], dim=1)
        
        # Forward pass
        action_probs = self.actor_network(actor_inputs)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Policy gradient loss (REINFORCE with baseline)
        baseline = rewards.mean()
        advantages = rewards - baseline
        loss = -(torch.log(selected_probs) * advantages).mean()
        
        # Backpropagation
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        return {
            'actor_loss': loss.item(),
            'actor_reward_mean': rewards.mean().item(),
            'actor_success_rate': (rewards > 0).float().mean().item()
        }
    
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> Dict[str, float]:
        """Update both locked and actor models"""
        # Update locked model (if training)
        locked_result = self.locked_model.update(state, action, reward, next_state, done)
        
        # Train actor model
        actor_result = self.train_actor()
        
        # Combine results
        result = {**locked_result, **actor_result}
        return result
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save both models"""
        # Save locked model
        locked_path = filepath.replace('.pt', '_locked.pt')
        self.locked_model.save_checkpoint(locked_path)
        
        # Save actor model
        actor_path = filepath.replace('.pt', '_actor.pt')
        torch.save({
            'actor_network_state_dict': self.actor_network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'actor_success_rate': self.actor_success_rate
        }, actor_path)
        
        print(f"Actor-Locked system saved: {locked_path}, {actor_path}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load both models"""
        # Load locked model
        locked_path = filepath.replace('.pt', '_locked.pt')
        if os.path.exists(locked_path):
            self.locked_model.load_checkpoint(locked_path)
        
        # Load actor model
        actor_path = filepath.replace('.pt', '_actor.pt')
        if os.path.exists(actor_path):
            checkpoint = torch.load(actor_path, map_location=self.device)
            self.actor_network.load_state_dict(checkpoint['actor_network_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.episode_count = checkpoint.get('episode_count', 0)
            self.actor_success_rate = checkpoint.get('actor_success_rate', 0.0)
        
        print(f"Actor-Locked system loaded from: {filepath}")
    
    def set_actor_trials(self, trials: int):
        """Set number of actor trials per state"""
        self.actor_trials = trials
        print(f"Actor trials set to: {trials}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get system information"""
        locked_info = self.locked_model.get_info()
        return {
            **locked_info,
            'system_type': 'ActorLockedSystem',
            'actor_trials': self.actor_trials,
            'actor_success_rate': self.actor_success_rate,
            'her_buffer_size': len(self.her_buffer),
            'training_mode': self.training_mode
        } 