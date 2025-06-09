#!/usr/bin/env python3
"""
Actor-Locked Hierarchical System with Option A Sequential Movement Execution

Architecture:
1. Locked Model: Selects target piece positions (x, y, rotation) 
2. Actor Model: Generates movement sequence to reach target
3. Sequential Execution: Execute movements step by step
4. Hindsight Relabelling: Rewards exact goal matching with random future goals
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

from .dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent
from .base_agent import BaseAgent

class MovementActorNetwork(nn.Module):
    """
    Movement Actor Network for generating piece movement sequences
    
    Input: Board state + current piece info + target placement position
    Output: Movement action probabilities (left, right, down, rotate, drop, etc.)
    """
    
    def __init__(self, input_dim: int = 212, movement_action_dim: int = 8):
        super(MovementActorNetwork, self).__init__()
        
        # Input: 206 (board) + 3 (current position) + 3 (target position) = 212
        # Output: 8 movement actions (left, right, down, rotate_cw, rotate_ccw, soft_drop, hard_drop, no_op)
        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, movement_action_dim)
        
        self.dropout = nn.Dropout(0.1)
        
        # Movement action mapping (direct environment actions)
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
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        action_probs = F.softmax(self.fc4(x), dim=-1)
        return action_probs

class HindsightExperienceBuffer:
    """Experience buffer with proper HER using random future goals"""
    
    def __init__(self, capacity: int = 50000, her_ratio: float = 0.4):
        self.capacity = capacity
        self.her_ratio = her_ratio
        self.buffer = deque(maxlen=capacity)
        self.goal_trajectory = deque(maxlen=10000)
        
    def store(self, experience: Dict[str, Any]):
        """Store experience and update goal trajectory"""
        self.buffer.append(experience)
        
        # Store achieved goals in trajectory for random HER sampling
        if 'achieved_goal' in experience:
            self.goal_trajectory.append(experience['achieved_goal'])
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch with proper HER using random future goals"""
        if len(self.buffer) < batch_size:
            return []
        
        experiences = random.sample(self.buffer, batch_size)
        
        # Apply HER relabelling with random future goals
        her_count = int(batch_size * self.her_ratio)
        for i in range(her_count):
            exp = experiences[i]
            
            # Use random future goal from trajectory (proper HER)
            if len(self.goal_trajectory) > 0:
                random_future_goal = random.choice(list(self.goal_trajectory))
                
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
        if np.array_equal(achieved_goal, desired_goal):
            return 100.0  # High reward for exact match
        else:
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return -distance * 10.0
    
    def __len__(self):
        return len(self.buffer)

class ActorLockedSystem(BaseAgent):
    """
    Option A: Sequential Movement Execution System
    
    1. Locked Model: Selects target position (x, y, rotation)
    2. Actor Model: Generates movement sequence to reach target  
    3. Sequential Execution: Execute movements step by step
    4. HER: Learn from achieved vs desired goals
    """
    
    def __init__(self,
                 device: str = 'cuda',
                 max_movement_steps: int = 20,
                 actor_learning_rate: float = 0.0001,
                 locked_model_path: Optional[str] = None,
                 epsilon_start: float = 0.95,
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 50000):
        
        super().__init__(action_space_size=800, observation_space_shape=(206,), device=device)
        
        self.max_movement_steps = max_movement_steps  # Max steps to reach target
        self.device = torch.device(device)
        
        # Initialize Locked Model (selects target positions) with custom epsilon schedule
        self.locked_model = RedesignedLockedStateDQNAgent(
            device=device,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps
        )
        if locked_model_path:
            self.locked_model.load_checkpoint(locked_model_path)
            print(f"Loaded locked model from: {locked_model_path}")
        
        # Initialize Movement Actor Model (generates movement sequences)
        self.actor_network = MovementActorNetwork().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=actor_learning_rate)
        
        # Hindsight Experience Replay with random future goals
        self.her_buffer = HindsightExperienceBuffer()
        
        # Training state
        self.training_mode = True
        self.episode_count = 0
        self.actor_success_rate = 0.0
        
        print(f"Actor-Locked System (Option A) initialized:")
        print(f"   Device: {self.device}")
        print(f"   Max movement steps: {self.max_movement_steps}")
        print(f"   Locked model parameters: {self.locked_model.get_parameter_count():,}")
        print(f"   Actor model parameters: {sum(p.numel() for p in self.actor_network.parameters()):,}")
    
    def _decode_locked_action(self, locked_action: int) -> Tuple[int, int, int]:
        """Decode locked model action to (x, y, rotation)"""
        # Locked action space: 800 = 10 * 20 * 4 (x * y * rotation)
        x = locked_action % 10
        temp = locked_action // 10
        y = temp % 20
        rotation = temp // 20
        return x, y, rotation
    
    def _encode_goal(self, x: int, y: int, rotation: int) -> np.ndarray:
        """Encode goal as normalized 3D vector"""
        return np.array([x / 9.0, y / 19.0, rotation / 3.0], dtype=np.float32)
    
    def _create_actor_input(self, board_state: np.ndarray, current_pos: Tuple[int, int, int], 
                           target_pos: Tuple[int, int, int]) -> torch.Tensor:
        """Create input for actor network"""
        # Normalize positions  
        current_norm = np.array([current_pos[0]/9.0, current_pos[1]/19.0, current_pos[2]/3.0])
        target_norm = np.array([target_pos[0]/9.0, target_pos[1]/19.0, target_pos[2]/3.0])
        
        # Concatenate: board (206) + current pos (3) + target pos (3) = 212
        actor_input = np.concatenate([board_state, current_norm, target_norm])
        return torch.FloatTensor(actor_input).unsqueeze(0).to(self.device)
    
    def select_action(self, observation: np.ndarray, training: bool = True, env=None) -> int:
        """
        Option A Sequential Movement Execution (Simplified):
        1. Use locked model to select target position
        2. Simulate movement sequence with actor to reach target (for HER training)
        3. Return final locked position action for compatibility
        """
        # Step 1: Get target position from locked model
        locked_action = self.locked_model.select_action(observation, training=training)
        target_x, target_y, target_rotation = self._decode_locked_action(locked_action)
        
        # Step 2: Simulate movement sequence to reach target (for HER training)
        if training:
            achieved_goal = self._simulate_movement_sequence(
                observation, (target_x, target_y, target_rotation)
            )
            
            # Step 3: Store experience for HER training
            desired_goal = self._encode_goal(target_x, target_y, target_rotation)
            experience = {
                'observation': observation.copy(),
                'desired_goal': desired_goal,
                'achieved_goal': achieved_goal,
                'reward': self._compute_reward(achieved_goal, desired_goal)
            }
            self.her_buffer.store(experience)
        
        # Return locked action for environment execution
        return locked_action
    
    def _simulate_movement_sequence(self, observation: np.ndarray, 
                                  target_pos: Tuple[int, int, int]) -> np.ndarray:
        """
        Simulate movement sequence to reach target position (for HER training)
        Returns achieved goal position
        """
        # Get current piece position (simplified - assume piece at center initially)
        current_x, current_y, current_rotation = 4, 0, 0  # Starting position approximation
        
        # Track movement sequence
        movements_executed = 0
        
        while movements_executed < self.max_movement_steps:
            # Check if we've reached target
            if (current_x, current_y, current_rotation) == target_pos:
                break
            
            # Create actor input
            actor_input = self._create_actor_input(
                observation, (current_x, current_y, current_rotation), target_pos
            )
            
            # Get movement action from actor
            with torch.no_grad():
                action_probs = self.actor_network(actor_input)
                # Sample action during simulation
                action_dist = torch.distributions.Categorical(action_probs)
                movement_action = action_dist.sample().item()
            
            # Update current position based on movement (simulated)
            current_x, current_y, current_rotation = self._update_position_from_movement(
                current_x, current_y, current_rotation, movement_action
            )
            
            movements_executed += 1
        
        # Return achieved goal
        return self._encode_goal(current_x, current_y, current_rotation)
    
    def _update_position_from_movement(self, x: int, y: int, rotation: int, 
                                     movement_action: int) -> Tuple[int, int, int]:
        """Update piece position based on movement action"""
        new_x, new_y, new_rotation = x, y, rotation
        
        if movement_action == 0:  # MOVE_LEFT
            new_x = max(0, x - 1)
        elif movement_action == 1:  # MOVE_RIGHT  
            new_x = min(9, x + 1)
        elif movement_action == 2:  # MOVE_DOWN
            new_y = min(19, y + 1)
        elif movement_action == 3:  # ROTATE_CW
            new_rotation = (rotation + 1) % 4
        elif movement_action == 4:  # ROTATE_CCW
            new_rotation = (rotation - 1) % 4
        elif movement_action == 5:  # SOFT_DROP
            new_y = min(19, y + 1)
        elif movement_action == 6:  # HARD_DROP
            new_y = 19  # Drop to bottom
        # movement_action == 7 is NO_OP, no change
        
        return new_x, new_y, new_rotation
    
    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """Compute reward for goal achievement"""
        if np.allclose(achieved_goal, desired_goal, atol=0.1):
            return 100.0  # High reward for reaching target
        else:
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return -distance * 10.0  # Distance penalty
    
    def train_actor(self, batch_size: int = 32) -> Dict[str, float]:
        """Train actor network using HER experiences"""
        if len(self.her_buffer) < batch_size:
            return {}
        
        experiences = self.her_buffer.sample(batch_size)
        if not experiences:
            return {}
        
        # Prepare training data
        observations = []
        desired_goals = []
        rewards = []
        
        for exp in experiences:
            observations.append(exp['observation'])
            desired_goals.append(exp['desired_goal'])
            rewards.append(exp['reward'])
        
        observations = torch.FloatTensor(np.array(observations)).to(self.device)
        desired_goals = torch.FloatTensor(np.array(desired_goals)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # Create actor inputs (simplified for training)
        current_pos = torch.zeros((batch_size, 3)).to(self.device)  # Simplified current position
        actor_inputs = torch.cat([observations, current_pos, desired_goals], dim=1)
        
        # Forward pass
        action_probs = self.actor_network(actor_inputs)
        
        # Compute loss (policy gradient style)
        # Use rewards as advantages for policy gradient
        log_probs = torch.log(action_probs + 1e-8)
        policy_loss = -(log_probs * rewards.unsqueeze(1)).mean()
        
        # Backward pass
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return {
            'actor_loss': policy_loss.item(),
            'avg_reward': rewards.mean().item(),
            'buffer_size': len(self.her_buffer)
        }
    
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> Dict[str, float]:
        """Update function for compatibility"""
        # Train actor network
        actor_metrics = self.train_actor()
        
        # Update locked model
        locked_metrics = self.locked_model.update(state, action, reward, next_state, done)
        
        # Combine metrics
        metrics = {**locked_metrics, **actor_metrics}
        
        if done:
            self.episode_count += 1
        
        return metrics
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save both locked and actor models"""
        checkpoint = {
            'actor_network_state_dict': self.actor_network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'actor_success_rate': self.actor_success_rate,
            'max_movement_steps': self.max_movement_steps
        }
        torch.save(checkpoint, filepath)
        
        # Save locked model separately
        locked_path = filepath.replace('.pth', '_locked.pth')
        self.locked_model.save_checkpoint(locked_path)
        
        print(f"Actor-Locked system checkpoint saved: {filepath}")
        print(f"Locked model checkpoint saved: {locked_path}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load both locked and actor models"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor_network.load_state_dict(checkpoint['actor_network_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.episode_count = checkpoint.get('episode_count', 0)
        self.actor_success_rate = checkpoint.get('actor_success_rate', 0.0)
        self.max_movement_steps = checkpoint.get('max_movement_steps', 20)
        
        # Load locked model
        locked_path = filepath.replace('.pth', '_locked.pth')
        if os.path.exists(locked_path):
            self.locked_model.load_checkpoint(locked_path)
        
        print(f"Actor-Locked system checkpoint loaded: {filepath}")
    
    def set_max_movement_steps(self, steps: int):
        """Set maximum movement steps per sequence"""
        self.max_movement_steps = steps
    
    def get_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'episode_count': self.episode_count,
            'actor_success_rate': self.actor_success_rate,
            'max_movement_steps': self.max_movement_steps,
            'locked_model_info': self.locked_model.get_info(),
            'her_buffer_size': len(self.her_buffer),
            'device': str(self.device)
        } 