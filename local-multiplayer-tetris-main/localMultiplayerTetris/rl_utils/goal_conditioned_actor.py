"""
Goal-Conditioned Actor - NO PPO Implementation
Simplified actor that learns direct goal→action mapping without PPO complexity.
Key features:
1. Goal-conditioned action selection
2. Direct supervised learning (no policy gradients)
3. Simple loss functions 
4. Compatible with Enhanced 6-Phase system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class GoalConditionedActor(nn.Module):
    """
    SIMPLIFIED Goal-Conditioned Actor (NO PPO)
    Direct goal→action mapping with supervised learning
    Uses 5D goals: rotation(2) + x + y + validity ONLY
    """
    
    def __init__(self, state_dim=210, goal_dim=5, action_dim=8, device='cpu'):
        super(GoalConditionedActor, self).__init__()
        
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.device = device
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Goal encoder (smaller since 5D instead of 6D)
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, 32),  # Adjusted for 5D input
            nn.ReLU(),
            nn.Linear(32, 16),        # Smaller since simpler goals
            nn.ReLU()
        )
        
        # Combined state+goal → action mapping
        self.action_head = nn.Sequential(
            nn.Linear(128 + 16, 128),  # state_features + goal_features (16 now)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # Action probabilities
        )
        
        # Action-specific heads for more precise control
        self.movement_head = nn.Linear(128 + 16, 4)  # Left, Right, Rotate, Down
        self.drop_head = nn.Linear(128 + 16, 2)      # Soft drop, Hard drop
        self.hold_head = nn.Linear(128 + 16, 2)      # Hold, No hold
        
    def forward(self, state, goal):
        """
        Forward pass: state + goal → action probabilities
        """
        # Ensure proper batch dimensions
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(goal.shape) == 1:
            goal = goal.unsqueeze(0)
        
        # Encode state and goal
        state_features = self.state_encoder(state)
        goal_features = self.goal_encoder(goal)
        
        # Combine features
        combined_features = torch.cat([state_features, goal_features], dim=-1)
        
        # Primary action distribution
        action_probs = self.action_head(combined_features)
        
        # Specialized action heads
        movement_logits = self.movement_head(combined_features)
        drop_logits = self.drop_head(combined_features)
        hold_logits = self.hold_head(combined_features)
        
        return {
            'action_probs': action_probs,
            'movement_logits': movement_logits,
            'drop_logits': drop_logits, 
            'hold_logits': hold_logits,
            'combined_features': combined_features
        }
    
    def select_action(self, state, goal, deterministic=False):
        """
        Select action given state and goal
        """
        with torch.no_grad():
            outputs = self.forward(state, goal)
            action_probs = outputs['action_probs']
            
            if deterministic:
                # Select most likely action
                action = torch.argmax(action_probs, dim=-1)
            else:
                # Sample from distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            return action, action_probs
    
    def get_action_vector(self, state, goal):
        """
        Get 8D action vector for Tetris environment
        """
        outputs = self.forward(state, goal)
        
        # Convert specialized outputs to 8D action vector
        movement_probs = F.softmax(outputs['movement_logits'], dim=-1)
        drop_probs = F.softmax(outputs['drop_logits'], dim=-1)
        hold_probs = F.softmax(outputs['hold_logits'], dim=-1)
        
        # Construct 8D action vector
        # [left, right, down, rotate, hold, hard_drop, soft_drop, no_op]
        action_vector = torch.zeros(state.shape[0], 8, device=self.device)
        
        action_vector[:, 0] = movement_probs[:, 0]  # Left
        action_vector[:, 1] = movement_probs[:, 1]  # Right
        action_vector[:, 2] = movement_probs[:, 3]  # Down
        action_vector[:, 3] = movement_probs[:, 2]  # Rotate
        action_vector[:, 4] = hold_probs[:, 0]      # Hold
        action_vector[:, 5] = drop_probs[:, 1]      # Hard drop
        action_vector[:, 6] = drop_probs[:, 0]      # Soft drop
        action_vector[:, 7] = 1.0 - action_vector[:, :7].sum(dim=1)  # No-op
        
        return action_vector


class GoalConditionedTrainer:
    """
    SIMPLIFIED Trainer for Goal-Conditioned Actor (NO PPO)
    Uses supervised learning with goal→action targets
    """
    
    def __init__(self, actor, learning_rate=0.001, device='cpu'):
        self.actor = actor
        self.device = device
        
        # Simple optimizer (no PPO complexity)
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
        
        # Training metrics
        self.training_steps = 0
        self.losses = []
        
    def train_on_goal_action_pairs(self, goal_action_data):
        """
        Train actor on goal→action pairs using supervised learning
        NO PPO - direct supervised learning approach
        """
        if not goal_action_data:
            return {'actor_loss': float('inf'), 'samples_trained': 0}
        
        total_loss = 0
        samples_trained = 0
        
        print(f"   🎯 Training actor on {len(goal_action_data)} goal→action pairs")
        
        for data in goal_action_data:
            # Extract training data
            state = torch.FloatTensor(data['state']).unsqueeze(0).to(self.device)
            goal = data['goal']  # Should be 5D tensor
            target_action = data['target_action']  # Target action vector or placement
            
            # Ensure goal is proper tensor
            if not isinstance(goal, torch.Tensor):
                goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
            elif len(goal.shape) == 1:
                goal = goal.unsqueeze(0).to(self.device)
            
            # Forward pass
            outputs = self.actor(state, goal)
            
            # Calculate supervised learning loss
            loss = self._calculate_supervised_loss(outputs, target_action, data)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            samples_trained += 1
            self.training_steps += 1
        
        avg_loss = total_loss / max(1, samples_trained)
        self.losses.append(avg_loss)
        
        return {
            'actor_loss': avg_loss,
            'samples_trained': samples_trained,
            'training_approach': 'supervised_goal_conditioning',
            'no_ppo': True
        }
    
    def _calculate_supervised_loss(self, outputs, target_action, data):
        """
        Calculate supervised learning loss for goal→action mapping - UPDATED for natural gameplay
        """
        # Extract placement target if available
        placement = data.get('placement', None)
        reward = data.get('terminal_reward', 0)
        
        # Primary action prediction loss
        action_probs = outputs['action_probs']
        
        # FIXED: Handle different target action formats
        if isinstance(target_action, (list, tuple)) and len(target_action) == 8:
            # Natural gameplay - 8D action vector
            target_action_tensor = torch.FloatTensor(target_action).to(self.device)
            
            # Normalize target action (ensure it's a probability distribution)
            if target_action_tensor.sum() > 0:
                target_action_tensor = target_action_tensor / target_action_tensor.sum()
            else:
                # All zeros - create uniform distribution
                target_action_tensor = torch.ones(8, device=self.device) / 8.0
            
            # Cross-entropy loss for action classification
            target_action_idx = torch.argmax(target_action_tensor)
            action_loss = F.cross_entropy(action_probs, target_action_idx.unsqueeze(0))
            
        elif isinstance(target_action, (list, tuple)) and len(target_action) == 3:
            # Old placement-based data (rotation, x, y) to action preferences
            rotation, x_pos, y_pos = target_action
            
            # Create target action distribution based on placement
            target_action_dist = self._placement_to_action_distribution(rotation, x_pos, y_pos)
            target_action_dist = torch.FloatTensor(target_action_dist).to(self.device)
            
            # KL divergence loss for action distribution
            action_loss = F.kl_div(torch.log(action_probs + 1e-8), target_action_dist, reduction='batchmean')
            
        elif isinstance(target_action, torch.Tensor) and target_action.shape[-1] == 8:
            # Direct 8D action tensor target
            target_action_normalized = target_action / (target_action.sum() + 1e-8)
            action_loss = F.mse_loss(action_probs, target_action_normalized)
            
        else:
            # Fallback - reward-based loss
            action_loss = -torch.log(action_probs.max() + 1e-8) * torch.tanh(torch.tensor(reward / 100.0, device=self.device))
        
        # Specialized head losses (simplified for natural gameplay)
        movement_loss = 0
        drop_loss = 0
        hold_loss = 0
        
        # For natural gameplay, use action vector to train specialized heads
        if isinstance(target_action, (list, tuple)) and len(target_action) == 8:
            # Movement head target (left=0, right=1, rotate=3, down=2 from action vector)
            movement_target_probs = torch.zeros(4, device=self.device)
            movement_target_probs[0] = target_action[0]  # Left
            movement_target_probs[1] = target_action[1]  # Right
            movement_target_probs[2] = target_action[3]  # Rotate
            movement_target_probs[3] = target_action[2]  # Down
            
            if movement_target_probs.sum() > 0:
                movement_target_probs = movement_target_probs / movement_target_probs.sum()
                movement_target_idx = torch.argmax(movement_target_probs)
                movement_loss = F.cross_entropy(outputs['movement_logits'], movement_target_idx.unsqueeze(0))
            
            # Drop head target (hard_drop=5, soft_drop via down=2)
            drop_target_probs = torch.zeros(2, device=self.device)
            drop_target_probs[0] = target_action[2]  # Soft drop (down)
            drop_target_probs[1] = target_action[5]  # Hard drop
            
            if drop_target_probs.sum() > 0:
                drop_target_probs = drop_target_probs / drop_target_probs.sum()
                drop_target_idx = torch.argmax(drop_target_probs)
                drop_loss = F.cross_entropy(outputs['drop_logits'], drop_target_idx.unsqueeze(0))
        
        # Combined loss with reduced weights for specialized heads
        total_loss = action_loss + 0.1 * movement_loss + 0.1 * drop_loss + 0.05 * hold_loss
        
        return total_loss
    
    def _placement_to_action_distribution(self, rotation, x_pos, y_pos):
        """
        Convert placement coordinates to action distribution
        """
        action_dist = np.zeros(8)
        
        # Based on placement, create action preferences
        if x_pos < 4:
            action_dist[0] = 0.3  # Left movement
        elif x_pos > 6:
            action_dist[1] = 0.3  # Right movement
        
        if rotation > 0:
            action_dist[3] = min(0.4, rotation * 0.15)  # Rotation preference
        
        if y_pos > 15:
            action_dist[5] = 0.4  # Hard drop for low placements
        else:
            action_dist[2] = 0.2  # Soft movement for high placements
        
        # Normalize
        action_dist = action_dist / (action_dist.sum() + 1e-8)
        
        return action_dist
    
    def _get_movement_target(self, placement):
        """
        Get movement target based on placement
        """
        rotation, x_pos, y_pos = placement
        
        # Simple movement target logic
        if x_pos < 3:
            return torch.tensor([0], device=self.device)  # Left
        elif x_pos > 7:
            return torch.tensor([1], device=self.device)  # Right  
        elif rotation > 0:
            return torch.tensor([2], device=self.device)  # Rotate
        else:
            return torch.tensor([3], device=self.device)  # Down
    
    def evaluate_goal_action_accuracy(self, test_data):
        """
        Evaluate how well the actor matches goal→action mappings
        """
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for data in test_data:
                state = torch.FloatTensor(data['state']).unsqueeze(0).to(self.device)
                goal = data['goal']
                
                if not isinstance(goal, torch.Tensor):
                    goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
                elif len(goal.shape) == 1:
                    goal = goal.unsqueeze(0).to(self.device)
                
                # Predict action
                outputs = self.actor(state, goal)
                predicted_action = torch.argmax(outputs['action_probs'], dim=-1)
                
                # Compare with target (simplified)
                target_placement = data.get('placement', (0, 0, 0))
                target_action = self._placement_to_dominant_action(target_placement)
                
                if predicted_action.item() == target_action:
                    correct_predictions += 1
                
                total_predictions += 1
        
        accuracy = correct_predictions / max(1, total_predictions)
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
    
    def _placement_to_dominant_action(self, placement):
        """
        Convert placement to dominant action for evaluation
        """
        rotation, x_pos, y_pos = placement
        
        # Simple heuristic for dominant action
        if rotation > 0:
            return 3  # Rotate
        elif x_pos < 4:
            return 0  # Left
        elif x_pos > 6:
            return 1  # Right
        elif y_pos > 15:
            return 5  # Hard drop
        else:
            return 2  # Down
    
    def save_checkpoint(self, path):
        """Save training checkpoint"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'losses': self.losses[-100:]  # Save last 100 losses
        }, path)
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.losses = checkpoint['losses']


# Integration with Enhanced 6-Phase System
class GoalConditionedActorIntegration:
    """
    Integration wrapper for Enhanced 6-Phase system
    Provides clean interface between goal generation and action execution
    """
    
    def __init__(self, enhanced_system, state_dim=210, goal_dim=5, action_dim=8, device='cpu'):
        self.enhanced_system = enhanced_system
        
        # Create goal-conditioned actor
        self.actor = GoalConditionedActor(state_dim, goal_dim, action_dim, device)
        self.trainer = GoalConditionedTrainer(self.actor, learning_rate=0.001, device=device)
        
        print("✅ Goal-Conditioned Actor Integration initialized:")
        print("   • NO PPO components")
        print("   • Direct goal→action mapping")
        print("   • Supervised learning approach")
        print("   • Compatible with 6-phase system")
    
    def train_actor_on_exploration_data(self, exploration_data):
        """
        Train actor using exploration data and enhanced system goals - UPDATED for natural gameplay
        """
        # Convert exploration data to goal→action pairs
        goal_action_pairs = []
        
        for data in exploration_data:
            # Get state
            state = data['resulting_state']
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Generate goal using enhanced system
            goal = self.enhanced_system.get_goal_for_actor(state_tensor)
            
            # FIXED: Handle natural gameplay data (action instead of placement)
            if 'action' in data:
                # Natural gameplay data - use actual action taken
                target_action = data['action']
            elif 'placement' in data:
                # Old placement data - convert to action
                target_action = data['placement']
            else:
                # Fallback - use hard drop action
                target_action = [0, 0, 0, 0, 0, 1, 0, 0]
            
            # Create training pair
            goal_action_pairs.append({
                'state': state,
                'goal': goal.squeeze(),
                'target_action': target_action,
                'terminal_reward': data['terminal_reward'],
                'lines_cleared': data.get('lines_cleared', 0),
                'natural_gameplay': data.get('natural_gameplay', False)
            })
        
        # Train actor
        results = self.trainer.train_on_goal_action_pairs(goal_action_pairs)
        
        return {
            **results,
            'goal_action_pairs': len(goal_action_pairs),
            'training_approach': 'supervised_goal_conditioning'
        }
    
    def get_action_for_state_goal(self, state, goal=None):
        """
        Get action for given state (with optional goal override)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0) if not isinstance(state, torch.Tensor) else state
        
        if goal is None:
            # Generate goal using enhanced system
            goal = self.enhanced_system.get_goal_for_actor(state_tensor)
        
        # Get action from actor
        action, action_probs = self.actor.select_action(state_tensor, goal)
        action_vector = self.actor.get_action_vector(state_tensor, goal)
        
        return {
            'action': action.item(),
            'action_probs': action_probs.squeeze(),
            'action_vector': action_vector.squeeze(),
            'goal_used': goal.squeeze()
        }
    
    def evaluate_integration(self, test_data):
        """
        Evaluate the complete goal→action pipeline
        """
        # Test goal generation
        goal_quality = []
        action_consistency = []
        
        for data in test_data[:10]:  # Test subset
            state = torch.FloatTensor(data['resulting_state']).unsqueeze(0)
            
            # Generate goal
            goal = self.enhanced_system.get_goal_for_actor(state)
            goal_quality.append(goal.norm().item())
            
            # Get action
            result = self.get_action_for_state_goal(state.squeeze())
            action_consistency.append(result['action_probs'].max().item())
        
        return {
            'avg_goal_quality': np.mean(goal_quality),
            'avg_action_consistency': np.mean(action_consistency), 
            'integration_status': 'functional',
            'no_ppo_confirmed': True
        } 