"""
Dream-Based Goal Achievement Framework
Revolutionary approach where actor learns from state_model through synthetic dreams

Core Components:
1. TetrisDreamEnvironment - Simulates realistic state transitions
2. ExplicitGoalMatcher - Learns direct goal->action mapping
3. DreamTrajectoryGenerator - Creates synthetic perfect goal trajectories
4. DreamRealityBridge - Transfers dream learning to real execution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class TetrisDreamEnvironment:
    """
    Synthetic environment for dreaming about goal achievement
    Generates realistic state transitions biased toward goal success
    """
    def __init__(self, state_model, tetris_env, device):
        self.state_model = state_model
        self.real_env = tetris_env
        self.device = device
        
        # Dream parameters
        self.dream_accuracy = 0.85  # How realistic dreams are (vs perfect)
        self.goal_bias_strength = 0.7  # How much to bias toward goals
        self.noise_level = 0.1  # Realistic noise to prevent overfitting
        
    def dream_step(self, state, action, goal_vector):
        """
        Simulate taking an action in a dream state
        Returns synthetic next_state that progresses toward goal
        
        Args:
            state: Current state vector (410D)
            action: Action to take (8D one-hot or index)
            goal_vector: Target goal from state_model (36D)
        Returns:
            dream_next_state: Synthetic next state biased toward goal
        """
        # Convert action to index if one-hot
        if hasattr(action, 'shape') and len(action.shape) > 0 and action.shape[0] > 1:
            action_idx = torch.argmax(action).item()
        else:
            action_idx = int(action)
        
        # Extract goal components
        goal_rotation = torch.argmax(goal_vector[0, :4]).item()
        goal_x_pos = torch.argmax(goal_vector[0, 4:14]).item()
        goal_y_pos = torch.argmax(goal_vector[0, 14:34]).item()
        goal_value = goal_vector[0, 34].item()
        goal_confidence = goal_vector[0, 35].item()
        
        # Start with current state
        dream_next_state = state.copy()
        
        # Apply action effects (simplified Tetris physics)
        dream_next_state = self._apply_dream_action_effects(
            dream_next_state, action_idx, goal_rotation, goal_x_pos, goal_y_pos
        )
        
        # Bias toward goal achievement (this is the "dream" magic)
        if goal_confidence > 0.5:  # Only bias for confident goals
            dream_next_state = self._bias_toward_goal_achievement(
                dream_next_state, goal_vector, self.goal_bias_strength
            )
        
        # Add realistic noise to prevent perfect dreams
        dream_next_state = self._add_realistic_noise(dream_next_state, self.noise_level)
        
        return dream_next_state
    
    def _apply_dream_action_effects(self, state, action_idx, goal_rot, goal_x, goal_y):
        """Apply simplified action effects in dream space"""
        dream_state = state.copy()
        
        # Update metadata to move toward goals (indices 407-409)
        current_rot = state[407] * 4  # Denormalize
        current_x = state[408] * 10
        current_y = state[409] * 20
        
        # Dream actions move toward goals
        if action_idx < 4:  # Rotation actions
            new_rot = min(max(action_idx, 0), 3)
            # Bias toward goal rotation
            new_rot = 0.7 * goal_rot + 0.3 * new_rot
        else:
            new_rot = current_rot
            
        if action_idx == 4:  # Move left
            new_x = max(current_x - 1, 0)
        elif action_idx == 5:  # Move right
            new_x = min(current_x + 1, 9)
        else:
            new_x = current_x
        
        # Bias positions toward goals
        new_x = 0.6 * goal_x + 0.4 * new_x
        new_y = 0.6 * goal_y + 0.4 * current_y
        
        # Update dream state
        dream_state[407] = new_rot / 4.0  # Normalize
        dream_state[408] = new_x / 10.0
        dream_state[409] = new_y / 20.0
        
        return dream_state
    
    def _bias_toward_goal_achievement(self, state, goal_vector, bias_strength):
        """Bias the dream state toward achieving the goal"""
        biased_state = state.copy()
        
        # Extract current piece grid (indices 0-199)
        current_piece_grid = state[:200].reshape(20, 10)
        
        # Dream about optimal piece placement
        goal_rotation = torch.argmax(goal_vector[0, :4]).item()
        goal_x_pos = torch.argmax(goal_vector[0, 4:14]).item()
        goal_y_pos = torch.argmax(goal_vector[0, 14:34]).item()
        
        # Create idealized piece placement in dream
        dream_piece_grid = current_piece_grid.copy()
        
        # Place piece at goal position with bias_strength probability
        if np.random.random() < bias_strength:
            # Clear current piece
            dream_piece_grid = dream_piece_grid * 0
            
            # Place at ideal goal position (simplified)
            if goal_y_pos < 18 and goal_x_pos < 8:
                dream_piece_grid[goal_y_pos:goal_y_pos+2, goal_x_pos:goal_x_pos+2] = 1.0
        
        # Update biased state
        biased_state[:200] = dream_piece_grid.flatten()
        
        return biased_state
    
    def _add_realistic_noise(self, state, noise_level):
        """Add realistic noise to prevent perfect dream overfitting"""
        noisy_state = state.copy()
        
        # Add small amounts of noise to continuous values
        noise = np.random.normal(0, noise_level, state.shape)
        noisy_state = noisy_state + noise
        
        # Clamp to valid ranges
        noisy_state = np.clip(noisy_state, -1.0, 1.0)
        
        return noisy_state

class ExplicitGoalMatcher(nn.Module):
    """
    Dedicated network that learns explicit goal->action mapping
    Dreams about achieving state_model goals before real execution
    """
    def __init__(self, state_dim, action_dim, goal_dim, device):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.device = device
        
        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # State encoder  
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Combined goal-state processor
        self.goal_action_predictor = nn.Sequential(
            nn.Linear(64 + 128, 256),  # goal + state encodings
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Move to device
        self.to(device)
        
    def forward(self, state, goal_vector):
        """
        Predict action that will achieve the given goal
        
        Args:
            state: State tensor (batch_size, state_dim)
            goal_vector: Goal tensor (batch_size, goal_dim)
        Returns:
            goal_action: Action probabilities (batch_size, action_dim)
        """
        # Ensure tensors are on correct device
        state = state.to(self.device)
        goal_vector = goal_vector.to(self.device)
        
        # Encode goal and state separately
        goal_encoding = self.goal_encoder(goal_vector)
        state_encoding = self.state_encoder(state)
        
        # Combine and predict optimal action
        combined = torch.cat([goal_encoding, state_encoding], dim=-1)
        goal_action = self.goal_action_predictor(combined)
        
        return goal_action

class DreamTrajectoryGenerator:
    """
    Generates synthetic trajectories where actor practices goal achievement
    Uses state_model predictions as explicit targets for learning
    """
    def __init__(self, state_model, goal_matcher, dream_env, device):
        self.state_model = state_model
        self.goal_matcher = goal_matcher
        self.dream_env = dream_env
        self.device = device
        
        # Dream quality tracking
        self.dream_quality_history = deque(maxlen=1000)
        
    def generate_dream_trajectory(self, initial_state, dream_length=15):
        """
        Generate a synthetic trajectory focused on goal achievement
        
        Args:
            initial_state: Starting state (410D vector)
            dream_length: Number of steps to dream
        Returns:
            dream_trajectory: List of dream experiences
        """
        dream_trajectory = []
        current_state = initial_state.copy()
        cumulative_dream_reward = 0
        
        for step in range(dream_length):
            # Get state_model goal for current state
            with torch.no_grad():
                state_tensor = torch.tensor(current_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                goal_vector = self.state_model.get_placement_goal_vector(state_tensor)
                
            if goal_vector is not None:
                # Ensure goal_vector is on the correct device
                goal_vector = goal_vector.to(self.device)
                
                # Dream about the perfect action to achieve this goal
                with torch.no_grad():
                    dream_action_probs = self.goal_matcher(state_tensor, goal_vector)
                    
                # Sample action from dream distribution (add exploration)
                dream_action_probs = F.softmax(dream_action_probs, dim=-1)  # Ensure proper probabilities
                dream_action_probs = dream_action_probs + 1e-8  # Add epsilon to prevent zeros
                dream_action_probs = dream_action_probs / dream_action_probs.sum(dim=-1, keepdim=True)  # Renormalize
                
                # Validate before sampling
                if torch.any(torch.isnan(dream_action_probs)) or torch.any(dream_action_probs <= 0):
                    print("⚠️ Invalid dream probabilities, using uniform distribution")
                    dream_action_probs = torch.ones_like(dream_action_probs) / dream_action_probs.shape[-1]
                
                try:
                    dream_action_idx = torch.multinomial(dream_action_probs, 1).item()
                except RuntimeError as e:
                    print(f"⚠️ Dream multinomial sampling failed: {e}, using argmax")
                    dream_action_idx = torch.argmax(dream_action_probs, dim=-1).item()
                    
                dream_action_onehot = F.one_hot(torch.tensor(dream_action_idx, device=self.device), self.goal_matcher.action_dim).float()
                
                # Simulate taking this action in dream environment
                dream_next_state = self.dream_env.dream_step(
                    current_state, dream_action_idx, goal_vector
                )
                
                # Calculate dream reward (how well we achieved the goal)
                dream_reward = self._calculate_dream_goal_achievement(
                    current_state, dream_action_onehot, dream_next_state, goal_vector
                )
                
                cumulative_dream_reward += dream_reward
                
                # Create dream experience
                dream_step = {
                    'state': current_state.copy(),
                    'action': dream_action_onehot.cpu().numpy(),
                    'action_idx': dream_action_idx,
                    'reward': dream_reward,
                    'next_state': dream_next_state.copy(),
                    'goal_vector': goal_vector.cpu().numpy(),
                    'is_dream': True,
                    'dream_step': step,
                    'dream_quality': 0.0  # Will be calculated later
                }
                
                dream_trajectory.append(dream_step)
                current_state = dream_next_state
            else:
                # No goal available, end dream
                break
        
        # Calculate overall dream quality
        avg_dream_reward = cumulative_dream_reward / max(1, len(dream_trajectory))
        dream_quality = min(1.0, max(0.0, (avg_dream_reward + 10) / 50.0))  # Normalize to [0,1]
        
        # Update dream quality for all steps
        for step in dream_trajectory:
            step['dream_quality'] = dream_quality
            
        self.dream_quality_history.append(dream_quality)
        
        # HINDSIGHT RELABELING FOR DREAMS: Create additional experiences using future states as goals
        hindsight_dream_experiences = self._create_dream_hindsight_experiences(dream_trajectory)
        dream_trajectory.extend(hindsight_dream_experiences)
        
        return dream_trajectory
    
    def _calculate_dream_goal_achievement(self, state, action, next_state, goal_vector):
        """
        Calculate how well a dream action achieved the goal
        
        Args:
            state: Current state
            action: Action taken (one-hot)
            next_state: Achieved next state
            goal_vector: Target goal
        Returns:
            dream_reward: Reward for goal achievement in dream
        """
        try:
            # Extract goal components
            goal_rotation = torch.argmax(goal_vector[0, :4]).item()
            goal_x_pos = torch.argmax(goal_vector[0, 4:14]).item()
            goal_y_pos = torch.argmax(goal_vector[0, 14:34]).item()
            goal_confidence = goal_vector[0, 35].item()
            
            # Extract achieved placement from next_state
            achieved_rotation = int(next_state[407] * 4)
            achieved_x_pos = int(next_state[408] * 10)
            achieved_y_pos = int(next_state[409] * 20)
            
            # Calculate direct goal matching
            rotation_match = 1.0 - abs(achieved_rotation - goal_rotation) / 4.0
            x_pos_match = 1.0 - abs(achieved_x_pos - goal_x_pos) / 10.0
            y_pos_match = 1.0 - abs(achieved_y_pos - goal_y_pos) / 20.0
            
            # Dream reward heavily favors goal achievement
            dream_reward = (
                rotation_match * 15.0 +      # Rotation alignment
                x_pos_match * 15.0 +         # X position alignment
                y_pos_match * 10.0 +         # Y position alignment
                goal_confidence * 5.0        # Confidence bonus
            )
            
            # Dream quality bonus (dreams should be better than reality)
            dream_bonus = 5.0  # Extra reward for dreaming
            
            return dream_reward + dream_bonus
            
        except Exception as e:
            print(f"Error in dream reward calculation: {e}")
            return 1.0  # Fallback
    
    def get_average_dream_quality(self):
        """Get recent average dream quality"""
        if len(self.dream_quality_history) == 0:
            return 0.0
        return np.mean(self.dream_quality_history)
    
    def _create_dream_hindsight_experiences(self, dream_trajectory):
        """
        Create hindsight experiences for dream rollouts where future dream states become goals
        
        Args:
            dream_trajectory: List of dream experiences from the trajectory
        Returns:
            hindsight_experiences: Additional dream experiences with hindsight relabeling
        """
        hindsight_experiences = []
        
        if len(dream_trajectory) < 2:
            return hindsight_experiences
        
        # For each dream step, create hindsight experiences using future states as goals
        for i, current_step in enumerate(dream_trajectory[:-1]):  # Exclude last step
            # Select future steps as potential hindsight goals
            future_steps = dream_trajectory[i + 1:]
            
            # Use top 50% of future steps by reward as hindsight goals
            future_rewards = [step['reward'] for step in future_steps]
            if not future_rewards:
                continue
                
            # Sort by reward and take top half
            sorted_indices = sorted(range(len(future_steps)), 
                                  key=lambda idx: future_rewards[idx], 
                                  reverse=True)
            top_half_count = max(1, len(sorted_indices) // 2)
            top_future_indices = sorted_indices[:top_half_count]
            
            # Create hindsight experiences for each top future step
            for future_idx in top_future_indices[:3]:  # Limit to top 3 to avoid explosion
                future_step = future_steps[future_idx]
                
                # Create synthetic goal from future state
                hindsight_goal = self._create_goal_from_state(future_step['next_state'])
                
                if hindsight_goal is not None:
                    # Calculate hindsight reward based on how well current action led to future state
                    hindsight_reward = self._calculate_dream_hindsight_reward(
                        current_step['state'],
                        current_step['action'],
                        current_step['next_state'],
                        future_step['next_state'],
                        hindsight_goal
                    )
                    
                    # Create hindsight dream experience
                    hindsight_step = {
                        'state': current_step['state'].copy(),
                        'action': current_step['action'].copy(),
                        'action_idx': current_step['action_idx'],
                        'reward': hindsight_reward,
                        'next_state': current_step['next_state'].copy(),
                        'goal_vector': hindsight_goal,
                        'is_dream': True,
                        'is_hindsight': True,  # Mark as hindsight experience
                        'dream_step': current_step['dream_step'],
                        'dream_quality': current_step['dream_quality'],
                        'hindsight_source': f"future_step_{i + 1 + future_idx}",
                        'temporal_distance': future_idx + 1  # Distance to future goal
                    }
                    
                    hindsight_experiences.append(hindsight_step)
        
        return hindsight_experiences
    
    def _create_goal_from_state(self, state):
        """
        Create a goal vector from an achieved state (reverse engineering)
        
        Args:
            state: Achieved state (410D vector)
        Returns:
            goal_vector: 36D goal vector derived from state
        """
        try:
            # Extract placement information from state metadata (indices 407-409)
            rotation = int(state[407] * 4)  # Denormalize rotation
            x_pos = int(state[408] * 10)    # Denormalize x position
            y_pos = int(state[409] * 20)    # Denormalize y position
            
            # Create goal vector with one-hot encodings
            goal_vector = np.zeros(36)
            
            # Rotation one-hot (indices 0-3)
            rotation = max(0, min(3, rotation))
            goal_vector[rotation] = 1.0
            
            # X position one-hot (indices 4-13)
            x_pos = max(0, min(9, x_pos))
            goal_vector[4 + x_pos] = 1.0
            
            # Y position one-hot (indices 14-33)
            y_pos = max(0, min(19, y_pos))
            goal_vector[14 + y_pos] = 1.0
            
            # Value and confidence (indices 34-35)
            # For hindsight goals, assume good placement with high confidence
            goal_vector[34] = 0.8  # High value
            goal_vector[35] = 0.9  # High confidence
            
            return goal_vector
            
        except Exception as e:
            print(f"Error creating goal from state: {e}")
            return None
    
    def _calculate_dream_hindsight_reward(self, current_state, action, achieved_state, future_state, hindsight_goal):
        """
        Calculate hindsight reward for dream experience
        
        Args:
            current_state: Starting state
            action: Action taken
            achieved_state: State after action
            future_state: Future state used as hindsight goal
            hindsight_goal: Goal vector created from future state
        Returns:
            hindsight_reward: Reward for achieving progress toward hindsight goal
        """
        try:
            # Calculate similarity between achieved state and future goal state
            achieved_tensor = torch.tensor(achieved_state, dtype=torch.float32, device=self.device)
            future_tensor = torch.tensor(future_state, dtype=torch.float32, device=self.device)
            
            # Cosine similarity for overall state alignment
            state_similarity = F.cosine_similarity(achieved_tensor, future_tensor, dim=0).item()
            state_similarity = max(0.0, state_similarity)
            
            # Extract goal components from hindsight goal
            goal_rotation = np.argmax(hindsight_goal[:4])
            goal_x_pos = np.argmax(hindsight_goal[4:14])
            goal_y_pos = np.argmax(hindsight_goal[14:34])
            
            # Extract achieved placement
            achieved_rotation = int(achieved_state[407] * 4)
            achieved_x_pos = int(achieved_state[408] * 10)
            achieved_y_pos = int(achieved_state[409] * 20)
            
            # Calculate hindsight goal alignment
            rotation_similarity = 1.0 - abs(achieved_rotation - goal_rotation) / 4.0
            x_similarity = 1.0 - abs(achieved_x_pos - goal_x_pos) / 10.0
            y_similarity = 1.0 - abs(achieved_y_pos - goal_y_pos) / 20.0
            
            # Combine similarities for hindsight reward
            hindsight_reward = (
                state_similarity * 8.0 +         # Overall state progression
                rotation_similarity * 6.0 +      # Rotation alignment
                x_similarity * 6.0 +             # X position alignment
                y_similarity * 5.0 +             # Y position alignment
                5.0                              # Hindsight bonus
            )
            
            return min(40.0, max(0.0, hindsight_reward))  # Clamp to reasonable range
            
        except Exception as e:
            print(f"Error in dream hindsight reward calculation: {e}")
            return 2.0  # Fallback reward

class DreamRealityBridge:
    """
    Transfers learning from dreams to reality
    Ensures dream-learned behaviors work in real environment
    """
    def __init__(self, actor_critic, goal_matcher, dream_generator, device):
        self.actor_critic = actor_critic
        self.goal_matcher = goal_matcher
        self.dream_generator = dream_generator
        self.device = device
        
        # Dream learning components
        self.dream_buffer = deque(maxlen=10000)
        self.goal_matcher_optimizer = torch.optim.Adam(goal_matcher.parameters(), lr=1e-3)
        
        # Transfer learning tracking
        self.transfer_loss_history = deque(maxlen=100)
        
    def dream_training_phase(self, num_dream_episodes=50):
        """
        Pure dream training where goal_matcher learns goal achievement
        
        Args:
            num_dream_episodes: Number of dream episodes to generate
        Returns:
            dream_experiences: Generated dream experiences
            avg_goal_loss: Average goal matching loss
        """
        print(f"🌙 Generating {num_dream_episodes} dream episodes...")
        
        dream_experiences = []
        goal_losses = []
        
        for episode in range(num_dream_episodes):
            # Generate random starting state
            initial_state = self._generate_random_tetris_state()
            
            # Generate dream trajectory
            dream_trajectory = self.dream_generator.generate_dream_trajectory(
                initial_state, dream_length=20
            )
            
            # Train goal_matcher on achieving state_model goals
            for step in dream_trajectory:
                state_tensor = torch.tensor(step['state'], dtype=torch.float32, device=self.device).unsqueeze(0)
                goal_tensor = torch.tensor(step['goal_vector'], dtype=torch.float32, device=self.device)
                target_action = torch.tensor(step['action_idx'], dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Predict action for goal
                predicted_action_probs = self.goal_matcher(state_tensor, goal_tensor)
                
                # Supervised learning: learn to predict optimal actions for goals
                goal_loss = F.cross_entropy(predicted_action_probs, target_action)
                
                # Backpropagation
                self.goal_matcher_optimizer.zero_grad()
                goal_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.goal_matcher.parameters(), 1.0)
                self.goal_matcher_optimizer.step()
                
                goal_losses.append(goal_loss.item())
            
            dream_experiences.extend(dream_trajectory)
            
        avg_goal_loss = np.mean(goal_losses) if goal_losses else 0.0
        
        # Store dreams for transfer
        self.dream_buffer.extend(dream_experiences)
        
        print(f"   ✨ Dream quality: {self.dream_generator.get_average_dream_quality():.3f}")
        print(f"   🎯 Goal matching loss: {avg_goal_loss:.4f}")
        
        return dream_experiences, avg_goal_loss
    
    def reality_transfer_phase(self, num_transfer_steps=200):
        """
        Transfer dream learning to actor_critic for real execution
        
        Args:
            num_transfer_steps: Number of transfer training steps
        Returns:
            avg_transfer_loss: Average distillation loss
        """
        if len(self.dream_buffer) == 0:
            return 0.0
            
        print(f"🌉 Transferring dream knowledge to actor ({num_transfer_steps} steps)...")
        
        # Filter high-quality dreams
        quality_dreams = self._filter_high_quality_dreams(self.dream_buffer)
        
        if len(quality_dreams) == 0:
            return 0.0
        
        transfer_losses = []
        
        for step in range(num_transfer_steps):
            # Sample dream experience
            dream_step = random.choice(quality_dreams)
            
            state_tensor = torch.tensor(dream_step['state'], dtype=torch.float32, device=self.device).unsqueeze(0)
            goal_tensor = torch.tensor(dream_step['goal_vector'], dtype=torch.float32, device=self.device)
            
            # Get optimal action from goal_matcher (teacher)
            with torch.no_grad():
                optimal_action_probs = self.goal_matcher(state_tensor, goal_tensor)
            
            # Get actor prediction (student) - use proper ActorCritic forward method
            actor_action_probs, _ = self.actor_critic.network(state_tensor, goal_tensor)
            
            # Distillation loss: actor learns to mimic goal_matcher
            transfer_loss = F.kl_div(
                F.log_softmax(actor_action_probs, dim=-1),
                optimal_action_probs,
                reduction='batchmean'
            )
            
            # Backpropagation for actor only
            self.actor_critic.actor_optimizer.zero_grad()
            transfer_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.network.actor.parameters(), 0.5)
            self.actor_critic.actor_optimizer.step()
            
            transfer_losses.append(transfer_loss.item())
        
        avg_transfer_loss = np.mean(transfer_losses)
        self.transfer_loss_history.append(avg_transfer_loss)
        
        print(f"   🎭 Actor-to-Dream alignment loss: {avg_transfer_loss:.4f}")
        print(f"   🎯 Dreams transferred: {len(quality_dreams)}")
        
        return avg_transfer_loss
    
    def get_dream_guided_action(self, state, goal_vector, dream_weight=0.5):
        """
        Get action guided by dream knowledge
        
        Args:
            state: Current state tensor
            goal_vector: Goal from state_model
            dream_weight: Weight for dream guidance vs actor exploration
        Returns:
            guided_action: Action blending dream guidance and actor policy
        """
        with torch.no_grad():
            # Ensure tensors are on the correct device
            state = state.to(self.device)
            goal_vector = goal_vector.to(self.device)
            
            # Get dream-optimal action
            dream_action_probs = self.goal_matcher(state, goal_vector)
            
            # Get actor action - use proper ActorCritic forward method
            actor_action_probs, _ = self.actor_critic.network(state, goal_vector)
            
            # Ensure probabilities are valid and normalized
            dream_action_probs = F.softmax(dream_action_probs, dim=-1)  # Convert to proper probabilities
            actor_action_probs = F.softmax(actor_action_probs, dim=-1)  # Convert to proper probabilities
            
            # Add small epsilon to prevent zeros
            epsilon = 1e-8
            dream_action_probs = dream_action_probs + epsilon
            actor_action_probs = actor_action_probs + epsilon
            
            # Renormalize after adding epsilon
            dream_action_probs = dream_action_probs / dream_action_probs.sum(dim=-1, keepdim=True)
            actor_action_probs = actor_action_probs / actor_action_probs.sum(dim=-1, keepdim=True)
            
            # Blend dream guidance with actor exploration
            guided_action_probs = (
                dream_weight * dream_action_probs + 
                (1 - dream_weight) * actor_action_probs
            )
            
            # Final normalization
            guided_action_probs = guided_action_probs / guided_action_probs.sum(dim=-1, keepdim=True)
            
            # Validate probabilities before sampling
            if torch.any(torch.isnan(guided_action_probs)) or torch.any(guided_action_probs <= 0):
                print("⚠️ Invalid probabilities detected, using uniform distribution")
                guided_action_probs = torch.ones_like(guided_action_probs) / guided_action_probs.shape[-1]
            
            # Sample from blended distribution
            try:
                guided_action_idx = torch.multinomial(guided_action_probs, 1).item()
            except RuntimeError as e:
                print(f"⚠️ Multinomial sampling failed: {e}, using argmax fallback")
                guided_action_idx = torch.argmax(guided_action_probs, dim=-1).item()
            
            guided_action_onehot = F.one_hot(torch.tensor(guided_action_idx, device=self.device), 8).float()
            
        return guided_action_onehot.cpu().numpy()
    
    def _filter_high_quality_dreams(self, dream_buffer, quality_threshold=0.6):
        """Filter dreams by quality score"""
        return [dream for dream in dream_buffer if dream['dream_quality'] >= quality_threshold]
    
    def _generate_random_tetris_state(self):
        """Generate random but valid Tetris state for dream starting point"""
        # Create random but plausible Tetris state
        state = np.zeros(410)
        
        # Random current piece grid (sparse)
        current_piece = np.random.choice([0, 1], size=200, p=[0.95, 0.05])
        state[:200] = current_piece
        
        # Random empty grid (mostly empty)
        empty_grid = np.random.choice([0, 1], size=200, p=[0.8, 0.2])
        state[200:400] = empty_grid
        
        # Random next piece (one-hot)
        next_piece_idx = np.random.randint(0, 7)
        state[400 + next_piece_idx] = 1.0
        
        # Random metadata
        state[407] = np.random.uniform(0, 1)  # rotation
        state[408] = np.random.uniform(0, 1)  # x_pos
        state[409] = np.random.uniform(0, 1)  # y_pos
        
        return state 