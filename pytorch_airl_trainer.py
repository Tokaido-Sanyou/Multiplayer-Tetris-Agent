#!/usr/bin/env python3
"""
PyTorch AIRL Trainer with TensorBoard Logging
Complete AIRL implementation without TensorFlow dependencies
"""

import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
from typing import Dict, List, Tuple, Optional
import logging

# Add paths
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')

class PyTorchDiscriminator(nn.Module):
    """PyTorch-based AIRL Discriminator Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 128, 64]):
        super(PyTorchDiscriminator, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State processing network
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        
        # Action processing network (one-hot encoded)
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_sizes[1] // 2),
            nn.ReLU()
        )
        
        # Combined processing
        combined_dim = hidden_sizes[1] + hidden_sizes[1] // 2
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1)
        )
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for expert vs learner classification."""
        state_features = self.state_net(states)
        action_features = self.action_net(actions)
        
        combined = torch.cat([state_features, action_features], dim=-1)
        logits = self.combined_net(combined)
        return logits
    
    def get_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get AIRL reward: log(D/(1-D))"""
        logits = self.forward(states, actions)
        probs = torch.sigmoid(logits)
        reward = torch.log(probs + 1e-8) - torch.log(1 - probs + 1e-8)
        return reward

class PyTorchActorCritic(nn.Module):
    """PyTorch-based Actor-Critic Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PyTorchActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action probabilities and state values."""
        features = self.feature_net(states)
        action_probs = self.actor(features)
        state_values = self.critic(features)
        return action_probs, state_values

class ExpertLoader:
    """Expert trajectory loader with PyTorch compatibility."""
    
    def __init__(self, trajectory_dir: str = "expert_trajectories_dqn_adapter"):
        self.trajectory_dir = trajectory_dir
        self.trajectories = []
        self.transitions = []
        
    def load_trajectories(self) -> int:
        """Load expert trajectories from directory."""
        if not os.path.exists(self.trajectory_dir):
            return 0
        
        trajectory_files = [f for f in os.listdir(self.trajectory_dir) if f.endswith('.pkl')]
        
        for filename in trajectory_files:
            filepath = os.path.join(self.trajectory_dir, filename)
            
            try:
                with open(filepath, 'rb') as f:
                    trajectory_data = pickle.load(f)
                
                steps = trajectory_data.get('steps', [])
                if len(steps) < 10:  # Skip very short episodes
                    continue
                
                # Process transitions
                for step in steps:
                    state_dict = step.get('state', {})
                    next_state_dict = step.get('next_state', {})
                    
                    if not state_dict or not next_state_dict:
                        continue
                    
                    # Extract features (207-dimensional)
                    state_features = self._extract_features(state_dict)
                    next_state_features = self._extract_features(next_state_dict)
                    
                    # Convert action to one-hot
                    action = step.get('action', 0)
                    action_onehot = self._action_to_onehot(action)
                    
                    transition = {
                        'state': state_features,
                        'action': action,
                        'action_onehot': action_onehot,
                        'reward': float(step.get('reward', 0.0)),
                        'next_state': next_state_features,
                        'done': bool(step.get('done', False))
                    }
                    
                    self.transitions.append(transition)
                
                self.trajectories.append(trajectory_data)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        return len(self.trajectories)
    
    def _extract_features(self, observation: Dict) -> np.ndarray:
        """Extract 207-dimensional features from observation."""
        grid = observation['grid'].flatten()  # 200 features
        next_piece = np.array([observation['next_piece']])  # 1 feature
        hold_piece = np.array([observation['hold_piece']])  # 1 feature
        current_shape = np.array([observation['current_shape']])  # 1 feature
        current_rotation = np.array([observation['current_rotation']])  # 1 feature
        current_x = np.array([observation['current_x']])  # 1 feature
        current_y = np.array([observation['current_y']])  # 1 feature
        can_hold = np.array([observation['can_hold']])  # 1 feature
        
        features = np.concatenate([
            grid, next_piece, hold_piece, current_shape,
            current_rotation, current_x, current_y, can_hold
        ]).astype(np.float32)
        
        return features
    
    def _action_to_onehot(self, action: int, num_actions: int = 41) -> np.ndarray:
        """Convert action to one-hot encoding."""
        onehot = np.zeros(num_actions, dtype=np.float32)
        if 0 <= action < num_actions:
            onehot[action] = 1.0
        return onehot
    
    def get_batch(self, batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Sample a batch of expert transitions."""
        if not self.transitions:
            raise ValueError("No expert transitions loaded")
        
        # Sample random transitions
        indices = np.random.choice(len(self.transitions), batch_size, replace=True)
        batch_transitions = [self.transitions[i] for i in indices]
        
        # Convert to tensors
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

class PyTorchAIRLTrainer:
    """Complete PyTorch AIRL Trainer with TensorBoard logging."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', True) else 'cpu')
        
        # Initialize environment
        from tetris_env import TetrisEnv
        self.env = TetrisEnv(single_player=True, headless=config.get('headless', True))
        
        # Dimensions
        self.state_dim = 207  # Full TetrisEnv observation
        self.action_dim = 41  # TetrisEnv action space
        
        # Initialize networks
        self.discriminator = PyTorchDiscriminator(self.state_dim, self.action_dim).to(self.device)
        self.policy = PyTorchActorCritic(self.state_dim, self.action_dim).to(self.device)
        
        # Optimizers
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=config.get('discriminator_lr', 3e-4)
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=config.get('policy_lr', 1e-4)
        )
        
        # Expert loader
        self.expert_loader = ExpertLoader(config.get('expert_trajectory_dir', 'expert_trajectories_dqn_adapter'))
        
        # Training buffers
        self.learner_buffer = []
        self.max_buffer_size = config.get('buffer_size', 10000)
        
        # TensorBoard logging
        self.use_tensorboard = config.get('use_tensorboard', True)
        if self.use_tensorboard:
            log_dir = os.path.join('logs', f"airl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.writer = SummaryWriter(log_dir)
            print(f"📊 TensorBoard logs: {log_dir}")
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.step_count = 0
        
    def extract_features(self, observation: Dict) -> np.ndarray:
        """Extract features from environment observation."""
        grid = observation['grid'].flatten()
        next_piece = np.array([observation['next_piece']])
        hold_piece = np.array([observation['hold_piece']])
        current_shape = np.array([observation['current_shape']])
        current_rotation = np.array([observation['current_rotation']])
        current_x = np.array([observation['current_x']])
        current_y = np.array([observation['current_y']])
        can_hold = np.array([observation['can_hold']])
        
        features = np.concatenate([
            grid, next_piece, hold_piece, current_shape,
            current_rotation, current_x, current_y, can_hold
        ]).astype(np.float32)
        
        return features
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        
        return action
    
    def collect_learner_data(self, num_episodes: int = 5) -> int:
        """Collect data from current learner policy."""
        transitions_collected = 0
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            episode_transitions = []
            
            for step in range(self.config.get('max_episode_steps', 500)):
                # Extract features and select action
                state_features = self.extract_features(obs)
                action = self.select_action(state_features)
                
                # Take action
                step_result = self.env.step(action)
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                    truncated = False
                else:
                    next_obs, reward, done, truncated, info = step_result
                
                done = done or truncated
                
                # Store transition
                next_state_features = self.extract_features(next_obs) if not done else np.zeros_like(state_features)
                
                # Convert action to one-hot
                action_onehot = np.zeros(self.action_dim, dtype=np.float32)
                action_onehot[action] = 1.0
                
                transition = {
                    'state': state_features,
                    'action': action,
                    'action_onehot': action_onehot,
                    'reward': reward,
                    'next_state': next_state_features,
                    'done': done
                }
                
                episode_transitions.append(transition)
                obs = next_obs
                
                if done:
                    break
            
            # Add to buffer
            self.learner_buffer.extend(episode_transitions)
            transitions_collected += len(episode_transitions)
            
            # Maintain buffer size
            if len(self.learner_buffer) > self.max_buffer_size:
                self.learner_buffer = self.learner_buffer[-self.max_buffer_size:]
        
        return transitions_collected
    
    def get_learner_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get batch from learner buffer."""
        if len(self.learner_buffer) < batch_size:
            raise ValueError("Not enough learner data")
        
        indices = np.random.choice(len(self.learner_buffer), batch_size, replace=True)
        batch_transitions = [self.learner_buffer[i] for i in indices]
        
        states = torch.FloatTensor([t['state'] for t in batch_transitions]).to(self.device)
        actions_onehot = torch.FloatTensor([t['action_onehot'] for t in batch_transitions]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in batch_transitions]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([t['next_state'] for t in batch_transitions]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in batch_transitions]).unsqueeze(1).to(self.device)
        
        return {
            'states': states,
            'actions': actions_onehot,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def update_discriminator(self, expert_batch: Dict, learner_batch: Dict) -> Dict[str, float]:
        """Update discriminator to distinguish expert from learner."""
        batch_size = expert_batch['states'].shape[0]
        
        # Get discriminator predictions
        expert_logits = self.discriminator(expert_batch['states'], expert_batch['actions'])
        learner_logits = self.discriminator(learner_batch['states'], learner_batch['actions'])
        
        # Labels: 1 for expert, 0 for learner
        expert_labels = torch.ones(batch_size, 1, device=self.device)
        learner_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Binary cross-entropy loss
        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, expert_labels)
        learner_loss = F.binary_cross_entropy_with_logits(learner_logits, learner_labels)
        discriminator_loss = expert_loss + learner_loss
        
        # Update discriminator
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        # Calculate accuracy
        expert_preds = (torch.sigmoid(expert_logits) > 0.5).float()
        learner_preds = (torch.sigmoid(learner_logits) < 0.5).float()
        expert_accuracy = expert_preds.mean().item()
        learner_accuracy = learner_preds.mean().item()
        overall_accuracy = (expert_accuracy + learner_accuracy) / 2
        
        return {
            'discriminator_loss': discriminator_loss.item(),
            'expert_loss': expert_loss.item(),
            'learner_loss': learner_loss.item(),
            'expert_accuracy': expert_accuracy,
            'learner_accuracy': learner_accuracy,
            'overall_accuracy': overall_accuracy
        }
    
    def update_policy(self, learner_batch: Dict) -> Dict[str, float]:
        """Update policy using AIRL rewards."""
        states = learner_batch['states']
        actions_onehot = learner_batch['actions']
        next_states = learner_batch['next_states']
        dones = learner_batch['dones']
        
        # Get AIRL rewards
        with torch.no_grad():
            airl_rewards = self.discriminator.get_reward(states, actions_onehot)
        
        # Get policy predictions
        action_probs, state_values = self.policy(states)
        next_state_values = self.policy(next_states)[1]
        
        # Calculate advantages
        gamma = self.config.get('gamma', 0.99)
        targets = airl_rewards + gamma * next_state_values * (1 - dones)
        advantages = targets - state_values
        
        # Actor loss (policy gradient)
        action_indices = actions_onehot.argmax(dim=-1)
        log_probs = torch.log(action_probs.gather(1, action_indices.unsqueeze(1)) + 1e-8)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(state_values, targets.detach())
        
        # Combined loss
        policy_loss = actor_loss + 0.5 * critic_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_airl_reward': airl_rewards.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def log_metrics(self, metrics: Dict[str, float], iteration: int):
        """Log metrics to TensorBoard."""
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, iteration)
    
    def train(self):
        """Main training loop."""
        print("🚀 STARTING PYTORCH AIRL TRAINING")
        print("=" * 60)
        
        # Load expert trajectories
        num_expert_trajectories = self.expert_loader.load_trajectories()
        if num_expert_trajectories == 0:
            raise ValueError("No expert trajectories loaded!")
        
        print(f"📂 Loaded {num_expert_trajectories} expert trajectories")
        print(f"🎯 Device: {self.device}")
        print(f"📊 State dim: {self.state_dim}, Action dim: {self.action_dim}")
        
        # Training parameters
        batch_size = self.config.get('batch_size', 64)
        max_iterations = self.config.get('max_iterations', 1000)
        episodes_per_iteration = self.config.get('episodes_per_iteration', 5)
        updates_per_iteration = self.config.get('updates_per_iteration', 10)
        
        print(f"🔧 Batch size: {batch_size}, Max iterations: {max_iterations}")
        print(f"📈 Episodes per iteration: {episodes_per_iteration}")
        
        # Initial data collection
        print("\n📦 Collecting initial learner data...")
        self.collect_learner_data(episodes_per_iteration * 2)
        
        # Training loop
        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Collect more learner data
            transitions_collected = self.collect_learner_data(episodes_per_iteration)
            
            # Training updates
            all_metrics = []
            
            for update in range(updates_per_iteration):
                try:
                    # Get batches
                    expert_batch = self.expert_loader.get_batch(batch_size, self.device)
                    learner_batch = self.get_learner_batch(batch_size)
                    
                    # Update discriminator
                    disc_metrics = self.update_discriminator(expert_batch, learner_batch)
                    
                    # Update policy
                    policy_metrics = self.update_policy(learner_batch)
                    
                    # Combine metrics
                    metrics = {**disc_metrics, **policy_metrics}
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"   ⚠️  Update {update} failed: {e}")
                    continue
            
            # Average metrics and log
            if all_metrics:
                avg_metrics = {}
                for key in all_metrics[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in all_metrics if key in m])
                
                # Console logging
                print(f"   Transitions: {transitions_collected}")
                print(f"   D_loss: {avg_metrics.get('discriminator_loss', 0):.4f}")
                print(f"   P_loss: {avg_metrics.get('policy_loss', 0):.4f}")
                print(f"   D_acc: {avg_metrics.get('overall_accuracy', 0):.3f}")
                print(f"   AIRL_reward: {avg_metrics.get('mean_airl_reward', 0):.4f}")
                
                # TensorBoard logging
                self.log_metrics(avg_metrics, iteration)
                
                # Add iteration-specific metrics
                self.log_metrics({
                    'learner/transitions_collected': transitions_collected,
                    'learner/buffer_size': len(self.learner_buffer)
                }, iteration)
        
        # Save models
        self.save_models("pytorch_airl_final")
        
        if self.use_tensorboard:
            self.writer.close()
        
        print(f"\n✅ Training completed!")
        print(f"📊 Total steps: {self.step_count}")
        
    def save_models(self, prefix: str):
        """Save trained models."""
        os.makedirs("models", exist_ok=True)
        
        discriminator_path = f"models/{prefix}_discriminator.pth"
        policy_path = f"models/{prefix}_policy.pth"
        
        torch.save(self.discriminator.state_dict(), discriminator_path)
        torch.save(self.policy.state_dict(), policy_path)
        
        print(f"💾 Models saved: {discriminator_path}, {policy_path}")

def create_training_config(visualized: bool = True) -> Dict:
    """Create training configuration."""
    return {
        # Environment
        'expert_trajectory_dir': 'expert_trajectories_dqn_adapter',
        'headless': not visualized,
        'max_episode_steps': 500,
        
        # Network parameters
        'discriminator_lr': 3e-4,
        'policy_lr': 1e-4,
        'gamma': 0.99,
        'batch_size': 64,
        'buffer_size': 10000,
        
        # Training schedule
        'max_iterations': 100 if visualized else 1000,
        'episodes_per_iteration': 3 if visualized else 5,
        'updates_per_iteration': 5 if visualized else 10,
        
        # Logging
        'use_tensorboard': True,
        'use_cuda': True,
    }

def main_visualized():
    """Run visualized AIRL training."""
    print("🎮 VISUALIZED AIRL TRAINING")
    config = create_training_config(visualized=True)
    trainer = PyTorchAIRLTrainer(config)
    trainer.train()

def main_headless():
    """Run headless AIRL training."""
    print("⚡ HEADLESS AIRL TRAINING")
    config = create_training_config(visualized=False)
    trainer = PyTorchAIRLTrainer(config)
    trainer.train()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "headless":
        main_headless()
    else:
        main_visualized() 