"""
DREAM Main Training Algorithm

Implements the complete DREAM training pipeline with consistent tuple-based observations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from dream.models.world_model import WorldModel
from dream.models.actor_critic import ActorCritic
from dream.buffers.replay_buffer import ReplayBuffer, ImaginationBuffer
from dream.configs.dream_config import DREAMConfig
from envs.tetris_env import TetrisEnv


class DREAMTrainer:
    """DREAM training orchestrator with consistent tuple-based data flow"""
    
    def __init__(self,
                 config: DREAMConfig,
                 env: Optional[TetrisEnv] = None,
                 experiment_name: str = "dream_tetris"):
        self.config = config
        self.experiment_name = experiment_name
        self.device = torch.device(config.device)
        
        # Create environment if not provided
        if env is None:
            self.env = TetrisEnv(
                num_agents=config.num_agents,
                headless=config.headless,
                action_mode=config.action_mode
            )
        else:
            self.env = env
        
        # Initialize models
        self.world_model = WorldModel(**config.world_model).to(self.device)
        self.actor_critic = ActorCritic(**config.actor_critic).to(self.device)
        
        # Initialize optimizers
        self.world_model_optimizer = optim.Adam(
            self.world_model.parameters(), 
            lr=config.world_model_lr
        )
        self.actor_optimizer = optim.Adam(
            self.actor_critic.parameters(), 
            lr=config.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.actor_critic.parameters(), 
            lr=config.critic_lr
        )
        
        # Initialize buffers
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_size,
            sequence_length=config.sequence_length,
            device=self.device
        )
        self.imagination_buffer = ImaginationBuffer(
            capacity=config.imagination_size,
            sequence_length=config.imagination_horizon,
            device=self.device
        )
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_reward = float('-inf')
        
        # Logging
        self.setup_logging()
        
        # GPU optimization
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print(f"🚀 GPU optimization enabled on {torch.cuda.get_device_name()}")
    
    def setup_logging(self):
        """Setup logging directories"""
        self.log_dir = Path(self.config.log_dir) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.config.save_config(str(self.log_dir / "config.json"))
        print(f"📁 Logs saved to: {self.log_dir}")
    
    def collect_real_experience(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Collect real experience from environment"""
        observations = []
        actions = []
        rewards = []
        dones = []
        
        obs = self.env.reset()  # array of 206 elements
        
        # FIXED: Pad observation dimensions (206→212) 
        if isinstance(obs, np.ndarray) and obs.shape[0] == 206:
            obs = np.concatenate([obs, np.zeros(6)], axis=0)
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config.max_episode_length):
            # Process observation to tensor for actor-critic
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            
            # Get action from current policy
            with torch.no_grad():
                action, log_prob, value = self.actor_critic.get_action_and_value(obs_tensor.unsqueeze(0))
                action = action.squeeze(0).cpu().item()
            
            # Store transition
            observations.append(obs)  # Store raw tuple
            actions.append(action)    # Store scalar action
            
            # Take environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # FIXED: Pad next observation dimensions (206→212)
            if isinstance(next_obs, np.ndarray) and next_obs.shape[0] == 206:
                next_obs = np.concatenate([next_obs, np.zeros(6)], axis=0)
            
            rewards.append(reward)
            dones.append(done)
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
                
            obs = next_obs
        
        # Create trajectory with consistent format
        trajectory = {
            'observations': observations,  # List of tuples
            'actions': actions,           # List of scalars
            'rewards': rewards,           # List of scalars
            'dones': dones               # List of booleans
        }
        
        episode_stats = {
            'episode_reward': episode_reward,
            'episode_length': episode_length
        }
        
        return trajectory, episode_stats
    
    def train(self, num_episodes: int):
        """Main training loop"""
        print(f"🎯 Starting DREAM training for {num_episodes} episodes")
        print(f"📱 Device: {self.device}")
        print(f"🎮 Action mode: {self.config.action_mode}")
        print("=" * 60)
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # 1. Collect real experience
            trajectory, episode_stats = self.collect_real_experience()
            self.replay_buffer.add_trajectory(trajectory)
            
            episode_reward = episode_stats['episode_reward']
            episode_length = episode_stats['episode_length']
            
            # 2. Train world model
            world_model_losses = {'world_model_loss': 0.0}
            if len(self.replay_buffer) >= self.config.min_buffer_size:
                world_model_losses = self._train_world_model()
            
            # 3. Generate imagination trajectories
            imagination_stats = {'num_trajectories': 0}
            if len(self.replay_buffer) >= self.config.min_buffer_size:
                imagination_stats = self._generate_imagination()
            
            # 4. Train actor-critic policy
            policy_losses = {'actor_loss': 0.0, 'critic_loss': 0.0}
            if len(self.imagination_buffer) >= self.config.batch_size:
                policy_losses = self._train_actor_critic()
            
            # 5. Logging
            episode_time = time.time() - episode_start_time
            
            if episode % 10 == 0:
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Length: {episode_length:3d} | "
                      f"Time: {episode_time:5.2f}s | "
                      f"Buffer: {len(self.replay_buffer):5d}")
            
            # Update counters
            self.episode_count = episode
            self.global_step += episode_length
            
            # Update best reward
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
        
        print("=" * 60)
        print(f"✅ Training completed! Best reward: {self.best_reward:.2f}")

    def select_action(self, observation):
        """Select action using current policy"""
        obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action_and_value(obs_tensor.unsqueeze(0))
            return action.squeeze(0).cpu().item()

    def _train_world_model(self) -> Dict[str, float]:
        """Train world model on collected experience"""
        batch = self.replay_buffer.sample_sequences(self.config.batch_size)
        
        # Convert to tensors with consistent shapes
        observations = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in batch['observations']]).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.bool).to(self.device)
        
        # Forward pass through world model
        model_outputs = self.world_model(observations, actions)
        
        # Compute losses
        reward_loss = F.mse_loss(model_outputs['predicted_rewards'], rewards)
        continue_loss = F.binary_cross_entropy_with_logits(
            model_outputs['predicted_continues'], (~dones).float()
        )
        kl_loss = model_outputs['kl_loss'].mean()
        
        total_loss = reward_loss + continue_loss + self.config.kl_weight * kl_loss
        
        # Backward pass
        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip_norm)
        self.world_model_optimizer.step()
        
        return {
            'world_model_loss': total_loss.item(),
            'reward_loss': reward_loss.item(),
            'continue_loss': continue_loss.item(),
            'kl_loss': kl_loss.item()
        }

    def _generate_imagination(self) -> Dict[str, int]:
        """Generate imagination trajectories using world model"""
        batch = self.replay_buffer.sample_sequences(self.config.batch_size)
        
        # Get initial observations
        initial_observations = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in batch['observations'][:, 0]]).to(self.device)
        initial_states = self.world_model.get_initial_state(self.config.batch_size, self.device)
        
        # Generate imagined trajectory
        imagined_actions = []
        imagined_rewards = []
        imagined_continues = []
        
        states = initial_states
        observations = initial_observations
        
        for i in range(self.config.imagination_horizon):
            # Get actions from current policy
            with torch.no_grad():
                actions, _, _ = self.actor_critic.get_action_and_value(observations)
            
            # Imagine next step
            imagined_outputs = self.world_model.imagine(states, actions.unsqueeze(1))
            
            imagined_actions.append(actions)
            imagined_rewards.append(imagined_outputs['predicted_rewards'].squeeze(1))
            imagined_continues.append(imagined_outputs['predicted_continues'].squeeze(1))
            
            # Update for next step
            states = imagined_outputs['final_state']
            observations = imagined_outputs['predicted_observations'].squeeze(1)
        
        # Create imagined trajectory
        imagined_trajectory = {
            'observations': [obs.cpu().numpy() for obs in imagined_actions],  # Use actions as proxy observations
            'actions': [act.cpu().numpy() for act in imagined_actions],
            'rewards': [rew.cpu().numpy() for rew in imagined_rewards],
            'dones': [~cont.cpu().numpy() for cont in imagined_continues]
        }
        
        self.imagination_buffer.add_trajectory(imagined_trajectory)
        
        return {'num_trajectories': 1}

    def _train_actor_critic(self) -> Dict[str, float]:
        """Train actor-critic on imagination trajectories"""
        batch = self.imagination_buffer.sample_sequences(self.config.batch_size)
        
        # Convert to tensors
        observations = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in batch['observations']]).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
        continues = torch.tensor(batch['continues'], dtype=torch.float32).to(self.device)
        
        # Forward pass
        action_probs, values = self.actor_critic(observations)
        
        # Compute returns
        returns = self._compute_returns(rewards, continues, values)
        
        # Compute losses
        log_probs = torch.log(action_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
        advantages = returns - values.detach()
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        
        # Backward pass
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        actor_loss.backward(retain_graph=True)
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.grad_clip_norm)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }

    def _compute_returns(self, rewards: torch.Tensor, continues: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns"""
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)
        
        for b in range(batch_size):
            next_value = 0.0
            for t in reversed(range(seq_len)):
                returns[b, t] = rewards[b, t] + self.config.gamma * next_value * continues[b, t]
                next_value = values[b, t]
        
        return returns
