"""
Deep Q-Network (DQN) Trainer for Tetris
Implements DQN with experience replay, target networks, and GPU support
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
import time
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.tetris_env import TetrisEnv
from models.tetris_cnn import TetrisCNN
from utils.logger import setup_training_logger, MetricsTracker
from utils.video_logger import TrainingVideoLogger

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer with prioritized sampling support"""
    
    def __init__(self, capacity: int = 100000, prioritized: bool = False):
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity) if prioritized else None
        self.alpha = 0.6  # Prioritization exponent
        self.beta = 0.4   # Importance sampling exponent
        self.epsilon = 1e-6  # Small constant to prevent zero priorities
        
    def push(self, state, action, reward, next_state, done, priority: float = None):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
        if self.prioritized:
            if priority is None:
                # Default to maximum priority for new experiences
                priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], List[float]]:
        """Sample batch of experiences"""
        if self.prioritized:
            return self._sample_prioritized(batch_size)
        else:
            return self._sample_uniform(batch_size)
    
    def _sample_uniform(self, batch_size: int) -> Tuple[List[Experience], List[int], List[float]]:
        """Uniform random sampling"""
        indices = random.sample(range(len(self.buffer)), batch_size)
        experiences = [self.buffer[i] for i in indices]
        weights = [1.0] * batch_size  # Equal weights for uniform sampling
        return experiences, indices, weights
    
    def _sample_prioritized(self, batch_size: int) -> Tuple[List[Experience], List[int], List[float]]:
        """Prioritized sampling based on TD error"""
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize by maximum weight
        
        return experiences, indices.tolist(), weights.tolist()
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for given indices"""
        if not self.prioritized:
            return
        
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority + self.epsilon
    
    def __len__(self):
        return len(self.buffer)

class DQNTrainer:
    """Deep Q-Network trainer for Tetris"""
    
    def __init__(self, 
                 env: TetrisEnv,
                 model_config: Dict[str, Any] = None,
                 training_config: Dict[str, Any] = None,
                 experiment_name: str = "dqn_tetris"):
        
        self.env = env
        self.experiment_name = experiment_name
        
        # Default model configuration
        default_model_config = {
            'output_size': 8,  # 8 possible actions
            'activation_type': 'identity',  # For DQN Q-values
            'use_dropout': True,
            'dropout_rate': 0.1
        }
        self.model_config = {**default_model_config, **(model_config or {})}
        
        # Default training configuration
        default_training_config = {
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay_steps': 50000,
            'batch_size': 32,
            'target_update_freq': 1000,
            'memory_size': 100000,
            'min_memory_size': 10000,
            'max_episodes': 10000,
            'max_steps_per_episode': 2000,
            'save_freq': 500,
            'eval_freq': 100,
            'prioritized_replay': True,
            'double_dqn': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        self.config = {**default_training_config, **(training_config or {})}
        
        # Setup device
        self.device = torch.device(self.config['device'])
        print(f"🎮 Using device: {self.device}")
        
        # Initialize networks
        self.q_network = TetrisCNN(**self.model_config).to(self.device)
        self.target_network = TetrisCNN(**self.model_config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(
            capacity=self.config['memory_size'],
            prioritized=self.config['prioritized_replay']
        )
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.epsilon = self.config['epsilon_start']
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker(window_size=100)
        
        # Initialize logger
        full_config = {**self.model_config, **self.config}
        self.logger = setup_training_logger(experiment_name, full_config)
        
        # Log GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_info = {
                'total': torch.cuda.get_device_properties(0).total_memory,
                'allocated': torch.cuda.memory_allocated(0),
                'cached': torch.cuda.memory_reserved(0)
            }
            self.logger.log_gpu_info(True, gpu_name, memory_info)
        else:
            self.logger.log_gpu_info(False)
        
        # Log hyperparameters
        self.logger.log_hyperparameters(full_config)
        
        # Create directories
        self.checkpoint_dir = f"models/checkpoints/{experiment_name}"
        self.results_dir = f"results/{experiment_name}"
        self.video_dir = f"videos/{experiment_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        
        # Initialize video logger
        self.video_logger = TrainingVideoLogger(
            output_dir=self.video_dir,
            record_frequency=self.config.get('video_frequency', 100),  # Record every 100 episodes by default
            record_best=True,
            record_evaluations=True
        )
    
    def preprocess_observation(self, obs) -> torch.Tensor:
        """Convert observation to tensor for neural network"""
        if isinstance(obs, (list, tuple)):
            # Binary tuple observation - convert to numpy array
            obs_array = np.array(obs, dtype=np.float32)
        else:
            obs_array = obs
        
        # Reshape to board format (20x10) + piece info
        board = obs_array[:200].reshape(20, 10)
        
        # Add batch and channel dimensions
        board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)  # [1, 1, 20, 10]
        
        return board_tensor.to(self.device)
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action
            return random.randint(0, 7)
        
        # Greedy action
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def update_epsilon(self):
        """Update epsilon for exploration"""
        decay_rate = (self.config['epsilon_start'] - self.config['epsilon_end']) / self.config['epsilon_decay_steps']
        self.epsilon = max(self.config['epsilon_end'], self.epsilon - decay_rate)
    
    def convert_action_to_tuple(self, action_idx: int) -> List[int]:
        """Convert action index to binary tuple format"""
        action_tuple = [0] * 8
        if 0 <= action_idx < 8:
            action_tuple[action_idx] = 1
        return action_tuple
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.config['min_memory_size']:
            return None
        
        # Sample batch from memory
        experiences, indices, weights = self.memory.sample(self.config['batch_size'])
        
        # Prepare batch tensors
        states = torch.cat([self.preprocess_observation(e.state) for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.cat([self.preprocess_observation(e.next_state) for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        with torch.no_grad():
            if self.config['double_dqn']:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0]
            
            target_q_values = rewards + (self.config['gamma'] * next_q_values * ~dones)
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        loss = (weights_tensor * (td_errors ** 2)).mean()
        
        # Update priorities if using prioritized replay
        if self.config['prioritized_replay']:
            priorities = torch.abs(td_errors).detach().cpu().numpy().tolist()
            self.memory.update_priorities(indices, priorities)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.debug("🎯 Target network updated")
    
    def save_checkpoint(self, episode: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'total_steps': self.total_steps,
            'model_state_dict': self.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config,
            'metrics': metrics
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"dqn_checkpoint_ep{episode}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if 'mean_reward' in metrics:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            if not os.path.exists(best_path):
                torch.save(checkpoint, best_path)
                self.logger.info(f"💾 First model saved as best: {metrics['mean_reward']:.3f}")
            else:
                best_checkpoint = torch.load(best_path)
                if metrics['mean_reward'] > best_checkpoint['metrics'].get('mean_reward', -float('inf')):
                    torch.save(checkpoint, best_path)
                    self.logger.info(f"🏆 New best model: {metrics['mean_reward']:.3f}")
        
        self.logger.log_model_checkpoint(episode, checkpoint_path, metrics)
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current policy"""
        self.logger.info(f"🧪 Starting evaluation ({num_episodes} episodes)")
        
        eval_rewards = []
        eval_steps = []
        
        for ep in range(num_episodes):
            obs = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < self.config['max_steps_per_episode']:
                state = self.preprocess_observation(obs)
                action_idx = self.select_action(state, training=False)
                action = self.convert_action_to_tuple(action_idx)
                
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
            
            eval_rewards.append(total_reward)
            eval_steps.append(steps)
        
        metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_steps': np.mean(eval_steps),
            'max_reward': np.max(eval_rewards),
            'min_reward': np.min(eval_rewards)
        }
        
        self.logger.log_performance_metrics(metrics)
        return metrics
    
    def record_gameplay_video(self, episode_id: str, metadata: dict = None):
        """Record a video of the agent playing"""
        try:
            # Create a visual environment for recording
            visual_env = TetrisEnv(
                num_agents=1,
                headless=False,  # Enable rendering for video capture
                action_mode=self.env.action_mode,
                step_mode=self.env.step_mode
            )
            
            video_path, episode_data = self.video_logger.record_episode(
                episode_id=episode_id,
                env=visual_env,
                agent=self,
                max_steps=500,  # Limit episode length for videos
                metadata=metadata
            )
            
            if video_path:
                self.logger.info(f"Video recorded: {video_path}")
            
            visual_env.close()
            
        except Exception as e:
            self.logger.error(f"Error recording video: {e}")
            # Don't fail training if video recording fails
    
    def train(self):
        """Main training loop"""
        self.logger.info("🚀 Starting DQN training")
        
        try:
            while self.episode < self.config['max_episodes']:
                # Start episode
                obs = self.env.reset()
                episode_reward = 0
                episode_steps = 0
                episode_losses = []
                
                self.logger.log_episode_start(self.episode)
                
                done = False
                while not done and episode_steps < self.config['max_steps_per_episode']:
                    # Select and execute action
                    state = self.preprocess_observation(obs)
                    action_idx = self.select_action(state, training=True)
                    action = self.convert_action_to_tuple(action_idx)
                    
                    next_obs, reward, done, info = self.env.step(action)
                    
                    # Store experience
                    self.memory.push(obs, action_idx, reward, next_obs, done)
                    
                    # Train if enough samples
                    if len(self.memory) >= self.config['min_memory_size']:
                        loss = self.train_step()
                        if loss is not None:
                            episode_losses.append(loss)
                    
                    # Update state
                    obs = next_obs
                    episode_reward += reward
                    episode_steps += 1
                    self.total_steps += 1
                    
                    # Update epsilon
                    self.update_epsilon()
                    
                    # Update target network
                    if self.total_steps % self.config['target_update_freq'] == 0:
                        self.update_target_network()
                    
                    # Log step
                    if episode_steps % 100 == 0:
                        self.logger.log_step(self.episode, episode_steps, reward, action_idx)
                
                # Episode finished
                self.episode += 1
                
                # Track metrics
                self.metrics_tracker.add_metric('reward', episode_reward)
                self.metrics_tracker.add_metric('steps', episode_steps)
                self.metrics_tracker.add_metric('epsilon', self.epsilon)
                
                if episode_losses:
                    self.metrics_tracker.add_metric('loss', np.mean(episode_losses))
                
                # Log episode end
                episode_info = {
                    'epsilon': self.epsilon,
                    'memory_size': len(self.memory),
                    'avg_loss': np.mean(episode_losses) if episode_losses else 0.0
                }
                self.logger.log_episode_end(self.episode, episode_reward, episode_steps, **episode_info)
                
                # Evaluation
                if self.episode % self.config['eval_freq'] == 0:
                    eval_metrics = self.evaluate()
                    
                    # Save checkpoint
                    if self.episode % self.config['save_freq'] == 0:
                        combined_metrics = {**eval_metrics, **self.metrics_tracker.get_metrics()}
                        self.save_checkpoint(self.episode, combined_metrics)
                    
                    # Record video if needed
                    if self.video_logger.should_record_episode(self.episode, eval_metrics.get('mean_reward'), is_evaluation=True):
                        self.record_gameplay_video(f"eval_ep_{self.episode}", eval_metrics)
                
                # Print progress
                if self.episode % 10 == 0:
                    metrics = self.metrics_tracker.get_metrics()
                    if metrics:
                        recent_metrics = {k: v['mean'] for k, v in metrics.items()}
                        self.logger.log_performance_metrics(recent_metrics)
        
        except KeyboardInterrupt:
            self.logger.warning("⚠️ Training interrupted by user")
        
        except Exception as e:
            self.logger.log_error_with_context(e, {
                'episode': self.episode,
                'total_steps': self.total_steps,
                'memory_size': len(self.memory)
            })
            raise
        
        finally:
            # Final save
            self.logger.info("Training completed")
            # Don't close logger here - let the training script handle final evaluation

def main():
    """Main training script"""
    # Configuration
    model_config = {
        'output_size': 8,
        'activation_type': 'identity',
        'use_dropout': True,
        'dropout_rate': 0.1
    }
    
    training_config = {
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_steps': 50000,
        'batch_size': 32,
        'target_update_freq': 1000,
        'memory_size': 100000,
        'min_memory_size': 1000,
        'max_episodes': 5000,
        'max_steps_per_episode': 2000,
        'save_freq': 500,
        'eval_freq': 100,
        'prioritized_replay': True,
        'double_dqn': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create environment
    env = TetrisEnv(
        num_agents=1,
        headless=True,
        action_mode='direct',
        step_mode='action'
    )
    
    # Create trainer
    trainer = DQNTrainer(
        env=env,
        model_config=model_config,
        training_config=training_config,
        experiment_name="dqn_tetris_v1"
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 