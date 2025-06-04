#!/usr/bin/env python3
"""
AIRL Training Script for Tetris
Adversarial Inverse Reinforcement Learning implementation
"""

import os
import sys
import torch
import numpy as np
import logging
import argparse
from datetime import datetime
import json
from typing import Dict, List

# Handle optional wandb import
try:
    import wandb
except ImportError:
    wandb = None

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Module-based imports with fallbacks
try:
    from localMultiplayerTetris.tetris_env import TetrisEnv
    from localMultiplayerTetris.rl_utils.airl_agent import AIRLAgent
    from localMultiplayerTetris.rl_utils.expert_loader import ExpertTrajectoryLoader
    from localMultiplayerTetris.rl_utils.actor_critic import ActorCritic
    from localMultiplayerTetris.rl_utils.replay_buffer import ReplayBuffer
except ImportError:
    # Add more paths to sys.path for imports
    grandparent_dir = os.path.dirname(os.path.dirname(parent_dir))
    if grandparent_dir not in sys.path:
        sys.path.append(grandparent_dir)
    
    try:
        from tetris_env import TetrisEnv
        from airl_agent import AIRLAgent
        from expert_loader import ExpertTrajectoryLoader
        from actor_critic import ActorCritic
        from replay_buffer import ReplayBuffer
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure you're running from the correct directory")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        sys.exit(1)

class AIRLTrainer:
    """
    Main AIRL training orchestrator.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize environment
        self.env = TetrisEnv(single_player=True, headless=True)
        
        # Get state and action dimensions
        sample_obs = self.env.reset()
        # Handle gym API version differences
        if isinstance(sample_obs, tuple):
            sample_obs = sample_obs[0]
        self.state_dim = self._extract_features(sample_obs).shape[0]  # 207 features
        self.action_dim = self.env.action_space.n  # 41 actions
        
        self.logger.info(f"State dimension: {self.state_dim}")
        self.logger.info(f"Action dimension: {self.action_dim}")
        
        # Initialize expert trajectory loader
        self.expert_loader = ExpertTrajectoryLoader(
            trajectory_dir=config['expert_trajectory_dir'],
            max_trajectories=config.get('max_expert_trajectories', None),
            min_episode_length=config.get('min_episode_length', 10),
            max_hold_percentage=config.get('max_hold_percentage', 50.0),  # Increased from 20% to 50%
            state_feature_extractor=self._extract_features
        )
        
        # Load expert trajectories
        num_loaded = self.expert_loader.load_trajectories()
        if num_loaded == 0:
            raise ValueError("No valid expert trajectories loaded!")
        
        # Print expert statistics
        expert_stats = self.expert_loader.get_statistics()
        self.logger.info(f"Expert data statistics: {expert_stats}")
        
        # Initialize policy network (Actor-Critic)
        self.policy = ActorCritic(
            input_dim=self.state_dim,
            output_dim=self.action_dim
        )
        
        # Initialize AIRL agent
        self.airl_agent = AIRLAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            policy_network=self.policy,
            lr_discriminator=config.get('discriminator_lr', 3e-4),
            lr_policy=config.get('policy_lr', 3e-4),
            discriminator_update_freq=config.get('discriminator_update_freq', 1),
            policy_update_freq=config.get('policy_update_freq', 1),
            gamma=config.get('gamma', 0.99),
            device=self.device
        )
        
        # Initialize learner replay buffer
        self.learner_buffer = ReplayBuffer(
            capacity=config.get('learner_buffer_size', 100000),
            alpha=config.get('per_alpha', 0.6),
            beta=config.get('per_beta', 0.4)
        )
        
        # Training state
        self.episode = 0
        self.step_count = 0
        self.best_score = -float('inf')
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'tetris-airl'),
                config=config,
                name=f"airl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('AIRLTrainer')
        self.logger.setLevel(log_level)
        
        # File handler
        fh = logging.FileHandler(f'logs/airl_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(log_level)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
    
    def _extract_features(self, observation: Dict) -> np.ndarray:
        """Extract features from TetrisEnv observation."""
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
    
    def _action_to_onehot(self, action: int) -> np.ndarray:
        """Convert action to one-hot encoding."""
        onehot = np.zeros(self.action_dim, dtype=np.float32)
        if 0 <= action < self.action_dim:
            onehot[action] = 1.0
        return onehot
    
    def collect_learner_data(self, num_episodes: int) -> int:
        """
        Collect learner trajectory data by running current policy.
        
        Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            Number of transitions collected
        """
        transitions_collected = 0
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            # Handle gym API version differences
            if isinstance(obs, tuple):
                obs = obs[0]
            state = self._extract_features(obs)
            done = False
            episode_transitions = []
            
            while not done:
                # Select action using current policy
                action = self.airl_agent.select_action(state, deterministic=False)
                
                # Execute action
                step_result = self.env.step(action)
                # Handle gym API version differences
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                    truncated = False
                else:
                    next_obs, reward, done, truncated, info = step_result
                
                # Combine done and truncated
                done = done or truncated
                next_state = self._extract_features(next_obs)
                
                # Store transition
                transition = {
                    'state': state.copy(),
                    'action': action,
                    'action_onehot': self._action_to_onehot(action),
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done
                }
                episode_transitions.append(transition)
                
                state = next_state
                self.step_count += 1
            
            # Add episode transitions to replay buffer
            for transition in episode_transitions:
                self.learner_buffer.push(**transition)
            
            transitions_collected += len(episode_transitions)
        
        return transitions_collected
    
    def get_learner_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from learner replay buffer."""
        if len(self.learner_buffer) < batch_size:
            raise ValueError(f"Not enough learner data: {len(self.learner_buffer)} < {batch_size}")
        
        batch = self.learner_buffer.sample(batch_size)
        
        return {
            'states': torch.FloatTensor(batch['state']).to(self.device),
            'actions': torch.FloatTensor([self._action_to_onehot(a) for a in batch['action']]).to(self.device),
            'rewards': torch.FloatTensor(batch['reward']).unsqueeze(1).to(self.device),
            'next_states': torch.FloatTensor(batch['next_state']).to(self.device),
            'dones': torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device)
        }
    
    def evaluate_policy(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate current policy performance."""
        scores = []
        episode_lengths = []
        lines_cleared_list = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            # Handle gym API version differences
            if isinstance(obs, tuple):
                obs = obs[0]
            state = self._extract_features(obs)
            done = False
            episode_length = 0
            total_reward = 0
            
            while not done:
                action = self.airl_agent.select_action(state, deterministic=True)
                step_result = self.env.step(action)
                # Handle gym API version differences
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                    truncated = False
                else:
                    next_obs, reward, done, truncated, info = step_result
                
                # Combine done and truncated
                done = done or truncated
                
                state = self._extract_features(next_obs)
                total_reward += reward
                episode_length += 1
            
            scores.append(total_reward)
            episode_lengths.append(episode_length)
            lines_cleared_list.append(info.get('lines_cleared', 0))
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_lines_cleared': np.mean(lines_cleared_list),
            'max_score': np.max(scores)
        }
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting AIRL training")
        
        # Training parameters
        episodes_per_iteration = self.config.get('episodes_per_iteration', 5)
        training_iterations = self.config.get('training_iterations', 1000)
        batch_size = self.config.get('batch_size', 64)
        eval_freq = self.config.get('eval_freq', 50)
        save_freq = self.config.get('save_freq', 100)
        
        # Initial learner data collection
        self.logger.info("Collecting initial learner data...")
        initial_transitions = self.collect_learner_data(episodes_per_iteration * 2)
        self.logger.info(f"Collected {initial_transitions} initial transitions")
        
        for iteration in range(training_iterations):
            # Collect more learner data
            learner_transitions = self.collect_learner_data(episodes_per_iteration)
            
            # Training phase - multiple updates per iteration
            training_metrics = []
            num_updates = self.config.get('updates_per_iteration', 10)
            
            for _ in range(num_updates):
                try:
                    # Get expert and learner batches
                    expert_batch = self.expert_loader.get_batch(batch_size, self.device)
                    learner_batch = self.get_learner_batch(batch_size)
                    
                    # AIRL training step
                    metrics = self.airl_agent.train_step(expert_batch, learner_batch)
                    training_metrics.append(metrics)
                    
                except Exception as e:
                    self.logger.error(f"Training error: {e}")
                    continue
            
            # Aggregate training metrics
            if training_metrics:
                avg_metrics = {}
                for key in training_metrics[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in training_metrics if key in m])
                
                # Log training metrics
                log_msg = f"Iteration {iteration:4d}: "
                log_msg += f"D_loss={avg_metrics.get('discriminator_loss', 0):.4f} "
                log_msg += f"P_loss={avg_metrics.get('policy_loss', 0):.4f} "
                log_msg += f"D_acc={avg_metrics.get('overall_accuracy', 0):.3f} "
                log_msg += f"AIRL_r={avg_metrics.get('mean_airl_reward', 0):.4f}"
                self.logger.info(log_msg)
                
                # Log to wandb
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'iteration': iteration,
                        'learner_transitions': learner_transitions,
                        **avg_metrics
                    })
            
            # Evaluation
            if iteration % eval_freq == 0:
                eval_metrics = self.evaluate_policy()
                eval_msg = f"Eval {iteration:4d}: "
                eval_msg += f"Score={eval_metrics['mean_score']:.2f}±{eval_metrics['std_score']:.2f} "
                eval_msg += f"Length={eval_metrics['mean_episode_length']:.1f} "
                eval_msg += f"Lines={eval_metrics['mean_lines_cleared']:.1f}"
                self.logger.info(eval_msg)
                
                # Save best model
                if eval_metrics['mean_score'] > self.best_score:
                    self.best_score = eval_metrics['mean_score']
                    self.save_checkpoint(f"checkpoints/airl_best_{iteration}.pt")
                    self.logger.info(f"New best score: {self.best_score:.2f}")
                
                # Log eval metrics to wandb
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'eval/mean_score': eval_metrics['mean_score'],
                        'eval/mean_episode_length': eval_metrics['mean_episode_length'],
                        'eval/mean_lines_cleared': eval_metrics['mean_lines_cleared'],
                        'iteration': iteration
                    })
            
            # Periodic save
            if iteration % save_freq == 0:
                self.save_checkpoint(f"checkpoints/airl_{iteration}.pt")
        
        self.logger.info("Training completed")
        
        # Final evaluation
        final_eval = self.evaluate_policy(num_episodes=20)
        self.logger.info(f"Final evaluation: {final_eval}")
        
        # Save final model
        self.save_checkpoint("checkpoints/airl_final.pt")
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'airl_agent_state': {
                'discriminator_state_dict': self.airl_agent.discriminator.state_dict(),
                'policy_state_dict': self.airl_agent.policy.state_dict(),
                'discriminator_optimizer_state_dict': self.airl_agent.discriminator_optimizer.state_dict(),
                'policy_optimizer_state_dict': self.airl_agent.policy_optimizer.state_dict(),
                'update_count': self.airl_agent.update_count
            },
            'training_state': {
                'episode': self.episode,
                'step_count': self.step_count,
                'best_score': self.best_score
            },
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")

def create_config() -> Dict:
    """Create default training configuration."""
    return {
        # Environment
        'expert_trajectory_dir': '../../../expert_trajectories',
        'max_expert_trajectories': None,
        'min_episode_length': 10,
        'max_hold_percentage': 50.0,
        
        # Network architecture
        'policy_hidden_dim': 256,
        'discriminator_hidden_sizes': [256, 128],
        
        # Training hyperparameters
        'policy_lr': 3e-4,
        'discriminator_lr': 3e-4,
        'gamma': 0.99,
        'batch_size': 64,
        'learner_buffer_size': 100000,
        
        # Training schedule
        'training_iterations': 1000,
        'episodes_per_iteration': 5,
        'updates_per_iteration': 10,
        'discriminator_update_freq': 1,
        'policy_update_freq': 1,
        
        # Evaluation and logging
        'eval_freq': 50,
        'save_freq': 100,
        'log_level': 'INFO',
        'use_wandb': False,
        'wandb_project': 'tetris-airl',
        
        # Hardware
        'use_cuda': True,
        
        # Prioritized Experience Replay
        'per_alpha': 0.6,
        'per_beta': 0.4
    }

def main():
    parser = argparse.ArgumentParser(description='AIRL Training for Tetris')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--expert-dir', type=str, default='../../../expert_trajectories',
                       help='Directory containing expert trajectories')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_config()
    
    # Override config with command line arguments
    if args.expert_dir:
        config['expert_trajectory_dir'] = args.expert_dir
    if args.iterations:
        config['training_iterations'] = args.iterations
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.use_wandb:
        config['use_wandb'] = True
    if args.device != 'auto':
        config['use_cuda'] = args.device == 'cuda'
    
    # Create trainer and start training
    trainer = AIRLTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 