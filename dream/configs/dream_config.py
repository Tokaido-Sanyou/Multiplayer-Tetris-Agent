"""
DREAM Configuration System

Provides configuration management for DREAM training with 
predefined presets and customizable hyperparameters.
"""

import torch
from typing import Dict, Any


class DREAMConfig:
    """Configuration for DREAM training"""
    
    def __init__(self, action_mode: str = 'direct', config_name: str = 'default'):
        """
        Initialize DREAM configuration
        
        Args:
            action_mode: 'direct' (8 actions) or 'locked_position' (200 actions)
            config_name: Name of configuration preset
        """
        self.action_mode = action_mode
        self.config_name = config_name
        
        # Environment configuration
        self.max_episode_length = 1000
        self.num_agents = 1
        self.headless = True
        
        # Model architecture - Reduced for 500k parameter limit
        self.world_model = {
            'observation_dim': 206,  # FIXED: Corrected to match actual environment dimension
            'action_dim': 8 if action_mode == 'direct' else 200,
            'hidden_dim': 128,
            'state_dim': 16
        }
        
        self.actor_critic = {
            'state_dim': 206,  # FIXED: Updated to match environment dimension
            'action_dim': 8 if action_mode == 'direct' else 200,
            'hidden_dim': 256,  # Reduced from 400
            'action_mode': action_mode
        }
        
        # Training hyperparameters
        self.world_model_lr = 3e-4
        self.actor_lr = 8e-5
        self.critic_lr = 8e-5
        self.gamma = 0.99
        self.lambda_param = 0.95
        self.kl_beta = 1.0
        self.kl_weight = 1.0  # Alias for kl_beta
        self.free_nats = 3.0
        self.grad_clip_norm = 10.0
        
        # Buffer configuration
        self.buffer_size = 100000
        self.imagination_size = 50000
        self.min_buffer_size = 1000
        self.sequence_length = 50
        
        # Training schedule
        self.world_model_batches = 10
        self.actor_critic_batches = 5
        self.batch_size = 32
        self.imagination_batch_size = 16
        self.imagination_horizon = 15
        
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Logging configuration
        self.log_dir = 'logs/dream'
        self.save_frequency = 100
        self.eval_frequency = 50
        self.video_frequency = 200
    
    @classmethod
    def get_default_config(cls, action_mode: str = 'direct') -> 'DREAMConfig':
        """Get default DREAM configuration"""
        return cls(action_mode=action_mode, config_name='default')
    
    @classmethod 
    def get_fast_config(cls, action_mode: str = 'direct') -> 'DREAMConfig':
        """Get fast training configuration for testing"""
        config = cls(action_mode=action_mode, config_name='fast')
        config.buffer_size = 10000
        config.imagination_size = 5000
        config.min_buffer_size = 100
        config.sequence_length = 25
        config.imagination_horizon = 8
        config.batch_size = 16
        config.imagination_batch_size = 8
        config.world_model_batches = 5
        config.actor_critic_batches = 3
        
        # Even smaller model for fast config
        config.world_model['state_dim'] = 12
        config.world_model['rnn_hidden_dim'] = 48
        config.world_model['stochastic_dim'] = 12
        config.world_model['hidden_dim'] = 96
        config.actor_critic['hidden_dim'] = 128
        
        return config
    
    @classmethod
    def get_research_config(cls, action_mode: str = 'direct') -> 'DREAMConfig':
        """Get high-performance research configuration - still under 500k params"""
        config = cls(action_mode=action_mode, config_name='research')
        config.buffer_size = 500000
        config.imagination_size = 200000
        config.sequence_length = 100
        config.imagination_horizon = 25
        config.batch_size = 64
        config.imagination_batch_size = 32
        config.world_model_batches = 20
        config.actor_critic_batches = 10
        
        # Larger models but still constrained
        config.world_model['state_dim'] = 20
        config.world_model['rnn_hidden_dim'] = 80
        config.world_model['stochastic_dim'] = 20
        config.world_model['hidden_dim'] = 160
        config.actor_critic['hidden_dim'] = 320
        
        return config
    
    @classmethod
    def get_line_clearing_config(cls, action_mode: str = 'direct') -> 'DREAMConfig':
        """Get configuration optimized for line clearing"""
        config = cls(action_mode=action_mode, config_name='line_clearing')
        config.buffer_size = 100000
        config.imagination_size = 50000
        config.sequence_length = 100
        config.imagination_horizon = 50
        config.batch_size = 8
        config.imagination_batch_size = 4
        config.world_model_batches = 15
        config.actor_critic_batches = 10
        
        # Enhanced model for complex patterns but still under 500k
        config.world_model['state_dim'] = 16
        config.world_model['rnn_hidden_dim'] = 64
        config.world_model['stochastic_dim'] = 16
        config.world_model['hidden_dim'] = 128
        config.actor_critic['hidden_dim'] = 256
        
        # Enhanced learning rates for line clearing
        config.world_model_lr = 0.00005
        config.actor_lr = 0.0005
        config.critic_lr = 0.0003
        config.gamma = 0.998
        
        return config
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary"""
        return {
            'action_mode': self.action_mode,
            'config_name': self.config_name,
            'max_episode_length': self.max_episode_length,
            'world_model': self.world_model,
            'actor_critic': self.actor_critic,
            'world_model_lr': self.world_model_lr,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'gamma': self.gamma,
            'lambda_param': self.lambda_param,
            'buffer_size': self.buffer_size,
            'imagination_size': self.imagination_size,
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'imagination_batch_size': self.imagination_batch_size,
            'imagination_horizon': self.imagination_horizon,
            'device': self.device,
            'log_dir': self.log_dir
        }
    
    def save_config(self, filepath: str = None) -> None:
        """Save configuration to JSON file"""
        import json
        from pathlib import Path
        
        if filepath is None:
            filepath = f'{self.log_dir}/config_{self.config_name}.json'
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.get_full_config(), f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'DREAMConfig':
        """Load configuration from JSON file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls(
            action_mode=config_dict['action_mode'],
            config_name=config_dict['config_name']
        )
        
        # Update attributes from loaded config
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config 

def get_fast_config():
    """Fast training configuration for testing and debugging"""
    return {
        # Model architecture
        'state_dim': 12,
        'rnn_hidden_dim': 48,
        'stochastic_dim': 12,
        'hidden_dim': 96,
        'actor_critic_hidden_dim': 128,
        
        # Training hyperparameters - improved for line clearing
        'world_lr': 0.0001,  # Reduced from 0.0003 for more stable learning
        'actor_lr': 0.0003,   # Increased from 8e-05 for better policy learning
        'critic_lr': 0.0002,  # Increased from 8e-05
        'gamma': 0.995,       # Increased from 0.99 for longer-term planning
        
        # Exploration and imagination - enhanced
        'imagination_horizon': 30,  # Increased from 15 for better planning
        'free_nats': 3.0,
        'kl_scale': 1.0,
        'exploration_noise': 0.3,  # Added for better exploration
        
        # Buffer and batch settings
        'sequence_length': 50,  # Increased from 20 for longer sequences
        'batch_size': 16,
        'buffer_size': 50000,   # Increased buffer size
        
        # Environment settings for better rewards
        'reward_scale': 1.0,
        'line_clear_bonus': 100,  # Explicit bonus for line clearing
        'piece_placement_bonus': 10,  # Bonus for successful piece placement
        'height_penalty_scale': 0.1,  # Penalty for high stacks
        
        # Training schedule
        'world_model_train_freq': 1,
        'policy_train_freq': 1,
        'target_update_freq': 100,
        'log_freq': 10,
        'save_freq': 1000
    }

def get_line_clearing_config():
    """Specialized configuration optimized for line clearing"""
    return {
        # Model architecture - slightly larger for complex patterns
        'state_dim': 16,
        'rnn_hidden_dim': 64,
        'stochastic_dim': 16,
        'hidden_dim': 128,
        'actor_critic_hidden_dim': 256,
        
        # Training hyperparameters optimized for line clearing
        'world_lr': 0.00005,  # Very stable world model learning
        'actor_lr': 0.0005,   # Higher actor learning for policy improvement
        'critic_lr': 0.0003,  # Higher critic learning for value estimation
        'gamma': 0.998,       # Even longer-term planning
        
        # Enhanced exploration and planning
        'imagination_horizon': 50,  # Much longer planning horizon
        'free_nats': 2.0,
        'kl_scale': 0.5,      # Reduced KL constraint for more flexibility
        'exploration_noise': 0.5,  # Higher exploration
        
        # Larger buffers for complex patterns
        'sequence_length': 100,
        'batch_size': 8,      # Smaller batch size for better gradients
        'buffer_size': 100000,
        
        # Strong reward shaping for line clearing
        'reward_scale': 1.0,
        'line_clear_bonus': 200,  # Very high bonus for line clearing
        'piece_placement_bonus': 20,
        'height_penalty_scale': 0.2,
        'hole_penalty_scale': 0.5,  # Penalty for creating holes
        'line_clear_multiplier': [100, 300, 500, 800],  # Tetris-style scoring
        
        # Training schedule
        'world_model_train_freq': 1,
        'policy_train_freq': 2,  # More frequent policy updates
        'target_update_freq': 50,
        'log_freq': 5,
        'save_freq': 500
    } 