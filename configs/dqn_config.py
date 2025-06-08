"""
Configuration file for DQN training
Centralized configuration management for hyperparameters and training settings
"""

import os
import torch
from typing import Dict, Any

class DQNConfig:
    """Configuration class for DQN training"""
    
    def __init__(self, config_name: str = "default"):
        self.config_name = config_name
        
        # Model Configuration
        self.model = {
            'output_size': 8,
            'activation_type': 'identity',
            'use_dropout': True,
            'dropout_rate': 0.1
        }
        
        # Training Configuration  
        self.training = {
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
        
        # Environment Configuration
        self.environment = {
            'num_agents': 1,
            'headless': True,
            'action_mode': 'direct',
            'step_mode': 'action',
            'enable_trajectory_tracking': True
        }
        
        # Logging Configuration
        self.logging = {
            'log_level': 'INFO',
            'console_output': True,
            'file_output': True,
            'json_output': True,
            'log_dir': f'logs/{config_name}',
            'tensorboard_logging': False,
            'video_logging': False
        }
        
        # Paths Configuration
        self.paths = {
            'models_dir': 'models',
            'checkpoints_dir': f'models/checkpoints/{config_name}',
            'results_dir': f'results/{config_name}',
            'data_dir': f'data/{config_name}',
            'logs_dir': f'logs/{config_name}'
        }
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary"""
        return {
            'config_name': self.config_name,
            'model': self.model,
            'training': self.training,
            'environment': self.environment,
            'logging': self.logging,
            'paths': self.paths
        }
    
    def save_config(self, filepath: str = None):
        """Save configuration to file"""
        if filepath is None:
            filepath = os.path.join(self.paths['logs_dir'], 'config.json')
        
        import json
        with open(filepath, 'w') as f:
            json.dump(self.get_full_config(), f, indent=2)
    
    def load_config(self, filepath: str):
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.config_name = config.get('config_name', self.config_name)
        self.model.update(config.get('model', {}))
        self.training.update(config.get('training', {}))
        self.environment.update(config.get('environment', {}))
        self.logging.update(config.get('logging', {}))
        self.paths.update(config.get('paths', {}))
        
        self._create_directories()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], config_name: str = "custom"):
        """Create config from dictionary"""
        config = cls(config_name)
        
        if 'model' in config_dict:
            config.model.update(config_dict['model'])
        if 'training' in config_dict:
            config.training.update(config_dict['training'])
        if 'environment' in config_dict:
            config.environment.update(config_dict['environment'])
        if 'logging' in config_dict:
            config.logging.update(config_dict['logging'])
        if 'paths' in config_dict:
            config.paths.update(config_dict['paths'])
        
        config._create_directories()
        return config

# Predefined configurations

def get_default_config() -> DQNConfig:
    """Get default DQN configuration"""
    return DQNConfig("dqn_default")

def get_fast_config() -> DQNConfig:
    """Get configuration for fast training/testing"""
    config = DQNConfig("dqn_fast")
    config.training.update({
        'max_episodes': 1000,
        'memory_size': 10000,
        'min_memory_size': 100,
        'eval_freq': 50,
        'save_freq': 200,
        'epsilon_decay_steps': 5000
    })
    return config

def get_research_config() -> DQNConfig:
    """Get configuration for research/high-performance training"""
    config = DQNConfig("dqn_research")
    config.training.update({
        'max_episodes': 20000,
        'memory_size': 500000,
        'min_memory_size': 10000,
        'batch_size': 64,
        'learning_rate': 0.00005,
        'epsilon_decay_steps': 100000,
        'eval_freq': 500,
        'save_freq': 1000
    })
    return config

def get_locked_position_config() -> DQNConfig:
    """Get configuration for locked position action mode"""
    config = DQNConfig("dqn_locked_position")
    config.environment.update({
        'action_mode': 'locked_position'
    })
    config.training.update({
        'epsilon_decay_steps': 30000,  # Faster exploration decay for position-based actions
        'batch_size': 64,
        'learning_rate': 0.0005
    })
    return config

def get_debug_config() -> DQNConfig:
    """Get configuration for debugging"""
    config = DQNConfig("dqn_debug")
    config.training.update({
        'max_episodes': 10,
        'memory_size': 1000,
        'min_memory_size': 10,
        'eval_freq': 2,
        'save_freq': 5,
        'epsilon_decay_steps': 100
    })
    config.logging.update({
        'log_level': 'DEBUG'
    })
    return config

# Configuration registry
CONFIGS = {
    'default': get_default_config,
    'fast': get_fast_config,
    'research': get_research_config,
    'locked_position': get_locked_position_config,
    'debug': get_debug_config
}

def get_config(name: str) -> DQNConfig:
    """Get configuration by name"""
    if name in CONFIGS:
        return CONFIGS[name]()
    else:
        raise ValueError(f"Unknown config name: {name}. Available: {list(CONFIGS.keys())}")

if __name__ == "__main__":
    # Test configurations
    print("Available configurations:")
    for name in CONFIGS.keys():
        config = get_config(name)
        print(f"  {name}: {config.training['max_episodes']} episodes, "
              f"{config.training['memory_size']} memory, "
              f"action_mode={config.environment['action_mode']}")
    
    # Save example config
    config = get_default_config()
    config.save_config("configs/example_config.json")
    print("\nExample config saved to configs/example_config.json") 