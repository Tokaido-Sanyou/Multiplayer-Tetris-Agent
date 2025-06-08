# Configuration System

This directory contains the configuration management system for DQN training in the Tetris environment.

## Overview

The configuration system provides centralized management of training parameters, model settings, and environment configurations. It supports multiple predefined configurations and allows for easy customization and extension.

## Files

### `dqn_config.py`
Main configuration management module containing:
- `DQNConfig` class for configuration management
- Predefined configuration functions
- Configuration registry system
- JSON serialization/deserialization

### `example_config.json`
Example configuration file showing the JSON format for saving/loading configurations.

## Usage

### Basic Usage

```python
from configs.dqn_config import get_config, DQNConfig

# Load a predefined configuration
config = get_config('default')

# Create custom configuration
custom_config = DQNConfig('my_experiment')
custom_config.training['learning_rate'] = 0.001
custom_config.training['max_episodes'] = 1000

# Save configuration
config.save_config('my_config.json')

# Load configuration
config = DQNConfig('loaded_config')
config.load_config('my_config.json')
```

### Command Line Usage

```bash
# List available configurations
python training/train_dqn.py --list-configs

# Use predefined configuration
python training/train_dqn.py --config default

# Override specific parameters
python training/train_dqn.py --config fast --episodes 500 --learning-rate 0.0005
```

## Predefined Configurations

### `default`
Standard training configuration suitable for most experiments.
- **Episodes**: 5,000
- **Memory Size**: 100,000
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Action Mode**: direct

### `fast`
Quick training configuration for testing and development.
- **Episodes**: 1,000
- **Memory Size**: 10,000
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Action Mode**: direct

### `research`
High-performance configuration for research experiments.
- **Episodes**: 20,000
- **Memory Size**: 500,000
- **Batch Size**: 64
- **Learning Rate**: 0.00005
- **Action Mode**: direct

### `locked_position`
Optimized configuration for locked position action mode.
- **Episodes**: 5,000
- **Memory Size**: 100,000
- **Batch Size**: 64
- **Learning Rate**: 0.0005
- **Action Mode**: locked_position
- **Epsilon Decay**: 30,000 steps (faster exploration decay)

### `debug`
Minimal configuration for debugging and quick tests.
- **Episodes**: 10
- **Memory Size**: 1,000
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Action Mode**: direct
- **Log Level**: DEBUG

## Configuration Structure

Each configuration contains the following sections:

### Model Configuration
```python
model = {
    'output_size': 8,           # Number of actions
    'activation_type': 'identity',  # Output activation
    'use_dropout': True,        # Enable dropout
    'dropout_rate': 0.1         # Dropout probability
}
```

### Training Configuration
```python
training = {
    'learning_rate': 0.0001,    # Adam learning rate
    'gamma': 0.99,              # Discount factor
    'epsilon_start': 1.0,       # Initial exploration
    'epsilon_end': 0.01,        # Final exploration
    'epsilon_decay_steps': 50000,  # Exploration decay
    'batch_size': 32,           # Training batch size
    'target_update_freq': 1000, # Target network update
    'memory_size': 100000,      # Replay buffer size
    'min_memory_size': 1000,    # Min samples before training
    'max_episodes': 5000,       # Maximum episodes
    'max_steps_per_episode': 2000,  # Max steps per episode
    'save_freq': 500,           # Checkpoint frequency
    'eval_freq': 100,           # Evaluation frequency
    'prioritized_replay': True, # Use prioritized replay
    'double_dqn': True,         # Use Double DQN
    'device': 'cuda'            # Training device
}
```

### Environment Configuration
```python
environment = {
    'num_agents': 1,            # Number of agents
    'headless': True,           # Headless mode
    'action_mode': 'direct',    # Action mode
    'step_mode': 'action',      # Step mode
    'enable_trajectory_tracking': True  # Trajectory tracking
}
```

### Logging Configuration
```python
logging = {
    'log_level': 'INFO',        # Logging level
    'console_output': True,     # Console logging
    'file_output': True,        # File logging
    'json_output': True,        # JSON structured logging
    'log_dir': 'logs/config_name',  # Log directory
    'tensorboard_logging': False,   # TensorBoard logging
    'video_logging': False      # Video logging
}
```

### Paths Configuration
```python
paths = {
    'models_dir': 'models',     # Models directory
    'checkpoints_dir': 'models/checkpoints/config_name',  # Checkpoints
    'results_dir': 'results/config_name',     # Results
    'data_dir': 'data/config_name',           # Data
    'logs_dir': 'logs/config_name'            # Logs
}
```

## Creating Custom Configurations

### Method 1: Modify Existing Configuration
```python
config = get_config('default')
config.training['learning_rate'] = 0.0005
config.training['batch_size'] = 64
config.environment['action_mode'] = 'locked_position'
```

### Method 2: Create from Dictionary
```python
custom_config = {
    'training': {
        'learning_rate': 0.001,
        'max_episodes': 2000
    },
    'environment': {
        'action_mode': 'locked_position'
    }
}

config = DQNConfig.from_dict(custom_config, 'my_experiment')
```

### Method 3: Create New Configuration Function
```python
def get_my_config() -> DQNConfig:
    config = DQNConfig("my_config")
    config.training.update({
        'learning_rate': 0.002,
        'max_episodes': 3000,
        'batch_size': 128
    })
    return config

# Register in CONFIGS dictionary
CONFIGS['my_config'] = get_my_config
```

## Directory Structure

When a configuration is used, it automatically creates the following directory structure:

```
├── models/checkpoints/[config_name]/  # Model checkpoints
├── results/[config_name]/             # Training results
├── logs/[config_name]/                # Training logs
├── data/[config_name]/                # Training data
└── videos/[config_name]/              # Training videos (if enabled)
```

## Best Practices

1. **Use Descriptive Names**: Choose configuration names that clearly indicate their purpose
2. **Document Changes**: When creating custom configurations, document the rationale
3. **Version Control**: Save important configurations as JSON files for reproducibility
4. **Test First**: Use the `debug` configuration to test changes before full training
5. **Monitor Resources**: Adjust memory and batch sizes based on available hardware

## Integration with Training

The configuration system integrates seamlessly with the training pipeline:

```python
# In training script
config = get_config(args.config)
trainer = DQNTrainer(
    env=TetrisEnv(**config.environment),
    model_config=config.model,
    training_config=config.training,
    experiment_name=config.config_name
)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `memory_size` and `batch_size`
2. **Slow Training**: Increase `batch_size` or reduce `memory_size`
3. **Poor Performance**: Adjust `learning_rate` and `epsilon_decay_steps`
4. **Directory Errors**: Ensure write permissions for output directories

### Configuration Validation

The system automatically validates configurations and provides helpful error messages for common issues like:
- Invalid parameter types
- Missing required parameters
- Incompatible parameter combinations
- Resource constraints

## Extension Points

The configuration system is designed for easy extension:

1. **New Parameters**: Add new parameters to any configuration section
2. **New Configurations**: Create new predefined configurations
3. **New Sections**: Add new configuration sections for new features
4. **Custom Validation**: Add validation logic for new parameters

This configuration system provides a robust foundation for managing complex training experiments while maintaining flexibility and ease of use. 