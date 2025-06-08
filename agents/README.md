# Agents Directory

This directory contains agent implementations for different reinforcement learning algorithms in the Tetris environment.

## Structure

```
agents/
├── __init__.py          # Package initialization
├── base_agent.py        # Abstract base agent class
├── dqn_agent.py         # Deep Q-Network agent implementation
└── README.md           # This file
```

## Base Agent

The `BaseAgent` class provides a common interface for all RL agents:

- **Action Selection**: `select_action(observation, training=True)`
- **Learning**: `update(*args, **kwargs)`
- **Checkpointing**: `save_checkpoint(filepath)` and `load_checkpoint(filepath)`
- **Observation Processing**: `preprocess_observation(observation)`
- **Action Conversion**: `convert_action_to_tuple(action_idx)`

## DQN Agent

The `DQNAgent` implements Deep Q-Network learning with:

- **Epsilon-greedy exploration** with configurable decay
- **Target network** for stable learning
- **Experience replay** with batch updates
- **GPU support** with automatic device detection
- **Gradient clipping** for training stability

### Key Features

- **Flexible Architecture**: Configurable neural network parameters
- **Training Modes**: Supports both training and evaluation modes
- **Checkpoint Management**: Complete state saving/loading
- **Metrics Tracking**: Returns training metrics for monitoring

### Usage Example

```python
from agents.dqn_agent import DQNAgent

# Create agent
agent = DQNAgent(
    action_space_size=8,
    observation_space_shape=(1, 20, 10),
    device='cuda',
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=50000
)

# Select action
action = agent.select_action(observation, training=True)

# Update agent
metrics = agent.update(state, action, reward, next_state, done)

# Save checkpoint
agent.save_checkpoint('models/checkpoints/agent.pt')
```

## Integration with Training

Agents are designed to work seamlessly with the training system:

1. **Environment Compatibility**: Agents handle both direct and locked position action modes
2. **Observation Processing**: Automatic conversion from environment observations to neural network inputs
3. **Action Conversion**: Seamless conversion between scalar actions and tuple formats
4. **Training Metrics**: Comprehensive metrics for monitoring and logging

## Adding New Agents

To add a new agent:

1. **Inherit from BaseAgent**: Implement all abstract methods
2. **Update __init__.py**: Add import and export
3. **Follow Interface**: Maintain compatibility with training system
4. **Add Documentation**: Update this README with usage examples

## GPU Support

All agents support GPU acceleration:

- **Automatic Detection**: Uses CUDA if available, falls back to CPU
- **Memory Management**: Efficient GPU memory usage
- **Device Consistency**: Ensures all tensors are on the correct device

## Performance Considerations

- **Batch Processing**: Agents support both single and batch updates
- **Memory Efficiency**: Optimized for large replay buffers
- **Training Speed**: Designed for fast action selection during training 