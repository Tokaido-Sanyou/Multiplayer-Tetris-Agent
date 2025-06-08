# Training System

This directory contains the training scripts and utilities for DQN training in the Tetris environment.

## Overview

The training system provides a command-line interface for training DQN agents with configurable parameters, automatic result saving, and comprehensive logging. It supports both direct and locked position action modes with GPU acceleration.

## Files

### `train_dqn.py`
Main training script with command-line interface for DQN training.

## Usage

### Basic Training

```bash
# Train with default configuration
python training/train_dqn.py

# Train with specific configuration
python training/train_dqn.py --config fast

# Train with parameter overrides
python training/train_dqn.py --config debug --episodes 50 --learning-rate 0.001
```

### Command Line Options

```bash
python training/train_dqn.py [OPTIONS]

Options:
  --config CONFIG_NAME          Configuration to use (default: 'default')
  --episodes EPISODES           Number of training episodes
  --learning-rate RATE          Learning rate for optimizer
  --batch-size SIZE             Training batch size
  --memory-size SIZE            Replay buffer size
  --epsilon-start VALUE         Initial exploration rate
  --epsilon-end VALUE           Final exploration rate
  --epsilon-decay STEPS         Exploration decay steps
  --gamma VALUE                 Discount factor
  --target-update-freq FREQ     Target network update frequency
  --save-freq FREQ              Model save frequency
  --eval-freq FREQ              Evaluation frequency
  --device DEVICE               Training device (cuda/cpu)
  --action-mode MODE            Action mode (direct/locked_position)
  --headless                    Run in headless mode
  --list-configs                List available configurations
  --help                        Show help message
```

### Configuration Management

```bash
# List all available configurations
python training/train_dqn.py --list-configs

# Available configurations:
# - default: Standard training configuration
# - fast: Quick training for testing
# - research: High-performance research settings
# - locked_position: Optimized for position-based actions
# - debug: Minimal configuration for debugging
```

## Training Process

### 1. Initialization
- Load configuration (predefined or custom)
- Apply command-line overrides
- Initialize environment with specified parameters
- Create DQN trainer with model and training configurations
- Setup logging and directory structure

### 2. Training Loop
- Episode execution with epsilon-greedy exploration
- Experience collection in replay buffer
- Batch training with prioritized replay (optional)
- Target network updates
- Periodic evaluation and checkpointing
- Video recording (if enabled)

### 3. Results Saving
- Training metrics and performance data
- Model checkpoints at specified intervals
- Final evaluation results
- Configuration used for reproducibility

## Output Structure

Training creates the following directory structure:

```
├── models/checkpoints/[experiment_name]/
│   ├── episode_500.pth         # Model checkpoints
│   ├── episode_1000.pth
│   └── final_model.pth
├── results/[experiment_name]/
│   ├── training_results.json   # Final results
│   ├── training_metrics.json   # Episode-by-episode metrics
│   └── config_used.json        # Configuration used
├── logs/[experiment_name]/
│   ├── training.log            # Text logs
│   ├── training.jsonl          # Structured JSON logs
│   └── gpu_metrics.json        # GPU usage metrics
└── videos/[experiment_name]/   # Training videos (if enabled)
    ├── eval_ep_100.gif
    ├── eval_ep_200.gif
    └── best_episode.gif
```

## Configuration Integration

The training system integrates seamlessly with the configuration system:

```python
# Example: Custom training with configuration override
from configs.dqn_config import get_config

config = get_config('default')
config.training['learning_rate'] = 0.0005
config.training['max_episodes'] = 2000

# Train with custom configuration
trainer = DQNTrainer(
    env=TetrisEnv(**config.environment),
    model_config=config.model,
    training_config=config.training,
    experiment_name='custom_experiment'
)
```

## Training Modes

### Direct Action Mode
Traditional Tetris controls with 8-bit binary action space:
- Move left/right, soft drop, rotate, hard drop, hold, no-op
- Fast action execution
- Suitable for real-time gameplay learning

```bash
python training/train_dqn.py --config default --action-mode direct
```

### Locked Position Mode
Grid-based piece placement with discrete action space:
- 200 possible grid positions (20x10)
- Intelligent piece placement algorithm
- Suitable for strategic placement learning

```bash
python training/train_dqn.py --config locked_position --action-mode locked_position
```

## Performance Optimization

### GPU Training
```bash
# Force GPU training
python training/train_dqn.py --config default --device cuda

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Memory Management
```bash
# Reduce memory usage for limited hardware
python training/train_dqn.py --config fast --memory-size 10000 --batch-size 16

# High-performance training
python training/train_dqn.py --config research --memory-size 500000 --batch-size 128
```

### Training Speed
```bash
# Fast training for testing
python training/train_dqn.py --config debug --episodes 10

# Quick evaluation
python training/train_dqn.py --config fast --episodes 1000 --eval-freq 50
```

## Monitoring and Logging

### Real-time Monitoring
The training script provides real-time progress updates:
```
Episode 100: reward=-156.5, steps=234, duration=0.15s, epsilon=0.85
Episode 200: reward=-142.3, steps=456, duration=0.18s, epsilon=0.70
Evaluation (100 episodes): mean_reward=-145.2, std=12.4, max=-120.5
```

### Log Files
- **training.log**: Human-readable training progress
- **training.jsonl**: Structured JSON logs for analysis
- **gpu_metrics.json**: GPU memory and utilization tracking

### Metrics Tracked
- Episode rewards and steps
- Training loss and Q-values
- Exploration rate (epsilon)
- GPU memory usage
- Training duration
- Evaluation performance

## Error Handling

The training system includes comprehensive error handling:

### Common Issues and Solutions

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size and memory
   python training/train_dqn.py --batch-size 16 --memory-size 50000
   ```

2. **Slow Training**
   ```bash
   # Increase batch size or reduce memory
   python training/train_dqn.py --batch-size 64 --memory-size 50000
   ```

3. **Poor Performance**
   ```bash
   # Adjust learning rate and exploration
   python training/train_dqn.py --learning-rate 0.0005 --epsilon-decay 30000
   ```

4. **Directory Permissions**
   ```bash
   # Ensure write permissions for output directories
   mkdir -p models/checkpoints results logs videos
   ```

### Graceful Interruption
Training can be safely interrupted (Ctrl+C) and will:
- Save current model checkpoint
- Write partial results
- Clean up resources
- Provide resumption instructions

## Advanced Usage

### Custom Training Scripts
```python
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_dqn import train_with_config
from configs.dqn_config import get_config

# Custom training logic
config = get_config('default')
config.training['learning_rate'] = 0.001

results = train_with_config('custom', config.get_full_config())
print(f"Training completed with final reward: {results['mean_reward']}")
```

### Batch Training
```bash
# Train multiple configurations
for config in default fast research; do
    python training/train_dqn.py --config $config --episodes 1000
done
```

### Hyperparameter Sweeps
```bash
# Learning rate sweep
for lr in 0.0001 0.0005 0.001 0.005; do
    python training/train_dqn.py --config fast --learning-rate $lr --episodes 500
done
```

## Integration with Other Components

### Environment Integration
```python
from envs.tetris_env import TetrisEnv

# Create environment with training configuration
env = TetrisEnv(
    num_agents=1,
    headless=True,
    action_mode='direct',
    step_mode='action'
)
```

### Model Integration
```python
from models.tetris_cnn import TetrisCNN

# Create model with configuration
model = TetrisCNN(
    output_size=8,
    activation_type='identity',
    use_dropout=True,
    dropout_rate=0.1
)
```

### Video Logging Integration
```python
from utils.video_logger import TrainingVideoLogger

# Setup video logging
video_logger = TrainingVideoLogger(
    output_dir='videos/experiment',
    record_frequency=100,
    record_best=True
)
```

## Best Practices

1. **Start Small**: Use `debug` configuration for initial testing
2. **Monitor Resources**: Watch GPU memory and adjust batch sizes accordingly
3. **Save Frequently**: Use appropriate save frequencies for long training runs
4. **Document Experiments**: Use descriptive experiment names and save configurations
5. **Evaluate Regularly**: Set reasonable evaluation frequencies to track progress
6. **Use Version Control**: Track configuration changes and results

## Troubleshooting

### Performance Issues
- Monitor GPU utilization with `nvidia-smi`
- Adjust batch size based on available memory
- Use appropriate replay buffer sizes
- Consider mixed precision training for large models

### Training Instability
- Reduce learning rate
- Increase target network update frequency
- Use gradient clipping
- Check for NaN values in loss

### Environment Issues
- Verify action space compatibility
- Check observation preprocessing
- Ensure proper reward scaling
- Validate episode termination conditions

This training system provides a robust, flexible foundation for DQN training with comprehensive monitoring, error handling, and optimization features. 