# ML Algorithms for Tetris Training

This directory contains machine learning algorithms implemented for Tetris training, with full GPU support and comprehensive logging.

## Overview

### Implemented Algorithms

#### 1. **Deep Q-Network (DQN)** - `dqn_trainer.py`
- **Status**: ✅ Fully Implemented
- **Features**:
  - Experience replay with prioritized sampling
  - Target network with periodic updates
  - Double DQN support for reduced overestimation
  - Epsilon-greedy exploration with decay
  - GPU acceleration with CUDA support
  - Comprehensive logging and metrics tracking
  - Model checkpointing and evaluation

#### 2. **DREAM** (Planned)
- **Status**: 🚧 Framework Ready
- **Features**: World model learning, imagined experience generation
- **Integration**: Uses existing world model data collection in `TetrisEnv`

#### 3. **RL2/Meta-Learning** (Planned)  
- **Status**: 🚧 Framework Ready
- **Features**: Episode history tracking, task adaptation
- **Integration**: Uses existing meta-learning infrastructure in `TetrisEnv`

## DQN Implementation Details

### Architecture
- **Input**: 425-bit binary observations (board state + piece info + opponent state)
- **Network**: CNN with 3 convolutional layers + fully connected layer
- **Output**: Q-values for 8 possible actions
- **Parameters**: ~306K total parameters

### Action Space (Direct Mode)
- `0`: Move left
- `1`: Move right  
- `2`: Soft drop (move down)
- `3`: Rotate clockwise
- `4`: Rotate counter-clockwise
- `5`: Hard drop (instant placement)
- `6`: Hold piece
- `7`: No-op

### Key Features

#### Experience Replay
```python
# Prioritized experience replay with importance sampling
buffer = ReplayBuffer(capacity=100000, prioritized=True)
experiences, indices, weights = buffer.sample(batch_size=32)
```

#### Target Network Updates
```python
# Periodic target network synchronization
if total_steps % target_update_freq == 0:
    target_network.load_state_dict(q_network.state_dict())
```

#### Double DQN
```python
# Reduced overestimation bias
next_actions = q_network(next_states).argmax(1)
next_q_values = target_network(next_states).gather(1, next_actions.unsqueeze(1))
```

### Training Configuration

#### Default Hyperparameters
```python
{
    'learning_rate': 0.0001,
    'gamma': 0.99,                    # Discount factor
    'epsilon_start': 1.0,             # Initial exploration
    'epsilon_end': 0.01,              # Final exploration
    'epsilon_decay_steps': 50000,     # Exploration decay
    'batch_size': 32,
    'target_update_freq': 1000,
    'memory_size': 100000,
    'min_memory_size': 1000,
    'max_episodes': 5000,
    'prioritized_replay': True,
    'double_dqn': True
}
```

## Alternative Action Modes

### Direct Actions (Default)
- Binary tuple format: `[0,0,0,1,0,0,0,0]` for rotate clockwise
- Immediate action execution
- Compatible with real-time gameplay

### Locked Position Selection (New)
- Position index: `0-199` (20×10 grid positions)
- Algorithm finds best piece placement near target
- Higher-level strategic control
- Useful for planning-based approaches

**Enable locked position mode:**
```python
env = TetrisEnv(action_mode='locked_position')
valid_positions = env.get_valid_positions(player)
action = random.choice(valid_positions)  # Select position index
```

## Usage

### Training DQN
```bash
cd algorithms
python dqn_trainer.py
```

### Custom Training
```python
from algorithms.dqn_trainer import DQNTrainer
from envs.tetris_env import TetrisEnv

# Create environment
env = TetrisEnv(
    num_agents=1,
    headless=True,
    action_mode='direct'
)

# Custom configuration
model_config = {
    'conv_filters': [32, 64, 64],
    'fc_units': 512,
    'dropout_rate': 0.2
}

training_config = {
    'learning_rate': 0.0005,
    'batch_size': 64,
    'max_episodes': 10000
}

# Train
trainer = DQNTrainer(env, model_config, training_config)
trainer.train()
```

### Loading Trained Models
```python
import torch
from models.tetris_cnn import TetrisCNN

# Load checkpoint
checkpoint = torch.load('models/checkpoints/dqn_tetris_v1/best_model.pt')
model = TetrisCNN(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    q_values = model(state)
    action = q_values.argmax().item()
```

## File Structure

```
algorithms/
├── README.md              # This file
├── dqn_trainer.py          # DQN implementation
├── dream_trainer.py        # (Planned) DREAM algorithm
└── rl2_trainer.py          # (Planned) RL2/Meta-learning
```

## Integration with Project Structure

### Environment Integration
- Uses `envs.tetris_env.TetrisEnv` for training
- Supports both single and multi-agent scenarios
- Compatible with binary observation format

### Model Integration
- Uses `models.tetris_cnn.TetrisCNN` architecture
- GPU acceleration through PyTorch CUDA
- Model checkpointing in `models/checkpoints/`

### Logging Integration
- Uses `utils.logger.TetrisLogger` for structured logging
- Training metrics saved to `logs/`
- JSON format for analysis and visualization

### Results Integration
- Training results saved to `results/`
- Model performance metrics tracking
- Evaluation and benchmark storage

## Performance

### Training Metrics
- **Episodes per hour**: ~500-1000 (GPU), ~200-400 (CPU)
- **Memory usage**: ~2-4GB GPU memory for default config
- **Convergence**: Typically 2000-5000 episodes for stable performance

### Hardware Requirements
- **Minimum**: 8GB RAM, modern CPU
- **Recommended**: 16GB RAM, NVIDIA GPU (GTX 1060+)
- **Optimal**: 32GB RAM, RTX 3080+ with 10GB+ VRAM

## Advanced Features

### Curriculum Learning
```python
# Progressive difficulty increase
def update_environment_difficulty(episode):
    if episode > 1000:
        env.game.fall_speed *= 0.95  # Increase speed
```

### Multi-Agent Training
```python
env = TetrisEnv(num_agents=2, action_mode='direct')
# Competitive or cooperative training scenarios
```

### Custom Reward Shaping
```python
# Override reward calculation in environment
def custom_reward(self, lines_cleared, height, holes):
    return lines_cleared * 10 - height * 0.1 - holes * 2
```

## Future Algorithms

### DREAM (World Model Learning)
- Model-based reinforcement learning
- Imagined experience generation
- Sample efficiency improvements

### RL2 (Learning to Reinforce Learn)
- Meta-learning across episodes
- Fast adaptation to new tasks
- Few-shot learning capabilities

### A3C/PPO (Actor-Critic Methods)
- Policy gradient methods
- Continuous action spaces
- Better exploration strategies

## Debugging and Analysis

### Training Logs
```bash
# View training progress
tail -f logs/dqn_tetris_v1/dqn_tetris_v1_*.log

# Analyze structured logs
cat logs/dqn_tetris_v1/dqn_tetris_v1_*.jsonl | jq '.event_type'
```

### Model Analysis
```python
# Visualize Q-values
import matplotlib.pyplot as plt

def plot_q_values(model, state):
    q_vals = model(state).detach().cpu().numpy()
    plt.bar(range(8), q_vals)
    plt.xlabel('Action')
    plt.ylabel('Q-Value')
    plt.show()
```

### Performance Profiling
```python
# GPU memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

# Training speed
import time
start_time = time.time()
# ... training code ...
print(f"Episodes per second: {episodes/(time.time()-start_time):.2f}")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `batch_size=16`
   - Reduce model size: `fc_units=128`
   - Use gradient accumulation

2. **Slow Training**
   - Enable GPU: `device='cuda'`
   - Increase batch size: `batch_size=64`
   - Use mixed precision training

3. **Poor Convergence**
   - Adjust learning rate: `lr=0.0005`
   - Increase exploration: `epsilon_decay_steps=100000`
   - Add reward shaping

4. **Memory Leaks**
   - Clear unused variables: `del experiences`
   - Use `torch.no_grad()` for inference
   - Periodic garbage collection

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{tetris_dqn_2024,
  title={Deep Q-Network Implementation for Tetris},
  author={Tetris ML Project},
  year={2024},
  url={https://github.com/your-repo/tetris-ml}
}
``` 