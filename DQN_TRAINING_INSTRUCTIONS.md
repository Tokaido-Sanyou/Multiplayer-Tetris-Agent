# Complete DQN Locked State Training Instructions

## Overview

The DQN Locked State system provides a comprehensive Deep Q-Network implementation specifically designed for Tetris gameplay with enhanced state representation and proper action space mapping.

## Key Features ✅

- **Corrected Action Space**: 1600 discrete actions (200 coordinates × 4 rotations × 2 lock states)
- **Enhanced State**: 585-dimensional state (425 observation + 160 selection tracking)
- **GPU Acceleration**: Full CUDA support with efficient tensor operations
- **Optimized Performance**: Fixed tensor conversion warnings for faster training
- **Packaged Training**: Streamlined script with command-line arguments and error handling
- **Batched Training**: Structured training with demonstrations every batch
- **Comprehensive Logging**: Real-time metrics, plots, and checkpoints
- **Multiple Training Modes**: Packaged, legacy, quick-train, and evaluation functions

## Quick Start

### Basic Training (100 Episodes) - RECOMMENDED
```powershell
# Run packaged training with optimized tensor operations (RECOMMENDED)
python train_dqn_packaged.py

# Or with custom episodes
python train_dqn_packaged.py --episodes 200

# Legacy training (still works but less optimized)
python training/train_dqn_locked.py
```

### Monitoring Training Progress
The trainer automatically creates:
- **Logs**: `results/dqn_locked/training_log_YYYYMMDD_HHMMSS.txt`
- **Plots**: `results/dqn_locked/training_progress_batch_XXX.png`
- **Checkpoints**: `results/dqn_locked/checkpoint_batch_XXX.pth`
- **Best Model**: `results/dqn_locked/best_model_batch_XXX.pth`

### Packaged Training Output Example
```
🚀 PACKAGED DQN LOCKED STATE TRAINING
==================================================
📊 Configuration:
   Total Episodes: 100
   Episodes per Batch: 10
   Device: CUDA
   Learning Rate: 0.0001
   Action Space: 1600 (200 coords × 4 rotations × 2 lock states)
   State Space: 585 (425 observation + 160 selection)
==================================================

🎯 Starting Training...
🔄 BATCH 1/10 - Training 10 episodes...
   Episode 1: Reward -150.50, Steps 12, Loss 3.90, Q-val 0.01, ε 0.985
   Episode 5: Reward -145.25, Steps 15, Loss 8.45, Q-val -0.12, ε 0.912

📈 BATCH 1 SUMMARY:
   Average Reward: -165.35
   🏆 NEW BEST REWARD: -165.35 - Model saved!

🎮 AGENT DEMONSTRATION (Batch 1):
   Demo Episode 1: -156.50 reward, 8 steps

🎉 Training Completed Successfully!
⏱️  Total Time: 1234.5s (20.6m)
📁 Results saved to: results/dqn_locked/
```

### Command Line Options
```powershell
python train_dqn_packaged.py --help

options:
  --episodes EPISODES           Total episodes to train (default: 100)
  --episodes-per-batch EPISODES Episodes per batch (default: 10) 
  --device {auto,cuda,cpu}      Device to use (default: auto)
  --learning-rate LEARNING_RATE Learning rate (default: 0.0001)
  --epsilon-decay EPSILON_DECAY Episodes to decay epsilon over (default: 800)
  --memory-size MEMORY_SIZE     Experience replay buffer size (default: 100000)
  --batch-size BATCH_SIZE       Training batch size (default: 32)
  --save-freq SAVE_FREQ         Save checkpoint every N batches (default: 1)
  --no-demo                     Disable agent demonstrations
  --quiet                       Reduce output verbosity
```

## Advanced Configuration

### Custom Training Parameters
```python
from training.train_dqn_locked import ComprehensiveDQNTrainer

# Create trainer with custom parameters
trainer = ComprehensiveDQNTrainer(
    device='cuda',                    # 'cuda' or 'cpu'
    episodes_per_batch=10,           # Episodes per batch
    learning_rate=0.0001,            # Learning rate
    epsilon_start=1.0,               # Initial exploration
    epsilon_end=0.01,                # Final exploration  
    epsilon_decay_episodes=800,      # Decay schedule
    target_update_freq=1000,         # Target network updates
    memory_size=100000,              # Experience buffer size
    batch_size=32                    # Training batch size
)

# Run training
trainer.train(total_episodes=200)   # Train for 200 episodes
```

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0001 | Adam optimizer learning rate |
| `epsilon_start` | 1.0 | Initial exploration probability |
| `epsilon_end` | 0.01 | Final exploration probability |
| `epsilon_decay_episodes` | 800 | Episodes to decay epsilon over |
| `gamma` | 0.99 | Discount factor |
| `batch_size` | 32 | Mini-batch size for training |
| `memory_size` | 100000 | Experience replay buffer capacity |
| `target_update_freq` | 1000 | Target network update frequency |

## Network Architecture

### Model Structure
```
Enhanced DQN Network:
Input Layer:     585 dimensions (425 obs + 160 selection)
Hidden Layer 1:  585 → 512 (ReLU + Dropout 0.1)
Hidden Layer 2:  512 → 512 (ReLU + Dropout 0.1)  
Hidden Layer 3:  512 → 256 (ReLU + Dropout 0.1)
Output Layer:    256 → 1600 (Q-values for all actions)

Total Parameters: ~1.2M
Optimizer: Adam with weight decay 1e-5
Loss Function: Huber Loss (for stability)
```

### Action Space Design
```python
# Action encoding: (x, y, rotation, lock_in) → action_index
# Total: 1600 actions = 10×20×4×2

action_idx = 0
for y in range(20):          # 20 rows
    for x in range(10):      # 10 columns
        for rotation in range(4):    # 4 rotations  
            for lock_in in range(2): # 2 lock states
                action_to_components[action_idx] = (x, y, rotation, lock_in)
                action_idx += 1

# Environment conversion:
env_action = y * 10 + x  # When lock_in=1 (place piece)
env_action = 0           # When lock_in=0 (select position)
```

## Training Phases

### Phase 1: Exploration (Episodes 1-200)
- **High Epsilon**: 1.0 → 0.5 (random exploration)
- **Expected Rewards**: -150 to -200 (learning basics)
- **Focus**: Experience collection and basic Q-learning

### Phase 2: Learning (Episodes 200-600)  
- **Medium Epsilon**: 0.5 → 0.1 (reducing exploration)
- **Expected Rewards**: -100 to -150 (improving play)
- **Focus**: Strategy development and action selection

### Phase 3: Optimization (Episodes 600+)
- **Low Epsilon**: 0.1 → 0.01 (exploitation focused)
- **Expected Rewards**: -50 to -100 (competent play)
- **Focus**: Fine-tuning and optimal strategies

## Performance Monitoring

### Key Metrics to Watch

1. **Average Reward per Batch**: Should gradually increase
2. **Training Loss**: Should decrease and stabilize  
3. **Q-Values**: Should develop structure (not random)
4. **Epsilon**: Should decay according to schedule
5. **Episode Length**: Should increase as agent improves

### Troubleshooting

#### Low Performance (Rewards < -200)
- Check epsilon schedule (too low = no exploration)
- Verify learning rate (too high = unstable, too low = slow)
- Ensure sufficient experience replay buffer

#### Training Instability  
- Reduce learning rate
- Increase target network update frequency
- Check for proper action masking

#### GPU Memory Issues
- Reduce batch size or memory buffer size
- Check for memory leaks in experience storage

## Checkpoint Management

### Loading Previous Models
```python
# Load specific checkpoint
trainer.agent.load_checkpoint('results/dqn_locked/checkpoint_batch_005.pth')

# Load best model
trainer.agent.load_checkpoint('results/dqn_locked/best_model_batch_008.pth')
```

### Model Evaluation
```python
# Run evaluation without training
trainer.agent.set_training_mode(False)
reward, length, _ = trainer.run_episode(training=False)
print(f"Evaluation: {reward:.2f} reward, {length} steps")
```

## Integration with Existing Codebase

### File Structure
```
agents/
├── dqn_locked_agent.py     # Core DQN implementation
├── base_agent.py           # Base agent interface

training/  
├── train_dqn_locked.py     # Comprehensive trainer

results/dqn_locked/
├── checkpoint_batch_XXX.pth      # Model checkpoints
├── training_progress_batch_XXX.png  # Progress plots  
├── training_log_YYYYMMDD.txt     # Training logs
└── best_model_batch_XXX.pth      # Best performing model
```

### Environment Compatibility
- **Standard Mode**: Compatible with `TetrisEnv(action_mode='direct')`
- **Locked Mode**: Uses `TetrisEnv(action_mode='locked_position')`
- **Observation**: 425-dimensional state vector from environment
- **Actions**: Converted from 1600 agent actions to environment format

## Advanced Features

### Action Masking
```python
# Get valid actions for current state
valid_actions = trainer.get_valid_actions_for_state(observation)

# Agent automatically filters Q-values to valid actions only
action = trainer.agent.select_action(observation, valid_actions=valid_actions)
```

### Experience Replay Optimization
```python
# Custom experience format for efficient storage
experience = {
    'state': enhanced_state,        # 585-dim state
    'action': action_info,          # Action components  
    'reward': reward,               # Environment reward
    'next_state': next_enhanced_state,  # Next 585-dim state
    'done': done                    # Episode termination
}
```

### Visualization System
- **Real-time Plots**: Training progress updated every batch
- **Agent Demonstrations**: Visual gameplay after each batch
- **Performance Metrics**: Comprehensive statistics tracking

## Comparison with Other Methods

### vs DREAM Training
- **DQN**: Model-free, direct Q-learning, faster per episode
- **DREAM**: Model-based, imagination training, more sample efficient

### vs Standard DQN  
- **Locked State**: Enhanced state representation, action masking
- **Standard**: Basic state, full action space, simpler implementation

## Future Enhancements

### Potential Improvements
1. **Prioritized Experience Replay**: Weight important experiences higher
2. **Dueling Architecture**: Separate value and advantage streams
3. **Rainbow Extensions**: Combine multiple DQN improvements
4. **Multi-step Learning**: N-step returns for faster learning

### Research Directions
1. **Hierarchical Actions**: High-level strategy selection
2. **Curriculum Learning**: Progressive difficulty increase
3. **Transfer Learning**: Share knowledge across Tetris variants
4. **Meta-Learning**: Adapt quickly to new Tetris configurations

---

## Quick Reference Commands

```powershell
# Basic training (recommended starting point) - OPTIMIZED
python train_dqn_packaged.py

# Extended training for better performance with custom parameters
python train_dqn_packaged.py --episodes 500 --epsilon-decay 1000 --learning-rate 0.0005

# Quick 50-episode training for testing
python train_dqn_packaged.py --episodes 50 --episodes-per-batch 5 --quiet

# CPU training (if GPU not available)
python train_dqn_packaged.py --device cpu --episodes 100

# Evaluation of trained model using packaged functions
python -c "
from train_dqn_packaged import evaluate_model
results = evaluate_model('results/dqn_locked/best_model_batch_010.pth', episodes=5)
print(f'Average Performance: {results[\"avg_reward\"]:.2f} reward')
"

# Programmatic training using quick_train function
python -c "
from train_dqn_packaged import quick_train
success = quick_train(episodes=100, device='cuda')
print(f'Training success: {success}')
"

# Legacy training (still works but tensor warnings may appear)
python training/train_dqn_locked.py
```

**Status**: Production-ready with comprehensive debugging and performance verification
**Last Updated**: December 7, 2024 