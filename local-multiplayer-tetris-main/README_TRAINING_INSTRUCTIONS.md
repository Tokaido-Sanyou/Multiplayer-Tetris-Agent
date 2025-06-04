# Multiplayer Tetris Agent - Training Instructions

## 🚀 Quick Start - NEW Enhanced Training Runner

This system implements a **goal-conditioned actor** with a **6-phase enhanced state model** for Tetris gameplay. The actor learns through **goal conditioning** rather than PPO, making it more efficient and focused.

### ⚡ Run Training Immediately
```bash
# Basic training (100 batches, iterative mode)
python enhanced_training_runner.py

# Quick test (10 batches)  
python enhanced_training_runner.py --num_batches 10

# Deterministic exploration (your preferred mode)
python enhanced_training_runner.py --num_batches 300 --exploration_mode deterministic

# Full production training
python enhanced_training_runner.py --num_batches 500 --exploration_mode iterative --max_pieces 5 --use_cuda
```

## 📋 System Overview

### Core Components
- **Enhanced 6-Phase State Model**: Predicts optimal piece placements
- **Goal-Conditioned Actor**: Uses 5D goal vectors for action selection (NO PPO)
- **Valid Locked Position Explorer**: Only explores realistic piece positions
- **Q-Learning Value Network**: Estimates placement values
- **3-Stage Training Pipeline**: State → Actor → Joint training

### Key Features
- ✅ **No PPO**: Uses goal-conditioned supervised learning instead
- ✅ **5D Goal Vectors**: Rotation bits + X + Y + validity (removed confidence/quality)
- ✅ **Valid Positions Only**: Based on actual Tetris physics using `tetris_env.valid_space()`
- ✅ **Iterative Exploration**: 96% efficiency improvement
- ✅ **Standalone Runner**: No dependency issues

## 🎯 Training Command Options

### Basic Commands
```bash
# Default training (recommended starting point)
python enhanced_training_runner.py

# Short test run
python enhanced_training_runner.py --num_batches 10

# Your preferred deterministic mode
python enhanced_training_runner.py --num_batches 300 --exploration_mode deterministic
```

### Exploration Mode Options
```bash
# Iterative mode (most efficient - 96% valid positions)
python enhanced_training_runner.py --exploration_mode iterative

# Deterministic mode (mapped to iterative for compatibility)
python enhanced_training_runner.py --exploration_mode deterministic  

# RND mode (random exploration)
python enhanced_training_runner.py --exploration_mode rnd
```

### Advanced Parameter Tuning
```bash
# Longer episodes with more pieces
python enhanced_training_runner.py --max_pieces 5 --boards_to_keep 6

# Faster learning rates
python enhanced_training_runner.py --state_lr 0.002 --q_lr 0.002

# GPU acceleration
python enhanced_training_runner.py --use_cuda

# Production training session
python enhanced_training_runner.py \
    --num_batches 500 \
    --exploration_mode iterative \
    --max_pieces 4 \
    --boards_to_keep 5 \
    --state_lr 0.001 \
    --q_lr 0.001 \
    --use_cuda
```

## 🔧 Training Stages Explained

### Stage 1: State Model Only (40% of batches)
- **Purpose**: Train state model to predict optimal placements
- **Status**: Actor is FROZEN
- **Training**: State model + Q-learning value estimation
- **Output**: State loss, Q loss, top performers used

### Stage 2: Actor Only (30% of batches)
- **Purpose**: Train goal-conditioned actor with frozen state model
- **Status**: State model is FROZEN
- **Training**: Goal→action mapping through supervised learning
- **Output**: Actor loss, goal→action pairs processed

### Stage 3: Joint Training (30% of batches)
- **Purpose**: Coordinate all components together  
- **Status**: All components training
- **Training**: State model + Q-learning + Actor
- **Output**: All loss metrics, coordination status

## ⚙️ Parameter Reference

### Core Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_batches` | 100 | Number of training batches |
| `--exploration_mode` | iterative | Mode: iterative, deterministic, rnd |
| `--max_pieces` | 3 | Pieces per exploration episode |
| `--boards_to_keep` | 4 | Top board states to continue |
| `--state_lr` | 0.001 | State model learning rate |
| `--q_lr` | 0.001 | Q-learning learning rate |
| `--use_cuda` | False | Enable GPU acceleration |

### Performance Tuning Examples
```bash
# Fast training (fewer pieces, higher learning rate)
python enhanced_training_runner.py --max_pieces 2 --state_lr 0.002

# Deep exploration (more pieces, more boards)
python enhanced_training_runner.py --max_pieces 7 --boards_to_keep 8

# GPU accelerated production training
python enhanced_training_runner.py --num_batches 1000 --use_cuda

# Memory efficient (fewer boards kept)
python enhanced_training_runner.py --boards_to_keep 2 --max_pieces 2
```

## 📊 Training Output & Monitoring

### Console Output
```
🚀 Enhanced 6-Phase Trainer Starting...
   • Device: cuda
   • Exploration mode: deterministic
   • Training batches: 300
   • Goal-conditioned actor: NO PPO

📦 BATCH 1/300
🎯 Stage: state_model_only
🎯 Stage 1: State Model Training (Actor FROZEN)
   • State loss: 2.4531
   • Q loss: 8.9234
   • Top performers: 15
   📊 Batch 1 Metrics:
      • Lines cleared: 3
      • Exploration samples: 47
      • Exploration efficiency: 89.4%
      • Epsilon: 0.0950
```

### Result Files
- **Checkpoint files**: `checkpoints/enhanced_6phase_batch_X/`
- **Final results**: `enhanced_6phase_results_TIMESTAMP.txt`
- **Training logs**: Console output with full metrics

### Key Metrics to Watch
- **State loss**: Should decrease from ~4.0 to ~1.5-2.0
- **Q loss**: Should stabilize around 3-15
- **Actor loss**: Should decrease during Stage 2 and 3
- **Lines cleared**: Should increase over time
- **Exploration efficiency**: Should stay >80%

## 🧪 Testing & Validation

### Quick Functionality Test
```bash
# 3-batch test to verify system works
python enhanced_training_runner.py --num_batches 3
```

### Stage Verification Test
```bash
# Test all 3 stages with 1 batch each
python enhanced_training_runner.py --num_batches 3 --max_pieces 2
```

### Troubleshooting Common Issues
```bash
# If CUDA memory issues
python enhanced_training_runner.py --max_pieces 2 --boards_to_keep 2

# If training too slow
python enhanced_training_runner.py --num_batches 10 --exploration_mode iterative

# If exploration efficiency low
python enhanced_training_runner.py --exploration_mode iterative --max_pieces 3
```

## 🔧 Installation & Setup

### Prerequisites
```bash
pip install torch numpy pygame gym
```

### Environment Setup
```bash
cd local-multiplayer-tetris-main
export PYTHONPATH=$PYTHONPATH:$(pwd)/localMultiplayerTetris
```

### Direct Training (No Setup Needed)
```bash
# The enhanced_training_runner.py handles all imports automatically
python enhanced_training_runner.py
```

## 🎮 Goal Vector Format

### 5D Goal Vector Structure (SIMPLIFIED)
```python
goal_vector = [
    rot_bit_0,      # [0]: Rotation bit 0 (0 or 1)
    rot_bit_1,      # [1]: Rotation bit 1 (0 or 1) 
    x_position,     # [2]: X coordinate (0-9)
    y_position,     # [3]: Y coordinate (0-19)
    validity_flag   # [4]: Placement validity ONLY (0 or 1)
]

# Example goal vectors:
[0, 1, 5, 15, 1]  # Rotation=2, X=5, Y=15, valid
[1, 0, 2, 18, 1]  # Rotation=1, X=2, Y=18, valid
[0, 0, 9, 10, 0]  # Rotation=0, X=9, Y=10, invalid
```

### Decoding Goal Vectors
```python
# Decode rotation from binary
rotation = rot_bit_0 + (rot_bit_1 << 1)  # 0-3

# Coordinates are direct
x_pos = int(goal_vector[2])  # 0-9
y_pos = int(goal_vector[3])  # 0-19

# Validity flag ONLY (no confidence/quality)
is_valid = goal_vector[4] > 0.5
```

## 📈 Monitoring Training Progress

### Key Metrics to Track
```python
# State model performance
state_loss = results['loss']                    # Should decrease over time
top_performers = results['top_performers_used'] # Should be consistent
threshold = results['threshold']                # Reward threshold for top performers

# Q-learning performance  
q_loss = results['q_loss']                     # Should decrease over time
trajectories = results['trajectories_trained'] # Number of samples used

# Exploration metrics
exploration_states = len(exploration_data)     # Number of valid positions found
lines_cleared = sum(d['lines_cleared'])        # Line clearing performance
avg_reward = mean([d['terminal_reward']])      # Average placement reward
```

### Expected Performance
```python
# Good state model performance
state_loss < 5.0                 # Well-trained model
top_performers > 50              # Sufficient training data

# Good Q-learning performance
q_loss < 50.0                    # Reasonable value estimation
trajectories > 100               # Sufficient experience

# Good exploration performance  
exploration_states > 30          # Finding valid positions
lines_cleared > 0                # Achieving line clears
efficiency > 90%                 # vs brute force approach
```

## 🛠️ Troubleshooting

### Common Issues

#### High State Loss (>10.0)
```python
# Solutions:
- Reduce learning rate: state_lr = 0.0005
- Train on more data: increase max_pieces
- Check target normalization: targets should be 0-1 range
```

#### High Q-Loss (>100.0)  
```python
# Solutions:
- Reduce learning rate: q_lr = 0.0005
- Check reward normalization: use tanh(reward/100)
- Increase n_step: n_step = 6
```

#### No Line Clearing
```python
# Solutions:
- Check board setup: ensure some near-complete rows
- Verify placement logic: pieces should lock at valid positions
- Increase exploration: higher epsilon or more pieces
```

#### Memory Issues
```python
# Solutions:
- Reduce max_pieces: max_pieces = 2
- Smaller batches: batch_size = 5
- Use CPU: device = 'cpu'
```

## 🔄 Complete Training Pipeline

### Full Training Script Example
```python
#!/usr/bin/env python3
import torch
from localMultiplayerTetris.rl_utils.enhanced_6phase_state_model import Enhanced6PhaseComponents
from localMultiplayerTetris.tetris_env import TetrisEnv

def full_training_pipeline():
    # Initialize
    env = TetrisEnv(single_player=True, headless=True)
    system = Enhanced6PhaseComponents(state_dim=210, goal_dim=6, device='cpu')
    system.set_optimizers(state_lr=0.001, q_lr=0.001)
    
    # Create exploration manager
    manager = system.create_piece_by_piece_exploration_manager(env)
    manager.max_pieces = 3
    manager.boards_to_keep = 4
    
    # Training loop
    for epoch in range(10):
        print(f"Epoch {epoch+1}/10")
        
        # Collect exploration data
        data = manager.collect_piece_by_piece_exploration_data('iterative')
        print(f"Collected {len(data)} exploration samples")
        
        # Train state model
        state_results = system.train_enhanced_state_model(data)
        print(f"State loss: {state_results['loss']:.4f}")
        
        # Train Q-learning
        q_results = system.train_simplified_q_learning(data)
        print(f"Q loss: {q_results['q_loss']:.4f}")
        
        # Update exploration
        system.update_epsilon()
        print(f"Epsilon: {system.goal_selector.epsilon:.4f}")
        print()
    
    print("Training completed!")

if __name__ == "__main__":
    full_training_pipeline()
```

## 📚 Additional Resources

### File Structure
```
local-multiplayer-tetris-main/
├── localMultiplayerTetris/
│   ├── rl_utils/
│   │   ├── enhanced_6phase_state_model.py  # Main training system
│   │   ├── actor_critic.py                 # Goal-conditioned actor (no PPO)
│   │   └── ...
│   ├── tetris_env.py                       # Tetris environment
│   └── config.py                           # Configuration settings
├── test_3_stages.py                        # 3-stage training test
├── quick_test.py                           # Quick functionality test
├── debug_full_system.py                    # Full system debug
└── README_TRAINING_INSTRUCTIONS.md         # This file
```

### Key Classes
- `Enhanced6PhaseComponents`: Main training system
- `PieceByPieceExplorationManager`: Exploration data collection
- `Top5TerminalStateModel`: State model for placement prediction
- `SimplifiedQLearning`: Value estimation network
- `EpsilonGreedyGoalSelector`: Goal generation and selection

---

## 🚀 Ready to Train!

```bash
# Quick start - test everything works
python test_3_stages.py

# Start basic training
python quick_test.py

# Full system check
python debug_full_system.py
```

For questions or issues, check the debug logs in `tetris_debug.log` or run the test scripts for validation. 