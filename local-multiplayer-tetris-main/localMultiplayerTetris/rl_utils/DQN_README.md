# Enhanced DQN Agent for Tetris

This directory contains an enhanced Deep Q-Network (DQN) implementation for training Tetris agents with comprehensive logging, checkpointing, and parallel training support.

## ✨ Features

### 🔄 Core DQN Features
- **Double Q-learning** to reduce overestimation bias
- **Experience replay** with configurable buffer size
- **Target network** with periodic updates for stability
- **Epsilon-greedy exploration** with configurable decay
- **Gradient clipping** to prevent exploding gradients

### 📊 TensorBoard Logging
- **Training metrics**: Loss, epsilon, buffer size, target network updates
- **Episode metrics**: Reward, length, score, lines cleared, level
- **Running averages**: 10-episode and 100-episode moving averages
- **Hyperparameter logging**: All configuration saved at start
- **Automatic timestamped log directories**

### 💾 Automatic Checkpointing
- **Saves every 1000 episodes** (configurable)
- **Complete training state**: Networks, optimizer, replay buffer stats
- **Resume training** from any checkpoint
- **Latest checkpoint** always available
- **Hyperparameter preservation**

### ⚡ Parallel Training Support
- **Vectorized environments** for efficient data collection
- **Batch action selection** and experience storage
- **Configurable update frequency** for batch training
- **Memory-efficient** parallel processing

## 🚀 Quick Start

### Basic Training (Single Environment)

```bash
cd local-multiplayer-tetris-main/localMultiplayerTetris/rl_utils
python dqn_train.py --mode single --num_episodes 5000
```

### Parallel Training (Recommended)

```bash
python dqn_train.py --mode vectorized --num_envs 8 --num_episodes 20000
```

### Resume from Checkpoint

```bash
python dqn_train.py --checkpoint checkpoints/dqn_checkpoint_episode_5000.pt
```

## 📁 File Structure

```
rl_utils/
├── dqn_new.py          # Enhanced DQN agent implementation
├── dqn_train.py        # Training script with parallel support
├── test_dqn.py         # Test suite for verification
├── DQN_README.md       # This documentation
├── logs/               # TensorBoard logs (auto-created)
│   └── dqn_tensorboard/
└── checkpoints/        # Model checkpoints (auto-created)
```

## 🎯 Usage Examples

### 1. Basic Training with Custom Hyperparameters

```bash
python dqn_train.py \
    --mode vectorized \
    --num_envs 6 \
    --num_episodes 15000 \
    --learning_rate 5e-4 \
    --gamma 0.995 \
    --epsilon_decay 0.998 \
    --batch_size 128 \
    --buffer_size 200000
```

### 2. Fast Prototyping (Single Environment)

```bash
python dqn_train.py \
    --mode single \
    --num_episodes 1000 \
    --eval_interval 100 \
    --save_interval 100
```

### 3. High-Performance Training

```bash
python dqn_train.py \
    --mode vectorized \
    --num_envs 16 \
    --num_episodes 50000 \
    --update_frequency 8 \
    --batch_size 256 \
    --target_update 1000
```

## 📊 Monitoring Training

### TensorBoard

Start TensorBoard to monitor training progress:

```bash
tensorboard --logdir logs/dqn_tensorboard
```

### Key Metrics to Watch

1. **Episode/Average_Reward_100**: Primary performance indicator
2. **Episode/Score**: Tetris game score (higher is better)
3. **Episode/Lines_Cleared**: Lines cleared per episode
4. **Training/Loss**: Should decrease over time
5. **Training/Epsilon**: Exploration rate (should decay)

### Expected Training Progress

- **Episodes 0-1000**: Random exploration, low scores
- **Episodes 1000-5000**: Learning basic patterns, improving scores
- **Episodes 5000-15000**: Stable improvement, higher scores
- **Episodes 15000+**: Fine-tuning, consistent performance

## 🔧 Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `vectorized` | Training mode: `single` or `vectorized` |
| `--num_episodes` | `10000` | Total episodes to train |
| `--num_envs` | `4` | Parallel environments (vectorized mode) |
| `--eval_interval` | `1000` | Episodes between evaluations |
| `--update_frequency` | `4` | Steps between agent updates (vectorized) |

### DQN Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning_rate` | `1e-4` | Adam optimizer learning rate |
| `--gamma` | `0.99` | Discount factor for future rewards |
| `--epsilon_start` | `1.0` | Initial exploration rate |
| `--epsilon_end` | `0.01` | Final exploration rate |
| `--epsilon_decay` | `0.995` | Exploration decay rate per episode |
| `--batch_size` | `64` | Training batch size |
| `--buffer_size` | `100000` | Experience replay buffer size |
| `--target_update` | `10` | Steps between target network updates |
| `--save_interval` | `1000` | Episodes between checkpoints |

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_dqn.py
```

Tests verify:
- ✅ Basic agent functionality
- ✅ Batch operations
- ✅ Save/load mechanics
- ✅ TensorBoard logging
- ✅ Environment compatibility

## 📈 Performance Tips

### For Faster Training
1. **Use vectorized mode** with 8-16 environments
2. **Increase batch size** to 128 or 256
3. **Tune update frequency** (4-8 steps per update)
4. **Use GPU** if available (automatic detection)

### For Better Performance
1. **Adjust epsilon decay** for longer exploration
2. **Increase buffer size** for more diverse experiences
3. **Tune learning rate** (try 5e-4 or 2e-4)
4. **Experiment with target update frequency**

### For Memory Efficiency
1. **Reduce buffer size** to 50k-100k
2. **Use smaller batch sizes** (32-64)
3. **Limit parallel environments** to 4-8

## 🔬 Architecture Details

### State Representation (207 dimensions)
- **Grid**: 200 values (20×10 flattened board state)
- **Metadata**: 7 values (next piece, hold piece, current piece info)

### Action Space (41 actions)
- **Placements**: 40 possible piece placements (4 rotations × 10 columns)
- **Hold**: 1 hold action

### Network Architecture
```
Input (207) → CNN Branch (Grid) + MLP Branch (Metadata)
           → Combined Features (13,120)
           → FC (512) → FC (256) → Output (41)
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `batch_size` or `num_envs`
   - Decrease `buffer_size`

2. **Training too slow**
   - Increase `num_envs` for parallel training
   - Check if GPU is being used

3. **Poor performance**
   - Check epsilon decay (might be too fast)
   - Verify reward function in environment
   - Monitor TensorBoard for trends

4. **Checkpoints not saving**
   - Check disk space
   - Verify write permissions in `checkpoints/` directory

### Getting Help

1. Check the test suite: `python test_dqn.py`
2. Monitor TensorBoard logs for anomalies
3. Verify environment compatibility
4. Check GPU memory usage if using CUDA

## 📊 Example Training Command for Production

```bash
# Recommended production training setup
python dqn_train.py \
    --mode vectorized \
    --num_envs 12 \
    --num_episodes 30000 \
    --learning_rate 3e-4 \
    --gamma 0.995 \
    --epsilon_decay 0.9995 \
    --batch_size 128 \
    --buffer_size 150000 \
    --update_frequency 6 \
    --eval_interval 500 \
    --save_interval 1000
```

This configuration provides a good balance of training speed, stability, and performance for most use cases. 