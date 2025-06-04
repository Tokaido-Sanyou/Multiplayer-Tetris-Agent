# Multiplayer Tetris AIRL Implementation

**Advanced Adversarial Inverse Reinforcement Learning for Competitive Tetris**

## 🎯 Overview

This project implements AIRL (Adversarial Inverse Reinforcement Learning) for multiplayer competitive Tetris training. Two AI agents learn to play Tetris competitively by imitating expert demonstrations and competing against each other.

## 🏗️ Architecture

```
Multiplayer-Tetris-Agent/
├── tetris-ai-master/              # Expert DQN model (4-feature states)
│   ├── sample.keras               # Pre-trained expert model
│   └── tetris.py                  # Expert environment
├── local-multiplayer-tetris-main/ # Multiplayer environment (20x10 grid)
│   ├── localMultiplayerTetris/
│   │   ├── tetris_env.py          # Base Tetris environment
│   │   └── rl_utils/
│   │       ├── airl_agent.py      # AIRL agent implementation
│   │       ├── actor_critic.py    # Policy network
│   │       ├── multiplayer_airl.py # Competitive trainer
│   │       ├── true_multiplayer_env.py # Fixed multiplayer wrapper
│   │       ├── visualized_training.py  # Real-time visualization
│   │       └── expert_loader.py   # Expert trajectory loader
├── expert_trajectories/           # Original expert data (30MB, high HOLD%)
├── expert_trajectories_new/       # Generated expert data (0.06MB, <5% HOLD)
└── enhanced_visualization_demo.py # Advanced analytics demo
```

## 🚀 Quick Start

### Prerequisites

**Windows:**
```powershell
# Install Python 3.8+ and dependencies
pip install torch torchvision torchaudio
pip install pygame numpy gymnasium matplotlib logging collections
pip install tensorflow  # For loading .keras models
```

**Mac/Linux:**
```bash
# Install Python 3.8+ and dependencies
pip install torch torchvision torchaudio
pip install pygame numpy gymnasium matplotlib
pip install tensorflow  # For loading .keras models
```

### Basic Training Commands

**Windows PowerShell:**
```powershell
# Set environment path
$env:PYTHONPATH="local-multiplayer-tetris-main"

# Run competitive visualization (recommended)
python local-multiplayer-tetris-main\localMultiplayerTetris\rl_utils\visualized_training.py

# Run enhanced analytics
python enhanced_visualization_demo.py

# Run integration tests
python test_airl_integration.py

# Generate new expert trajectories
python generate_expert_trajectories.py
```

**Mac/Linux:**
```bash
# Set environment path
export PYTHONPATH="local-multiplayer-tetris-main"

# Run competitive visualization
python local-multiplayer-tetris-main/localMultiplayerTetris/rl_utils/visualized_training.py

# Run enhanced analytics
python enhanced_visualization_demo.py

# Run integration tests
python test_airl_integration.py
```

## 🎮 Training Parameters

### Core AIRL Configuration

```python
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'learning_rate_discriminator': 3e-4,
    'learning_rate_policy': 1e-4,
    'batch_size': 64,
    'buffer_size': 100000,
    'gamma': 0.99,
    'lambda_gae': 0.95,
    'clip_ratio': 0.2,
    'train_discriminator_iters': 5,
    'train_policy_iters': 5,
    'max_episodes': 1000,
    'max_steps_per_episode': 1000
}
```

### Network Architecture

**State Space:** 207 dimensions
- Grid: 20×10 = 200 features (flattened game board)
- Metadata: 7 features (next_piece, hold_piece, current_shape, rotation, x, y, can_hold)

**Action Space:** 41 discrete actions
- Actions 0-39: Placement actions (rotation × 10 + column)
- Action 40: Hold piece

**Discriminator Network:**
```
Input(207) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Linear(1) → Sigmoid
Parameters: 121,857
```

**Policy Network (Actor-Critic):**
```
Feature Extractor: CNN(20×10×3) → Flatten → Linear(416)
Actor: Linear(416) → Linear(128) → Linear(128) → Linear(41) → Softmax
Critic: Linear(416) → Linear(128) → Linear(128) → Linear(1)
Parameters: 145,652
```

### Visualization Parameters

```python
visualization_config = {
    'render_delay': 0.1,      # Seconds between moves (0.05-0.5)
    'show_metrics': True,     # Display real-time analytics
    'save_screenshots': False, # Save episode screenshots
    'episode_length': 500,    # Max steps per visualized episode
    'num_episodes': 5         # Episodes to run
}
```

## 🏆 Competitive Training

### True Multiplayer Environment

The system uses `TrueMultiplayerTetrisEnv` which manages two independent TetrisEnv instances for genuine competitive play:

**Key Features:**
- Independent game states for each player
- Distinct block sequences (separate RNG streams)
- Competitive reward shaping
- Win/loss/draw detection
- Real-time visualization support

### Training Modes

1. **Visualized Training:** Real-time competitive episodes with pygame rendering
2. **Headless Training:** Fast training without visualization
3. **Analytics Mode:** Detailed performance metrics and learning curves

### Competitive Rewards

**Base Rewards (per player):**
- Line clears: 10-80 × level (1-4 lines)
- Game over penalty: -20
- Time penalty: -0.01 per step
- Feature-based shaping: holes, height, bumpiness

**Competitive Bonuses:**
- Survival bonus: +0.1 per step alive
- Winning bonus: +10.0
- Losing penalty: -5.0
- Score advantage: ±0.001 × score_difference (capped at ±2.0)

## 📊 Expert Trajectories

### Dataset Specifications

**Original Expert Data:**
- Files: 11 pickle files (~2.7MB each)
- Total size: 30.09 MB
- Issue: 99-100% HOLD actions (problematic)
- Episodes: 5000 steps each

**Generated Expert Data:**
- Files: 10 pickle files (~0.006MB each)  
- Total size: 0.06 MB
- HOLD usage: <5% (reasonable)
- Episodes: Variable length (terminated properly)

### Expert Data Format

```python
trajectory_structure = {
    'observations': List[Dict],  # TetrisEnv observations
    'actions': List[int],        # Action indices (0-40)
    'rewards': List[float],      # Reward values
    'dones': List[bool],         # Episode termination flags
    'infos': List[Dict]          # Additional metadata
}

observation_format = {
    'grid': np.ndarray,          # Shape (20, 10), dtype=int8
    'next_piece': int,           # Shape ID (0-7)
    'hold_piece': int,           # Shape ID (0-7)
    'current_shape': int,        # Shape ID (0-7)
    'current_rotation': int,     # Rotation (0-3)
    'current_x': int,            # X position (0-9)
    'current_y': int,            # Y position (-4 to 19)
    'can_hold': int              # Boolean flag (0-1)
}
```

## 🔧 Technical Specifications

### System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 1GB free disk space
- CPU: Intel i5 or equivalent

**Recommended:**
- Python 3.9+
- 8GB RAM
- 2GB free disk space
- GPU: NVIDIA RTX series (for faster training)
- CPU: Intel i7 or equivalent

### Performance Benchmarks

**Training Speed:**
- CPU (Intel i7): ~50-100 episodes/hour
- GPU (RTX 3070): ~200-400 episodes/hour

**Memory Usage:**
- Base system: ~200MB
- With visualization: ~500MB
- Large buffer (100k): ~1GB

### Known Issues & Solutions

**Issue 1: Import Errors**
```
Error: "attempted relative import beyond top-level package"
Solution: Set PYTHONPATH correctly and run from project root
```

**Issue 2: Pygame Window Issues**
```
Error: Display surface quit unexpectedly
Solution: Run with headless=True for training, headless=False for visualization
```

**Issue 3: CUDA Out of Memory**
```
Error: RuntimeError: CUDA out of memory
Solution: Reduce batch_size or use device='cpu'
```

## 📈 Monitoring Training

### Key Metrics to Watch

1. **Win Rate Balance:** Should trend toward 50/50 as agents improve
2. **Action Diversity:** Agents should explore all 41 actions
3. **Reward Progression:** Should increase over time
4. **Episode Length:** Should increase as agents get better
5. **Learning Stability:** Loss functions should converge

### Training Logs

Logs are saved in `logs/` directory:
- `training.log`: Main training events
- `airl_demo.log`: Demonstration results
- `tetris_debug.log`: Debug information

### Visualization Metrics

Real-time display includes:
- Win/loss/draw ratios
- Average rewards (last 10 episodes)
- Action diversity scores
- Performance trends
- Episode duration statistics

## 🧪 Testing & Validation

### Integration Tests

Run comprehensive test suite:
```powershell
python test_airl_integration.py
```

Tests include:
- Environment functionality
- Network forward passes
- Expert data loading
- Discriminator training
- Policy updates
- Multiplayer coordination

### Performance Benchmarks

```powershell
python performance_benchmark.py
```

Measures:
- Training speed (episodes/hour)
- Memory usage
- Network inference time
- Environment step time

## 🔬 Advanced Usage

### Custom Expert Generation

```python
# Generate new expert trajectories
python generate_expert_trajectories.py --episodes 20 --max_steps 1000
```

### Hyperparameter Tuning

Key parameters to adjust:
- `learning_rate_discriminator`: 1e-5 to 1e-3
- `learning_rate_policy`: 1e-5 to 1e-3  
- `batch_size`: 32 to 128
- `train_discriminator_iters`: 1 to 10
- `render_delay`: 0.05 to 0.5 (visualization speed)

### Multi-GPU Training

```python
config = {
    'device': 'cuda:0',  # Specify GPU
    'distributed': True,  # Enable if multiple GPUs
    'batch_size': 128     # Increase for better GPU utilization
}
```

## 📚 Research Background

This implementation is based on:
- **AIRL Paper:** "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning"
- **Actor-Critic Methods:** PPO-style policy optimization
- **Competitive Multi-Agent RL:** Self-play training paradigms

### Key Innovations

1. **True Multiplayer:** Independent environments ensuring genuine competition
2. **Competitive Reward Shaping:** Win/loss bonuses and relative performance rewards  
3. **Expert Quality Filtering:** Automatic detection and filtering of low-quality demonstrations
4. **Real-time Visualization:** Live training monitoring with detailed analytics

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Follow code style: Use type hints, docstrings, and comments
4. Add tests for new functionality
5. Submit pull request with detailed description

### Code Organization

- Keep AIRL logic in `rl_utils/`
- Environment modifications in `tetris_env.py`
- Visualization code in `visualized_training.py`
- Tests in `test_*.py` files
- Documentation in `README.md`

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 References

- [AIRL Paper](https://arxiv.org/abs/1710.11248)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Tetris AI Research](https://arxiv.org/abs/1905.01652)

---

**Need Help?** Check the test suite, logs, or create an issue with your specific configuration and error messages. 