# AIRL Multiplayer Tetris Project

## 🎯 Project Overview

This project implements **Adversarial Inverse Reinforcement Learning (AIRL)** for competitive multiplayer Tetris. The system learns to play Tetris by observing expert demonstrations from a pre-trained DQN agent, with the ultimate goal of training competitive agents that can engage in multiplayer matches.

### Key Features
- ✅ **Single-Player AIRL**: Complete implementation of AIRL for single-player Tetris
- ✅ **Expert Trajectory Integration**: Seamless integration with DQN expert demonstrations  
- ✅ **Flexible Architecture**: Modular design supporting both single-player and future multiplayer scenarios
- 🔧 **Multiplayer Extension**: Planned implementation for competitive two-player AIRL training

## 🏗️ Architecture Overview

```
┌─────────────────────┐    ┌─────────────────────┐
│   tetris-ai-master  │    │local-multiplayer-   │
│   (Expert Source)   │───▶│tetris-main (AIRL)   │
│                     │    │                     │
│ • DQN Agent         │    │ • TetrisEnv         │
│ • 4-feature states  │    │ • Rich observations │
│ • Expert traject.   │    │ • AIRL training     │
└─────────────────────┘    └─────────────────────┘
```

### Core Components

1. **Expert Source**: `tetris-ai-master` - DQN model providing expert demonstrations
2. **Training Environment**: `localMultiplayerTetris` - Gym-style Tetris environment
3. **AIRL Framework**: Discriminator + Policy networks for imitation learning
4. **State Bridge**: Unified feature extraction (207 dimensions) for consistent training

## 📦 Installation & Setup

### Prerequisites
```powershell
# Ensure Python 3.8+ is installed
python --version

# Install required packages
pip install torch torchvision numpy gym pygame matplotlib seaborn
```

### Optional Dependencies
```powershell
# For training monitoring and logging
pip install wandb

# For evaluation visualizations  
pip install pandas
```

### Project Setup
```powershell
# Clone or navigate to project directory
cd C:\path\to\Multiplayer-Tetris-Agent

# Verify installation
python test_airl_integration.py
```

## 🚀 Quick Start Guide

### 1. Verify Installation
```powershell
# Run integration tests
python test_airl_integration.py
```
Expected output: `🎉 All tests passed! AIRL implementation is ready for training.`

### 2. Prepare Expert Data
Ensure expert trajectories are available:
```powershell
# Check expert trajectories directory
dir expert_trajectories\
# Should contain: trajectory_ep000000.pkl, trajectory_ep000001.pkl, etc.

# Optional: Filter low-quality expert data
python filter_expert_trajectories.py

# Optional: Analyze expert action patterns
python analyze_expert_actions.py
```

### 3. Train AIRL Agent
```powershell
# Navigate to training directory
cd local-multiplayer-tetris-main\localMultiplayerTetris\rl_utils

# Basic training (500 iterations)
python airl_train.py --expert-dir ..\..\..\expert_trajectories --iterations 500

# Training with Weights & Biases logging
python airl_train.py --expert-dir ..\..\..\expert_trajectories --use-wandb --iterations 1000

# Custom configuration
python airl_train.py --config custom_config.json
```

### 4. Evaluate Trained Agent
```powershell
# Evaluate best checkpoint
python airl_evaluate.py --checkpoint checkpoints\airl_best_500.pt --episodes 100

# Compare with baselines
python airl_evaluate.py --checkpoint checkpoints\airl_final.pt --output-dir results_final
```

## 📁 Directory Structure

```
Multiplayer-Tetris-Agent/
├── local-multiplayer-tetris-main/
│   └── localMultiplayerTetris/
│       ├── rl_utils/                     # AIRL Implementation
│       │   ├── airl_agent.py            # Core AIRL discriminator + agent
│       │   ├── expert_loader.py         # Expert trajectory processing
│       │   ├── airl_train.py            # Training orchestrator
│       │   ├── airl_evaluate.py         # Evaluation framework
│       │   ├── actor_critic.py          # Policy network (existing)
│       │   └── replay_buffer.py         # Experience replay (existing)
│       ├── tetris_env.py                # Single-player Tetris environment
│       ├── game.py                      # Multiplayer game mechanics
│       ├── player.py                    # Player state management
│       └── utils.py                     # Game utilities
├── tetris-ai-master/                    # Expert DQN source
│   ├── dqn_agent.py                    # Pre-trained DQN expert
│   ├── tetris.py                       # DQN game logic
│   └── sample.keras                    # Expert model weights
├── expert_trajectories/                 # Expert demonstration data
│   ├── trajectory_ep000000.pkl
│   └── ...
├── test_airl_integration.py            # Integration test suite
├── AIRL_IMPLEMENTATION_PLAN.md         # Technical implementation details
├── MULTIPLAYER_AIRL_PLAN.md           # Multiplayer extension roadmap
└── README.md                           # This file
```

## ⚙️ Configuration Options

### Training Configuration
```json
{
  "expert_trajectory_dir": "../../../expert_trajectories",
  "max_expert_trajectories": 10,
  "max_hold_percentage": 15.0,
  
  "policy_lr": 3e-4,
  "discriminator_lr": 1e-4,
  "batch_size": 64,
  "gamma": 0.99,
  
  "training_iterations": 1000,
  "episodes_per_iteration": 5,
  "updates_per_iteration": 10,
  
  "eval_freq": 50,
  "save_freq": 100,
  "use_wandb": true
}
```

### Environment Parameters
- **State Space**: 207 dimensions (grid + piece metadata)
- **Action Space**: 41 discrete actions (40 placements + 1 hold)
- **Episode Length**: Maximum 25,000 steps
- **Reward Structure**: Line clearing + height management + AIRL rewards

## 📊 Evaluation Metrics

### Quantitative Metrics
- **Performance**: Score, episode length, lines cleared
- **Efficiency**: Sample efficiency vs pure RL
- **Discriminator**: Accuracy distinguishing expert vs learner
- **Stability**: Consistency across training runs

### Qualitative Assessment
- **Play Style**: Similarity to expert action patterns
- **Strategy**: Piece placement efficiency and line clearing
- **Robustness**: Performance across different game states

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```powershell
# If you see "attempted relative import" errors:
cd local-multiplayer-tetris-main\localMultiplayerTetris\rl_utils
python -c "import sys; print(sys.path)"

# Ensure the project root is in Python path
set PYTHONPATH=%PYTHONPATH%;C:\path\to\Multiplayer-Tetris-Agent
```

#### Memory Issues
```powershell
# Reduce batch size for limited memory
python airl_train.py --batch-size 32 --expert-dir ..\..\..\expert_trajectories

# Use CPU instead of GPU
python airl_train.py --device cpu --expert-dir ..\..\..\expert_trajectories
```

#### Training Instability
- **Discriminator overpowering**: Reduce discriminator learning rate
- **Policy not learning**: Increase policy learning rate
- **Mode collapse**: Reduce update frequencies, add regularization

### Performance Optimization
```powershell
# Use GPU acceleration (if available)
python -c "import torch; print(torch.cuda.is_available())"

# Monitor training with TensorBoard (if wandb unavailable)
pip install tensorboard
# (Add tensorboard logging to training script)
```

## 🎯 Next Steps & Roadmap

### Phase 1: Single-Player AIRL ✅
- [x] Core AIRL implementation
- [x] Expert trajectory integration
- [x] Training and evaluation framework
- [x] Comprehensive testing

### Phase 2: Multiplayer Environment 🔧
- [ ] Two-player competitive environment
- [ ] Extended state representation (421 dimensions)
- [ ] Garbage line mechanics integration
- [ ] Competitive reward structure

### Phase 3: Competitive AIRL 🎯
- [ ] Self-play training infrastructure
- [ ] Population-based training
- [ ] Competitive strategy emergence
- [ ] Tournament evaluation system

## 📖 Technical Details

For in-depth technical information, see:
- **[AIRL_IMPLEMENTATION_PLAN.md](AIRL_IMPLEMENTATION_PLAN.md)**: Single-player AIRL details
- **[MULTIPLAYER_AIRL_PLAN.md](MULTIPLAYER_AIRL_PLAN.md)**: Multiplayer extension strategy

### Key Algorithms
- **AIRL**: Adversarial Inverse Reinforcement Learning
- **Actor-Critic**: Policy gradient with value function baseline
- **Experience Replay**: Prioritized experience replay buffer
- **SRS**: Super Rotation System for authentic Tetris mechanics

### State Representation
```python
# Single-player state (207 dimensions)
grid_features = observation['grid'].flatten()        # 200 (20x10)
piece_features = [
    observation['next_piece'],                       # 1
    observation['hold_piece'],                       # 1  
    observation['current_shape'],                    # 1
    observation['current_rotation'],                 # 1
    observation['current_x'],                        # 1
    observation['current_y'],                        # 1
    observation['can_hold']                          # 1
]
# Total: 200 + 7 = 207 dimensions
```

## 🤝 Contributing

### Code Standards
- Use PowerShell for Windows commands
- Maintain import consistency (module-based preferred)
- Include comprehensive error handling
- Add unit tests for new features

### Development Workflow
```powershell
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Implement changes
# ... code changes ...

# 3. Test implementation
python test_airl_integration.py

# 4. Run training verification
python airl_train.py --expert-dir ..\..\..\expert_trajectories --iterations 50

# 5. Submit pull request
```

## 📝 License

This project builds upon existing Tetris implementations and incorporates AIRL research. Please respect the original licenses and cite appropriate research papers when using this code.

## 🙏 Acknowledgments

- **Tetris AI Master**: Original DQN implementation for expert demonstrations
- **Local Multiplayer Tetris**: Base multiplayer game mechanics
- **AIRL Research**: Adversarial Inverse Reinforcement Learning methodology

---

## 🆘 Getting Help

If you encounter issues:

1. **Check Integration Tests**: Run `python test_airl_integration.py`
2. **Review Logs**: Check training logs in `logs/` directory
3. **Verify Configuration**: Ensure expert trajectories are accessible
4. **Resource Monitoring**: Monitor GPU/CPU usage during training

For technical questions about the AIRL implementation, refer to the detailed documentation in `AIRL_IMPLEMENTATION_PLAN.md`.
