# Multiplayer Tetris AIRL Implementation

This repository implements Adversarial Imitation Learning (AIRL) for Tetris agents with support for both single-agent learning and dual-agent competitive play.

## Overview

The implementation includes:
- Expert trajectory collection from trained agents
- AIRL framework with compact neural networks
- Dual-agent competitive environment
- Comprehensive training and evaluation scripts

## Network Architecture

### AIRL Discriminator Network
The discriminator learns to distinguish expert from policy trajectories and provides shaped rewards.

**Architecture:**
- **Input**: State vector (207 dimensions) + Action (41 classes)
- **Feature Extractor**: 
  - Grid processing: 200 → 64 → 32 (ReLU)
  - Scalar processing: 7 → 16 (ReLU)
  - Combined: 48 → 64 (ReLU)
- **Reward Network**: Feature(64) + Action(41) → 64 → 32 → 1 (ReLU)
- **Value Network**: Feature(64) → 32 → 1 (ReLU)
- **Total Parameters**: ~23,000

### AIRL Policy Network (Actor-Critic)
The policy network learns from AIRL-shaped rewards.

**Architecture:**
- **Input**: State vector (207 dimensions)
- **Shared Features**: 207 → 128 → 128 (ReLU)
- **Policy Head**: 128 → 64 → 41 (ReLU → Softmax)
- **Value Head**: 128 → 64 → 1 (ReLU)
- **Total Parameters**: ~63,000

### Dual-Agent Extensions
For competitive play, the state space is extended to include opponent information.

**Extended State**: 227 dimensions
- Standard Tetris state: 207 dimensions
- Opponent grid summary: 10 dimensions (column heights)
- Opponent metadata: 10 dimensions (score, danger zones, etc.)

### Total Network Complexity
- **Single Agent**: ~86,000 parameters
- **Dual Agent**: ~94,000 parameters (per agent)
- **Maximum per agent**: <100,000 parameters (well under 300k limit)

## Configuration Parameters

### Training Parameters
```python
# AIRL Configuration
batch_size = 128
learning_rate_discriminator = 1e-4
learning_rate_policy = 1e-4
gamma = 0.99  # Discount factor
tau = 0.95    # GAE parameter

# Network Architecture
hidden_dim = 128     # Policy network hidden dimension
feature_dim = 64     # Discriminator feature dimension

# Training Schedule
max_episodes = 5000
discriminator_steps = 5  # Disc updates per policy update
gradient_penalty_coeff = 10.0
entropy_coeff = 0.01
value_loss_coeff = 0.5

# Evaluation
eval_interval = 100   # Episodes between evaluations
save_interval = 200   # Episodes between checkpoints
log_interval = 10     # Episodes between logs
```

### Environment Parameters
```python
# Tetris Environment
state_dim = 207      # Grid(200) + scalars(7)
action_dim = 41      # 40 placements + 1 hold
max_steps = 25000    # Max steps per episode

# Dual Agent Environment
dual_state_dim = 227 # Standard(207) + opponent(20)
max_game_steps = 10000
```

## Installation and Setup

### Prerequisites
```bash
pip install torch torchvision
pip install gym pygame numpy
pip install tensorboard
```

### Repository Structure
```
local-multiplayer-tetris-main/
├── localMultiplayerTetris/
│   ├── tetris_env.py              # Single-agent environment
│   ├── dual_agent_env.py          # Dual-agent environment
│   ├── rl_utils/
│   │   ├── airl.py                # AIRL implementation
│   │   ├── trajectory_collector.py # Trajectory saving
│   │   └── actor_critic.py        # Original AC agent
├── replay_agent_with_trajectories.py  # Expert data collection
├── train_airl.py                  # AIRL training script
├── train_dual_agents.py           # Dual-agent training
└── README_AIRL.md                 # This file
```

## Usage Instructions

### 1. Train Base Agent (Optional)
If you don't have a trained agent, first train a baseline:
```bash
cd local-multiplayer-tetris-main
python -m localMultiplayerTetris.rl_utils.single_player_train \
    --num_episodes 5000 \
    --save-interval 500 \
    --visualize False
```

### 2. Collect Expert Trajectories
Generate expert demonstrations from a trained agent:
```bash
python replay_agent_with_trajectories.py \
    --checkpoint checkpoints/actor_critic_episode_5000.pt \
    --num-episodes 100 \
    --min-reward 0
```

Alternative with auto-checkpoint detection:
```bash
python replay_agent_with_trajectories.py \
    --auto-checkpoint \
    --num-episodes 100 \
    --visualize
```

### 3. Train AIRL Agent
Train an agent using adversarial imitation learning:
```bash
python train_airl.py \
    --max-episodes 5000 \
    --batch-size 128 \
    --lr-discriminator 1e-4 \
    --lr-policy 1e-4 \
    --expert-data-path expert_trajectories/expert_dataset.pkl
```

With visualization (slower):
```bash
python train_airl.py \
    --max-episodes 2000 \
    --visualize \
    --log-level DEBUG
```

### 4. Train Dual Agents (Competitive)
Train two agents to compete against each other:
```bash
python train_dual_agents.py \
    --max-episodes 10000 \
    --batch-size 128 \
    --lr-policy 1e-4 \
    --save-trajectories
```

With pretrained initialization:
```bash
python train_dual_agents.py \
    --max-episodes 10000 \
    --pretrained-agent1 airl_checkpoints/airl_final.pt \
    --pretrained-agent2 airl_checkpoints/airl_final.pt \
    --visualize
```

## Advanced Configuration

### Custom Network Architecture
Modify network sizes in the configuration:
```bash
python train_airl.py \
    --hidden-dim 64 \      # Smaller networks
    --feature-dim 32 \
    --batch-size 64
```

### Expert Data Quality Control
Filter expert trajectories by performance:
```bash
python replay_agent_with_trajectories.py \
    --auto-checkpoint \
    --num-episodes 200 \
    --min-reward 10.0      # Only keep high-reward episodes
```

### Hyperparameter Tuning
Key parameters to adjust:
- `--lr-discriminator`: Higher values (1e-3) for faster discriminator learning
- `--lr-policy`: Lower values (5e-5) for more stable policy learning
- `--batch-size`: Larger batches (256) for more stable gradients
- `--hidden-dim`: Smaller networks (64) for faster training
- `--feature-dim`: Balance between capacity and efficiency

## Monitoring and Evaluation

### TensorBoard Monitoring
```bash
tensorboard --logdir logs/airl_tensorboard
# or for dual agents
tensorboard --logdir logs/dual_agent_tensorboard
```

**Key Metrics to Monitor:**
- `Discriminator/expert_accuracy`: Should stay around 0.7-0.9
- `Discriminator/policy_accuracy`: Should stay around 0.1-0.3
- `Policy/policy_loss`: Should decrease over time
- `Episode/Reward`: Should improve with training
- `Eval/Score`: Periodic evaluation performance

### Log Analysis
Training logs provide detailed information:
```bash
tail -f airl_training.log           # Monitor AIRL training
tail -f dual_agent_training.log     # Monitor dual-agent training
```

## Performance Expectations

### Training Time
- **AIRL Training**: ~2-4 hours for 5000 episodes (GPU)
- **Dual-Agent Training**: ~4-8 hours for 10000 episodes (GPU)
- **Expert Collection**: ~30-60 minutes for 100 episodes

### Expected Performance
- **Expert Agent**: 50-200 average score, 10-50 lines cleared
- **AIRL Agent**: Should reach 70-80% of expert performance
- **Dual Agents**: Balanced win rates (45-55%) indicate good competition

### Resource Requirements
- **Memory**: ~2-4 GB RAM during training
- **Storage**: ~100 MB for checkpoints, ~50 MB for trajectories
- **GPU**: Optional but recommended (10x speedup)

## Troubleshooting

### Common Issues

**1. Expert data not found**
```bash
# Ensure expert trajectories were collected first
ls expert_trajectories/expert_dataset.pkl
```

**2. Low discriminator accuracy**
- Reduce discriminator learning rate
- Increase batch size
- Check expert data quality

**3. Policy not improving**
- Increase policy learning rate
- Reduce discriminator steps per policy update
- Check reward shaping in discriminator

**4. Memory issues**
- Reduce batch size
- Reduce network dimensions
- Clear trajectory buffers more frequently

### Debug Mode
Run with debug logging for detailed information:
```bash
python train_airl.py --log-level DEBUG
```

## File Outputs

### Checkpoints
- `airl_checkpoints/airl_episode_*.pt`: Periodic AIRL checkpoints
- `airl_checkpoints/airl_final.pt`: Final trained AIRL agent
- `dual_agent_checkpoints/agent*_*.pt`: Dual agent checkpoints

### Trajectories
- `expert_trajectories/`: Expert demonstration data
- `trajectories_agent1/`: Competitive agent 1 trajectories
- `trajectories_agent2/`: Competitive agent 2 trajectories

### Logs
- `airl_training.log`: AIRL training logs
- `dual_agent_training.log`: Dual-agent training logs
- `logs/*/`: TensorBoard event files

## Research and Extensions

### Potential Improvements
1. **Curriculum Learning**: Gradually increase game difficulty
2. **Population Training**: Train against multiple opponent types
3. **Transfer Learning**: Pre-train on simpler Tetris variants
4. **Multi-Objective Rewards**: Balance multiple game objectives
5. **Hierarchical Actions**: Learn high-level strategies

### Experimental Configurations
```bash
# Fast prototyping
python train_airl.py --max-episodes 1000 --hidden-dim 64 --batch-size 64

# High-capacity training
python train_airl.py --max-episodes 10000 --hidden-dim 256 --feature-dim 128

# Competitive evaluation
python train_dual_agents.py --max-episodes 20000 --save-trajectories
```

## Citation and References

This implementation is based on:
- Adversarial Imitation Learning (Ho & Ermon, 2016)
- Actor-Critic methods for reinforcement learning
- OpenAI Gym environment standards

For research use, please cite the original AIRL paper and this implementation. 