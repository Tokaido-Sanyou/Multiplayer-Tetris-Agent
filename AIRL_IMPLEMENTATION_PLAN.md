# AIRL Implementation Plan for Multiplayer Tetris

## Project Overview

This document outlines the comprehensive plan for implementing Adversarial Inverse Reinforcement Learning (AIRL) for the multiplayer Tetris environment, using expert trajectories from the DQN-based `tetris-ai-master` model.

## Architecture Overview

### Two-Stage System Design

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

### Key Components

1. **Expert Trajectory Source**: `tetris-ai-master` DQN model
2. **Training Environment**: `local-multiplayer-tetris-main` TetrisEnv
3. **Bridge**: `dqn_adapter.py` (already exists)
4. **AIRL Implementation**: New discriminator + policy networks

## Implementation Phases

### Phase 1: Core AIRL Components (Week 1-2)

#### 1.1 AIRL Agent (`airl_agent.py`) ✅ COMPLETED
- **Discriminator Network**: Distinguishes expert vs learner trajectories
  - Input: State-action pairs (207 + 41 dimensions)
  - Architecture: [256, 128] hidden layers
  - Output: Binary classification logit
  - AIRL reward: `log(D/(1-D))`

- **AIRLAgent Class**: Main training orchestrator
  - Integrates discriminator + existing actor-critic policy
  - Alternating updates between discriminator and policy
  - Uses AIRL rewards instead of environment rewards

#### 1.2 Expert Trajectory Loader (`expert_loader.py`) ✅ COMPLETED
- **ExpertTrajectoryLoader**: Processes expert data
  - Loads from `expert_trajectories/*.pkl` files
  - Filters based on HOLD action percentage (<20%)
  - Converts to unified state representation (207 features)
  - Provides batching for training

### Phase 2: Training Infrastructure (Week 2-3)

#### 2.1 AIRL Training Script (`airl_train.py`) ✅ COMPLETED
- **AIRLTrainer**: Main training orchestrator
  - Environment setup (TetrisEnv single-player, headless)
  - Expert data loading and validation
  - Learner data collection via policy rollouts
  - Alternating discriminator/policy updates
  - Evaluation and checkpointing

#### 2.2 State Representation Bridge
```python
# Unified feature extraction (207 dimensions)
def extract_features(observation):
    grid = observation['grid'].flatten()        # 200 features (20x10)
    next_piece = [observation['next_piece']]    # 1 feature
    hold_piece = [observation['hold_piece']]    # 1 feature  
    current_shape = [observation['current_shape']]  # 1 feature
    current_rotation = [observation['current_rotation']]  # 1 feature
    current_x = [observation['current_x']]      # 1 feature
    current_y = [observation['current_y']]      # 1 feature
    can_hold = [observation['can_hold']]        # 1 feature
    return np.concatenate([grid, next_piece, hold_piece, current_shape, 
                          current_rotation, current_x, current_y, can_hold])
```

### Phase 3: Evaluation and Analysis (Week 3-4)

#### 3.1 Evaluation Framework (`airl_evaluate.py`) ✅ COMPLETED
- **Comprehensive Testing**:
  - AIRL agent vs Expert DQN vs Random baseline
  - Performance metrics: score, episode length, lines cleared
  - Action distribution analysis
  - Discriminator accuracy evaluation

#### 3.2 Visualization and Analysis
- Training progress plots
- Agent comparison charts
- Action distribution heatmaps
- Discriminator performance analysis

### Phase 4: Integration and Optimization (Week 4)

#### 4.1 Hyperparameter Tuning
- Discriminator/policy learning rates balance
- Update frequency scheduling
- Network architecture optimization
- Training stability improvements

#### 4.2 Multiplayer Extension
- Adapt single-player AIRL to multiplayer setting
- Multi-agent discriminator training
- Competitive/cooperative reward structures

## Technical Specifications

### Network Architectures

#### Discriminator Network
```python
Input: [state(207) + action_onehot(41)] = 248 dimensions
├── Linear(248 → 256) + ReLU
├── Linear(256 → 128) + ReLU  
├── Linear(128 → 64) + ReLU
└── Linear(64 → 1)  # Binary classification logit
```

#### Policy Network (Existing Actor-Critic)
```python
Feature Extractor:
├── Grid processing: Conv layers or fully connected
├── Piece embeddings: Learned embeddings for pieces
└── Concatenated features → 256 dimensions

Actor: 256 → 128 → 41 (action probabilities)
Critic: 256 → 128 → 1 (state value)
```

### Training Algorithm

```python
for iteration in range(training_iterations):
    # 1. Collect learner data
    learner_data = collect_episodes(policy, num_episodes=5)
    
    # 2. Update discriminator
    expert_batch = expert_loader.get_batch(batch_size=64)
    learner_batch = sample_from_buffer(learner_data, batch_size=64)
    discriminator_loss = train_discriminator(expert_batch, learner_batch)
    
    # 3. Update policy using AIRL rewards
    airl_rewards = discriminator.get_reward(learner_batch)
    policy_loss = train_policy(learner_batch, airl_rewards)
    
    # 4. Evaluate and log
    if iteration % eval_freq == 0:
        metrics = evaluate_policy(policy, num_episodes=10)
        log_metrics(discriminator_loss, policy_loss, metrics)
```

### Data Flow

```
Expert Trajectories (tetris-ai-master)
├── Load from expert_trajectories/*.pkl
├── Filter by quality (HOLD% < 20%)
├── Extract state features (207D)
├── Convert actions to one-hot (41D)
└── Store in ExpertTrajectoryLoader

Learner Data Collection (AIRL policy)
├── Run policy in TetrisEnv
├── Extract same state features (207D)
├── Store transitions in ReplayBuffer
└── Sample for discriminator training

AIRL Training Loop
├── Sample expert batch
├── Sample learner batch  
├── Update discriminator (expert=1, learner=0)
├── Compute AIRL rewards for learner data
├── Update policy using AIRL rewards
└── Repeat
```

## Expected Challenges and Solutions

### 1. State Representation Mismatch
**Challenge**: Expert uses 4-feature DQN states, environment has rich observations
**Solution**: ✅ Unified feature extraction function converts rich obs → 207D vectors

### 2. Action Space Alignment  
**Challenge**: Expert actions may not directly map to environment actions
**Solution**: Use existing `dqn_adapter.py` + action enumeration functions

### 3. Training Stability
**Challenge**: GAN-like training can be unstable
**Solutions**:
- Careful learning rate tuning (discriminator vs policy)
- Gradient clipping
- Update frequency scheduling
- Early stopping based on discriminator accuracy

### 4. Expert Data Quality
**Challenge**: Expert trajectories may have suboptimal patterns (high HOLD usage)
**Solution**: ✅ Filtering pipeline removes poor-quality trajectories

### 5. Evaluation Methodology
**Challenge**: How to measure AIRL success vs expert performance
**Solutions**:
- Multi-metric evaluation (score, episode length, lines cleared)
- Action distribution similarity analysis
- Discriminator confidence analysis
- Human evaluation of play quality

## Usage Instructions

### Training
```bash
cd local-multiplayer-tetris-main/localMultiplayerTetris/rl_utils

# Basic training
python airl_train.py --expert-dir ../../../expert_trajectories --iterations 1000

# With Weights & Biases logging
python airl_train.py --expert-dir ../../../expert_trajectories --use-wandb --iterations 2000

# Custom configuration
python airl_train.py --config custom_config.json
```

### Evaluation
```bash
# Evaluate trained model
python airl_evaluate.py --checkpoint checkpoints/airl_best_500.pt --episodes 100

# Compare with baseline
python airl_evaluate.py --checkpoint checkpoints/airl_final.pt --output-dir results_final
```

### Configuration Example
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

## Success Metrics

### Quantitative Metrics
1. **Performance**: AIRL agent score within 80% of expert performance
2. **Stability**: Consistent performance across multiple training runs
3. **Efficiency**: Faster learning than pure RL (sample efficiency)
4. **Discriminator**: >80% accuracy distinguishing expert vs learner

### Qualitative Metrics  
1. **Play Style**: Similar action patterns to expert (low HOLD usage)
2. **Strategy**: Efficient piece placement and line clearing
3. **Robustness**: Good performance across different game states

## Timeline and Milestones

- ✅ **Week 1**: Core AIRL components (discriminator, agent) - COMPLETED
- ✅ **Week 2**: Expert loader and training infrastructure - COMPLETED  
- ✅ **Week 3**: Training script and evaluation framework - COMPLETED
- **Week 4**: Integration testing, hyperparameter tuning, evaluation

## File Structure

```
local-multiplayer-tetris-main/localMultiplayerTetris/
├── rl_utils/
│   ├── airl_agent.py          # ✅ Core AIRL implementation
│   ├── expert_loader.py       # ✅ Expert trajectory processing  
│   ├── airl_train.py          # ✅ Training script
│   ├── airl_evaluate.py       # ✅ Evaluation framework
│   └── actor_critic.py        # Existing policy network
├── tetris_env.py              # Existing environment
├── dqn_adapter.py             # Existing state/action bridge
└── checkpoints/               # Model saves

expert_trajectories/           # Expert data source
├── trajectory_ep000000.pkl
├── trajectory_ep000001.pkl
└── ...

tetris-ai-master/             # Expert model source  
├── dqn_agent.py
├── tetris.py
└── sample.keras
```

## Next Steps

1. **Immediate**: Test the complete pipeline end-to-end
2. **Short-term**: Hyperparameter tuning and stability improvements
3. **Medium-term**: Multiplayer extension and competitive scenarios
4. **Long-term**: Advanced IRL techniques (SQIL, ValueDice) comparison

## Research Extensions

1. **Multi-Agent AIRL**: Extend to competitive multiplayer setting
2. **Curriculum Learning**: Progressive difficulty in expert demonstrations
3. **Domain Adaptation**: Transfer to other puzzle games
4. **Preference Learning**: Human feedback integration
5. **State-only IRL**: Learning from observations without actions 