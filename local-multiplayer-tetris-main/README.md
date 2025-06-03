# Multiplayer Tetris Agent - Hierarchical Reinforcement Learning System

A sophisticated multi-agent reinforcement learning system for Tetris that implements a 6-phase hierarchical training pipeline with goal-conditioned policies, Random Network Distillation (RND) exploration, and adaptive reward systems.

## 🎯 System Overview

This project implements a state-of-the-art hierarchical RL system for Tetris using:
- **State Model**: Predicts optimal piece placements from current game states
- **Goal-Conditioned Actor-Critic**: Uses state model outputs as goals for action selection
- **Future Reward Predictor**: Estimates long-term value of terminal block placements
- **RND Exploration**: Random Network Distillation for curiosity-driven exploration
- **Adaptive Reward System**: Piece presence rewards that decay over training
- **Unified Training Pipeline**: 6-phase algorithm for robust learning

### Key Features
- ✅ **Goal-Conditioned Learning**: State model outputs directly feed as goals to the actor
- ✅ **RND Exploration**: Curiosity-driven exploration using Random Network Distillation
- ✅ **Adaptive Piece Presence Rewards**: Rewards decrease from 1.0 to 0.0 over first half of training
- ✅ **Consistent Network Parameters**: Centralized configuration across all components
- ✅ **Lines Cleared Tracking**: Episode-by-episode performance monitoring
- ✅ **Blended Reward Prediction**: Future reward predictor integrates with state model
- ✅ **Extended Training**: 1000 total episodes across 50 batches
- ✅ **Updated Feature Weights**: Reduced penalty weights for more balanced learning

## 🏗️ Architecture

### State Representation (410-dimensional)
The system uses a simplified yet comprehensive state vector:
```
- Current Piece Grid: 200 values (20×10 binary grid for falling piece)
- Empty Grid: 200 values (20×10 binary grid for empty spaces)  
- Next Piece: 7 values (one-hot encoding for next piece)
- Metadata: 3 values (current_rotation, current_x, current_y)
Total: 410 dimensions
```

### Action Representation (8-dimensional one-hot)
Actions are encoded as binary vectors:
```
[1,0,0,0,0,0,0,0] = Move Left
[0,1,0,0,0,0,0,0] = Move Right  
[0,0,1,0,0,0,0,0] = Move Down
[0,0,0,1,0,0,0,0] = Rotate Clockwise
[0,0,0,0,1,0,0,0] = Rotate Counter-clockwise
[0,0,0,0,0,1,0,0] = Hard Drop
[0,0,0,0,0,0,1,0] = Hold Piece
[0,0,0,0,0,0,0,1] = No-op
```

### Goal Vector (36-dimensional)
State model outputs optimal placements as goal vectors:
```
- Rotation One-Hot: 4 values (rotation 0-3)
- X Position One-Hot: 10 values (x coordinate 0-9)
- Y Position One-Hot: 20 values (y coordinate 0-19, landing position)
- Value Prediction: 1 value (terminal reward estimate)
- Confidence Score: 1 value (placement confidence)
Total: 36 dimensions
```

## 🧠 Neural Network Components

### 1. State Model
**Purpose**: Predicts optimal piece placements from game states
```python
Input: State vector (410D)
Encoder: 410 → 256 → 256 → 128 (with dropout)
Outputs:
  - Rotation logits (4 classes)
  - X position logits (10 classes) 
  - Y position logits (20 classes)
  - Terminal value (1 value)
```

### 2. Actor-Critic Network (Goal-Conditioned)
**Purpose**: Selects actions based on current state + optimal placement goals
```python
State Features: 410 → 512 → 256 → 128
Goal Encoder: 36 → 64 → 64
Combined Features: 192 (128 state + 64 goal)

Actor: 192 → 256 → 128 → 8 (sigmoid)
Critic: 192 → 256 → 128 → 1
```

### 3. Future Reward Predictor
**Purpose**: Estimates long-term rewards for terminal placements, blends with state model
```python
State Encoder: 410 → 256 → 128
Action Encoder: 8 → 32
Combined: 160 → 256 → 128
Outputs:
  - Immediate reward (1 value)
  - Future value (1 value)
```

### 4. NEW: Random Network Distillation (RND)
**Purpose**: Provides intrinsic motivation for curiosity-driven exploration

#### Random Target Network (Fixed)
```python
Input: State vector (410D)
Architecture: 410 → 512 → 256 → 128 → 64
Parameters: Frozen (never trained)
Output: Fixed random features (64D)
```

#### Predictor Network (Trainable)
```python
Input: State vector (410D)
Architecture: 410 → 512 → 256 → 128 → 64 (with dropout)
Parameters: Trainable
Output: Predicted features (64D)
Loss: MSE(predicted_features, target_features)
```

#### Intrinsic Reward Calculation
```python
prediction_error = MSE(predictor_output, target_output)
intrinsic_reward = normalize(prediction_error)  # Running statistics
exploration_bonus = intrinsic_reward * scale_factor
```

## 🔄 6-Phase Training Algorithm

### Phase 1: RND-Driven Exploration Data Collection
- **RND Exploration Actor** uses Random Network Distillation for curiosity-driven exploration
- **Intrinsic Motivation**: Prediction error on random network provides exploration signal
- Records **ONLY terminal states** (final placement outcomes) with intrinsic rewards
- Generates 2-6 terminal placement simulations per game step (scaled by intrinsic reward)
- Collects (state, placement, terminal_reward, resulting_state, intrinsic_reward) tuples
- **Adaptive Exploration**: High intrinsic reward → more diverse placements, low → conservative

### Phase 2: State Model Learning
- Trains state model on terminal placement data with intrinsic motivation
- Learns to predict optimal rotations and positions
- Uses terminal rewards + intrinsic rewards to weight training importance
- Outputs confidence scores for placement quality

### Phase 3: Future Reward Prediction
- Trains future reward predictor on terminal placements
- Learns to estimate long-term consequences
- Blends predictions with state model values using confidence weighting
- Focuses on terminal state value estimation with RND-enhanced data

### Phase 4: Goal-Conditioned Exploitation Episodes  
- **Goal-Conditioned Policy Rollouts** with adaptive piece presence rewards
- State model generates optimal placement goals
- Actor-critic uses goals for action selection
- **Piece Presence Rewards**: Start at 1.0 per piece, decay to 0.0 over first 500 episodes
- Tracks lines cleared, rewards, and episode length
- Collects experience for policy improvement

### Phase 5: PPO Training
- Trains actor-critic with PPO clipping using enhanced rewards
- Incorporates auxiliary state model loss
- Updates both policy and value functions
- Uses prioritized experience replay with RND-enhanced data

### Phase 6: Evaluation
- Pure exploitation episodes (ε = 0) with piece presence rewards
- Measures policy performance with adaptive reward system
- Tracks improvement over training with RND statistics

## 🎯 Reward Function Design

### Environment Rewards
```python
# Line clearing rewards (base scores)
LINE_CLEAR_BASE = {
    1: 100,    # Single line
    2: 200,    # Double lines  
    3: 400,    # Triple lines
    4: 1600    # Tetris (4 lines)
}

# Additional factors
LEVEL_MULTIPLIER = True      # Multiply by (level + 1)
GAME_OVER_PENALTY = -200     # Heavy penalty for losing
TIME_PENALTY = -0.01         # Small penalty per step
```

### Feature-Based Shaping (UPDATED WEIGHTS)
```python
# Reduced penalty weights for more balanced learning
HOLE_WEIGHT = 0.5           # Penalty per hole created (reduced from 4.0)
MAX_HEIGHT_WEIGHT = 5.0     # Penalty for tall stacks (reduced from 10.0)
BUMPINESS_WEIGHT = 0.2      # Penalty for uneven surface (reduced from 1.0)
```

### NEW: Adaptive Piece Presence Reward System
```python
# Encourages longer games early in training, phases out as agent improves
PIECE_PRESENCE_REWARD = 1.0        # Base reward per piece on board
PIECE_PRESENCE_DECAY_STEPS = 500   # Decay over first 500 episodes (first half)
PIECE_PRESENCE_MIN = 0.0           # No reward after episode 500

# Calculation:
# Episodes 0-500: reward = (1.0 - episode/500) * pieces_on_board
# Episodes 500+:  reward = 0.0
```

### RND Intrinsic Motivation
```python
# Random Network Distillation parameters
INTRINSIC_REWARD_SCALE = 10.0      # Scale factor for intrinsic rewards
RND_LEARNING_RATE = 1e-4           # Learning rate for predictor network
REWARD_NORMALIZATION = True        # Normalize intrinsic rewards
REWARD_CLIPPING = [-5.0, 5.0]      # Clip normalized rewards

# Exploration guidance:
# High prediction error → diverse exploration
# Low prediction error → conservative exploitation  
```

### Exploration rewards (terminal placement evaluation)
```python
EXPLORATION_MAX_HEIGHT_WEIGHT = -0.5   # Negative for height
EXPLORATION_HOLE_WEIGHT = -10.0        # Heavy hole penalty  
EXPLORATION_BUMPINESS_WEIGHT = -0.1    # Surface smoothness
```

### State Model Training Weights
```python
# Higher terminal rewards get more training weight
reward_weight = max(0.1, (terminal_reward + 100) / 200)
total_loss = reward_weight * (rotation_loss + x_loss) + value_loss
```

## 📊 Training Configuration

### Episode Structure (1000 Total Episodes)
```
Total Batches: 50
Episodes per Batch: 20
Total Episodes: 50 × 20 = 1000

Breakdown:
- Exploration: 20 episodes/batch × 50 batches = 1000 exploration episodes
- Exploitation: 20 episodes/batch × 50 batches = 1000 policy episodes  
- Evaluation: 10 episodes/batch × 50 batches = 500 evaluation episodes
```

### Network Hyperparameters
```python
# Learning rates
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3  
STATE_LEARNING_RATE = 1e-3
REWARD_LEARNING_RATE = 1e-3

# Training parameters
GAMMA = 0.99                # Discount factor
PPO_CLIP_RATIO = 0.2        # PPO clipping
EPSILON_START = 1.0         # Initial exploration
EPSILON_MIN = 0.01          # Minimum exploration
EPSILON_DECAY = 0.995       # Decay rate

# Episode limits
MAX_EPISODE_STEPS = 2000    # Extended for longer games
```

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install torch torchvision numpy pygame tensorboard
```

### Training
```bash
# Standard training (1000 episodes)
python -m localMultiplayerTetris.rl_utils.unified_trainer

# With visualization (last episode of each batch)
python -m localMultiplayerTetris.rl_utils.unified_trainer --visualize

# Custom configuration
python -m localMultiplayerTetris.rl_utils.unified_trainer \
    --num_batches 100 \
    --log_dir logs/custom_run \
    --checkpoint_dir checkpoints/custom
```

### Monitoring Progress
```bash
# Launch TensorBoard
tensorboard --logdir logs/unified_training

# Available metrics:
# - Exploration: Terminal rewards, placement success rates, RND intrinsic rewards
# - StateModel: Training losses, auxiliary losses  
# - RewardPredictor: Prediction accuracy, loss trends
# - Exploitation: Episode rewards, steps, lines cleared, piece presence rewards
# - PPO: Actor/critic losses, training efficiency
# - Evaluation: Pure policy performance
# - RND: Intrinsic rewards, prediction errors, exploration diversity
```

## 📈 Key Metrics Tracked

### Episode-Level Metrics
- **Episode Reward**: Total reward accumulated (includes piece presence + intrinsic rewards)
- **Episode Steps**: Number of actions taken
- **Lines Cleared**: Tetris lines completed (primary success metric)
- **Episode Length**: Game duration
- **NEW: Piece Presence Reward**: Adaptive reward based on pieces on board
- **NEW: Piece Presence Decay Factor**: Current decay multiplier (1.0 → 0.0 over 500 episodes)

### RND Exploration Metrics
- **Intrinsic Reward**: Curiosity-driven exploration bonus from prediction error
- **Prediction Error**: MSE between predictor and target network outputs
- **Exploration Diversity**: Variance in intrinsic rewards across states
- **RND Training Loss**: Predictor network learning progress
- **Exploration/Exploitation Balance**: Proportion of high vs low intrinsic reward states

### Training Metrics  
- **State Model Losses**: Rotation, position, and value prediction accuracy
- **Reward Predictor**: Terminal placement value estimation with RND enhancement
- **Actor-Critic**: Policy gradient and value function losses with enhanced rewards
- **Auxiliary Losses**: Joint state model training during policy learning
- **RND Predictor Loss**: Training progress of curiosity network

### Performance Trends
- **Batch Averages**: Mean performance across episodes with adaptive rewards
- **Success Rates**: Fraction of high-reward placements (intrinsic + extrinsic)
- **Training Efficiency**: Successful optimization iterations
- **Confidence Scores**: State model placement certainty
- **Exploration Effectiveness**: RND-driven discovery rate of novel states

## 🔧 Customization

### Modifying Network Architecture
Edit `localMultiplayerTetris/config.py`:
```python
class NetworkConfig:
    class StateModel:
        ENCODER_LAYERS = [410, 512, 256, 128]  # Custom encoder
        DROPOUT_RATE = 0.2                     # Increased dropout
        
    class ActorCritic:
        ACTOR_HIDDEN_LAYERS = [512, 256]      # Larger actor
        GOAL_FEATURES = 128                    # More goal encoding
```

### Adjusting Reward Function
```python
class RewardConfig:
    LINE_CLEAR_BASE = {1: 200, 2: 500, 3: 800, 4: 2000}  # Higher rewards
    HOLE_WEIGHT = 0.2                                      # Even lighter penalty
    MAX_HEIGHT_WEIGHT = 3.0                                # Reduced height penalty
    BUMPINESS_WEIGHT = 0.1                                 # Minimal bumpiness penalty
    GAME_OVER_PENALTY = -500                               # Harsher penalty
    
    # NEW: Piece presence reward customization
    PIECE_PRESENCE_REWARD = 2.0                            # Higher base reward
    PIECE_PRESENCE_DECAY_STEPS = 750                       # Extend decay period
    PIECE_PRESENCE_MIN = 0.5                               # Minimum reward floor
```

### RND Exploration Parameters
```python
class RNDConfig:
    INTRINSIC_REWARD_SCALE = 20.0                          # Stronger curiosity signal
    RND_LEARNING_RATE = 5e-4                               # Faster predictor learning
    REWARD_CLIPPING = [-10.0, 10.0]                        # Wider reward range
    STATE_BUFFER_SIZE = 20000                              # Larger experience buffer
    PREDICTOR_DROPOUT = 0.2                                # Higher dropout for regularization
```

### Training Schedule
```python
class TrainingConfig:
    NUM_BATCHES = 100                    # More batches
    EXPLORATION_EPISODES = 30            # More exploration
    PPO_EPOCHS = 6                       # More PPO updates
    MAX_EPISODE_STEPS = 3000             # Longer episodes
```

## 🧪 Experimental Features

### Advanced Goal Conditioning
The system supports sophisticated goal conditioning where:
- State model confidence affects goal weighting
- Multiple goal candidates can be evaluated
- Goal adaptation based on learning progress

### Multi-Objective Learning
- Simultaneous optimization of multiple metrics
- Weighted combination of rewards
- Pareto-optimal policy discovery

### Curriculum Learning
- Progressive difficulty increase
- Adaptive exploration strategies
- Dynamic reward shaping

## 📚 Research Context

This implementation draws from several key areas:
- **Hierarchical RL**: Goal-conditioned policies with high-level planning
- **Model-Based RL**: State transition learning for better sample efficiency  
- **Multi-Task Learning**: Joint training of multiple prediction tasks
- **Exploration Strategies**: Systematic discovery of terminal state rewards

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**🎮 Ready to train your own Tetris AI? Start with the unified trainer and watch your agent learn to master the game through hierarchical reinforcement learning!** 