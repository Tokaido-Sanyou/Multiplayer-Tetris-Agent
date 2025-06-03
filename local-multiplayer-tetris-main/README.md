# Multiplayer Tetris Agent - Revolutionary Dream-Enhanced Reinforcement Learning System

A sophisticated multi-agent reinforcement learning system for Tetris that implements both a **6-phase hierarchical training pipeline** and a revolutionary **8-phase dream-enhanced training system** with explicit goal achievement through synthetic dream practice.

## 🎯 System Overview

This project implements two state-of-the-art RL training systems for Tetris:

### **🌙 NEW: Dream-Enhanced Training System**
Revolutionary approach where actor learns explicit goal achievement through synthetic dream practice:
- **Dream Environment**: Simulates realistic state transitions biased toward goal achievement  
- **Explicit Goal Matcher**: Neural network learning direct goal→action mapping
- **Dream Trajectory Generator**: Creates synthetic perfect goal achievement sequences
- **Dream-Reality Bridge**: Transfers dream knowledge to real actor execution
- **8-Phase Training**: Enhanced pipeline with dream generation and transfer phases
- **Expected Improvement**: Goal success rate from 8.8% → 60-80%

### **🎯 Standard Hierarchical Training System** 
- **State Model**: Predicts optimal piece placements from current game states
- **Goal-Conditioned Actor-Critic**: Uses state model outputs as goals for action selection with future state prediction
- **Future Reward Predictor**: Estimates long-term value of terminal block placements
- **Enhanced RND Exploration**: Terminal value-based curiosity with novelty tracking for unvisited states
- **Adaptive Reward System**: Piece presence rewards that decay over training
- **6-Phase Training Pipeline**: Robust learning with comprehensive monitoring

### Key Features
- ✅ **🌙 REVOLUTIONARY DREAM FRAMEWORK**: Explicit goal achievement through synthetic practice
- ✅ **Goal-Conditioned Learning**: State model outputs directly feed as goals to the actor
- ✅ **Enhanced RND Exploration**: Terminal value-based curiosity with novelty detection for unvisited states
- ✅ **Future State Prediction**: Actor-critic enhanced with future state prediction head
- ✅ **Adaptive Piece Presence Rewards**: Rewards decrease from 1.0 to 0.0 over first half of training
- ✅ **Deterministic Terminal State Exploration**: Systematic coverage of state space
- ✅ **Comprehensive Model Persistence**: Save/load all components including RND state and future predictors
- ✅ **Consistent Network Parameters**: Centralized configuration across all components
- ✅ **Multi-Attempt + HER System**: Enhanced exploration with hindsight experience replay

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

## 🌙 NEW: Dream-Enhanced Training Components

### 1. TetrisDreamEnvironment
**Purpose**: Simulates realistic state transitions biased toward goal achievement
```python
Input: Current state (410D) + Action + Goal vector (36D)
Output: Dream next state (410D) biased toward goal completion
Bias Strength: 0.7 (configurable)
Noise Level: 0.1 (prevents overfitting)
```

### 2. ExplicitGoalMatcher
**Purpose**: Learns direct goal→action mapping through dream practice
```python
State Encoder: 410 → 256 → 128
Goal Encoder: 36 → 128 → 64  
Combined: 192 → 256 → 128 → 8 (softmax)
Training: Supervised learning on dream trajectories
```

### 3. DreamTrajectoryGenerator
**Purpose**: Creates synthetic trajectories where actor practices goal achievement
```python
Dream Length: 15-20 steps per trajectory
Quality Scoring: 0-1 based on goal achievement
Quality Threshold: 0.6 for high-quality dreams
Trajectory Buffer: 10,000 experiences
```

### 4. DreamRealityBridge
**Purpose**: Transfers dream learning to real actor execution
```python
Dream Training Phase: Goal matcher learns from synthetic trajectories
Reality Transfer Phase: Knowledge distillation to actor network
Dream Guidance: Weighted blending of dream + actor policies
Dream Weaning: Gradual reduction from 1.0 to 0.1 over training
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

### 2. Enhanced Actor-Critic Network (Goal-Conditioned + Future State Prediction)
**Purpose**: Selects actions based on current state + optimal placement goals + predicts future states
```python
State Features: 410 → 512 → 256 → 128
Goal Encoder: 36 → 64 → 64
Combined Features: 192 (128 state + 64 goal)

Actor: 192 → 256 → 128 → 8 (sigmoid)
Critic: 192 → 256 → 128 → 1
Future State Predictor: 192 → 256 → 128 → 410 (predicts next state)
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

### 4. Random Network Distillation (RND)
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

## 🔄 Training Algorithms

### 🌟 NEW: Staged Unified Training Algorithm
**Purpose**: To address the "moving target" problem where the actor tries to learn from a state model that is also rapidly changing. This staged approach first stabilizes the state model, then trains the actor on these stable goals, and finally fine-tunes both.

**3-Stage Schedule** (configurable, e.g., 300 batches total):
1.  **Stage 1: State Model Pretraining** (e.g., Batches 0-149)
    *   **Focus**: Intensively train the `StateModel`.
    *   `ActorCritic` training is **disabled**.
    *   **Goal**: Learn optimal placements, line clearing strategies, and produce stable, high-quality goal vectors.
    *   *Intensity*: State model may undergo multiple training epochs per batch.
    *   *Evaluation*: At the end of this stage, the quality of the state model's goals (consistency, optimality, diversity) is evaluated.

2.  **Stage 2: Actor Training with Frozen Goals** (e.g., Batches 150-249)
    *   **Focus**: Intensively train the `ActorCritic` agent.
    *   `StateModel` parameters are **frozen**.
    *   **Goal**: Actor learns to achieve the stable, high-quality goals provided by the pretrained state model. Gradients from the actor **do not** flow back to the state model (`goal_vector.detach()` is used).
    *   *Intensity*: Actor (PPO) may undergo multiple training epochs per batch.

3.  **Stage 3: Joint Fine-tuning** (e.g., Batches 250-299)
    *   **Focus**: Fine-tune both the `StateModel` and `ActorCritic` agent.
    *   Both models are trained, and gradients are allowed to flow through the goals.
    *   **Goal**: Achieve optimal alignment and performance by allowing both models to adapt to each other.

**Benefits**:
-   **Stable Learning**: Reduces oscillations and instability caused by the actor chasing a moving target (the state model).
-   **Improved Goal Quality**: Ensures the state model produces meaningful and reliable goals before the actor starts learning from them.
-   **Better Actor Performance**: The actor can learn more effectively when guided by consistent and sensible goals.

### 🌙 8-Phase Dream-Enhanced Training Algorithm

**Revolutionary training system with explicit goal achievement through synthetic dreams:**

#### Phase 1: Enhanced Terminal Value-Based RND Exploration
- Same as standard training: **Enhanced RND Exploration Actor** for curiosity-driven exploration
- Records terminal states with enhanced rewards for dream learning

#### Phase 2: State Model Learning  
- Trains state model on terminal placement data to generate placement goals
- Outputs confidence scores for placement quality estimation

#### Phase 3: Future Reward Prediction
- Trains future reward predictor on terminal placements for value estimation
- Blends predictions with state model values using confidence weighting

#### **Phase 4: 🌙 DREAM GENERATION (NEW)**
- **Dream Trajectory Generator** creates synthetic perfect goal achievement sequences
- **Explicit Goal Matcher** learns direct goal→action mapping through supervised learning on dreams
- Generates 30+ dream episodes per batch with 15-20 steps each
- Tracks dream quality scores (0-1) based on goal achievement success

#### **Phase 5: 🌉 DREAM-REALITY TRANSFER (NEW)**
- **Dream-Reality Bridge** transfers dream knowledge to real actor execution
- Knowledge distillation: Actor learns to mimic goal matcher's optimal actions
- 150+ transfer steps per batch with gradient clipping for stability
- Validates dream quality (>0.6 threshold) before transfer

#### **Phase 6: 🎮 DREAM-GUIDED EXPLOITATION (ENHANCED)**
- Actor uses **dream guidance** for real environment actions
- Weighted blending: `dream_weight × dream_policy + (1-dream_weight) × actor_policy`
- **Dream weaning**: Gradually reduce dream dependence from 1.0 to 0.1 over training
- Multi-attempt mechanism with hindsight experience replay for enhanced learning

#### Phase 7: Dream-Enhanced PPO Training
- Standard PPO training but with dream-enhanced actor network
- Actor has learned explicit goal achievement from dream practice

#### Phase 8: Dual Evaluation (Goal + Game Performance)
- Evaluates both goal achievement rate and traditional game performance
- Tracks alignment between goal-focused learning and actual game success

**Expected Results**: Goal success rate improvement from 8.8% → 60-80%

---

### 🎯 6-Phase Standard Hierarchical Training Algorithm

#### Phase 1: Enhanced Terminal Value-Based RND Exploration
- **Enhanced RND Exploration Actor** uses terminal value prediction for curiosity-driven exploration
- **Terminal Value Focus**: RND rewards based on terminal state values rather than prediction error
- **Piece-Focused Episodes**: Each episode explores one specific piece type (I, O, T, S, Z, J, L) for consistent learning
- **Episode Distribution**: Automatically adjusts episode count to multiples of 7 for even piece coverage
- **Novelty Detection**: Tracks visited terminal states and strongly encourages exploration of unvisited ones
- **Smart Terminal Generation**: 4-10 terminal placement attempts per step, scaled by intrinsic reward
- Records **ONLY terminal states** with enhanced rewards: terminal_value + intrinsic_bonus + novelty_bonus
- **Unvisited State Bias**: Novel terminal states receive 5x bonus, revisited states get minimal reward
- **Distinct State Tracking**: Reports new terminal states discovered per batch for progress monitoring
- Collects (state, placement, enhanced_terminal_reward, resulting_state, novelty_score, target_piece_type) tuples

#### Phase 2: State Model Learning
- Trains state model on terminal placement data with intrinsic motivation
- Learns to predict optimal rotations and positions
- Uses terminal rewards + intrinsic rewards to weight training importance
- Outputs confidence scores for placement quality

#### Phase 3: Future Reward Prediction
- Trains future reward predictor on terminal placements
- Learns to estimate long-term consequences
- Blends predictions with state model values using confidence weighting
- Focuses on terminal state value estimation with RND-enhanced data

#### Phase 4: Goal-Conditioned Exploitation Episodes  
- **Goal-Conditioned Policy Rollouts** with adaptive piece presence rewards
- State model generates optimal placement goals
- Actor-critic uses goals for action selection
- **Piece Presence Rewards**: Start at 1.0 per piece, decay to 0.0 over first 500 episodes
- Tracks lines cleared, rewards, and episode length
- Collects experience for policy improvement

#### Phase 5: PPO Training
- Trains actor-critic with PPO clipping using enhanced rewards
- Incorporates auxiliary state model loss
- Updates both policy and value functions
- Uses prioritized experience replay with RND-enhanced data

#### Phase 6: Evaluation
- Pure exploitation episodes (ε = 0) with piece presence rewards
- Measures policy performance with adaptive reward system
- Tracks improvement over training with RND statistics

## 🚀 Training Commands

### **🌙 Dream-Enhanced Training (RECOMMENDED)**
```bash
# Navigate to the RL utils directory
cd local-multiplayer-tetris-main/localMultiplayerTetris/rl_utils

# Run dream-enhanced training with minimal configuration
python unified_trainer_dream.py --num_batches 1 --dream_episodes 5

# Run extended dream-enhanced training  
python unified_trainer_dream.py --num_batches 25 --dream_episodes 30

# Run with deterministic exploration for reproducible results
python unified_trainer_dream.py --num_batches 10 --dream_episodes 20 --exploration_mode deterministic

# Run with visualization for the last episode of each batch
python unified_trainer_dream.py --num_batches 5 --dream_episodes 15 --visualize
```

### **🎯 Standard Hierarchical Training**
```bash
# Navigate to the RL utils directory  
cd local-multiplayer-tetris-main/localMultiplayerTetris/rl_utils

# Run standard training with RND exploration
python unified_trainer.py --num_batches 25 --exploration_mode rnd

# Run with deterministic exploration for systematic coverage
python unified_trainer.py --num_batches 20 --exploration_mode deterministic  

# Run with visualization enabled
python unified_trainer.py --num_batches 10 --visualize --exploration_mode rnd
```

### **⚙️ Command Line Options**

#### Dream-Enhanced Training Options:
```bash
--num_batches INT        # Number of training batches (default: 25)
--dream_episodes INT     # Dream episodes per batch (default: 30) 
--dream_transfer_steps INT # Transfer learning steps (default: 150)
--exploration_mode STR   # 'rnd', 'random', 'deterministic' (default: 'rnd')
--visualize             # Enable visualization for last episode per batch
```

#### Standard Training Options:
```bash
--num_batches INT       # Number of training batches (default: 50)
--exploration_mode STR  # 'rnd', 'random', 'deterministic' (default: 'rnd')  
--visualize            # Enable visualization for last episode per batch
--log_dir STR          # Custom logging directory
--checkpoint_dir STR   # Custom checkpoint directory
```

### **🧪 Testing and Validation**
```bash
# Test the dream framework components
python test_dream_framework.py

# Test deterministic exploration system
python test_deterministic_explorer.py

# Test multi-attempt + HER training
python test_multi_attempt_training.py

# Test system integration
python test_integration.py
```

### **📁 Directory Structure**
```
local-multiplayer-tetris-main/
├── localMultiplayerTetris/
│   ├── rl_utils/
│   │   ├── unified_trainer_dream.py    # 🌙 Dream-enhanced training
│   │   ├── unified_trainer.py          # 🎯 Standard training
│   │   ├── dream_framework.py          # 🌙 Dream components
│   │   ├── actor_critic.py             # Goal-conditioned actor-critic
│   │   ├── state_model.py              # Placement prediction model
│   │   ├── rnd_exploration.py          # RND exploration system
│   │   └── ...
│   └── config.py                       # Centralized configuration
├── test_dream_framework.py             # 🌙 Dream framework tests
├── test_deterministic_explorer.py      # Deterministic exploration tests
└── README.md                           # This file
```

## 📊 Streamlined Console Reporting

The system provides clean, real-time monitoring with immediate phase results and concise batch summaries:

### Phase-by-Phase Reporting
Each training phase reports its results immediately upon completion:

```
🔍 Phase 1: RND Terminal State Exploration (Batch 1)
Starting RND exploration: 21 episodes (3 per piece type)...
✅ RND Exploration completed:
   • Total terminal placements: 800
   • Episodes per piece type: 3
   • Unique terminal states discovered: 156
   • New distinct states this batch: 47
   • Average terminal value: 12.45
   • RND learning progress - Mean: 0.1234, Std: 0.0567

📊 Phase 1 Results:
   • Terminal rewards: 12.46 ± 8.23
   • Success rate: 67.5% (540/800)
   • Intrinsic motivation: 0.345 ± 0.123
   • Novel states discovered: 156
   • Distinct terminal states this batch: 47

🎯 Phase 2: State Model Training (Batch 1)
📊 Phase 2 Results:
   • Total loss: 0.0457 (improvement: 12.3%)
   • Rotation loss: 0.0123
   • Position loss: 0.0235
   • Value loss: 0.0099

🔮 Phase 3: Future Reward Prediction (Batch 1)
📊 Phase 3 Results:
   • Total loss: 0.0346
   • Reward prediction: 0.0189
   • Value prediction: 0.0157

🎮 Phase 4: Policy Exploitation Episodes (Batch 1)
📊 Phase 4 Results:
   • Episode rewards: 145.67 ± 23.45
   • Episode steps: 89.3 ± 15.2
   • Lines cleared: 3.2 ± 1.8
   • Piece presence decay: 0.876 (reward: 2.456)

🏋️ Phase 5: PPO Policy Training (Batch 1)
📊 Phase 5 Results:
   • Actor loss: 0.001234
   • Critic loss: 0.002345
   • Future state loss: 0.003456
   • Training success rate: 100.0%

📊 Phase 6: Policy Evaluation (Batch 1)
📊 Phase 6 Results:
   • Pure policy reward: 156.78
   • Pure policy steps: 94.5
```

### Concise Batch Summaries
At the end of each batch, a clean summary shows key metrics:

```
================================================================================
📊 BATCH 1 SUMMARY
================================================================================
📈 PROGRESS: 2.0% complete • Episode 40/1000 • ε=0.9950

🔍 EXPLORATION: 12.5 • 68% • +47
🎯 STATE MODEL: 0.0457 • +12%
🔮 REWARD PRED: 0.0346
🎮 EXPLOITATION: 145.7 • 3.2 • 88%
🏋️ PPO TRAINING: 0.0012 • 0.0023 • 0.0035
📊 EVALUATION: 156.8
================================================================================
```

### Enhanced Model Persistence
The system now saves comprehensive checkpoints including:
- **All Network States**: Actor-critic, state model, future reward predictor
- **All Optimizers**: Including future state predictor optimizer
- **RND Exploration State**: Visited terminal states, value history, learning progress
- **Training Progress**: Episodes completed, epsilon decay, performance metrics
- **Backward Compatibility**: Graceful loading of older checkpoints

### Checkpoint Management
```python
# Automatic saving every 10 batches + latest checkpoint
checkpoint_batch_10.pt    # Batch 10 state
checkpoint_batch_20.pt    # Batch 20 state
latest_checkpoint.pt      # Most recent state (updated every batch)

# Loading checkpoints
trainer.load_checkpoint('checkpoints/latest_checkpoint.pt')
# ✅ Checkpoint loaded successfully!
#    • Batch: 25, Episodes: 500
#    • Exploration data: 800 placements
#    • Epsilon: 0.7788
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

## 🚀 How to Run

### 1. Standard Unified Training
```bash
python -m localMultiplayerTetris.rl_utils.unified_trainer --num_batches 50 --exploration_mode rnd --visualize
```
-   `--num_batches`: Total number of training batches (e.g., 50 batches * 20 exploration + 20 exploitation episodes = 2000 total episodes by default).
-   `--exploration_mode`: `rnd` (default), `random`, or `deterministic`.
-   `--visualize`: Add this flag to see the Tetris game rendered for the last exploitation episode of each batch.
-   `--log_dir`: Specify log directory (default: `logs/unified_training`).
-   `--checkpoint_dir`: Specify checkpoint directory (default: `checkpoints/unified`).

### 2. 🌟 NEW: Staged Unified Training (Recommended for Improved Stability)
```bash
python -m localMultiplayerTetris.rl_utils.staged_unified_trainer --num_batches 300 --exploration_mode rnd
```
-   `--num_batches`: Total number of training batches across all stages (e.g., 300). The script internally divides these into pretraining, actor training, and joint fine-tuning.
-   `--exploration_mode`: `rnd` (default), `random`, or `deterministic`.
-   `--visualize`: Add this flag to see rendering.
-   `--log_dir`: Specify log directory (default: `logs/staged_unified_training`).
-   `--checkpoint_dir`: Specify checkpoint directory (default: `checkpoints/staged_unified`).

### 3. Dream-Enhanced Training
```bash
python -m localMultiplayerTetris.rl_utils.unified_trainer_dream --num_batches 100 --exploration_mode rnd --visualize
```
-   Uses the `unified_trainer_dream.py` script. Parameters are similar to standard training.

### 💡 Configuration
// ... existing code ...
### 📈 Monitoring Training
-   **TensorBoard**: Comprehensive logs are saved in the specified `log_dir`.
    ```bash
    tensorboard --logdir logs/staged_unified_training
    # or logs/unified_training, logs/dream_enhanced_training
    ```
-   **Console Output**: Detailed batch summaries and phase-specific results are printed.
-   **Log Files**: `unified_training.log`, `staged_unified_training.log`, or `dream_training.log` capture all console output.
// ... existing code ... 