# Tetris Reinforcement Learning

This project implements a model-based Monte Carlo Tree Search (MCTS) planning agent using learned StateModel and RewardModel to play Tetris. It includes components for:  
1) *Exploratory state-action probing* via `StateExplorer` to discover valid transitions.  
2) *State transition modeling* via `StateModel` to predict next‐state given (state, action).  
3) *Instruction-conditioned policy* via a modified ActorCritic that takes both current state and a target instruction (the next‐state mask).  
4) *Joint training*: supervised pretraining of actor on one-step expert actions, reinforcement learning with an auxiliary state-model loss.

## Key Components

### 1. StateExplorer (`rl_utils/state_explorer.py`)
- Probes the environment by resetting and applying every action once.  
- Records `(initial_obs, action, next_obs, validity)` tuples and summarizes valid vs invalid actions.
- Bootstraps initial rules for action masking and dataset for state‐model pretraining.

### 2. StateModel (`rl_utils/state_model.py`)
- A neural network that predicts the ideal placement for the current piece.
- Input: flattened state vector (200 grid cells + 6 piece metadata).
- Outputs:
  • `rot_logits` (batch_size × num_rotations): logits over rotations of the current piece.
  • `x_logits` (batch_size × board_width): logits over x-positions for landing.

### 3. MCTSAgent (`rl_utils/mcts_agent.py`)
- Uses learned `StateModel` to simulate next-state given (state, action).
- Uses learned `RewardModel` to predict immediate placement score for (state, action).
- Performs configurable Monte Carlo rollouts (num_simulations, max_depth) to estimate cumulative reward per action.
- `select_action(state)`: picks the action with highest average rollout reward.

### 4. Training Pipeline (`rl_utils/train.py` & `single_player_train.py`)
1. **Exploration**: use `StateExplorer` to collect one-step transitions and derive action‐validity rules.  
2. **Supervised Pretraining**:
   - (Optional) Pretrain `StateModel` on collected transitions.  
   - Pretrain `Actor` to match expert one-step actions using cross-entropy.
3. **Reinforcement Learning**:
   - At each step, generate a target instruction by sampling `StateModel` predictions over valid actions.  
   - Feed `(state, instruction)` into the actor to select actions.  
   - Store transitions `(state, action, reward, next_state, done, info, instruction)` in replay buffer.  
   - Train ActorCritic on policy gradient + value loss conditioned on instructions.  
   - Jointly update `StateModel` via auxiliary MSE loss.

## Configurable Parameters

- **ActorCriticAgent** (`actor_critic.py`): learning rates, gamma, epsilon schedule, batch size, gradient clipping, and auxiliary loss weights.
- **Training Scripts**: number of episodes, save/eval intervals, pretrain epochs, and visualization options.
- **Replay Buffer**: capacity, priority exponent alpha, importance sampling beta.

## Network Architecture and Hyperparameters

### Shared Feature Extractor
- **Grid CNN** (input shape: 1×20×10)  
- Layers:  
  • Conv2d(1 → 32, kernel_size=3, padding=1) + ReLU  (→ 32×20×10)  
  • Conv2d(32 → 32, kernel_size=3, padding=1) + ReLU  (→ 32×20×10)  
  • Conv2d(32 → 32, kernel_size=3, padding=1) + ReLU  (→ 32×20×10)  
- Flatten → feature dim = 32×20×10 = 6400  

- **Piece Embed** (6-dimensional: current_shape, rotation, x, y, next_id, hold_id)  
- Layers:  
  • Linear(6 → 64) + ReLU  
  • Linear(64 → 64) + ReLU  

Combined state feature dimension = 6400 + 64 = 6464

### Instruction Embed  
- Input: flattened next-state vector (200 grid cells + 2 piece IDs = 202 dims)  
- MLP: Linear(202 → 128) + ReLU → outputs 128-d instruction embedding

### Actor Head (Policy)  
- Input: concatenated state+instruction features (6464 + 128 = 6592 dims)  
- MLP:  
  • Linear(6592 → 512) + ReLU  
  • Linear(512 → 256) + ReLU  
  • Linear(256 → *A*) + Softmax  
- *A* = action_dim (8 actions)

### Critic Head (Value)  
- Input: concatenated state+instruction features (6464 + 128 = 6592 dims)  
- MLP:  
  • Linear(6592 → 512) + ReLU  
  • Linear(512 → 256) + ReLU  
  • Linear(256 → 1) → state-value estimate

### StateModel (Next-State Predictor)
- Input: concatenated [state (202 dims), action one-hot (8 dims)] = 210 dims  
- MLP:  
  • Linear(210 → 256) + ReLU  
  • Linear(256 → 256) + ReLU  
- Outputs:  
  • **grid_out**: Linear(256 → 200) + Sigmoid  
  • **piece_out**: Linear(256 → 2) (logits)

### Hyperparameters
```yaml
actor_learning_rate: 1e-4
critic_learning_rate: 1e-3
gamma (discount): 0.99
epsilon_start: 1.0
epsilon_min: 0.01
epsilon_decay: 0.995
batch_size: 64
gradient_clip_norm: 1.0
state_model_lr: 1e-3
pretrain_epochs: 5
replay_buffer_capacity: 100_000
priority_alpha: 0.6
importance_beta_start: 0.4
importance_beta_inc: 0.001
```

## Project Structure
```
localMultiplayerTetris/
├── rl_utils/
│   ├── mcts_agent.py     # Monte Carlo Tree Search planning agent
│   ├── state_model.py    # Next-state prediction network
│   ├── reward_model.py   # Immediate reward predictor
│   ├── state_explorer.py # Exploratory transition collector
│   ├── replay_buffer.py  # Prioritized experience replay
│   └── single_player_train.py  # Single-player MCTS training script
├── tetris_env.py         # Gym‐style Tetris environment
├── game.py               # Core Tetris mechanics
├── piece_utils.py        # Validity and shape formatting
└── constants.py          # Game constants (grid size, colors)
```  

## Installation & Run

1. Install dependencies:
```powershell
pip install -r requirements.txt
```
2. Launch training:
```powershell
python -m localMultiplayerTetris.rl_utils.train
```
3. For single-player specific options:
```powershell
python -m localMultiplayerTetris.rl_utils.single_player_train --visualize False
```

## Monitoring
- Checkpoints saved under `checkpoints/`.  
- TensorBoard logs under `logs/tensorboard/`:  
```powershell
tensorboard --logdir logs/tensorboard
```

---
*Experiment and customize further: adjust pretraining epochs, instruction sampling strategy, or network sizes to optimize performance.*
