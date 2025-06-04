# Tetris Reinforcement Learning

This project implements an Actor-Critic agent to play Tetris. The agent leverages a hybrid neural network architecture combining Convolutional Neural Networks (CNNs) for grid processing and Multi-Layer Perceptrons (MLPs) for piece information, enabling effective learning of Tetris strategies.

## Network Architecture

The architecture is defined in `localMultiplayerTetris/rl_utils/actor_critic.py` and consists of the following components:

1. **Shared Feature Extractor**
   - **Inputs:**
     - 20×10 Tetris grid (game state)
     - Next piece ID and Hold piece ID (scalar)
     - Current piece metadata:
       - shape ID (scalar)
       - rotation index (0–3)
       - x and y coordinates (integers)
   - **Grid Processing (CNN):**
     - `Conv2d(1, 8, kernel_size=3, padding=1)` + ReLU  # maintains 20×10
     - `Conv2d(8, 8, kernel_size=3, padding=1)` + ReLU  # maintains 20×10
     - Output is flattened (8×20×10 = 1600 features)
   - **Piece Processing (MLP):**
     - `Linear(6, 32)` + ReLU    # next, hold, curr_shape, rotation, x, y
     - `Linear(32, 32)` + ReLU
   - **Output:** Concatenated feature vector (1600 + 32 = 1632 dimensions)

2. **Actor Network (Policy)**
   - **Input:** Shared features (1632-dim)
   - **Architecture:**
     - `Linear(1632, 128)` + ReLU
     - `Linear(128, 64)` + ReLU
     - `Linear(64, 8)` + Softmax    # 8 possible actions
   - **Output:** Action probabilities

3. **Critic Network (Value)**
   - **Input:** Shared features (1632-dim)
   - **Architecture:**
     - `Linear(1632, 128)` + ReLU
     - `Linear(128, 64)` + ReLU
     - `Linear(64, 1)`              # state value estimate
   - **Output:** Scalar value

4. **Epsilon-Greedy Actor**
   - Implements an exploration strategy by selecting random actions with probability ε (epsilon), which decays over time.
   - Uses the actor network for action selection when not exploring.

## Configurable Parameters

### Actor-Critic Parameters (`actor_critic.py`)
```python
# Network & Scheduling Parameters
actor_lr = 1e-4          # Learning rate for actor
critic_lr = 1e-3         # Learning rate for critic
# Discount factor schedule: starts at 0.9, ends at 0.99
gamma_start = 0.9
gamma_end = 0.99
# Exploration schedule: starts at 1.0, ends at 0.05 over schedule_episodes
epsilon_start = 1.0
epsilon_end = 0.05
schedule_episodes = 10000
batch_size = 64          # Batch size for training
gradient_clip = 1.0      # Maximum gradient norm
```

### Training Parameters (`single_player_train.py`)
```bash
# Training script CLI arguments
--num_episodes N        # Total number of training episodes
--save_interval N       # Save model every N episodes
--eval_interval N       # Evaluate agent every N episodes
--visualize [True|False] # Render environment during training
--checkpoint PATH       # Load weights from checkpoint
--no_eval               # Disable evaluation during training
--verbose               # Enable per-step logging
```

## Vectorized Training (`vector_train.py`)

The vectorized trainer leverages multiple environments in parallel for more efficient data collection and supports periodic evaluation.

**Usage:**
```bash
python -m localMultiplayerTetris.rl_utils.vector_train \
  --num-envs 4 \
  --episodes 10000 \
  --save-interval 500 \
  --eval-interval 1000 \
  --checkpoint checkpoints/actor_critic_episode_1000.pt \
  --no-eval \
  --headless-eval True
```

**Arguments:**
- `--num-envs N`           : Number of parallel environments (default: 4)
- `--episodes N`           : Total number of training episodes (global across envs) (default: 10000)
- `--save-interval N`      : Save checkpoints every N completed episodes (default: 100)
- `--eval-interval N`      : Run evaluation every N completed episodes (default: 500)
- `--checkpoint PATH`      : Path to load an existing model checkpoint
- `--no-eval`              : Disable periodic evaluation during training
- `--headless-eval [True|False]`: Run evaluation without rendering (default: True)

**TensorBoard:**
Training logs are written to `logs/vectorized_tensorboard`. To monitor:
```bash
tensorboard --logdir logs/vectorized_tensorboard
```

## Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
numpy>=1.19.2
torch>=1.7.0
pygame>=2.0.0
gym>=0.17.3
matplotlib>=3.3.2  # For visualization
tensorboard>=2.4.0  # For training visualization
```

## Installation Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/local-multiplayer-tetris.git
cd local-multiplayer-tetris
```

2. Create a virtual environment (recommended):
```bash
# Using venv (Python 3)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the training script:
```bash
python -m localMultiplayerTetris.rl_utils.single_player_train --num_episodes 10000 --visualize False
```

## Project Structure
```
local-multiplayer-tetris-main/
├── play_agent.py         # CLI launcher to load and play agent
├── setup.py              # Package setup
├── README.md             # Project overview and instructions
├── requirements.txt      # Python dependencies
├── checkpoints/          # Saved model weights
├── logs/                 # Training metrics and TensorBoard logs
└── localMultiplayerTetris/
    ├── __main__.py       # Entry point for `python -m localMultiplayerTetris`
    ├── tetris_env.py     # Gym environment implementation
    ├── game.py           # Core game mechanics
    ├── piece.py          # Piece definitions and rotation
    ├── player.py         # Player state and update logic
    ├── action_handler.py # Map actions to game moves
    ├── piece_utils.py    # Shape formatting and collision checks
    ├── utils.py          # Grid creation and loss checking
    ├── constants.py      # Game constants and wall kick data
    ├── play_agent.py     # Module to launch GUI play
    └── rl_utils/         # RL training utilities
        ├── actor_critic.py         # Actor-Critic network implementation
        ├── replay_buffer.py        # Prioritized experience replay
        ├── train.py                # Single-player training script (deprecated)
        ├── single_player_train.py  # Active training entrypoint
        └── vector_train.py         # Vectorized parallel training script
```

## Training Output

The training process will create two directories:
- `checkpoints/`: Contains saved model weights
- `logs/`: Contains training metrics and TensorBoard logs

You can monitor training progress using TensorBoard:
```bash
tensorboard --logdir=logs
```

## Customization

1. **Network Architecture:** Modify the `ActorCritic` class in `actor_critic.py` to change the network structure.
2. **Training Parameters:** Adjust parameters in `train.py` to modify training behavior.
3. **Reward Structure:** Modify the reward function in `tetris_env.py` to change the learning objective.
4. **Action Space:** Modify the action space in `tetris_env.py` to add or remove possible actions.
5. **Exploration Strategy:** Modify the epsilon-greedy actor in `actor_critic.py` to implement different exploration strategies.

## Contributing

Feel free to submit issues and enhancement requests!
