# Tetris Reinforcement Learning

This project implements a Deep Q-Learning (DQN) agent to play Tetris. The agent uses a combination of Convolutional Neural Networks (CNN) and Multi-Layer Perceptrons (MLP) to learn optimal strategies for playing Tetris.

## Network Design

The network architecture is defined in `localMultiplayerTetris/rl_utils/dqn_agent.py` and consists of three main components:

1. **Grid Processing (CNN)**
   - Input: 20x10 grid matrix
   - Architecture:
     - Conv2D(1→32, kernel=3, padding=1) + ReLU
     - Conv2D(32→64, kernel=3, padding=1) + ReLU
     - Conv2D(64→64, kernel=3, padding=1) + ReLU

2. **Piece Processing (MLP)**
   - Input: 48 features (3 pieces × 16 features each)
   - Architecture:
     - Linear(48→128) + ReLU
     - Linear(128→128) + ReLU

3. **Combined Network**
   - Input: Concatenated features from CNN and MLP
   - Architecture:
     - Linear(12800 + 128→512) + ReLU
     - Linear(512→256) + ReLU
     - Linear(256→7)  # 7 possible actions

## Configurable Parameters

### DQN Agent Parameters (`dqn_agent.py`)
```python
# Network Parameters
learning_rate = 1e-4      # Learning rate for optimizer
gamma = 0.99             # Discount factor for future rewards
epsilon = 1.0            # Initial exploration rate
epsilon_min = 0.01       # Minimum exploration rate
epsilon_decay = 0.995    # Decay rate for exploration
batch_size = 64          # Batch size for training
target_update = 10       # Frequency of target network updates
gradient_clip = 1.0      # Maximum gradient norm
```

### Training Parameters (`train.py`)
```python
# Training Parameters
num_episodes = 1000      # Total number of training episodes
save_interval = 100      # Save model every N episodes
eval_interval = 50       # Evaluate agent every N episodes
```

### Replay Buffer Parameters (`replay_buffer.py`)
```python
# Buffer Parameters
capacity = 100000        # Maximum number of transitions to store
alpha = 0.6             # Priority exponent for sampling
beta = 0.4              # Importance sampling exponent
beta_increment = 0.001  # Rate of beta increase
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
python -m localMultiplayerTetris.rl_utils.train
```

## Project Structure

```
localMultiplayerTetris/
├── rl_utils/
│   ├── dqn_agent.py     # DQN network and agent implementation
│   ├── replay_buffer.py # Prioritized experience replay buffer
│   └── train.py         # Training script
├── tetris_env.py        # Tetris environment implementation
├── game.py             # Core game mechanics
├── piece.py            # Tetris piece definitions
└── constants.py        # Game constants and configurations
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

1. **Network Architecture**: Modify the `DQN` class in `dqn_agent.py` to change the network structure.
2. **Training Parameters**: Adjust parameters in `train.py` to modify training behavior.
3. **Reward Structure**: Modify the reward function in `tetris_env.py` to change the learning objective.
4. **Action Space**: Modify the action space in `tetris_env.py` to add or remove possible actions.

## Contributing

Feel free to submit issues and enhancement requests!
