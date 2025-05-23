# Tetris Reinforcement Learning

This project implements an Actor-Critic agent to play Tetris. The agent uses a combination of Convolutional Neural Networks (CNN) and Multi-Layer Perceptrons (MLP) to learn optimal strategies for playing Tetris.

## Network Design

The network architecture is defined in `localMultiplayerTetris/rl_utils/actor_critic.py` and consists of four main components:

1. **Shared Feature Extractor**
   - Input: 20x10 grid matrix and piece information
   - Architecture:
     - Grid Processing (CNN):
       - Conv2D(1→32, kernel=3, padding=1) + ReLU
       - Conv2D(32→64, kernel=3, padding=1) + ReLU
       - Conv2D(64→64, kernel=3, padding=1) + ReLU
     - Piece Processing (MLP):
       - Linear(48→128) + ReLU
       - Linear(128→128) + ReLU

2. **Actor Network (Policy)**
   - Input: Shared features
   - Architecture:
     - Linear(12800 + 128→512) + ReLU
     - Linear(512→256) + ReLU
     - Linear(256→7) + Softmax  # 7 possible actions
   - Output: Action probabilities

3. **Critic Network (Value)**
   - Input: Shared features
   - Architecture:
     - Linear(12800 + 128→512) + ReLU
     - Linear(512→256) + ReLU
     - Linear(256→1)  # State value
   - Output: State value estimate

4. **Epsilon-Greedy Actor**
   - Implements exploration strategy
   - Gradually reduces exploration rate
   - Uses actor network for action selection

## Configurable Parameters

### Actor-Critic Parameters (`actor_critic.py`)
```python
# Network Parameters
actor_lr = 1e-4          # Learning rate for actor
critic_lr = 1e-3         # Learning rate for critic
gamma = 0.99             # Discount factor for future rewards
epsilon = 1.0            # Initial exploration rate
epsilon_min = 0.01       # Minimum exploration rate
epsilon_decay = 0.995    # Decay rate for exploration
batch_size = 64          # Batch size for training
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
│   ├── actor_critic.py  # Actor-Critic network implementation
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

1. **Network Architecture**: Modify the `ActorCritic` class in `actor_critic.py` to change the network structure.
2. **Training Parameters**: Adjust parameters in `train.py` to modify training behavior.
3. **Reward Structure**: Modify the reward function in `tetris_env.py` to change the learning objective.
4. **Action Space**: Modify the action space in `tetris_env.py` to add or remove possible actions.
5. **Exploration Strategy**: Modify the epsilon-greedy actor in `actor_critic.py` to implement different exploration strategies.

## Contributing

Feel free to submit issues and enhancement requests!
