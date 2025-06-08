# Training System Commands

This document lists all available commands for training the Tetris AI systems.

## 1. Basic DQN Training (Locked Model)

### Command: `python train_redesigned_agent.py`

Train the redesigned locked state DQN agent with stable architecture.

#### Basic Usage
```bash
# Default training (1000 episodes, auto-resume)
python train_redesigned_agent.py

# Custom episode count
python train_redesigned_agent.py --episodes 2000

# Start fresh training (ignore checkpoints)
python train_redesigned_agent.py --no-resume
```

#### All Parameters
```bash
python train_redesigned_agent.py \
  --episodes 1000 \                    # Number of episodes (default: 1000)
  --save-interval 100 \                # Save checkpoint every N episodes (default: 100)
  --no-resume \                        # Start fresh training (default: auto-resume)
  --learning-rate 0.00005 \            # Learning rate (default: 0.00005)
  --gamma 0.99 \                       # Discount factor (default: 0.99)
  --epsilon-start 1.0 \                # Initial epsilon (default: 1.0)
  --epsilon-end 0.01 \                 # Final epsilon (default: 0.01)
  --epsilon-decay-steps 50000 \        # Epsilon decay steps (default: 50000)
  --batch-size 32 \                    # Batch size (default: 32)
  --memory-size 100000 \               # Memory buffer size (default: 100000)
  --target-update-freq 1000 \          # Target network update frequency (default: 1000)
  --device cuda                        # Device: auto/cuda/cpu (default: auto)
```

#### Expected Output
- Episode progress with reward, pieces, lines, loss, epsilon
- Automatic checkpoint saving every 100 episodes
- Final performance metrics
- Training time and statistics

---

## 2. Actor-Locked Hierarchical Training

### Command: `python train_actor_locked_system.py`

Train the hierarchical Actor-Locked system with Hindsight Experience Replay.

#### Basic Usage
```bash
# Default training (10 actor trials per state)
python train_actor_locked_system.py

# Custom actor trials
python train_actor_locked_system.py --actor-trials 20

# With visualization
python train_actor_locked_system.py --show-visualization
```

#### All Parameters
```bash
python train_actor_locked_system.py \
  --episodes 1000 \                     # Number of episodes (default: 1000)
  --save-interval 100 \                 # Save checkpoint every N episodes (default: 100)
  --no-resume \                         # Start fresh training (default: auto-resume)
  --actor-trials 10 \                   # Actor trials per state (default: 10)
  --locked-model-path path/to/model.pt \ # Pre-trained locked model (optional)
  --actor-learning-rate 0.0001 \        # Actor learning rate (default: 0.0001)
  --show-visualization \                # Show text-based game visualization
  --visualization-interval 100 \        # Show visualization every N episodes (default: 100)
  --device cuda                         # Device: auto/cuda/cpu (default: auto)
```

#### Expected Output
- Episode progress with locked loss, actor loss, actor success rate
- Action comparisons when visualization is enabled
- Hindsight experience replay statistics
- Both locked and actor model checkpoints

---

## 3. Checkpoint Management

### Resume Training
Both training scripts automatically resume from the latest checkpoint unless `--no-resume` is specified.

### Checkpoint Locations
```
checkpoints/
├── redesigned_agent_episode_100.pt        # Basic DQN checkpoints
├── redesigned_agent_episode_100_history.json
├── actor_locked_episode_100_locked.pt     # Actor-Locked checkpoints
├── actor_locked_episode_100_actor.pt
└── actor_locked_episode_100_history.json
```

### Manual Checkpoint Loading
```bash
# Load specific checkpoint for basic training
python train_redesigned_agent.py --episodes 500  # Will auto-resume from latest

# Load specific locked model for Actor-Locked training
python train_actor_locked_system.py --locked-model-path checkpoints/redesigned_agent_final.pt
```

---

## 4. Visualization Modes

### Text-Based Game Visualization
```bash
# Show visualization every 100 episodes
python train_actor_locked_system.py --show-visualization

# Show visualization every 50 episodes
python train_actor_locked_system.py --show-visualization --visualization-interval 50
```

#### Visualization Output
- **INITIAL STATE**: Board representation and piece info
- **Action Comparison**: Locked vs Actor action choices per step
- **Episode Summary**: Total reward, pieces placed, lines cleared

### Visualization Types Available
1. **Locked Model Only**: Basic DQN playing (via train_redesigned_agent.py)
2. **Actor Training**: Actor model trials during training (via --show-visualization)
3. **Combined System**: Both locked and actor working together (default mode)

---

## 5. Performance Monitoring

### Training Metrics
Both training scripts provide comprehensive metrics:

#### Basic DQN Metrics
- Episode reward (average of last 10)
- Pieces placed per episode
- Lines cleared per episode
- Training loss
- Epsilon value (exploration rate)

#### Actor-Locked Metrics
- Locked model loss
- Actor model loss
- Actor success rate (goal achievement)
- Action comparison statistics

### Expected Performance Ranges
```
Basic DQN (after 1000 episodes):
- Reward: -200 to -150 (higher is better)
- Pieces: 20-30 per episode
- Lines: 0-2 per episode
- Loss: Stable around 5-20

Actor-Locked (after 1000 episodes):
- Actor Success Rate: 0.1-0.8 (higher is better)
- Combined reward improvement over basic DQN
```

---

## 6. System Requirements

### Hardware
- **GPU**: CUDA-compatible GPU recommended (automatic detection)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 1GB for checkpoints and logs

### Software
- **OS**: Windows 10/11 with PowerShell
- **Python**: 3.8+
- **PyTorch**: CUDA-enabled version recommended

### Verification Commands
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check device
python -c "import torch; print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
```

---

## 7. Troubleshooting Commands

### Debug Training Issues
```bash
# Test basic functionality (5 episodes)
python train_redesigned_agent.py --episodes 5 --no-resume

# Test Actor-Locked with minimal setup
python train_actor_locked_system.py --episodes 5 --actor-trials 3 --no-resume

# Force CPU mode if GPU issues
python train_redesigned_agent.py --device cpu
```

### Common Issues
1. **CUDA Out of Memory**: Reduce `--batch-size` or use `--device cpu`
2. **Checkpoint Corruption**: Use `--no-resume` to start fresh
3. **Slow Training**: Ensure CUDA is available and functioning

### Performance Testing
```bash
# Quick performance check (100 episodes)
python train_redesigned_agent.py --episodes 100 --save-interval 50

# Actor-Locked quick test
python train_actor_locked_system.py --episodes 50 --actor-trials 5
```

---

## 8. Advanced Usage

### Sequential Training Pipeline
```bash
# Step 1: Train basic DQN (locked model)
python train_redesigned_agent.py --episodes 2000

# Step 2: Train Actor-Locked using pre-trained locked model
python train_actor_locked_system.py \
  --locked-model-path checkpoints/redesigned_agent_final.pt \
  --episodes 1000 \
  --actor-trials 15

# Step 3: Fine-tune with higher actor trials
python train_actor_locked_system.py \
  --actor-trials 25 \
  --episodes 500
```

### Hyperparameter Tuning
```bash
# Conservative training (stable but slow)
python train_redesigned_agent.py \
  --learning-rate 0.00001 \
  --epsilon-decay-steps 100000 \
  --episodes 3000

# Aggressive training (faster but potentially unstable)
python train_redesigned_agent.py \
  --learning-rate 0.0001 \
  --epsilon-decay-steps 25000 \
  --episodes 1500
```

### Batch Processing
```bash
# Multiple training runs
for trials in 5 10 15 20; do
  python train_actor_locked_system.py \
    --actor-trials $trials \
    --episodes 500 \
    --no-resume
done
```

---

## 9. Output File Locations

### Checkpoints
- `checkpoints/redesigned_agent_episode_*.pt` - Basic DQN states
- `checkpoints/actor_locked_episode_*_locked.pt` - Locked model states
- `checkpoints/actor_locked_episode_*_actor.pt` - Actor model states

### Training History
- `checkpoints/*_history.json` - Complete training metrics and progress

### Logs
- Console output provides real-time training progress
- All metrics are saved in history files for analysis

---

## 10. Quick Reference

### Most Common Commands
```bash
# Basic training
python train_redesigned_agent.py

# Actor-Locked training with visualization
python train_actor_locked_system.py --show-visualization

# Resume training with custom episodes
python train_redesigned_agent.py --episodes 2000

# Actor-Locked with custom trials
python train_actor_locked_system.py --actor-trials 20

# Start fresh training
python train_redesigned_agent.py --no-resume
python train_actor_locked_system.py --no-resume
```

### Parameter Ranges
- `--episodes`: 100-10000 (typical: 1000-3000)
- `--actor-trials`: 3-50 (typical: 10-20)
- `--learning-rate`: 0.00001-0.001 (typical: 0.00005)
- `--batch-size`: 16-128 (typical: 32-64)
- `--save-interval`: 10-500 (typical: 50-100) 