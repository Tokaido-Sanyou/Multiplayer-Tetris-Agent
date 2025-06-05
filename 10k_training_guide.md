# 🎯 10,000 Episode AIRL Training Guide

## 🚀 Quick Start Commands

### Single-Player AIRL (1000 iterations × 10 episodes = 10K)
```bash
python pytorch_airl_complete.py --mode 10k --type single
```

### Multiplayer Competitive (10K episodes directly)
```bash
python pytorch_airl_complete.py --mode 10k --type multiplayer
```

## 📊 What Happens During Training?

### 🔄 Per Episode/Batch Process

#### Single-Player AIRL Process:
**Per Iteration (10 episodes + 20 network updates):**
1. **Collect Learner Data**: Run 10 episodes with current policy
2. **Batch Updates** (20 times per iteration):
   - Sample 64 expert transitions from expert trajectories
   - Sample 64 learner transitions from collected episodes
   - **Discriminator Update**: Train to classify expert vs learner
   - **Policy Update**: Use AIRL rewards to improve policy
3. **Logging**: TensorBoard metrics and console output

**Expert Imitation Learning: ✅ YES - Every batch update!**
- Expert trajectories sampled randomly each batch
- Discriminator learns expert behavioral patterns
- Policy receives AIRL rewards to mimic expert behavior

#### Multiplayer Competitive Process:
**Per Episode:**
1. **Reset Environments**: Both P1 and P2 start fresh games
2. **Competitive Episode**: Players compete until game over
3. **Winner Determination**: Based on score and survival
4. **Competitive Rewards**: Win/loss bonuses applied
5. **Logging**: Episode results to TensorBoard

**Expert Learning in Multiplayer: ⚠️ Currently Basic**
- Could be enhanced with periodic AIRL updates
- Expert demonstration replay for both players
- Self-play + expert guidance integration

### 🧮 Training Mathematics

#### Single-Player (1000 iterations):
- **Total Episodes**: 1000 × 10 = 10,000
- **Network Updates**: 1000 × 20 = 20,000
- **Expert Samples**: 20,000 × 64 = 1,280,000
- **Expected Duration**: 8-12 hours

#### Multiplayer (10K episodes):
- **Total Episodes**: 10,000 direct episodes
- **Expected Duration**: 6-10 hours

## 📈 TensorBoard Monitoring

### 1️⃣ Start TensorBoard
```bash
# In a separate terminal/PowerShell window
tensorboard --logdir=logs

# Alternative port if 6006 is busy
tensorboard --logdir=logs --port=6007
```

**Access at**: http://localhost:6006

### 2️⃣ Key Metrics to Watch

#### Single-Player AIRL Metrics:
- **`discriminator_loss`**: Should stabilize around 0.7-1.0
- **`overall_accuracy`**: Should reach 0.7-0.9 (discriminator learning)
- **`policy_loss`**: Should decrease over time
- **`mean_airl_reward`**: Should become less negative
- **`learner/buffer_size`**: Should grow then stabilize at 50K
- **`learner/transitions_collected`**: Episodes collected per iteration

#### Multiplayer Competitive Metrics:
- **`Episode/P1_Reward`** & **`Episode/P2_Reward`**: Should increase
- **`Episode/Winner`**: Should approach 50/50 balance as agents improve
- **`Episode/Steps`**: Should increase as agents get better
- **`Training/Total_Steps`**: Cumulative step counter

### 3️⃣ Log Directory Structure
```
logs/
├── airl_20250604_HHMMSS/              # Single-player logs
└── multiplayer_airl_20250604_HHMMSS/  # Multiplayer logs
```

### 4️⃣ TensorBoard Commands for Specific Runs
```bash
# View specific run
tensorboard --logdir=logs/airl_20250604_030413

# Compare multiple runs
tensorboard --logdir=logs

# Multiple runs with aliases
tensorboard --logdir=run1:logs/airl_20250604_030413,run2:logs/airl_20250604_031205
```

## 🔍 Training Progress Indicators

### Good Progress Signs:
✅ **Discriminator accuracy 0.7-0.9**: Learning to distinguish expert vs learner  
✅ **Policy loss decreasing**: Agent improving  
✅ **AIRL rewards less negative**: Better expert imitation  
✅ **Episode steps increasing**: Agents surviving longer  
✅ **Buffer size growing steadily**: Collecting diverse data  

### Warning Signs:
⚠️ **Discriminator accuracy stuck at 0.5**: Not learning  
⚠️ **Policy loss exploding**: Learning rate too high  
⚠️ **AIRL rewards becoming more negative**: Getting worse  
⚠️ **Episode steps decreasing**: Agents getting worse  

## 🎛️ Configuration Details

### 10K Mode Settings:
```python
{
    'headless': True,                    # No visualization for speed
    'max_iterations': 1000,              # Single-player: 1000 iterations
    'max_episodes': 10000,               # Multiplayer: 10K episodes
    'episodes_per_iteration': 10,        # Single-player: 10 episodes/iteration
    'updates_per_iteration': 20,         # Single-player: 20 updates/iteration
    'batch_size': 64,                    # Larger batch for stability
    'buffer_size': 50000,                # Large buffer for diversity
    'discriminator_lr': 3e-4,            # Discriminator learning rate
    'policy_lr': 1e-4,                   # Policy learning rate
    'gamma': 0.99,                       # Discount factor
}
```

## 🔧 Troubleshooting

### Common Issues:
1. **CUDA out of memory**: Reduce `batch_size` to 32
2. **Training too slow**: Ensure `headless=True` and GPU available
3. **TensorBoard not showing data**: Check logs directory exists
4. **Import errors**: Run from correct directory

### Performance Tips:
- Monitor GPU/CPU usage during training
- Use SSD storage for faster I/O
- Close other GPU-intensive applications
- Consider distributed training for even longer runs

## 📁 Output Files

After training, you'll have:
```
models/
├── pytorch_airl_final_discriminator.pth
└── pytorch_airl_final_policy.pth

logs/
└── [training_run_logs]/
    ├── events.out.tfevents.*
    └── [TensorBoard files]
```

## 🎯 Expected Results

### Single-Player AIRL:
- Agent should learn expert-like Tetris playing
- Discriminator should achieve 70-90% accuracy
- Policy should show improved game performance
- AIRL rewards should trend toward expert levels

### Multiplayer Competitive:
- Both agents should improve over time
- Win rates should become more balanced
- Episode lengths should increase
- Competitive strategies should emerge

---

**Ready to start your 10K episode training!** 🚀 