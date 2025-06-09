# 🔍 DREAM Model Diagnosis & TensorBoard Setup

## 📊 TensorBoard Commands

### Quick Start:
```bash
# View all logs
tensorboard --logdir=logs --port=6006

# View specific algorithm
tensorboard --logdir=logs/dream_training --port=6007
tensorboard --logdir=logs/dqn_locked_position --port=6008

# Compare algorithms
tensorboard --logdir_spec=dream:logs/dream_training,dqn:logs/dqn_locked_position
```

### Issue: No TensorBoard Logs Currently Generated
Current training files lack TensorBoard integration. Need to add logging.

## 🎯 DREAM Model Issues Diagnosed

### Issue 1: Never Clears Lines (0 lines in all episodes)
**Root Cause**: Action selection and environment interaction problems

**Evidence**:
- Episodes: 20/20 completed with 0 lines cleared
- Rewards: -57 to -214 (overly optimistic for no progress)
- Episode lengths: 339-500 steps (hitting max steps, not game over)

### Issue 2: Reward Function Mismatch
**Current Reward Issues**:
1. **Standard reward mode** gives positive rewards for small improvements
2. **No actual game termination** - episodes end at max steps, not game over
3. **Optimistic shaping rewards** mask poor performance

### Issue 3: Action Space Problems
**Direct Action Mode (8 actions)**:
- Actions 0-7: LEFT, RIGHT, DOWN, ROTATE_CW, ROTATE_CCW, SOFT_DROP, HARD_DROP, NO_OP
- **Problem**: DREAM using Independent Bernoulli distributions
- **Issue**: Can select multiple contradictory actions simultaneously

### Issue 4: State Representation Mismatch
**206→212 padding issue**:
- Environment provides 206-dim observations
- DREAM expects 212-dim inputs
- Padding with zeros may confuse learning

## 🔧 Comprehensive Fixes

### Fix 1: Add TensorBoard Logging
```python
from torch.utils.tensorboard import SummaryWriter

class DREAMTrainerWithTensorBoard:
    def __init__(self):
        self.writer = SummaryWriter('logs/dream_training')
    
    def log_metrics(self, episode, reward, lines, world_loss, actor_loss):
        self.writer.add_scalar('Episode/Reward', reward, episode)
        self.writer.add_scalar('Episode/Lines_Cleared', lines, episode)
        self.writer.add_scalar('Training/World_Loss', world_loss, episode)
        self.writer.add_scalar('Training/Actor_Loss', actor_loss, episode)
```

### Fix 2: Correct Action Selection
**Current Problem**: Independent Bernoulli allows multiple actions
**Solution**: Use Categorical distribution for mutually exclusive actions

```python
# In actor_critic.py
if self.action_mode == 'direct':
    # Change from Independent Bernoulli to Categorical
    return torch.distributions.Categorical(logits=logits)
```

### Fix 3: Fix Reward Function
**Current**: Optimistic shaping rewards
**Solution**: Use sparse 'lines_only' mode for clearer learning signal

```python
# Use lines_only reward mode
trainer = DREAMTrainer(reward_mode='lines_only')
```

### Fix 4: Diagnose Environment Integration
**Test actual game mechanics**:
```python
def test_action_execution():
    env = TetrisEnv(reward_mode='lines_only')
    obs = env.reset()
    
    for action in range(8):
        print(f"Testing action {action}")
        next_obs, reward, done, info = env.step(action)
        print(f"  Result: reward={reward}, done={done}, info={info}")
```

## 🎯 Immediate Action Plan

### Step 1: Add TensorBoard Logging
- Implement logging in train_dream.py
- Log episode metrics, training losses, action distributions

### Step 2: Fix Action Space
- Change from Independent Bernoulli to Categorical
- Ensure mutually exclusive action selection

### Step 3: Test with Sparse Rewards
- Use 'lines_only' reward mode
- Eliminate confusing shaping rewards

### Step 4: Debug Environment Integration
- Verify action execution leads to piece movement
- Check if game actually ends (vs hitting step limit)
- Validate state transitions

### Step 5: Verify State Representation
- Test if 206→212 padding affects learning
- Consider using native 206-dim inputs

## 🔍 Debugging Commands

### Test Environment:
```bash
python -c "
from envs.tetris_env import TetrisEnv
env = TetrisEnv(reward_mode='lines_only', headless=True)
obs = env.reset()
print(f'Obs shape: {obs.shape}')
for i in range(10):
    obs, r, done, info = env.step(i%8)
    print(f'Action {i%8}: reward={r}, done={done}, info={info}')
"
```

### Test DREAM Action Selection:
```bash
python -c "
import torch
from dream.models.actor_critic import ActorCritic
model = ActorCritic(state_dim=212, action_dim=8, action_mode='direct')
state = torch.randn(1, 212)
dist, value = model(state)
action = dist.sample()
print(f'Action: {action}, Value: {value}')
print(f'Action shape: {action.shape}, sum: {action.sum()}')
"
```

## 📈 Expected Improvements

After fixes:
1. **Lines cleared**: Should see >0 lines cleared in some episodes
2. **Rewards**: Sparse but meaningful (only when lines cleared)
3. **Game termination**: Episodes should end with game_over=True
4. **Action coherence**: Only one action selected per step
5. **TensorBoard**: Rich visualizations of training progress

## 🎯 Success Metrics

### Training Success Indicators:
- Lines cleared: >0 in at least 10% of episodes
- Episode termination: Game over (not step limit)
- Reward signal: Sparse but correlated with performance
- Action selection: Single action per step
- World model convergence: Decreasing loss over time 