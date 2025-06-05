# Multiplayer Tetris AIRL Implementation Plan

## 🎮 **Understanding the Multiplayer Architecture**

Based on my analysis of the codebase, here's how multiplayer Tetris works:

### Current Multiplayer Structure
```
Game Class
├── Player1 (ActionHandler, KeyHandler, own grid)
├── Player2 (ActionHandler, KeyHandler, own grid)  
├── Shared BlockPool (synchronized piece sequences)
├── Garbage line mechanics (lines cleared → opponent gets garbage)
└── Competitive win/loss conditions
```

### Key Multiplayer Components
1. **Two Independent Players**: Each has separate grids, pieces, scores
2. **Shared Block Pool**: Both players get same piece sequence 
3. **Garbage Lines**: Line clears send garbage to opponent
4. **Competitive Objective**: Defeat opponent by making them lose

## 🚨 **Critical Issues with Current AIRL Implementation**

### ❌ Problem 1: Single-Player Focus
- Current `tetris_env.py` disables Player2 in single_player mode
- AIRL implementation assumes single-player state representation
- No consideration for opponent state or competitive dynamics

### ❌ Problem 2: State Representation Mismatch
- Expert trajectories from `tetris-ai-master` are single-player
- Multiplayer requires opponent awareness and competitive strategy
- Current 207-dimensional state lacks opponent information

### ❌ Problem 3: Reward Structure Conflict
- Single-player rewards focus on lines cleared, survival time
- Multiplayer rewards should focus on defeating opponent
- Garbage mechanics not considered in reward function

### ❌ Problem 4: Action Space Limitations
- Current action space designed for single-player optimization
- Multiplayer strategy requires tempo, defense, timing considerations

## 🎯 **Revised Implementation Strategy**

### Option A: **Hierarchical AIRL** (Recommended)
1. **Phase 1**: Train single-player AIRL agents using current implementation
2. **Phase 2**: Create multiplayer environment with two AIRL agents
3. **Phase 3**: Fine-tune agents for competitive play using self-play

### Option B: **Direct Multiplayer AIRL** (Complex)
1. Create expert trajectories from multiplayer games
2. Design state representation including opponent information
3. Train AIRL with competitive rewards from the start

## 📋 **Detailed Implementation Plan - Option A (Hierarchical)**

### Phase 1: Single-Player AIRL Foundation ✅ (Already Implemented)
**Status**: Complete with current implementation
- ✅ Core AIRL components (`airl_agent.py`)
- ✅ Expert trajectory loader (`expert_loader.py`) 
- ✅ Training infrastructure (`airl_train.py`)
- ✅ Evaluation framework (`airl_evaluate.py`)

### Phase 2: Multiplayer Environment Extension

#### 2.1 MultiplayerTetrisEnv (`multiplayer_tetris_env.py`) 🔧
```python
class MultiplayerTetrisEnv(gym.Env):
    def __init__(self, agent1=None, agent2=None, headless=True):
        # Two-player environment
        # Each player has independent observation/action spaces
        # Shared game state with garbage mechanics
        
    def step(self, actions):  # actions = [action1, action2]
        # Execute both players' actions simultaneously
        # Handle garbage line mechanics
        # Return [obs1, obs2], [reward1, reward2], done, info
        
    def reset(self):
        # Reset to two-player competitive state
        # Return [obs1, obs2]
```

#### 2.2 Enhanced State Representation (421 dimensions)
```python
def extract_multiplayer_features(player1_obs, player2_obs):
    # Player 1 state (207 dims) - same as single-player
    p1_features = extract_single_player_features(player1_obs)
    
    # Player 2 state (207 dims) - opponent awareness
    p2_features = extract_single_player_features(player2_obs)
    
    # Competitive metrics (7 dims)
    competitive_features = [
        score_difference,        # p1_score - p2_score
        height_difference,       # p1_max_height - p2_max_height  
        lines_difference,        # p1_lines - p2_lines
        garbage_pending_p1,      # garbage lines queued for p1
        garbage_pending_p2,      # garbage lines queued for p2
        game_time,              # current game duration
        tempo_difference        # piece placement speed difference
    ]
    
    return np.concatenate([p1_features, p2_features, competitive_features])
    # Total: 207 + 207 + 7 = 421 dimensions
```

#### 2.3 Competitive Reward Structure
```python
def get_multiplayer_reward(p1_state, p2_state, actions, game_result):
    """Competitive reward focusing on defeating opponent"""
    
    base_reward = 0
    
    # 1. Winning/Losing (primary objective)
    if game_result == "p1_wins":
        base_reward += 100
    elif game_result == "p2_wins":
        base_reward -= 100
    
    # 2. Lines cleared → garbage sent (offensive)
    lines_sent = calculate_garbage_sent(p1_action)
    base_reward += lines_sent * 5
    
    # 3. Defensive play (surviving opponent's garbage)
    if p1_state['garbage_received'] > 0:
        if not p1_lost:
            base_reward += 2  # survival bonus
    
    # 4. Tempo and efficiency
    placement_efficiency = evaluate_placement_efficiency(p1_action, p1_state)
    base_reward += placement_efficiency * 0.5
    
    # 5. Height management (defensive)
    height_penalty = -0.1 * max(get_column_heights(p1_state['grid']))
    base_reward += height_penalty
    
    return base_reward
```

### Phase 3: Multi-Agent AIRL Training

#### 3.1 Competitive AIRL Agent (`competitive_airl_agent.py`)
```python
class CompetitiveAIRLAgent(AIRLAgent):
    def __init__(self, state_dim=421, action_dim=41, **kwargs):
        # Extended discriminator for multiplayer states
        super().__init__(state_dim, action_dim, **kwargs)
        
    def update_discriminator_competitive(self, expert_batch, learner_batch):
        # Discriminator considers both players' states and competitive context
        # Expert data includes successful competitive play patterns
        
    def update_policy_competitive(self, batch):
        # Policy update with competitive rewards
        # Includes opponent modeling and strategic planning
```

#### 3.2 Self-Play Training Loop
```python
def train_competitive_airl():
    # 1. Initialize two AIRL agents (both start with single-player training)
    agent1 = CompetitiveAIRLAgent.load_from_single_player_checkpoint()
    agent2 = CompetitiveAIRLAgent.load_from_single_player_checkpoint()
    
    # 2. Self-play training
    for episode in range(max_episodes):
        # Play game: agent1 vs agent2
        game_trajectory = play_competitive_game(agent1, agent2)
        
        # 3. Update both agents based on competitive outcomes
        agent1.update_from_competitive_game(game_trajectory, player_id=1)
        agent2.update_from_competitive_game(game_trajectory, player_id=2)
        
        # 4. Periodically evaluate against baselines
        if episode % eval_freq == 0:
            evaluate_competitive_performance(agent1, agent2)
```

## 🔧 **Implementation Steps & Cautions**

### Step 1: Complete Single-Player AIRL (Current Priority)
**Caution**: Ensure single-player AIRL works perfectly before attempting multiplayer
```powershell
# Test current implementation
cd local-multiplayer-tetris-main\localMultiplayerTetris\rl_utils
python test_airl_integration.py

# Train single-player AIRL
python airl_train.py --expert-dir ..\..\..\expert_trajectories --iterations 500
```

### Step 2: Create Multiplayer Environment Extension
**Caution**: 
- Don't break existing single-player functionality
- Use composition, not inheritance for multiplayer env
- Handle action synchronization carefully
- Test garbage mechanics thoroughly

```python
# New file: multiplayer_tetris_env.py
class MultiplayerTetrisEnv:
    def __init__(self):
        # Wrap two single-player environments
        self.env1 = TetrisEnv(single_player=True, headless=True)
        self.env2 = TetrisEnv(single_player=True, headless=True)
        # Add competitive game logic
```

### Step 3: State Representation Bridge
**Caution**: 
- Maintain backward compatibility with single-player states
- Normalize competitive features properly
- Handle variable-length games

### Step 4: Expert Data for Multiplayer
**Caution**: 
- Current expert trajectories are single-player only
- Need to generate competitive expert data
- Consider using rule-based competitive agents initially

## 📁 **Proposed File Structure**

```
local-multiplayer-tetris-main/localMultiplayerTetris/
├── rl_utils/
│   ├── airl_agent.py                    # ✅ Single-player AIRL
│   ├── expert_loader.py                 # ✅ Single-player expert data
│   ├── airl_train.py                    # ✅ Single-player training
│   ├── airl_evaluate.py                 # ✅ Single-player evaluation
│   ├── competitive_airl_agent.py        # 🔧 NEW: Multiplayer AIRL
│   ├── multiplayer_expert_loader.py    # 🔧 NEW: Competitive expert data
│   ├── competitive_airl_train.py        # 🔧 NEW: Self-play training
│   └── competitive_airl_evaluate.py     # 🔧 NEW: Competitive evaluation
├── multiplayer_tetris_env.py            # 🔧 NEW: Two-player environment
├── tetris_env.py                        # ✅ Single-player environment
└── competitive_utils.py                 # 🔧 NEW: Competitive game utilities
```

## 🎯 **Success Metrics for Multiplayer AIRL**

### Phase 1 Success (Single-Player AIRL)
- ✅ Agent achieves 80% of expert performance in single-player
- ✅ Discriminator accuracy > 80%
- ✅ Stable training across multiple runs

### Phase 2 Success (Multiplayer Environment)
- 🎯 Two agents can play competitive games without crashes
- 🎯 Garbage mechanics work correctly
- 🎯 State representation captures competitive dynamics

### Phase 3 Success (Competitive AIRL)
- 🎯 AIRL agents can defeat rule-based opponents
- 🎯 Emergent competitive strategies (tempo, defense, offense)
- 🎯 Stable self-play training without mode collapse

## ⚠️ **Major Risks & Mitigation Strategies**

### Risk 1: Expert Data Scarcity for Multiplayer
**Mitigation**: 
- Start with single-player expert data
- Generate competitive data using rule-based agents
- Use curriculum learning from single → multiplayer

### Risk 2: Training Instability in Self-Play
**Mitigation**:
- Use population-based training
- Periodic checkpoint reversion
- Careful hyperparameter tuning

### Risk 3: State Space Explosion (421 dims)
**Mitigation**:
- Dimensionality reduction techniques
- Hierarchical state representation
- Feature importance analysis

### Risk 4: Reward Engineering Difficulty
**Mitigation**:
- Start with simple win/loss rewards
- Gradually add competitive sophistication
- Human evaluation of play quality

## 🚀 **Immediate Next Steps**

1. **[HIGH PRIORITY]** Complete and debug single-player AIRL
2. **[MEDIUM]** Design multiplayer environment wrapper
3. **[MEDIUM]** Create competitive expert data generation
4. **[LOW]** Implement self-play training infrastructure

## 💻 **Usage Commands (PowerShell)**

```powershell
# Phase 1: Single-player AIRL
cd local-multiplayer-tetris-main\localMultiplayerTetris\rl_utils
python airl_train.py --expert-dir ..\..\..\expert_trajectories

# Phase 2: Multiplayer environment testing  
python test_multiplayer_env.py

# Phase 3: Competitive training
python competitive_airl_train.py --agent1-checkpoint checkpoints\airl_best.pt --agent2-checkpoint checkpoints\airl_best.pt
```

This plan provides a clear path from single-player AIRL to competitive multiplayer AIRL while minimizing risks and maintaining existing functionality. 