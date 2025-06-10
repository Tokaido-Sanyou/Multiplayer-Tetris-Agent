# Dreamer vs DREAM: Implementation Comparison

## Overview

This document compares two different approaches to model-based reinforcement learning for Tetris:

1. **DREAM** (Previous Implementation): Actor-critic with sequence training
2. **Standard Dreamer** (New Implementation): True world model-based learning

## Key Architectural Differences

### 1. World Model Structure

#### DREAM (Previous):
- **No explicit world model**
- Direct actor-critic operating on raw observations
- "Sequence training" processes entire episode fragments
- Single shared backbone with actor/critic heads

#### Standard Dreamer (New):
- **Explicit world model with 4 components:**
  - `RepresentationModel`: Encodes observations → latent space (VAE)
  - `DynamicsModel`: Predicts next latent state given state + action
  - `RewardModel`: Predicts rewards in latent space
  - `ContinueModel`: Predicts episode termination probability
- **Separate actor-critic networks** operating in latent space

### 2. Training Philosophy

#### DREAM (Previous):
```python
# Single-phase training
# Computes R-V = A directly and uses advantage in policy gradient
advantage = returns - values
actor_loss = -(log_probs * advantage.detach()).mean()
```

#### Standard Dreamer (New):
```python
# Two-phase training
# Phase 1: Train world model on real data
world_loss = kl_loss + dynamics_loss + reward_loss + continue_loss

# Phase 2: Train policy using imagined rollouts
rollout = imagine_rollout(initial_latent, horizon=15)
# Policy trained on imagined data, world model frozen during policy updates
```

### 3. Data Utilization

#### DREAM (Previous):
- **Real experience only**
- Trains on sequences from replay buffer
- No imagination or world model rollouts

#### Standard Dreamer (New):
- **Real + Imagined experience**
- Phase 1: Random exploration to pretrain world model
- Phase 2: Policy rollouts in imagination using learned world model
- Much more sample efficient due to imagined experience

### 4. Loss Functions

#### DREAM (Previous):
```python
# Actor-critic losses only
actor_loss = -log_prob * advantage + entropy_bonus
critic_loss = MSE(values, returns)
```

#### Standard Dreamer (New):
```python
# World model losses
kl_loss = KL_divergence(posterior, prior)  # VAE regularization
dynamics_loss = MSE(predicted_next_state, target_next_state)
reward_loss = MSE(predicted_reward, actual_reward)
continue_loss = BCE(predicted_continue, actual_continue)

# Policy losses (computed on imagined rollouts)
actor_loss = -log_prob * advantage  # No direct R-V computation
critic_loss = MSE(values, lambda_returns)
```

## Parameter Counts

### DREAM (Previous):
- **Basic Mode**: 407,209 parameters
- **Enhanced Mode**: 635,913 parameters (with curiosity)
- Single network with shared backbone

### Standard Dreamer (New):
- **World Model**: ~380,000 parameters
  - Representation: 206→512→256→256 (VAE encoder)
  - Dynamics: 136→256→256→256 (state+action → next state)
  - Reward: 136→256→256→1 (state+action → reward)
  - Continue: 128→256→1 (state → continue probability)
- **Policy Networks**: ~135,000 parameters
  - Actor: 128→256→256→8
  - Critic: 128→256→256→1
- **Total**: ~515,000 parameters

## Training Procedure Comparison

### DREAM (Previous):
```python
for episode in episodes:
    # Collect real experience
    state, action, reward, next_state = env.step()
    buffer.add(state, action, reward, next_state)
    
    # Train on sequences
    sequences = buffer.sample_sequences()
    actor_loss, critic_loss = train_on_sequences(sequences)
    update_networks(actor_loss, critic_loss)
```

### Standard Dreamer (New):
```python
# Phase 1: World model pretraining
for episode in pretrain_episodes:
    # Random data collection
    action = random_action()
    experience = env.step(action)
    buffer.add(experience)
    
    # Train world model
    world_loss = train_world_model(buffer.sample())

# Phase 2: Policy training with imagination
for episode in episodes:
    # Policy data collection
    action = policy.select_action(state)
    experience = env.step(action)
    buffer.add(experience)
    
    # Alternate training
    if episode % 2 == 0:
        world_loss = train_world_model(buffer.sample())
    
    # Train policy using imagination
    initial_states = buffer.sample_states()
    imagined_rollouts = world_model.imagine(initial_states, horizon=15)
    policy_loss = train_policy(imagined_rollouts)
```

## Key Advantages

### DREAM (Previous):
- ✅ Simpler architecture and implementation
- ✅ Direct advantage computation (R-V)
- ✅ Single training loop
- ✅ Lower computational overhead
- ✅ Works well with sparse rewards

### Standard Dreamer (New):
- ✅ **True model-based learning** with explicit world model
- ✅ **Sample efficiency** through imagination
- ✅ **Decoupled training** allows specialized optimization
- ✅ **Latent space learning** can capture complex dynamics
- ✅ **Scalable** to longer horizons and complex environments
- ✅ **Interpretable** world model components

## Disadvantages

### DREAM (Previous):
- ❌ Limited to real experience only
- ❌ No explicit world understanding
- ❌ Sample inefficient
- ❌ Not truly "model-based"

### Standard Dreamer (New):
- ❌ More complex implementation
- ❌ Requires careful hyperparameter tuning
- ❌ World model quality critical for performance
- ❌ Higher computational cost
- ❌ Potential compounding errors in imagination

## Implementation Files

### DREAM (Previous):
- `train_dream.py` - Original implementation with 4 modes
- Multiple training modes: basic, enhanced_exploration, fixed_logging, comprehensive

### Standard Dreamer (New):
- `train_dreamer_standard.py` - True Dreamer implementation
- Clean separation of world model and policy components
- Two-phase training procedure

## Usage Recommendations

### Use DREAM when:
- Simple, direct actor-critic approach is sufficient
- Computational resources are limited
- Quick prototyping and experimentation
- Sparse reward environments

### Use Standard Dreamer when:
- Sample efficiency is critical
- Want to leverage imagination for learning
- Need interpretable world model
- Planning and model-based reasoning is important
- Working with complex, long-horizon tasks

## Conclusion

The **Standard Dreamer** represents a true model-based approach with explicit world modeling, imagination-based training, and the theoretical advantages of the Dreamer framework. The **DREAM** implementation is more of an enhanced actor-critic with some naming inspiration from Dreamer but lacks the core world modeling components.

For Tetris specifically, both approaches have merit:
- **DREAM** for quick experimentation and simpler debugging
- **Standard Dreamer** for leveraging the full power of model-based RL and achieving better sample efficiency
