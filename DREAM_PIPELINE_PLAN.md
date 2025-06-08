# DREAM Single Player Pipeline - Comprehensive Implementation Plan

## Overview

DREAM (Dreamer) is a model-based reinforcement learning algorithm that learns a world model to generate imagined trajectories for training a policy. This document outlines the complete implementation plan for integrating DREAM into the Tetris environment.

## Core Algorithm

### DREAM Components
1. **World Model**: Learns environment dynamics
2. **Actor**: Policy network for action selection  
3. **Critic**: Value function estimation
4. **Experience Buffer**: Stores real environment interactions
5. **Imagination**: Generates synthetic trajectories using world model

### Key Advantages for Tetris
- **Sample Efficiency**: Learn from imagined experiences
- **Long-term Planning**: Lookahead through world model
- **Continuous Learning**: Update world model during training
- **Complex State Spaces**: Handle Tetris grid complexity effectively

## File Structure

```
dream/                              # DREAM implementation directory
├── __init__.py                     # Package initialization
├── models/                         # DREAM-specific models
│   ├── __init__.py
│   ├── world_model.py              # Recurrent State Space Model (RSSM)
│   ├── observation_model.py        # Observation decoder/encoder
│   ├── reward_model.py             # Reward prediction model
│   ├── actor_critic.py             # Policy and value networks
│   └── ensemble_model.py           # Ensemble world models
├── algorithms/                     # DREAM training algorithms
│   ├── __init__.py
│   ├── dream_trainer.py            # Main DREAM training loop
│   ├── world_model_trainer.py      # World model training
│   ├── actor_critic_trainer.py     # Policy training
│   └── imagination_trainer.py      # Imagined trajectory training
├── buffers/                        # Experience management
│   ├── __init__.py
│   ├── replay_buffer.py            # Experience replay buffer
│   ├── sequence_buffer.py          # Sequential experience buffer
│   └── imagination_buffer.py       # Imagined experience buffer
├── utils/                          # DREAM utilities
│   ├── __init__.py
│   ├── state_representation.py     # State encoding/decoding
│   ├── trajectory_utils.py         # Trajectory manipulation
│   ├── dream_logger.py             # DREAM-specific logging
│   └── visualization.py           # World model visualization
├── configs/                        # DREAM configurations
│   ├── __init__.py
│   ├── dream_config.py             # DREAM hyperparameters
│   └── model_configs.py           # Network architectures
└── agents/                         # DREAM agents
    ├── __init__.py
    ├── dream_agent.py              # Main DREAM agent
    └── planning_agent.py           # Planning-based agent
```

## Network Architectures

### 1. World Model (RSSM - Recurrent State Space Model)

```python
class WorldModel(nn.Module):
    """
    Recurrent State Space Model for learning environment dynamics
    
    Components:
    - Representation Model: h_t, z_t = representation(o_t, h_{t-1}, a_{t-1})
    - Transition Model: h_t = transition(h_{t-1}, z_{t-1}, a_{t-1})
    - Observation Model: o_t = observation(h_t, z_t)
    - Reward Model: r_t = reward(h_t, z_t)
    - Continue Model: c_t = continue(h_t, z_t)
    """
    
    def __init__(self, 
                 state_dim: int = 30,
                 rnn_hidden_dim: int = 200,
                 stochastic_dim: int = 30,
                 hidden_dim: int = 400,
                 action_dim: int = 8,
                 observation_shape: tuple = (1, 20, 10)):
        
        # Representation model (encoder)
        self.representation = ConvEncoder(observation_shape, state_dim)
        
        # Recurrent transition model  
        self.transition_rnn = nn.GRUCell(stochastic_dim + action_dim, rnn_hidden_dim)
        self.transition_mlp = MLP(rnn_hidden_dim, stochastic_dim * 2)  # mean + std
        
        # Observation model (decoder)
        self.observation = ConvDecoder(state_dim + rnn_hidden_dim, observation_shape)
        
        # Reward model
        self.reward = MLP(state_dim + rnn_hidden_dim, 1)
        
        # Continue model (episode termination)
        self.continue = MLP(state_dim + rnn_hidden_dim, 1)
```

### 2. Actor-Critic Networks

```python
class ActorCritic(nn.Module):
    """
    Actor-Critic network for policy learning
    """
    
    def __init__(self,
                 state_dim: int = 230,  # rnn_hidden + stochastic
                 action_dim: int = 8,
                 hidden_dim: int = 400,
                 action_mode: str = 'direct'):
        
        # Shared feature extraction
        self.feature_net = MLP(state_dim, hidden_dim)
        
        # Actor network (policy)
        if action_mode == 'direct':
            # 8 binary actions
            self.actor = MLP(hidden_dim, action_dim)
        elif action_mode == 'locked_position':
            # 200 discrete positions
            self.actor = MLP(hidden_dim, 200)
        
        # Critic network (value function)
        self.critic = MLP(hidden_dim, 1)
        
        # Value target network for stability
        self.target_critic = MLP(hidden_dim, 1)
        self.target_critic.load_state_dict(self.critic.state_dict())
```

### 3. Tetris-Specific Observation Encoder

```python
class TetrisEncoder(nn.Module):
    """
    Tetris-specific observation encoder for world model
    """
    
    def __init__(self, observation_shape: tuple = (1, 20, 10), latent_dim: int = 256):
        # Convolutional layers for grid processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2))  # Reduce to manageable size
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 2 + 32, latent_dim),  # +32 for piece info
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
```

## Training Pipeline

### Main Training Loop

```python
class DREAMTrainer:
    """Main DREAM training orchestrator"""
    
    def train(self, num_episodes: int):
        for episode in range(num_episodes):
            # 1. Collect real experience
            real_trajectory = self.collect_real_experience()
            self.replay_buffer.add_trajectory(real_trajectory)
            
            # 2. Train world model on real data
            if len(self.replay_buffer) > self.config.min_buffer_size:
                world_model_loss = self.train_world_model()
                
                # 3. Generate imagined trajectories
                imagined_trajectories = self.generate_imagined_trajectories()
                self.imagination_buffer.add_trajectories(imagined_trajectories)
                
                # 4. Train actor-critic on imagined data
                actor_loss, critic_loss = self.train_actor_critic()
                
                # 5. Log metrics
                self.log_training_metrics(episode, world_model_loss, actor_loss, critic_loss)
```

## Configuration System

```python
class DREAMConfig:
    """Configuration for DREAM training"""
    
    def __init__(self, action_mode: str = 'direct'):
        # Environment configuration
        self.action_mode = action_mode
        self.max_episode_length = 1000
        
        # Model architecture
        self.world_model = {
            'state_dim': 30,
            'rnn_hidden_dim': 200,
            'stochastic_dim': 30,
            'hidden_dim': 400,
            'action_dim': 8 if action_mode == 'direct' else 200
        }
        
        self.actor_critic = {
            'state_dim': 230,  # rnn_hidden + stochastic
            'action_dim': 8 if action_mode == 'direct' else 200,
            'hidden_dim': 400,
            'action_mode': action_mode
        }
        
        # Training hyperparameters
        self.world_model_lr = 3e-4
        self.actor_lr = 8e-5
        self.critic_lr = 8e-5
        self.gamma = 0.99
        self.lambda_param = 0.95
        
        # Buffer configuration
        self.buffer_size = 100000
        self.imagination_size = 50000
        self.min_buffer_size = 1000
        self.sequence_length = 50
        
        # Training schedule
        self.world_model_batches = 10
        self.actor_critic_batches = 5
        self.batch_size = 32
        self.imagination_batch_size = 16
        self.imagination_horizon = 15
```

## Logging and Monitoring

```python
class DREAMLogger:
    """Enhanced logging for DREAM training"""
    
    def log_episode(self, episode: int, real_reward: float, 
                   imagined_reward: float, world_model_loss: float):
        """Log episode metrics"""
        metrics = {
            'episode': episode,
            'real_reward': real_reward,
            'imagined_reward': imagined_reward,
            'world_model_loss': world_model_loss,
            'timestamp': time.time()
        }
        
        # TensorBoard logging
        self.tensorboard.add_scalar('Episode/Real_Reward', real_reward, episode)
        self.tensorboard.add_scalar('Episode/Imagined_Reward', imagined_reward, episode)
        self.tensorboard.add_scalar('Training/World_Model_Loss', world_model_loss, episode)
```

## GPU Support and Optimization

```python
class GPUManager:
    """Manage GPU memory for DREAM training"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if device == 'cuda' and torch.cuda.is_available():
            # Enable memory optimization
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.8)
    
    def optimize_memory(self):
        """Optimize GPU memory usage"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
```

## Expected Performance Characteristics

### Training Efficiency
- **Sample Efficiency**: 2-3x better than model-free methods
- **Wall Clock Time**: Longer due to world model training
- **GPU Memory**: Higher usage due to multiple models
- **Convergence**: Smoother learning curves

### Tetris-Specific Benefits
- **Long-term Planning**: Better understanding of piece stacking
- **State Prediction**: Accurate grid state forecasting
- **Strategic Play**: Improved line clearing strategies
- **Robustness**: Better handling of different piece sequences

### Scalability
- **Action Modes**: Support for both direct and locked position
- **Multi-GPU**: Parallel world model training
- **Large Networks**: Gradient accumulation support
- **Memory Efficient**: Buffer management and cleanup

## Implementation Timeline

### Phase 1: Core Infrastructure (Week 1-2)
- World model architecture
- Basic replay buffers
- DREAM configuration system
- GPU memory management

### Phase 2: Training Pipeline (Week 3-4)
- World model training loop
- Imagination generation
- Actor-critic training
- Integration with Tetris environment

### Phase 3: Optimization & Evaluation (Week 5-6)
- Performance optimization
- Comprehensive evaluation metrics
- Logging and visualization
- Command line interface

### Phase 4: Testing & Documentation (Week 7-8)
- Extensive testing
- Documentation completion
- Performance benchmarking
- Integration with existing codebase

This comprehensive DREAM implementation will provide state-of-the-art model-based reinforcement learning capabilities for the Tetris environment, enabling more sample-efficient training and better long-term strategic planning.
