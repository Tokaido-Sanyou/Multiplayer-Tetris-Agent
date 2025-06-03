"""
Comprehensive Configuration File for Tetris Multi-Agent RL System

This file centralizes all configuration parameters for the hierarchical Tetris RL system.
The system implements a 6-phase training pipeline:
1. Exploration Phase: Systematic piece placement trials
2. State Model Training: Learning optimal placements from terminal rewards  
3. Future Reward Prediction: Training value estimators
4. Exploitation Phase: Policy rollouts with experience collection
5. PPO Training: Actor-critic optimization with clipping
6. Evaluation Phase: Performance assessment

Architecture Overview:
===================
- State Model: Predicts rotation/position from state → 4 rotations, 10 x-positions, 20 y-positions
- Actor-Critic: Shared feature extractor → actor (8 binary actions) + critic (value)
- Future Reward Predictor: State-action encoder → immediate reward + future value
- Exploration Actor: Systematic placement generation with terminal reward evaluation

Action Representation:
====================
Actions are represented as 8-dimensional one-hot vectors:
- [1,0,0,0,0,0,0,0] = Move Left
- [0,1,0,0,0,0,0,0] = Move Right  
- [0,0,1,0,0,0,0,0] = Move Down
- [0,0,0,1,0,0,0,0] = Rotate Clockwise
- [0,0,0,0,1,0,0,0] = Rotate Counter-clockwise
- [0,0,0,0,0,1,0,0] = Hard Drop
- [0,0,0,0,0,0,1,0] = Hold Piece
- [0,0,0,0,0,0,0,1] = No-op

State Representation (NEW SIMPLIFIED):
====================================
State vector is 410-dimensional (reduced from 1817):
- Current Piece Grid: 200 values (20×10 binary grid for falling piece)
- Empty Grid: 200 values (20×10 binary grid for empty spaces)  
- Next Piece: 7 values (one-hot encoding for next piece only)
- Metadata: 3 values (current_rotation, current_x, current_y)
REMOVED: 7-piece type grids (was 1400 values), hold piece (was 7 values)
"""

import torch
import os

# Handle both direct execution and module import
try:
    pass
except ImportError:
    # Direct execution - imports without relative paths
    pass

class TetrisConfig:
    """
    Centralized configuration for all Tetris RL components
    """
    
    # ===========================================
    # ENVIRONMENT CONFIGURATION
    # ===========================================
    
    # Basic environment settings
    SINGLE_PLAYER = True
    HEADLESS = True  # Set to False for visualization during training
    MAX_EPISODE_STEPS = 2000
    GRAVITY_INTERVAL = 5  # Agent steps per gravity drop
    
    # State space configuration (SIMPLIFIED ARCHITECTURE)
    GRID_HEIGHT = 20
    GRID_WIDTH = 10
    STATE_DIM = 410  # NEW: 200 + 200 + 7 + 3 = 410 (removed 7-piece grids + hold piece)
    ACTION_DIM = 8   # One-hot encoded actions
    GOAL_DIM = 36    # Goal vector from state model: 4 (rotation) + 10 (x) + 20 (y) + 1 (value) + 1 (confidence)
    
    # State vector breakdown:
    CURRENT_PIECE_GRID_DIM = GRID_HEIGHT * GRID_WIDTH  # 200
    EMPTY_GRID_DIM = GRID_HEIGHT * GRID_WIDTH          # 200  
    NEXT_PIECE_DIM = 7                                 # 7 (one-hot for next piece only)
    METADATA_DIM = 3                                   # 3 (rotation, x, y)
    
    # ===========================================
    # REWARD SYSTEM CONFIGURATION
    # ===========================================
    
    class RewardConfig:
        """
        Detailed reward parameter configuration across all models
        
        Reward Philosophy:
        - Line clearing rewards encourage efficient play
        - Height penalties discourage stacking too high
        - Hole penalties encourage clean placement
        - Bumpiness penalties encourage smooth surfaces
        - Terminal penalties heavily discourage game over
        - Piece presence rewards encourage longer games (decreasing over training)
        """
        
        # Environment reward parameters (tetris_env.py)
        LINE_CLEAR_BASE = {1: 100, 2: 200, 3: 400, 4: 1600}  # Base scores for line clears
        LEVEL_MULTIPLIER = True  # Multiply by (level + 1)
        GAME_OVER_PENALTY = -200  # Heavy penalty for losing
        TIME_PENALTY = -0.01  # Small penalty per step (disabled)
        
        # Feature-based reward shaping weights (UPDATED)
        HOLE_WEIGHT = 0.5      # Penalty per hole created/filled (reduced from 4.0)
        MAX_HEIGHT_WEIGHT = 5.0   # Penalty for maximum column height changes (reduced from 10.0)
        BUMPINESS_WEIGHT = 0.2    # Penalty for surface irregularity changes (reduced from 1.0)
        
        # NEW: Piece presence reward system
        PIECE_PRESENCE_REWARD = 1.0      # Base reward per piece on board
        PIECE_PRESENCE_DECAY_STEPS = 500  # Steps over which to decay to 0 (first half of 1000 episodes)
        PIECE_PRESENCE_MIN = 0.0         # Minimum piece presence reward
        
        # Exploration actor reward parameters (exploration_actor.py)
        # Used for evaluating terminal states during placement trials
        EXPLORATION_MAX_HEIGHT_WEIGHT = -0.5   # Negative reward for tall stacks
        EXPLORATION_HOLE_WEIGHT = -10.0        # Heavy penalty for holes
        EXPLORATION_BUMPINESS_WEIGHT = -0.1    # Penalty for uneven surface
        
        # State model reward weighting (state_model.py)
        # Higher terminal rewards get more training weight
        MIN_REWARD_WEIGHT = 0.1   # Minimum weight for bad placements
        REWARD_WEIGHT_SCALE = 200  # Scale factor for weight normalization
        
        # Future reward predictor parameters (future_reward_predictor.py)
        DISCOUNT_FACTOR = 0.99    # Gamma for trajectory value calculation
        HORIZON_STEPS = 10        # Number of future steps to consider
        
    # ===========================================
    # NEURAL NETWORK ARCHITECTURE (CENTRALIZED)
    # ===========================================
    
    class NetworkConfig:
        """
        Centralized neural network architecture specifications
        
        Design Philosophy:
        - All network dimensions defined here for easy modification
        - Pure MLP architecture for simplified 410D state vector
        - Separate heads for different prediction tasks
        """
        
        # Global network dimensions
        STATE_DIM = 410          # NEW: Simplified state dimension
        ACTION_DIM = 8           # One-hot action space
        
        # Shared network sizes
        LARGE_HIDDEN = 512       # Large hidden layer size
        MEDIUM_HIDDEN = 256      # Medium hidden layer size  
        SMALL_HIDDEN = 128       # Small hidden layer size
        TINY_HIDDEN = 64         # Tiny hidden layer size
        
        # Shared Feature Extractor (actor_critic.py)
        class SharedFeatureExtractor:
            INPUT_DIM = 410                                    # NEW: Simplified input
            HIDDEN_LAYERS = [512, 256, 128]                   # Progressive reduction
            DROPOUT_RATE = 0.1
            OUTPUT_FEATURES = 128                              # Final feature dimension
            
        # Actor-Critic Network (actor_critic.py)
        class ActorCritic:
            # Input processing
            STATE_DIM = 410                                    # NEW: Simplified state
            GOAL_DIM = 36                                      # Goal vector from state model
            SHARED_HIDDEN = [512, 256, 128]                   # Shared feature layers
            
            # Goal encoder dimensions
            GOAL_ENCODER_LAYERS = [36, 64, 64]                # Goal processing layers
            GOAL_FEATURES = 64                                # Output from goal encoder
            
            # Combined features = STATE_FEATURES + GOAL_FEATURES = 128 + 64 = 192
            COMBINED_FEATURES = 192                           # Total feature dimension
            
            # Actor network (policy) - outputs 8 binary decisions
            ACTOR_HIDDEN_LAYERS = [256, 128]                  # Actor-specific layers
            ACTOR_OUTPUT_DIM = 8                              # One-hot action space
            ACTOR_ACTIVATION = 'sigmoid'                      # For binary outputs
            
            # Critic network (value function)
            CRITIC_HIDDEN_LAYERS = [256, 128]                 # Critic-specific layers
            CRITIC_OUTPUT_DIM = 1
            
        # State Model (state_model.py)
        class StateModel:
            INPUT_DIM = 410                                    # NEW: Simplified input
            ENCODER_LAYERS = [410, 256, 256, 128]             # Encoder with dropout
            DROPOUT_RATE = 0.1
            
            # Output heads
            ROTATION_CLASSES = 4                              # 0, 1, 2, 3 rotations
            X_POSITION_CLASSES = 10                           # 0-9 x coordinates
            Y_POSITION_CLASSES = 20                           # 0-19 y coordinates (landing position)
            VALUE_OUTPUT = 1                                  # Terminal reward prediction
            
        # Future Reward Predictor (future_reward_predictor.py)  
        class FutureRewardPredictor:
            STATE_DIM = 410                                   # NEW: Simplified state
            ACTION_DIM = 8                                    # One-hot encoded
            
            # State encoder
            STATE_ENCODER_LAYERS = [410, 256, 128]            # State processing
            
            # Action encoder  
            ACTION_ENCODER_LAYERS = [8, 32]                   # Action processing
            
            # Combined encoder
            COMBINED_INPUT = 128 + 32                         # 160 total
            COMBINED_LAYERS = [160, 256, 128]                 # Joint processing
            
            # Output heads
            REWARD_HEAD_OUTPUT = 1                            # Immediate reward prediction
            VALUE_HEAD_OUTPUT = 1                             # Future value estimate
    
    # ===========================================
    # TRAINING CONFIGURATION
    # ===========================================
    
    class TrainingConfig:
        """
        Training hyperparameters and pipeline configuration
        """
        
        # Device configuration
        DEVICE = 'auto'  # 'auto', 'cuda', 'mps', or 'cpu'
        
        # Overall training schedule - Extended to 1000 total episodes
        NUM_BATCHES = 50          # Total training batches  
        BATCH_SIZE = 32            # Experience batch size
        
        # Phase 1: Exploration
        EXPLORATION_EPISODES = 20   # Episodes per batch for data collection (50 * 20 = 1000 total)
        
        # Phase 2: State Model Training  
        STATE_TRAINING_SAMPLES = 1000  # Max samples to use per batch
        STATE_EPOCHS = 5               # Training epochs per batch
        STATE_LEARNING_RATE = 1e-3
        
        # Phase 3: Future Reward Prediction
        REWARD_BATCH_SIZE = 64      # Batch size for reward predictor
        REWARD_LEARNING_RATE = 1e-3
        MIN_BUFFER_SIZE = 1000      # Minimum experiences before training
        
        # Phase 4: Exploitation  
        EXPLOITATION_EPISODES = 20   # Policy rollout episodes per batch (50 * 20 = 1000 total)
        
        # Phase 5: PPO Training
        PPO_ITERATIONS = 3          # PPO update iterations per batch
        PPO_EPOCHS = 4              # Epochs per PPO iteration
        PPO_BATCH_SIZE = 64         # Batch size for PPO updates
        PPO_CLIP_RATIO = 0.2        # Clipping parameter
        
        # Phase 6: Evaluation
        EVAL_EPISODES = 10          # Pure evaluation episodes per batch
        
        # Episode limits
        MAX_EPISODE_STEPS = 2000    # Maximum steps per episode (extended for longer games)
        
        # Actor-Critic specific parameters
        ACTOR_LEARNING_RATE = 1e-4
        CRITIC_LEARNING_RATE = 1e-3
        GAMMA = 0.99                # Discount factor
        
        # Exploration parameters
        EPSILON_START = 1.0         # Initial exploration rate
        EPSILON_MIN = 0.01          # Minimum exploration rate  
        EPSILON_DECAY = 0.995       # Decay rate per update
        
        # Gradient clipping
        GRADIENT_CLIP_NORM = 1.0    # Max gradient norm
        
        # Experience replay
        BUFFER_SIZE = 100000        # Replay buffer capacity
        
        # Prioritized Experience Replay parameters
        PRIORITY_ALPHA = 0.6        # Priority exponent
        PRIORITY_BETA_START = 0.4   # Importance sampling start
        PRIORITY_BETA_INCREMENT = 0.001  # Beta increment per sample
    
    # ===========================================
    # LOGGING AND CHECKPOINTING
    # ===========================================
    
    class LoggingConfig:
        """
        Logging and checkpointing configuration
        """
        
        # Directories
        LOG_DIR = 'logs/unified_training'
        CHECKPOINT_DIR = 'checkpoints/unified'
        
        # Checkpoint frequency
        SAVE_INTERVAL = 10  # Save every N batches
        
        # TensorBoard logging groups
        METRICS_TO_LOG = {
            'Exploration': [
                'AvgTerminalReward', 'StdTerminalReward', 'MaxTerminalReward',
                'MinTerminalReward', 'NumPlacements', 'PositivePlacementRate',
                'AvgPositiveReward', 'AvgNegativeReward', 'SuccessfulPlacementRate'
            ],
            'StateModel': [
                'TotalLoss', 'RotationLoss', 'XPositionLoss', 'ValueLoss',
                'TotalLossImprovement', 'FinalEpochLoss', 'AuxiliaryTotalLoss',
                'AuxiliaryRotationLoss', 'AuxiliaryXPositionLoss', 'AuxiliaryYPositionLoss'
            ],
            'RewardPredictor': [
                'TotalLoss', 'RewardPredictionLoss', 'ValuePredictionLoss',
                'RewardMAE', 'ValueMAE'
            ],
            'Exploitation': [
                'BatchAvgReward', 'BatchAvgSteps', 'BatchStdReward', 'BatchStdSteps',
                'BatchMaxReward', 'BatchMinReward', 'BatchRewardTrend', 'EpisodeReward', 'EpisodeSteps'
            ],
            'PPO': [
                'ActorLoss', 'CriticLoss', 'RewardLoss', 'AuxiliaryLoss',
                'ActorLossRatio', 'CriticLossRatio', 'RewardLossRatio',
                'SuccessfulIterations', 'IterationSuccessRate'
            ],
            'Evaluation': [
                'AvgReward', 'AvgSteps'
            ]
        }
        
        # Log file names
        TRAINING_LOG = 'unified_training.log'
        DEBUG_LOG = 'tetris_debug.log'
        
    # ===========================================
    # ALGORITHM PARAMETERS
    # ===========================================
    
    class AlgorithmConfig:
        """
        Algorithm-specific parameters and thresholds
        """
        
        # Success thresholds for exploration
        HIGH_REWARD_THRESHOLD = -50  # Threshold for "successful" placements
        
        # State model training
        REWARD_WEIGHT_NORMALIZATION = 200  # Normalization for reward weighting
        
        # PPO specific
        ADVANTAGE_NORMALIZATION = True    # Normalize advantages
        VALUE_CLIPPING = False           # Clip value function updates
        
        # Early stopping criteria
        EARLY_STOP_PATIENCE = 5          # Batches without improvement
        EARLY_STOP_THRESHOLD = 0.001     # Minimum improvement threshold
        
        # Model evaluation
        EVALUATION_FREQUENCY = 1          # Evaluate every N batches
        BEST_MODEL_METRIC = 'avg_reward'  # Metric for best model selection

def get_device():
    """
    Automatically detect and return the best available device
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU detected: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Apple Silicon GPU (MPS) detected")
    else:
        device = 'cpu'
        print("Using CPU - no GPU detected")
    return device

def create_directories(config):
    """
    Create necessary directories for logging and checkpointing
    """
    os.makedirs(config.LoggingConfig.LOG_DIR, exist_ok=True)
    os.makedirs(config.LoggingConfig.CHECKPOINT_DIR, exist_ok=True)

# Example usage:
if __name__ == '__main__':
    config = TetrisConfig()
    print("Tetris RL Configuration loaded successfully!")
    print(f"State dimension: {config.STATE_DIM}")
    print(f"Action dimension: {config.ACTION_DIM}")
    print(f"Training batches: {config.TrainingConfig.NUM_BATCHES}")
    print(f"Device: {get_device()}") 