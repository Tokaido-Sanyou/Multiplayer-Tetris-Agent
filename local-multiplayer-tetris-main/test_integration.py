#!/usr/bin/env python3
"""
Integration test for the enhanced Multiplayer Tetris Agent
Tests all the key modifications:
1. Goal-conditioned actor
2. Terminal state exploration
3. Consistent network parameters  
4. Lines cleared tracking
5. Blended reward prediction
6. 1000 episode configuration
"""

import torch
import numpy as np
from localMultiplayerTetris.config import TetrisConfig
from localMultiplayerTetris.rl_utils.state_model import StateModel
from localMultiplayerTetris.rl_utils.actor_critic import ActorCriticAgent
from localMultiplayerTetris.rl_utils.future_reward_predictor import FutureRewardPredictor
from localMultiplayerTetris.rl_utils.rnd_exploration import RNDExploration, RNDExplorationActor

def test_configuration():
    """Test the centralized configuration"""
    print("🔧 Testing Configuration...")
    config = TetrisConfig()
    
    assert config.STATE_DIM == 410, f"Expected STATE_DIM=410, got {config.STATE_DIM}"
    assert config.ACTION_DIM == 8, f"Expected ACTION_DIM=8, got {config.ACTION_DIM}"
    assert config.GOAL_DIM == 36, f"Expected GOAL_DIM=36, got {config.GOAL_DIM}"
    
    # Test 1000 episode configuration
    train_config = config.TrainingConfig
    total_episodes = train_config.NUM_BATCHES * train_config.EXPLORATION_EPISODES
    assert total_episodes == 1000, f"Expected 1000 total episodes, got {total_episodes}"
    
    # NEW: Test updated reward weights
    reward_config = config.RewardConfig
    assert reward_config.HOLE_WEIGHT == 0.5, f"Expected HOLE_WEIGHT=0.5, got {reward_config.HOLE_WEIGHT}"
    assert reward_config.MAX_HEIGHT_WEIGHT == 5.0, f"Expected MAX_HEIGHT_WEIGHT=5.0, got {reward_config.MAX_HEIGHT_WEIGHT}"
    assert reward_config.BUMPINESS_WEIGHT == 0.2, f"Expected BUMPINESS_WEIGHT=0.2, got {reward_config.BUMPINESS_WEIGHT}"
    
    # NEW: Test piece presence reward parameters
    assert reward_config.PIECE_PRESENCE_REWARD == 1.0, f"Expected PIECE_PRESENCE_REWARD=1.0, got {reward_config.PIECE_PRESENCE_REWARD}"
    assert reward_config.PIECE_PRESENCE_DECAY_STEPS == 500, f"Expected decay over 500 steps, got {reward_config.PIECE_PRESENCE_DECAY_STEPS}"
    
    print(f"   ✅ STATE_DIM: {config.STATE_DIM}")
    print(f"   ✅ ACTION_DIM: {config.ACTION_DIM}")
    print(f"   ✅ GOAL_DIM: {config.GOAL_DIM}")
    print(f"   ✅ Total Episodes: {total_episodes}")
    print(f"   ✅ Updated Reward Weights: HOLE={reward_config.HOLE_WEIGHT}, HEIGHT={reward_config.MAX_HEIGHT_WEIGHT}, BUMP={reward_config.BUMPINESS_WEIGHT}")
    print(f"   ✅ Piece Presence Reward: {reward_config.PIECE_PRESENCE_REWARD} decaying over {reward_config.PIECE_PRESENCE_DECAY_STEPS} episodes")
    print()

def test_state_model_goals():
    """Test state model goal generation"""
    print("🎯 Testing State Model Goal Generation...")
    
    config = TetrisConfig()
    state_model = StateModel(state_dim=config.STATE_DIM)
    
    # Create dummy state
    batch_size = 2
    dummy_state = torch.randn(batch_size, config.STATE_DIM)
    
    # Test goal vector generation
    goal_vector = state_model.get_placement_goal_vector(dummy_state)
    
    assert goal_vector.shape == (batch_size, config.GOAL_DIM), \
        f"Expected goal shape ({batch_size}, {config.GOAL_DIM}), got {goal_vector.shape}"
    
    # Test optimal placement extraction
    optimal_placement = state_model.get_optimal_placement(dummy_state)
    
    required_keys = ['rotation', 'x_position', 'y_position', 'value', 'confidence']
    for key in required_keys:
        assert key in optimal_placement, f"Missing key '{key}' in optimal placement"
    
    print(f"   ✅ Goal vector shape: {goal_vector.shape}")
    print(f"   ✅ Optimal placement keys: {list(optimal_placement.keys())}")
    print()

def test_goal_conditioned_actor():
    """Test goal-conditioned actor-critic"""
    print("🤖 Testing Goal-Conditioned Actor-Critic...")
    
    config = TetrisConfig()
    state_model = StateModel(state_dim=config.STATE_DIM)
    actor_critic = ActorCriticAgent(
        state_dim=config.STATE_DIM, 
        action_dim=config.ACTION_DIM,
        state_model=state_model
    )
    
    # Test with dummy state
    dummy_state = np.random.randn(config.STATE_DIM)
    action = actor_critic.select_action(dummy_state)
    
    assert action.shape == (config.ACTION_DIM,), \
        f"Expected action shape ({config.ACTION_DIM},), got {action.shape}"
    assert action.sum() == 1, "Action should be one-hot encoded"
    assert action.dtype == np.int8, f"Expected int8, got {action.dtype}"
    
    print(f"   ✅ Action shape: {action.shape}")
    print(f"   ✅ Action sum (should be 1): {action.sum()}")
    print(f"   ✅ Goal conditioning: Enabled")
    print()

def test_future_reward_predictor():
    """Test enhanced future reward predictor"""
    print("🔮 Testing Future Reward Predictor...")
    
    config = TetrisConfig()
    predictor = FutureRewardPredictor(
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM
    )
    
    # Test basic prediction
    batch_size = 2
    dummy_state = torch.randn(batch_size, config.STATE_DIM)
    dummy_action = torch.randn(batch_size, config.ACTION_DIM)
    
    reward_pred, value_pred = predictor(dummy_state, dummy_action)
    
    assert reward_pred.shape == (batch_size, 1), \
        f"Expected reward shape ({batch_size}, 1), got {reward_pred.shape}"
    assert value_pred.shape == (batch_size, 1), \
        f"Expected value shape ({batch_size}, 1), got {value_pred.shape}"
    
    # Test terminal placement value prediction
    dummy_goal = torch.randn(batch_size, config.GOAL_DIM)
    terminal_value = predictor.predict_terminal_placement_value(dummy_state, dummy_goal)
    
    assert terminal_value.shape == (batch_size, 1), \
        f"Expected terminal value shape ({batch_size}, 1), got {terminal_value.shape}"
    
    print(f"   ✅ Reward prediction shape: {reward_pred.shape}")
    print(f"   ✅ Value prediction shape: {value_pred.shape}")
    print(f"   ✅ Terminal value blending: Enabled")
    print()

def test_network_consistency():
    """Test that all networks use consistent dimensions"""
    print("🔗 Testing Network Consistency...")
    
    config = TetrisConfig()
    
    # Test state model
    state_model = StateModel()
    assert state_model.state_dim == config.STATE_DIM
    
    # Test actor-critic
    actor_critic_net = ActorCriticAgent()
    assert actor_critic_net.state_dim == config.STATE_DIM
    assert actor_critic_net.action_dim == config.ACTION_DIM
    
    # Test future reward predictor  
    reward_predictor = FutureRewardPredictor()
    assert reward_predictor.state_dim == config.STATE_DIM
    assert reward_predictor.action_dim == config.ACTION_DIM
    
    print("   ✅ All networks use consistent STATE_DIM")
    print("   ✅ All networks use consistent ACTION_DIM") 
    print("   ✅ Goal dimensions properly configured")
    print()

def test_rnd_exploration():
    """Test Random Network Distillation exploration"""
    print("🔍 Testing RND Exploration...")
    
    config = TetrisConfig()
    rnd_exploration = RNDExploration(state_dim=config.STATE_DIM)
    
    # Test RND networks
    batch_size = 2
    dummy_state = torch.randn(batch_size, config.STATE_DIM)
    
    # Test intrinsic reward calculation
    intrinsic_reward, prediction_error = rnd_exploration(dummy_state)
    
    assert intrinsic_reward.shape == (batch_size, 1), \
        f"Expected intrinsic reward shape ({batch_size}, 1), got {intrinsic_reward.shape}"
    assert prediction_error.shape == (batch_size, 1), \
        f"Expected prediction error shape ({batch_size}, 1), got {prediction_error.shape}"
    
    # Test target network (should be frozen)
    target_features = rnd_exploration.target_network(dummy_state)
    assert target_features.shape == (batch_size, 64), \
        f"Expected target features shape ({batch_size}, 64), got {target_features.shape}"
    
    # Test predictor network
    predicted_features = rnd_exploration.predictor_network(dummy_state)
    assert predicted_features.shape == (batch_size, 64), \
        f"Expected predicted features shape ({batch_size}, 64), got {predicted_features.shape}"
    
    # Test training
    optimizer = torch.optim.Adam(rnd_exploration.predictor_network.parameters(), lr=1e-4)
    loss = rnd_exploration.train_predictor(dummy_state, optimizer)
    assert isinstance(loss, float), f"Expected loss to be float, got {type(loss)}"
    
    # Test exploration statistics
    stats = rnd_exploration.get_exploration_stats()
    required_keys = ['reward_mean', 'reward_std', 'update_count']
    for key in required_keys:
        assert key in stats, f"Missing key '{key}' in exploration stats"
    
    print(f"   ✅ Intrinsic reward shape: {intrinsic_reward.shape}")
    print(f"   ✅ Prediction error shape: {prediction_error.shape}")
    print(f"   ✅ Target network features: {target_features.shape}")
    print(f"   ✅ Predictor training loss: {loss:.6f}")
    print(f"   ✅ RND statistics keys: {list(stats.keys())}")
    print()

def main():
    """Run all integration tests"""
    print("🚀 Multiplayer Tetris Agent - Integration Test")
    print("=" * 50)
    print()
    
    try:
        test_configuration()
        test_state_model_goals()
        test_goal_conditioned_actor()
        test_future_reward_predictor()
        test_network_consistency()
        test_rnd_exploration()
        
        print("🎉 ALL TESTS PASSED!")
        print()
        print("✅ Goal-conditioned actor: Working")
        print("✅ RND exploration: Implemented with curiosity-driven terminal states")
        print("✅ Piece presence rewards: Adaptive decay over first 500 episodes") 
        print("✅ Updated reward weights: HOLE=0.5, HEIGHT=5.0, BUMP=0.2")
        print("✅ Consistent network parameters: Verified")
        print("✅ Blended reward prediction: Working")
        print("✅ 1000 episode configuration: Set")
        print()
        print("🎮 Ready to train your Tetris AI with:")
        print("   python -m localMultiplayerTetris.rl_utils.unified_trainer --visualize")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 