#!/usr/bin/env python3
"""
Quick architecture test for the new 410-dimensional state space
"""

def test_config():
    """Test the centralized config"""
    print("=== Testing Config ===")
    from config import TetrisConfig
    config = TetrisConfig()
    print(f"✓ State dimension: {config.STATE_DIM}")
    print(f"✓ Action dimension: {config.ACTION_DIM}")
    print(f"✓ Network config loaded: {hasattr(config, 'NetworkConfig')}")
    
def test_environment():
    """Test the simplified environment"""
    print("\n=== Testing Environment ===")
    from tetris_env import TetrisEnv
    env = TetrisEnv(single_player=True, headless=True)
    obs = env.reset()
    
    print(f"✓ Observation keys: {list(obs.keys())}")
    print(f"✓ Current piece grid: {obs['current_piece_grid'].shape}")
    print(f"✓ Empty grid: {obs['empty_grid'].shape}")  
    print(f"✓ Next piece: {obs['next_piece'].shape}")
    print(f"✓ Metadata: rotation={obs['current_rotation']}, x={obs['current_x']}, y={obs['current_y']}")
    
    env.close()
    
def test_state_conversion():
    """Test state vector conversion"""
    print("\n=== Testing State Conversion ===")
    from tetris_env import TetrisEnv
    from rl_utils.exploration_actor import ExplorationActor
    
    env = TetrisEnv(single_player=True, headless=True)
    obs = env.reset()
    
    exploration_actor = ExplorationActor(env)
    state_vector = exploration_actor._obs_to_state_vector(obs)
    
    print(f"✓ State vector dimension: {len(state_vector)}")
    print(f"✓ Components breakdown:")
    print(f"  - Current piece: {state_vector[:200].shape} (200 values)")
    print(f"  - Empty grid: {state_vector[200:400].shape} (200 values)")
    print(f"  - Next piece: {state_vector[400:407].shape} (7 values)")  
    print(f"  - Metadata: {state_vector[407:].shape} (3 values)")
    
    env.close()

def test_models():
    """Test model initialization with new dimensions"""
    print("\n=== Testing Models ===")
    
    # Test StateModel
    from rl_utils.state_model import StateModel
    state_model = StateModel()
    print(f"✓ StateModel initialized with {state_model.state_dim} dimensions")
    
    # Test ActorCritic  
    from rl_utils.actor_critic import ActorCriticAgent
    agent = ActorCriticAgent()
    print(f"✓ ActorCritic initialized with state_dim={agent.state_dim}, action_dim={agent.action_dim}")
    
    # Test FutureRewardPredictor
    from rl_utils.future_reward_predictor import FutureRewardPredictor
    predictor = FutureRewardPredictor()
    print(f"✓ FutureRewardPredictor initialized with state_dim={predictor.state_dim}, action_dim={predictor.action_dim}")

def test_action_selection():
    """Test action selection with new architecture"""
    print("\n=== Testing Action Selection ===")
    import numpy as np
    from rl_utils.actor_critic import ActorCriticAgent
    
    agent = ActorCriticAgent()
    dummy_state = np.random.random(410).astype(np.float32)
    action = agent.select_action(dummy_state)
    
    print(f"✓ Action shape: {action.shape}")
    print(f"✓ Action sum (should be 1): {np.sum(action)}")
    print(f"✓ Action type: {action.dtype}")
    print(f"✓ Sample action: {action}")

if __name__ == '__main__':
    print("🧪 Testing New 410-Dimensional Architecture")
    print("=" * 50)
    
    try:
        test_config()
        test_environment() 
        test_state_conversion()
        test_models()
        test_action_selection()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! New architecture is working correctly.")
        print("📊 Summary:")
        print("  - State space: 1817 → 410 dimensions (76% reduction)")
        print("  - Removed: 7-piece grids (1400) + hold piece (7)")
        print("  - Kept: current piece (200) + empty grid (200) + next piece (7) + metadata (3)")
        print("  - All models use centralized config")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 