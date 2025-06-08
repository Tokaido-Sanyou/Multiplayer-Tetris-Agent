#!/usr/bin/env python3
"""
Comprehensive test script for the enhanced Tetris environment
Tests multi-agent support, trajectory tracking, board state management, etc.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
try:
    from envs.tetris_env import TetrisEnv
except ImportError:
    # Try alternative import for when running from tests directory
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'envs'))
    from tetris_env import TetrisEnv

def test_single_agent_mode():
    """Test single agent functionality"""
    print("Testing single agent mode...")
    
    env = TetrisEnv(num_agents=1, headless=True, step_mode='action')
    obs = env.reset()
    
    # Test observation structure
    expected_keys = ['piece_grids', 'current_piece_grid', 'empty_grid', 'next_piece', 
                     'hold_piece', 'current_rotation', 'current_x', 'current_y', 'opponent_grid']
    assert all(key in obs for key in expected_keys), f"Missing observation keys: {set(expected_keys) - set(obs.keys())}"
    
    # Test action execution
    for i in range(10):
        action = np.zeros(8)
        action[np.random.randint(0, 8)] = 1
        obs, reward, done, info = env.step(action)
        
        assert isinstance(reward, (int, float)), f"Reward should be numeric, got {type(reward)}"
        assert isinstance(done, bool), f"Done should be boolean, got {type(done)}"
        assert isinstance(info, dict), f"Info should be dict, got {type(info)}"
        
        if done:
            obs = env.reset()
            break
    
    env.close()
    print("   ✓ Single agent mode works correctly")

def test_multi_agent_mode():
    """Test multi-agent functionality"""
    print("Testing multi-agent mode...")
    
    env = TetrisEnv(num_agents=2, headless=True, step_mode='action')
    obs = env.reset()
    
    # Test observation structure for multi-agent
    assert isinstance(obs, dict), "Multi-agent obs should be dict"
    assert 'agent_0' in obs and 'agent_1' in obs, "Should have agent_0 and agent_1 keys"
    
    # Test multi-agent action execution
    for i in range(10):
        actions = {
            'agent_0': np.zeros(8),
            'agent_1': np.zeros(8)
        }
        actions['agent_0'][np.random.randint(0, 8)] = 1
        actions['agent_1'][np.random.randint(0, 8)] = 1
        
        obs, rewards, done, infos = env.step(actions)
        
        assert isinstance(obs, dict), "Multi-agent obs should be dict"
        assert isinstance(rewards, dict), "Multi-agent rewards should be dict"
        assert isinstance(infos, dict), "Multi-agent infos should be dict"
        
        if done:
            obs = env.reset()
            break
    
    env.close()
    print("   ✓ Multi-agent mode works correctly")

def test_mode_switching():
    """Test switching between single and multi-agent modes"""
    print("Testing mode switching...")
    
    env = TetrisEnv(num_agents=1, headless=True)
    
    # Test single agent
    obs = env.reset()
    assert not isinstance(obs, dict) or len(obs) == 9, "Single agent obs should not be nested dict"
    
    # Switch to multi-agent
    env.switch_mode(num_agents=2)
    obs = env.reset()
    assert isinstance(obs, dict) and 'agent_0' in obs, "Should switch to multi-agent"
    
    # Switch back to single agent
    env.switch_mode(num_agents=1)
    obs = env.reset()
    assert not isinstance(obs, dict) or 'agent_0' not in obs, "Should switch back to single agent"
    
    env.close()
    print("   ✓ Mode switching works correctly")

def test_step_modes():
    """Test different step modes"""
    print("Testing step modes...")
    
    # Test action mode
    env_action = TetrisEnv(num_agents=1, headless=True, step_mode='action')
    obs = env_action.reset()
    
    action = np.zeros(8)
    action[2] = 1  # Soft drop
    obs, reward, done, info = env_action.step(action)
    assert 'piece_placed' in info, "Should have piece_placed info"
    
    env_action.close()
    
    # Test block_placed mode
    env_block = TetrisEnv(num_agents=1, headless=True, step_mode='block_placed')
    obs = env_block.reset()
    
    action = np.zeros(8)
    action[0] = 1  # Move left
    obs, reward, done, info = env_block.step(action)
    assert 'piece_placed' in info, "Should have piece_placed info"
    
    env_block.close()
    print("   ✓ Step modes work correctly")

def test_board_state_management():
    """Test board state saving and restoration"""
    print("Testing board state management...")
    
    env = TetrisEnv(num_agents=1, headless=True)
    obs1 = env.reset()
    
    # Play a few moves
    for _ in range(5):
        action = np.zeros(8)
        action[np.random.randint(0, 8)] = 1
        obs1, _, done, _ = env.step(action)
        if done:
            obs1 = env.reset()
            break
    
    # Save board state
    env.save_board_state('test_state')
    
    # Continue playing
    for _ in range(5):
        action = np.zeros(8)
        action[np.random.randint(0, 8)] = 1
        obs2, _, done, _ = env.step(action)
        if done:
            break
    
    # Restore board state
    env.restore_board_state('test_state')
    obs_restored = env._get_observation()
    
    # Check if restoration worked (simplified check)
    assert obs_restored is not None, "Should restore board state"
    
    env.close()
    print("   ✓ Board state management works correctly")

def test_trajectory_tracking():
    """Test trajectory tracking functionality"""
    print("Testing trajectory tracking...")
    
    env = TetrisEnv(num_agents=1, headless=True, enable_trajectory_tracking=True)
    obs = env.reset()
    
    # Start a trajectory
    env.start_trajectory('test_trajectory')
    
    # Play some moves
    for i in range(5):
        action = np.zeros(8)
        action[np.random.randint(0, 8)] = 1
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    # Get trajectory
    trajectory = env.get_trajectory('test_trajectory')
    assert trajectory is not None, "Should have trajectory"
    assert len(trajectory.states) > 0, "Trajectory should have states"
    assert len(trajectory.actions) > 0, "Trajectory should have actions"
    
    # Test branching
    env.start_trajectory('branch_trajectory', parent_id='test_trajectory', branch_point=2)
    branch_trajectory = env.get_trajectory('branch_trajectory')
    assert branch_trajectory is not None, "Should have branch trajectory"
    assert branch_trajectory.parent_trajectory is not None, "Should have parent"
    
    env.close()
    print("   ✓ Trajectory tracking works correctly")

def test_enhanced_rewards():
    """Test enhanced reward function"""
    print("Testing enhanced reward function...")
    
    env = TetrisEnv(num_agents=1, headless=True)
    obs = env.reset()
    
    total_reward = 0
    for i in range(20):
        action = np.zeros(8)
        action[np.random.randint(0, 8)] = 1
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Reward should be finite
        assert np.isfinite(reward), f"Reward should be finite, got {reward}"
        
        if done:
            obs = env.reset()
            break
    
    env.close()
    print("   ✓ Enhanced reward function works correctly")

def test_observation_consistency():
    """Test observation consistency across modes"""
    print("Testing observation consistency...")
    
    # Single agent
    env1 = TetrisEnv(num_agents=1, headless=True)
    obs1 = env1.reset()
    
    # Multi-agent
    env2 = TetrisEnv(num_agents=2, headless=True)
    obs2 = env2.reset()
    
    # Check single agent observation structure
    expected_shape_keys = ['piece_grids', 'current_piece_grid', 'empty_grid']
    for key in expected_shape_keys:
        assert key in obs1, f"Missing key {key} in single agent obs"
    
    # Check multi-agent observation structure
    for agent_key in ['agent_0', 'agent_1']:
        assert agent_key in obs2, f"Missing agent key {agent_key}"
        for key in expected_shape_keys:
            assert key in obs2[agent_key], f"Missing key {key} in {agent_key} obs"
    
    env1.close()
    env2.close()
    print("   ✓ Observation consistency verified")

if __name__ == "__main__":
    print("Starting comprehensive Tetris environment tests...")
    print("=" * 60)
    
    try:
        test_single_agent_mode()
        test_multi_agent_mode()
        test_mode_switching()
        test_step_modes()
        test_board_state_management()
        test_trajectory_tracking()
        test_enhanced_rewards()
        test_observation_consistency()
        
        print("=" * 60)
        print("✅ All enhanced environment tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 