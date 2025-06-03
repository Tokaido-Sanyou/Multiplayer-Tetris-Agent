#!/usr/bin/env python3
"""Test script to verify the consistency fixes in the training code"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from localMultiplayerTetris.tetris_env import TetrisEnv
from localMultiplayerTetris.rl_utils.train import preprocess_state, evaluate_agent
from localMultiplayerTetris.rl_utils.actor_critic import ActorCriticAgent
from localMultiplayerTetris.rl_utils.replay_buffer import ReplayBuffer

def test_state_dimensions():
    """Test that preprocess_state returns the expected dimensions"""
    print("Testing state dimensions...")
    
    # Create a mock state dictionary
    state = {
        'grid': np.zeros((20, 10)),
        'next_piece': 3,
        'hold_piece': 1,
        'current_shape': 2,
        'current_rotation': 0,
        'current_x': 5,
        'current_y': 0
    }
    
    # Preprocess the state
    processed = preprocess_state(state)
    
    print(f"State shape: {processed.shape}")
    print(f"Expected: (206,), Actual: {processed.shape}")
    assert processed.shape == (206,), f"State dimension mismatch! Expected 206, got {processed.shape[0]}"
    print("✓ State dimensions correct\n")

def test_replay_buffer():
    """Test that replay buffer works with state dictionaries"""
    print("Testing replay buffer...")
    
    buffer = ReplayBuffer(100)
    
    # Create mock states
    state = {
        'grid': np.zeros((20, 10)),
        'next_piece': 3,
        'hold_piece': 1
    }
    
    next_state = {
        'grid': np.zeros((20, 10)),
        'next_piece': 4,
        'hold_piece': 1
    }
    
    info = {
        'lines_cleared': 2,
        'score': 200,
        'level': 1,
        'episode_steps': 10
    }
    
    # Test push
    try:
        buffer.push(state, 0, 10.0, next_state, False, info)
        print("✓ Replay buffer push successful\n")
    except Exception as e:
        print(f"✗ Replay buffer push failed: {e}\n")
        raise

def test_evaluate_agent_return():
    """Test that evaluate_agent returns three values"""
    print("Testing evaluate_agent return values...")
    
    # Create environment and agent
    env = TetrisEnv(single_player=True, headless=True)
    agent = ActorCriticAgent(206, 8)
    
    # Test evaluate_agent
    try:
        # Note: We need to handle the fact that env.reset() might return (obs, info) tuple
        # Let's do a quick test first
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            print(f"  Note: env.reset() returns tuple format (obs, info)")
        
        result = evaluate_agent(env, agent, num_episodes=1)
        assert isinstance(result, tuple), "evaluate_agent should return a tuple"
        assert len(result) == 3, f"evaluate_agent should return 3 values, got {len(result)}"
        rewards, scores, lines = result
        assert isinstance(rewards, list), "First return value should be a list"
        assert isinstance(scores, list), "Second return value should be a list"
        assert isinstance(lines, list), "Third return value should be a list"
        print(f"✓ evaluate_agent returns correct format: (rewards, scores, lines)")
        print(f"  Sample: rewards={rewards}, scores={scores}, lines={lines}\n")
    except Exception as e:
        print(f"✗ evaluate_agent test failed: {e}\n")
        raise
    finally:
        env.close()

def test_agent_memory_integration():
    """Test that agent's memory buffer works correctly"""
    print("Testing agent memory integration...")
    
    # Create agent
    agent = ActorCriticAgent(206, 8)
    
    # Create mock states
    state = {
        'grid': np.zeros((20, 10)),
        'next_piece': 3,
        'hold_piece': 1
    }
    
    next_state = {
        'grid': np.zeros((20, 10)),
        'next_piece': 4,
        'hold_piece': 1
    }
    
    info = {'lines_cleared': 0, 'score': 0, 'level': 1}
    
    # Test memory push
    try:
        agent.memory.push(state, 0, 1.0, next_state, False, info)
        print(f"✓ Agent memory push successful")
        print(f"  Memory size: {len(agent.memory)}\n")
    except Exception as e:
        print(f"✗ Agent memory push failed: {e}\n")
        raise

if __name__ == "__main__":
    print("Running consistency tests...\n")
    
    try:
        test_state_dimensions()
        test_replay_buffer()
        test_evaluate_agent_return()
        test_agent_memory_integration()
        
        print("All tests passed! ✓")
    except Exception as e:
        print(f"Tests failed: {e}")
        sys.exit(1)
