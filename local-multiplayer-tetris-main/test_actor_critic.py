#!/usr/bin/env python3
"""
Simple test script to verify Actor-Critic integration with TetrisEnv
"""
import sys
import os
import numpy as np
import torch

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from localMultiplayerTetris.tetris_env import TetrisEnv
from localMultiplayerTetris.rl_utils.actor_critic import ActorCriticAgent
from localMultiplayerTetris.rl_utils.train import preprocess_state

def test_actor_critic_integration():
    """Test that Actor-Critic agent can interact with TetrisEnv"""
    print("Testing Actor-Critic integration with TetrisEnv...")
    
    try:
        # Create environment
        print("Creating TetrisEnv...")
        env = TetrisEnv(single_player=True, headless=True, total_episodes=10)
        
        # Create agent
        print("Creating ActorCriticAgent...")
        state_dim = 202  # 20x10 grid + next_piece + hold_piece scalars
        action_dim = 8   # 8 possible actions
        agent = ActorCriticAgent(state_dim, action_dim)
        
        # Test one episode
        print("Running test episode...")
        obs = env.reset()
        print(f"Initial observation type: {type(obs)}")
        print(f"Initial observation keys: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}")
        
        # Test state preprocessing
        print("Testing state preprocessing...")
        state_array = preprocess_state(obs)
        print(f"Preprocessed state shape: {state_array.shape}")
        print(f"Preprocessed state type: {type(state_array)}")
        
        # Test action selection
        print("Testing action selection...")
        action = agent.select_action(state_array)
        print(f"Selected action: {action} (type: {type(action)})")
        
        # Test environment step
        print("Testing environment step...")
        step_result = env.step(action)
        print(f"Step result type: {type(step_result)}")
        
        if isinstance(step_result, tuple) and len(step_result) == 4:
            next_obs, reward, done, info = step_result
            print(f"Next obs type: {type(next_obs)}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print(f"Info: {info}")
            
            # Test storing in replay buffer
            print("Testing replay buffer storage...")
            agent.memory.push(obs, action, reward, next_obs, done, info)
            print("Successfully stored transition in replay buffer")
            
            # Test training (if we have enough samples)
            print("Testing agent training...")
            train_result = agent.train()
            if train_result is not None:
                actor_loss, critic_loss = train_result
                print(f"Training successful - Actor loss: {actor_loss:.4f}, Critic loss: {critic_loss:.4f}")
            else:
                print("Training skipped (not enough samples in buffer)")
        else:
            print(f"Unexpected step result: {step_result}")
        
        env.close()
        print("✅ Actor-Critic integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Actor-Critic integration test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_actor_critic_integration()
    sys.exit(0 if success else 1)
