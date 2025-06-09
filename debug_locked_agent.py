#!/usr/bin/env python3
"""
Debug script for locked agent to identify issues
"""

import numpy as np
from envs.tetris_env import TetrisEnv
from agents.dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent

def debug_locked_agent():
    print("🔍 DEBUGGING LOCKED AGENT")
    print("=" * 50)
    
    # Create environment
    env = TetrisEnv(
        num_agents=1,
        headless=True,
        action_mode='locked_position',
        reward_mode='standard'
    )
    
    # Create agent
    agent = RedesignedLockedStateDQNAgent(
        input_dim=206,
        num_actions=800,
        device='cuda',
        epsilon_start=0.1  # Low epsilon for mostly greedy actions
    )
    
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for step in range(50):
        try:
            # Get action from agent
            action = agent.select_action(obs, training=True, env=env)
            print(f"Step {step}: Action {action}")
            
            # Check if action is valid before executing
            is_valid = agent.is_valid_action(action, env)
            print(f"  Action valid: {is_valid}")
            
            if not is_valid:
                print(f"  Invalid action! Trying fallback...")
                # Try first 10 actions to find a valid one
                for fallback_action in range(10):
                    if agent.is_valid_action(fallback_action, env):
                        action = fallback_action
                        print(f"  Using fallback action: {action}")
                        break
                else:
                    print(f"  No valid fallback found, using action 0")
                    action = 0
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            print(f"  Reward: {reward}, Done: {done}")
            
            if 'lines_cleared' in info:
                print(f"  Lines cleared: {info['lines_cleared']}")
            
            if done:
                print(f"Episode finished at step {step}")
                break
                
            obs = next_obs
            
        except Exception as e:
            print(f"ERROR at step {step}: {e}")
            break
    
    print(f"Invalid action count: {agent.invalid_action_count}")

if __name__ == "__main__":
    debug_locked_agent() 