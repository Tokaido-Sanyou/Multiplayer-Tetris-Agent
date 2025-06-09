#!/usr/bin/env python3
"""
Debug script for movement agent
"""

import numpy as np
from envs.tetris_env import TetrisEnv
from agents.dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent
from agents.dqn_movement_agent_redesigned import RedesignedMovementAgent

def debug_movement_agent():
    print("🔍 DEBUGGING MOVEMENT AGENT")
    print("=" * 50)
    
    try:
        # Create environment
        env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='direct',
            reward_mode='standard'
        )
        
        # Create locked agent
        locked_agent = RedesignedLockedStateDQNAgent(
            input_dim=206,
            num_actions=800,
            device='cuda'
        )
        
        # Create movement agent
        movement_agent = RedesignedMovementAgent(
            input_dim=1012,
            num_actions=8,
            device='cuda'
        )
        
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Test state composition
        print("Testing state composition...")
        
        # Extract state components
        board_flat = obs[:200]
        current_piece = obs[200:206]
        next_piece = np.zeros(6)
        
        print(f"Board shape: {board_flat.shape}")
        print(f"Current piece shape: {current_piece.shape}")
        print(f"Next piece shape: {next_piece.shape}")
        
        # Get locked Q-values
        locked_q_values = locked_agent.get_q_values(obs)
        print(f"Locked Q-values shape: {locked_q_values.shape}")
        
        # Combine movement state
        movement_state = np.concatenate([
            board_flat,
            current_piece,
            next_piece,
            locked_q_values
        ])
        print(f"Movement state shape: {movement_state.shape}")
        
        # Test action selection
        action = movement_agent.select_action(movement_state, training=False)
        print(f"Selected action: {action}")
        
        # Execute one step
        next_obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        
        print("✅ Movement agent debug completed successfully")
        
    except Exception as e:
        print(f"❌ Error in movement agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_movement_agent() 