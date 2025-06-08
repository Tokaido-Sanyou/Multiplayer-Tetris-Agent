#!/usr/bin/env python3
"""
Demo script showcasing the key features of the Tetris ML environment
"""

import numpy as np
from envs.tetris_env import TetrisEnv
import time

def demo_single_agent():
    """Demonstrate single agent functionality"""
    print("🎮 Demo: Single Agent Mode")
    print("=" * 40)
    
    env = TetrisEnv(num_agents=1, headless=True, step_mode='action')
    obs = env.reset()
    
    print(f"Observation space keys: {list(obs.keys())}")
    print(f"Grid shape: {obs['empty_grid'].shape}")
    print(f"Next piece one-hot: {obs['next_piece']}")
    
    total_reward = 0
    for step in range(20):
        action = np.zeros(8)
        action[np.random.randint(0, 8)] = 1
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if info['lines_cleared'] > 0:
            print(f"  Step {step}: Cleared {info['lines_cleared']} lines! Reward: {reward:.2f}")
        
        if done:
            print(f"  Game over at step {step}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.close()
    print()

def demo_multi_agent():
    """Demonstrate multi-agent functionality"""
    print("👥 Demo: Multi-Agent Mode")
    print("=" * 40)
    
    env = TetrisEnv(num_agents=2, headless=True, step_mode='action')
    obs = env.reset()
    
    print(f"Agents: {list(obs.keys())}")
    print(f"Agent 0 has opponent grid: {'opponent_grid' in obs['agent_0']}")
    
    total_rewards = {'agent_0': 0, 'agent_1': 0}
    
    for step in range(15):
        actions = {
            'agent_0': np.zeros(8),
            'agent_1': np.zeros(8)
        }
        actions['agent_0'][np.random.randint(0, 8)] = 1
        actions['agent_1'][np.random.randint(0, 8)] = 1
        
        obs, rewards, done, infos = env.step(actions)
        
        for agent in ['agent_0', 'agent_1']:
            total_rewards[agent] += rewards[agent]
            if infos[agent]['lines_cleared'] > 0:
                print(f"  {agent} cleared {infos[agent]['lines_cleared']} lines at step {step}")
        
        if done:
            print(f"  Game over at step {step}")
            break
    
    print(f"Final rewards - Agent 0: {total_rewards['agent_0']:.2f}, Agent 1: {total_rewards['agent_1']:.2f}")
    env.close()
    print()

def demo_board_state_management():
    """Demonstrate board state saving and restoration"""
    print("💾 Demo: Board State Management")
    print("=" * 40)
    
    env = TetrisEnv(num_agents=1, headless=True)
    obs = env.reset()
    
    # Play some moves
    print("Playing 5 moves...")
    for i in range(5):
        action = np.zeros(8)
        action[np.random.randint(0, 8)] = 1
        obs, reward, done, info = env.step(action)
        print(f"  Move {i+1}: Score = {info['score']}")
    
    # Save state
    env.save_board_state('demo_checkpoint')
    print("\n📍 Board state saved as 'demo_checkpoint'")
    
    # Continue playing
    print("\nPlaying 5 more moves...")
    for i in range(5):
        action = np.zeros(8)
        action[np.random.randint(0, 8)] = 1
        obs, reward, done, info = env.step(action)
        print(f"  Move {i+6}: Score = {info['score']}")
    
    # Restore state
    print("\n🔄 Restoring board state...")
    obs = env.restore_board_state('demo_checkpoint')
    
    # Verify restoration
    action = np.zeros(8)
    action[7] = 1  # No-op
    obs, reward, done, info = env.step(action)
    print(f"After restoration - Score: {info['score']}")
    
    env.close()
    print()

def demo_trajectory_tracking():
    """Demonstrate trajectory tracking with branching"""
    print("🌳 Demo: Trajectory Tracking")
    print("=" * 40)
    
    env = TetrisEnv(num_agents=1, headless=True, enable_trajectory_tracking=True)
    obs = env.reset()
    
    # Start main trajectory
    env.start_trajectory('main_path')
    print("Started main trajectory")
    
    # Play some moves
    for i in range(8):
        action = np.zeros(8)
        action[np.random.randint(0, 8)] = 1
        obs, reward, done, info = env.step(action)
        
        if i == 4:  # Branch at step 5
            env.start_trajectory('branch_path', parent_id='main_path', branch_point=4)
            print("  🌿 Created branch trajectory at step 5")
    
    # Get trajectories
    main_traj = env.get_trajectory('main_path')
    branch_traj = env.get_trajectory('branch_path')
    
    print(f"\nTrajectory Results:")
    print(f"  Main trajectory: {len(main_traj.states)} states, {len(main_traj.actions)} actions")
    print(f"  Branch trajectory: {len(branch_traj.states)} states, {len(branch_traj.actions)} actions")
    print(f"  Branch parent: {branch_traj.parent_trajectory.trajectory_id}")
    print(f"  Branch point: {branch_traj.branch_point}")
    
    env.close()
    print()

def demo_mode_switching():
    """Demonstrate dynamic mode switching"""
    print("🔄 Demo: Mode Switching")
    print("=" * 40)
    
    env = TetrisEnv(num_agents=1, headless=True)
    obs = env.reset()
    
    print("Started in single-agent mode")
    print(f"Observation type: {type(obs)}")
    
    # Switch to multi-agent
    obs = env.switch_mode(num_agents=2)
    print("\nSwitched to multi-agent mode")
    print(f"Observation type: {type(obs)}")
    print(f"Agent keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
    
    # Switch step mode
    obs = env.switch_mode(step_mode='block_placed')
    print("\nSwitched to 'block_placed' step mode")
    
    # Switch back to single agent
    obs = env.switch_mode(num_agents=1, step_mode='action')
    print("\nSwitched back to single-agent, action mode")
    
    env.close()
    print()

def demo_enhanced_rewards():
    """Demonstrate the enhanced reward system"""
    print("💰 Demo: Enhanced Reward System")
    print("=" * 40)
    
    env = TetrisEnv(num_agents=1, headless=True)
    obs = env.reset()
    
    print("Testing different action patterns...")
    
    # Test different actions and their rewards
    action_names = ['Left', 'Right', 'Soft Drop', 'Rotate CW', 'Rotate CCW', 'Hard Drop', 'Hold', 'No-op']
    
    for i, action_name in enumerate(action_names[:4]):
        action = np.zeros(8)
        action[i] = 1
        
        obs, reward, done, info = env.step(action)
        print(f"  {action_name}: Reward = {reward:.3f}")
        
        if done:
            obs = env.reset()
    
    env.close()
    print()

def main():
    """Run all demos"""
    print("🎯 Tetris ML Environment Feature Demo")
    print("=" * 50)
    print()
    
    try:
        demo_single_agent()
        demo_multi_agent()
        demo_board_state_management()
        demo_trajectory_tracking()
        demo_mode_switching()
        demo_enhanced_rewards()
        
        print("✅ All demos completed successfully!")
        print("\nThe environment is ready for your ML experiments!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 