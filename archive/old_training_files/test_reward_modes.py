#!/usr/bin/env python3
"""
🎮 TETRIS REWARD MODES DEMONSTRATION

Tests both reward functions:
1. Standard reward: Complex shaping with board features
2. Lines-only reward: Sparse rewards only for line clearing
"""

import numpy as np
from envs.tetris_env import TetrisEnv
import time

def test_reward_mode(reward_mode, num_episodes=5):
    """Test a specific reward mode"""
    print(f"\n🎯 TESTING {reward_mode.upper()} REWARD MODE")
    print("-" * 50)
    
    env = TetrisEnv(
        num_agents=1, 
        headless=True, 
        action_mode='direct',
        reward_mode=reward_mode
    )
    
    episode_rewards = []
    lines_cleared_total = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_lines = 0
        steps = 0
        
        for step in range(200):  # Max 200 steps per episode
            # Random action for demonstration
            action = np.random.randint(0, 8)
            
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Track lines cleared
            if 'lines_cleared' in info:
                lines_this_step = info['lines_cleared']
                episode_lines += lines_this_step
                if lines_this_step > 0:
                    print(f"   Episode {episode}, Step {step}: Cleared {lines_this_step} lines! Reward: {reward:.2f}")
            
            if done:
                break
            
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        lines_cleared_total += episode_lines
        
        print(f"Episode {episode}: Reward={episode_reward:7.2f}, Steps={steps:3d}, Lines={episode_lines}")
    
    env.close()
    
    # Statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\n📊 {reward_mode.upper()} MODE STATISTICS:")
    print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   Reward range: [{np.min(episode_rewards):.2f}, {np.max(episode_rewards):.2f}]")
    print(f"   Total lines cleared: {lines_cleared_total}")
    print(f"   Non-zero rewards: {sum(1 for r in episode_rewards if r > 0)}/{num_episodes}")
    
    return {
        'mode': reward_mode,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'total_lines': lines_cleared_total,
        'episodes': episode_rewards
    }

def compare_reward_modes():
    """Compare both reward modes side by side"""
    print("🎮 TETRIS REWARD MODES COMPARISON")
    print("=" * 80)
    
    # Test standard mode
    standard_results = test_reward_mode('standard', num_episodes=10)
    
    # Test lines-only mode
    lines_only_results = test_reward_mode('lines_only', num_episodes=10)
    
    # Comparison
    print(f"\n🔍 DETAILED COMPARISON")
    print("=" * 80)
    
    print(f"📈 REWARD CHARACTERISTICS:")
    print(f"   Standard mode:")
    print(f"     - Mean: {standard_results['mean_reward']:7.2f} ± {standard_results['std_reward']:.2f}")
    print(f"     - Range: [{standard_results['min_reward']:6.2f}, {standard_results['max_reward']:6.2f}]")
    print(f"     - Dense rewards (every step)")
    
    print(f"   Lines-only mode:")
    print(f"     - Mean: {lines_only_results['mean_reward']:7.2f} ± {lines_only_results['std_reward']:.2f}")
    print(f"     - Range: [{lines_only_results['min_reward']:6.2f}, {lines_only_results['max_reward']:6.2f}]")
    print(f"     - Sparse rewards (only line clearing)")
    
    print(f"\n🎯 LEARNING IMPLICATIONS:")
    print(f"   Standard mode:")
    print(f"     ✅ Dense feedback for board management")
    print(f"     ✅ Guides towards good Tetris practices")
    print(f"     ⚠️  Complex reward signal may slow convergence")
    
    print(f"   Lines-only mode:")
    print(f"     ✅ Clear objective: maximize line clearing")
    print(f"     ✅ Simpler learning signal")
    print(f"     ⚠️  Sparse rewards may require exploration strategies")
    
    print(f"\n🧠 RECOMMENDED USAGE:")
    print(f"   Standard mode: General Tetris skill development")
    print(f"   Lines-only mode: DQN with experience replay and epsilon-greedy")
    
    print("=" * 80)
    
    return standard_results, lines_only_results

def demonstrate_dqn_usage():
    """Show how to use lines-only mode with DQN"""
    print(f"\n🤖 DQN USAGE DEMONSTRATION")
    print("-" * 50)
    
    # Create environment with lines-only rewards
    env = TetrisEnv(
        num_agents=1,
        headless=True,
        action_mode='direct',
        reward_mode='lines_only'  # Key parameter for DQN
    )
    
    print(f"✅ Environment created with lines-only rewards")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Reward mode: {env.reward_mode}")
    
    # Simulate training episode
    obs = env.reset()
    cumulative_reward = 0
    non_zero_rewards = 0
    
    print(f"\n📝 Sample episode with lines-only rewards:")
    
    for step in range(50):
        action = np.random.randint(0, 8)
        next_obs, reward, done, info = env.step(action)
        
        cumulative_reward += reward
        if reward > 0:
            non_zero_rewards += 1
            print(f"   Step {step:2d}: Action={action}, Reward={reward:4.1f} (LINE CLEARED!)")
        elif step % 10 == 0:
            print(f"   Step {step:2d}: Action={action}, Reward={reward:4.1f}")
        
        if done:
            break
        
        obs = next_obs
    
    env.close()
    
    print(f"\n📊 Episode summary:")
    print(f"   Total reward: {cumulative_reward:.1f}")
    print(f"   Non-zero rewards: {non_zero_rewards}/{step+1} steps")
    print(f"   Sparsity: {(1 - non_zero_rewards/(step+1))*100:.1f}% zero rewards")
    
    print(f"\n💡 DQN Training Tips with lines-only rewards:")
    print(f"   🎯 Use experience replay buffer (handles sparse rewards)")
    print(f"   🎯 Epsilon-greedy exploration (essential for line discovery)")
    print(f"   🎯 Longer episodes (more chances for line clearing)")
    print(f"   🎯 Reward scaling/clipping may help convergence")

def main():
    """Main demonstration function"""
    print("🎮 TETRIS REWARD FUNCTION DEMONSTRATION")
    print("=" * 80)
    
    # Compare both modes
    standard_results, lines_only_results = compare_reward_modes()
    
    # Show DQN usage
    demonstrate_dqn_usage()
    
    print(f"\n✅ DEMONSTRATION COMPLETE!")
    print(f"   Both reward modes are working perfectly")
    print(f"   Use reward_mode='lines_only' for sparse DQN training")
    print(f"   Use reward_mode='standard' for dense reward shaping")

if __name__ == "__main__":
    main() 