#!/usr/bin/env python3
"""
Simple test to show lines cleared statistics tracking
"""
import numpy as np
from tetris_env import TetrisEnv

def test_lines_tracking():
    """Test basic lines cleared tracking"""
    print("🎮 Testing Lines Cleared Tracking\n")
    
    env = TetrisEnv(reward_mode='lines_only')
    
    # Simulate some episodes
    episode_lines_cleared = []
    episode_rewards = []
    
    print("Running test episodes...")
    
    for episode in range(50):
        state = env.reset()
        episode_reward = 0
        lines_cleared = 0
        steps = 0
        
        while steps < 100:  # Short episodes
            action = env.action_space.sample()  # Random actions
            next_state, reward, done, info = env.step(action)
            
            # Track lines cleared
            if 'lines_cleared' in info:
                lines_cleared += info['lines_cleared']
            
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_lines_cleared.append(lines_cleared)
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 9:
            recent_lines = episode_lines_cleared[-10:]
            recent_rewards = episode_rewards[-10:]
            
            avg_lines = np.mean(recent_lines)
            max_lines = max(recent_lines)
            avg_reward = np.mean(recent_rewards)
            
            print(f"Episodes {episode-9:2d}-{episode:2d}: "
                  f"Avg Lines: {avg_lines:.2f}, Max Lines: {max_lines}, "
                  f"Avg Reward: {avg_reward:.2f}")
    
    # Final statistics for last 50 episodes
    print(f"\n📊 Final Statistics (Last 50 episodes):")
    print(f"  🏆 Highest Lines Cleared: {max(episode_lines_cleared)}")
    print(f"  📈 Average Lines/Episode: {np.mean(episode_lines_cleared):.2f}")
    print(f"  💰 Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"  🎯 Max Reward: {max(episode_rewards):.2f}")
    
    # Top 5 performances
    sorted_indices = np.argsort(episode_lines_cleared)[::-1]
    top_5_lines = [episode_lines_cleared[i] for i in sorted_indices[:5]]
    top_5_rewards = [episode_rewards[i] for i in sorted_indices[:5]]
    
    print(f"  🔥 Top 5 Line Performances: {top_5_lines}")
    print(f"  💎 Corresponding Rewards: {[f'{r:.1f}' for r in top_5_rewards]}")
    
    # Distribution analysis
    print(f"\n📋 Performance Distribution:")
    print(f"  Episodes with 0 lines: {len([x for x in episode_lines_cleared if x == 0])}")
    print(f"  Episodes with 1+ lines: {len([x for x in episode_lines_cleared if x >= 1])}")
    print(f"  Episodes with 2+ lines: {len([x for x in episode_lines_cleared if x >= 2])}")
    print(f"  Episodes with 3+ lines: {len([x for x in episode_lines_cleared if x >= 3])}")
    print(f"  Episodes with 4+ lines: {len([x for x in episode_lines_cleared if x >= 4])}")
    
    if any(x > 0 for x in episode_lines_cleared):
        non_zero_lines = [x for x in episode_lines_cleared if x > 0]
        success_rate = len(non_zero_lines) / len(episode_lines_cleared) * 100
        print(f"  Success rate (clearing lines): {success_rate:.1f}%")
        print(f"  Average lines when successful: {np.mean(non_zero_lines):.2f}")
    
    return episode_lines_cleared, episode_rewards

def show_dreamer_tracking_format():
    """Show the format that Dreamer will use for tracking"""
    print("\n🧠 Dreamer Performance Tracking Format:")
    print("=" * 60)
    
    # Simulate some performance data
    np.random.seed(42)
    episode_lines = np.random.poisson(0.8, 50)  # Realistic Tetris line clearing
    episode_lines = np.clip(episode_lines, 0, 8)  # Cap at reasonable maximum
    
    # Show last 50 episodes stats
    recent_count = min(50, len(episode_lines))
    recent_lines = episode_lines[-recent_count:]
    
    avg_lines = np.mean(recent_lines)
    max_lines = max(recent_lines)
    
    # Top performances
    sorted_indices = np.argsort(recent_lines)[::-1]
    top_5_lines = [recent_lines[i] for i in sorted_indices[:5]]
    
    print(f"📊 Episode XXX Performance Report (Last {recent_count} episodes):")
    print(f"  🏆 Highest Lines Cleared: {max_lines}")
    print(f"  📈 Average Lines/Episode: {avg_lines:.2f}")
    print(f"  🔥 Top 5 Line Performances: {top_5_lines}")
    print(f"  📋 Updates - World Model: XXX, Policy: XXX")
    
    print(f"\n🎉 Training Completed! Final Performance Summary:")
    print(f"  🏆 Best Lines Cleared (All Time): {max(episode_lines)}")
    print(f"  📊 Average Lines (Last 50): {np.mean(recent_lines):.2f}")
    print(f"  🔥 Best Lines (Last 50): {max(recent_lines)}")
    
    return episode_lines

if __name__ == "__main__":
    print("🚀 Lines Cleared Statistics Demo\n")
    
    # Test actual environment tracking
    lines_data, rewards_data = test_lines_tracking()
    
    # Show Dreamer format
    dreamer_data = show_dreamer_tracking_format()
    
    print(f"\n✨ Demo completed!")
    print(f"This shows how the enhanced Dreamer will track and report")
    print(f"the highest lines cleared in the last 50 episodes during training.") 