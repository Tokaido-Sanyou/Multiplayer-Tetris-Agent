#!/usr/bin/env python3
"""
🎯 FOCUSED LINE CLEARING TEST

Test line clearing with optimized strategy until we see results
"""

import numpy as np
from envs.tetris_env import TetrisEnv

def test_line_clearing_focused():
    """Test with heavy hard drop strategy"""
    print("🎯 FOCUSED LINE CLEARING TEST")
    print("=" * 50)
    
    env = TetrisEnv(
        num_agents=1,
        headless=True,
        action_mode='direct',
        reward_mode='lines_only'
    )
    
    total_lines = 0
    
    for episode in range(50):  # Try up to 50 episodes
        obs = env.reset()
        episode_lines = 0
        
        # Strategy: 90% hard drops, 10% other actions for variety
        for step in range(100):
            if np.random.random() < 0.9:
                action = 5  # Hard drop
            else:
                action = np.random.choice([0, 1, 2, 3, 4, 7])  # Other actions
            
            obs, reward, done, info = env.step(action)
            
            # Check for line clearing
            if 'lines_cleared' in info and info['lines_cleared'] > 0:
                lines = info['lines_cleared']
                episode_lines += lines
                total_lines += lines
                print(f"🎉 Episode {episode}, Step {step}: Cleared {lines} lines!")
                print(f"   Reward: {reward}, Total lines this episode: {episode_lines}")
                
                # Success - we can stop here
                if total_lines > 0:
                    print(f"\n✅ SUCCESS! Total lines cleared: {total_lines}")
                    env.close()
                    return True
            
            if done:
                break
        
        if episode % 10 == 0:
            print(f"Episode {episode}: {episode_lines} lines")
    
    env.close()
    print(f"\n❌ No lines cleared after 50 episodes")
    return False

if __name__ == "__main__":
    test_line_clearing_focused() 