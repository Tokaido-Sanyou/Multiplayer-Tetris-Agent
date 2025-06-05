#!/usr/bin/env python3
"""
Analyze action distribution in expert trajectories
"""

import pickle
import os
import numpy as np
from collections import Counter

def analyze_expert_actions():
    """Analyze action distribution across all expert trajectories"""
    expert_dir = "expert_trajectories"
    
    print("🎯 Analyzing Expert Action Distribution")
    print("=" * 50)
    
    if not os.path.exists(expert_dir):
        print(f"❌ Directory not found: {expert_dir}")
        return
    
    trajectory_files = sorted([f for f in os.listdir(expert_dir) if f.endswith('.pkl')])
    
    if not trajectory_files:
        print(f"❌ No trajectory files found in {expert_dir}")
        return
    
    all_actions = []
    all_rewards = []
    episode_summaries = []
    
    for filename in trajectory_files:
        filepath = os.path.join(expert_dir, filename)
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            steps = data.get('steps', [])
            if not steps:
                continue
                
            episode_actions = [step.get('action', -1) for step in steps]
            episode_rewards = [step.get('reward', 0) for step in steps]
            
            # Count actions in this episode
            action_counts = Counter(episode_actions)
            total_reward = sum(episode_rewards)
            
            # Count how many times HOLD (action 40) was used
            hold_count = action_counts.get(40, 0)
            hold_percentage = (hold_count / len(episode_actions)) * 100 if episode_actions else 0
            
            episode_summaries.append({
                'filename': filename,
                'total_steps': len(episode_actions),
                'total_reward': total_reward,
                'hold_count': hold_count,
                'hold_percentage': hold_percentage,
                'unique_actions': len(action_counts),
                'action_counts': action_counts
            })
            
            all_actions.extend(episode_actions)
            all_rewards.extend(episode_rewards)
            
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
    
    # Overall statistics
    print(f"📊 Overall Statistics:")
    print(f"  Total episodes: {len(episode_summaries)}")
    print(f"  Total steps: {len(all_actions)}")
    print(f"  Total reward sum: {sum(all_rewards):.3f}")
    print(f"  Mean reward per step: {np.mean(all_rewards):.6f}")
    
    # Action distribution
    overall_action_counts = Counter(all_actions)
    print(f"\n🎯 Action Distribution (Overall):")
    for action in sorted(overall_action_counts.keys()):
        count = overall_action_counts[action]
        percentage = (count / len(all_actions)) * 100
        action_name = "HOLD" if action == 40 else f"PLACE"
        print(f"  Action {action:2d} ({action_name}): {count:6d} times ({percentage:5.1f}%)")
    
    # Episode-by-episode analysis
    print(f"\n📋 Episode-by-Episode Analysis:")
    for ep in episode_summaries[:5]:  # Show first 5 episodes
        print(f"  {ep['filename']}:")
        print(f"    Steps: {ep['total_steps']}, Reward: {ep['total_reward']:.3f}")
        print(f"    HOLD usage: {ep['hold_count']}/{ep['total_steps']} ({ep['hold_percentage']:.1f}%)")
        print(f"    Unique actions: {ep['unique_actions']}")
        
        # Show top 5 most used actions
        top_actions = ep['action_counts'].most_common(5)
        print(f"    Top actions: {top_actions}")
    
    # Identify problem episodes
    problem_episodes = [ep for ep in episode_summaries if ep['hold_percentage'] > 90]
    print(f"\n⚠️  Problem Episodes (>90% HOLD actions): {len(problem_episodes)}")
    for ep in problem_episodes[:3]:
        print(f"  {ep['filename']}: {ep['hold_percentage']:.1f}% HOLD, reward: {ep['total_reward']:.3f}")

if __name__ == "__main__":
    analyze_expert_actions() 