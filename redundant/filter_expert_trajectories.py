#!/usr/bin/env python3
"""
Filter expert trajectories to remove those with excessive HOLD actions
"""

import pickle
import os
import shutil
from collections import Counter

def filter_expert_trajectories(max_hold_percentage=10.0):
    """
    Filter expert trajectories to keep only those with reasonable HOLD usage
    
    Args:
        max_hold_percentage: Maximum percentage of HOLD actions allowed
    """
    expert_dir = "expert_trajectories"
    filtered_dir = "expert_trajectories_filtered"
    backup_dir = "expert_trajectories_backup"
    
    print(f"🔍 Filtering Expert Trajectories (max HOLD: {max_hold_percentage}%)")
    print("=" * 60)
    
    # Create directories
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    
    trajectory_files = sorted([f for f in os.listdir(expert_dir) if f.endswith('.pkl')])
    
    good_trajectories = []
    bad_trajectories = []
    
    for filename in trajectory_files:
        filepath = os.path.join(expert_dir, filename)
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            steps = data.get('steps', [])
            if not steps:
                continue
                
            episode_actions = [step.get('action', -1) for step in steps]
            action_counts = Counter(episode_actions)
            
            hold_count = action_counts.get(40, 0)
            hold_percentage = (hold_count / len(episode_actions)) * 100 if episode_actions else 0
            
            total_reward = sum(step.get('reward', 0) for step in steps)
            
            if hold_percentage <= max_hold_percentage:
                # Good trajectory
                good_filepath = os.path.join(filtered_dir, filename)
                shutil.copy2(filepath, good_filepath)
                good_trajectories.append({
                    'filename': filename,
                    'hold_percentage': hold_percentage,
                    'total_reward': total_reward,
                    'steps': len(steps)
                })
                print(f"✅ KEPT: {filename} ({hold_percentage:.1f}% HOLD, reward: {total_reward:.1f})")
            else:
                # Bad trajectory
                backup_filepath = os.path.join(backup_dir, filename)
                shutil.copy2(filepath, backup_filepath)
                bad_trajectories.append({
                    'filename': filename,
                    'hold_percentage': hold_percentage,
                    'total_reward': total_reward,
                    'steps': len(steps)
                })
                print(f"❌ REMOVED: {filename} ({hold_percentage:.1f}% HOLD, reward: {total_reward:.1f})")
                
        except Exception as e:
            print(f"⚠️  Error processing {filename}: {e}")
    
    print(f"\n📊 Filtering Results:")
    print(f"  Good trajectories: {len(good_trajectories)}")
    print(f"  Bad trajectories: {len(bad_trajectories)}")
    print(f"  Filtered directory: {filtered_dir}")
    print(f"  Backup directory: {backup_dir}")
    
    if good_trajectories:
        print(f"\n✅ Good Trajectories Summary:")
        for traj in good_trajectories:
            print(f"  {traj['filename']}: {traj['hold_percentage']:.1f}% HOLD, {traj['total_reward']:.1f} reward")
    else:
        print(f"\n❌ No good trajectories found! All trajectories have >{max_hold_percentage}% HOLD actions.")
        print(f"   You need to retrain your expert or use a different expert policy.")

def main():
    # First try very strict filtering
    filter_expert_trajectories(max_hold_percentage=5.0)
    
    # If no good trajectories, try more lenient
    filtered_dir = "expert_trajectories_filtered"
    if not os.listdir(filtered_dir):
        print(f"\n🔄 No trajectories passed strict filter, trying lenient filter...")
        filter_expert_trajectories(max_hold_percentage=20.0)
    
    # If still no good trajectories, try very lenient
    if not os.listdir(filtered_dir):
        print(f"\n🔄 No trajectories passed lenient filter, trying very lenient filter...")
        filter_expert_trajectories(max_hold_percentage=50.0)

if __name__ == "__main__":
    main() 