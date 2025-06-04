#!/usr/bin/env python3
"""
Analyze New Expert Trajectories
"""

import pickle
import os

def analyze_trajectory(filepath):
    """Analyze a single trajectory file."""
    print(f"\n📄 Analyzing: {os.path.basename(filepath)}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"   Type: {type(data)}")
        print(f"   Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        if isinstance(data, dict):
            if 'steps' in data:
                steps = data['steps']
                print(f"   Steps: {len(steps)} total")
                
                if len(steps) > 0:
                    # Count HOLD actions
                    hold_count = sum(1 for step in steps if step.get('action') == 40)
                    hold_percentage = (hold_count / len(steps)) * 100.0
                    print(f"   HOLD actions: {hold_count}/{len(steps)} ({hold_percentage:.1f}%)")
                    
                    # Sample first few actions
                    print(f"   First 10 actions: {[step.get('action') for step in steps[:10]]}")
                    
            if 'total_reward' in data:
                print(f"   Total reward: {data['total_reward']}")
            
            if 'length' in data:
                print(f"   Episode length: {data['length']}")
                
            if 'hold_percentage' in data:
                print(f"   Recorded HOLD%: {data['hold_percentage']:.1f}%")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    trajectory_dir = "expert_trajectories_new"
    
    if not os.path.exists(trajectory_dir):
        print(f"❌ Directory not found: {trajectory_dir}")
        return
    
    print("🔍 Analyzing New Expert Trajectories")
    print("=" * 50)
    
    # List all trajectory files
    files = [f for f in os.listdir(trajectory_dir) if f.endswith('.pkl')]
    print(f"Found {len(files)} trajectory files")
    
    # Analyze first few files
    for filename in sorted(files)[:3]:
        filepath = os.path.join(trajectory_dir, filename)
        analyze_trajectory(filepath)
    
    print(f"\n📊 Summary:")
    print(f"   Total files: {len(files)}")
    print(f"   Directory: {trajectory_dir}")

if __name__ == "__main__":
    main() 