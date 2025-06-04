#!/usr/bin/env python3
"""
Debug trajectory file format
"""

import pickle
import os
import numpy as np

def debug_trajectory_file(filepath):
    """Debug a single trajectory file"""
    print(f"🔍 Debugging: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"  Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")
            for key, value in data.items():
                if key == 'steps':
                    print(f"    {key}: length={len(value)}, type={type(value)}")
                    if isinstance(value, list) and len(value) > 0:
                        step_sample = value[0]
                        print(f"      First step type: {type(step_sample)}")
                        if isinstance(step_sample, dict):
                            print(f"      First step keys: {list(step_sample.keys())}")
                            for step_key, step_value in step_sample.items():
                                if hasattr(step_value, 'shape'):
                                    print(f"        {step_key}: shape={step_value.shape}, dtype={step_value.dtype}")
                                else:
                                    print(f"        {step_key}: {type(step_value)} = {step_value}")
                elif hasattr(value, 'shape'):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (list, tuple)):
                    print(f"    {key}: length={len(value)}, type={type(value[0]) if value else 'empty'}")
                else:
                    print(f"    {key}: {type(value)} = {value}")
        else:
            print(f"  Data: {data}")
            
    except Exception as e:
        print(f"  Error: {e}")

def main():
    """Debug all trajectory files"""
    expert_dir = "expert_trajectories"
    
    print("🔍 Debugging Expert Trajectory Files")
    print("=" * 50)
    
    if not os.path.exists(expert_dir):
        print(f"❌ Directory not found: {expert_dir}")
        return
    
    trajectory_files = sorted([f for f in os.listdir(expert_dir) if f.endswith('.pkl')])
    
    if not trajectory_files:
        print(f"❌ No trajectory files found in {expert_dir}")
        return
    
    print(f"Found {len(trajectory_files)} trajectory files")
    
    # Debug first file in detail
    filepath = os.path.join(expert_dir, trajectory_files[0])
    debug_trajectory_file(filepath)

if __name__ == "__main__":
    main() 