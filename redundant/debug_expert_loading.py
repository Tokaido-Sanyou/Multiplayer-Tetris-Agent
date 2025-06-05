#!/usr/bin/env python3
"""
Debug script to test expert trajectory loading
"""

import sys
import os

# Add paths for imports
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris/rl_utils')

import numpy as np
import pickle

def dummy_feature_extractor(obs):
    """Dummy feature extractor for testing."""
    return np.random.randn(207).astype(np.float32)

def inspect_trajectory_file(filepath):
    """Inspect what's in a trajectory file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"📄 File: {os.path.basename(filepath)}")
        print(f"   Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            for key, value in data.items():
                if hasattr(value, '__len__') and key != 'steps':  # Don't print all steps
                    print(f"   {key}: length {len(value)}, type {type(value)}")
                elif key == 'steps':
                    print(f"   {key}: length {len(value)}, type {type(value)}")
                    if len(value) > 0:
                        print(f"      First step: {type(value[0])}")
                        if isinstance(value[0], dict):
                            print(f"      First step keys: {list(value[0].keys())}")
                        print(f"      Sample steps (first 3):")
                        for i, step in enumerate(value[:3]):
                            print(f"        Step {i}: {step}")
                else:
                    print(f"   {key}: {type(value)}")
        elif isinstance(data, (list, tuple)):
            print(f"   Length: {len(data)}")
            if len(data) > 0:
                print(f"   First element type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"   First element keys: {list(data[0].keys())}")
        else:
            print(f"   Data: {data}")
            
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")

def main():
    # First, check trajectory files directly
    print("🔍 Inspecting expert trajectory files...")
    traj_dir = 'expert_trajectories'
    
    if os.path.exists(traj_dir):
        files = [f for f in os.listdir(traj_dir) if f.endswith('.pkl')]
        print(f"📁 Found {len(files)} .pkl files")
        
        # Inspect first few files
        for filepath in sorted(files)[:3]:
            inspect_trajectory_file(os.path.join(traj_dir, filepath))
            print()
    else:
        print(f"❌ Directory {traj_dir} does not exist")
        return
    
    try:
        from expert_loader import ExpertTrajectoryLoader
        print("✅ Expert loader imported successfully")
        
        # Test with expert trajectories directory
        loader = ExpertTrajectoryLoader(
            trajectory_dir='expert_trajectories',
            state_feature_extractor=dummy_feature_extractor,
            max_trajectories=2,  # Load only 2 for testing
            max_hold_percentage=50.0  # Relaxed filtering
        )
        
        print("📁 Loading expert trajectories...")
        num_loaded = loader.load_trajectories()
        print(f"📊 Loaded {num_loaded} trajectories")
        
        if num_loaded > 0:
            stats = loader.get_statistics()
            print(f"📈 Statistics: {stats}")
            
            # Test batch sampling
            try:
                batch = loader.get_batch(batch_size=5)
                print(f"✅ Batch sampling successful")
                print(f"   States shape: {batch['states'].shape}")
                print(f"   Actions shape: {batch['actions'].shape}")
            except Exception as e:
                print(f"❌ Batch sampling failed: {e}")
        else:
            print("🔍 Debugging why no trajectories were loaded...")
            # Let's check what's happening in the loader
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 