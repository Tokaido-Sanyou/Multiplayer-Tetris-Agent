#!/usr/bin/env python3
"""
Dream Agent Training Script with Correct Dimensions

This script runs the dream agent with proper dimension handling
for the current 206→212 environment setup.
"""

import torch
import numpy as np
import sys
import os

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.tetris_env import TetrisEnv

def pad_observation(obs):
    """Pad 206-dim observation to 212 dimensions for dream agent compatibility"""
    if isinstance(obs, np.ndarray) and obs.shape[0] == 206:
        # Pad with zeros for next piece info (6 dimensions)
        return np.concatenate([obs, np.zeros(6)], axis=0)
    return obs

def main():
    """Main training function"""
    print("=== DREAM AGENT TRAINING ===")
    print("✅ Environment: Tetris with 206→212 dimension padding")
    print("✅ Configuration: Using dream_tetris_clean.py with corrected dimensions")
    
    # Test environment
    env = TetrisEnv(num_agents=1, headless=True)
    obs = env.reset()
    print(f"✅ Environment test - base observation shape: {obs.shape}")
    
    # Test padding
    padded_obs = pad_observation(obs)
    print(f"✅ Padding test - padded observation shape: {padded_obs.shape}")
    
    # Run dream training
    print("\n=== STARTING DREAM TRAINING ===")
    try:
        from dream_tetris_clean import ImprovedDREAMTrainer
        trainer = ImprovedDREAMTrainer(device='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ DREAM trainer initialized")
        
        results = trainer.train(num_episodes=100)
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 