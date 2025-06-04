"""
Simplified test to isolate the hanging issue
"""
import sys
import os

def test_basic_imports():
    """Test basic imports one by one"""
    print("Testing basic imports...")
    
    try:
        print("1. Testing numpy...")
        import numpy as np
        print("   ✅ numpy imported successfully")
        
        print("2. Testing torch...")
        import torch
        print("   ✅ torch imported successfully")
        
        print("3. Testing torch.nn...")
        import torch.nn as nn
        print("   ✅ torch.nn imported successfully")
        
        print("4. Testing device detection...")
        device = torch.device("cpu")  # Force CPU to avoid CUDA issues
        print(f"   ✅ device set to: {device}")
        
        print("5. Testing simple tensor operations...")
        x = torch.randn(5, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)
        print(f"   ✅ tensor operations work, result shape: {z.shape}")
        
        print("6. Testing simple neural network...")
        net = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        print("   ✅ simple neural network created")
        
        print("7. Testing TetrisEnv import...")
        from localMultiplayerTetris.tetris_env import TetrisEnv
        print("   ✅ TetrisEnv imported successfully")
        
        print("8. Testing ReplayBuffer import...")
        from localMultiplayerTetris.rl_utils.replay_buffer import ReplayBuffer
        print("   ✅ ReplayBuffer imported successfully")
        
        print("All basic imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error during import test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_imports()
    if success:
        print("✅ All basic tests passed!")
    else:
        print("❌ Some tests failed!")
    sys.exit(0 if success else 1)
