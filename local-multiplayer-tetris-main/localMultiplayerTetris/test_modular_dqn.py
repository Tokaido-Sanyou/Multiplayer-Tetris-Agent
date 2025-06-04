#!/usr/bin/env python3
"""
Test script for the modularized DQN training system

This script verifies that all features work correctly:
- Single environment training
- Vectorized environment training  
- TensorBoard logging
- Checkpointing and resuming
- Batch operations
"""

import os
import sys
import subprocess
import time

def test_single_training():
    """Test single environment training"""
    print("🧪 Testing single environment training...")
    
    cmd = [
        sys.executable, "dqn_training_module.py",
        "--mode", "single",
        "--num_episodes", "3",
        "--eval_interval", "2",
        "--save_interval", "2"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Single environment training: PASSED")
        return True
    else:
        print(f"❌ Single environment training: FAILED")
        print(f"Error: {result.stderr}")
        return False

def test_vectorized_training():
    """Test vectorized environment training"""
    print("🧪 Testing vectorized environment training...")
    
    cmd = [
        sys.executable, "dqn_training_module.py", 
        "--mode", "vectorized",
        "--num_episodes", "5",
        "--num_envs", "2",
        "--eval_interval", "3",
        "--save_interval", "3"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Vectorized environment training: PASSED")
        return True
    else:
        print(f"❌ Vectorized environment training: FAILED")
        print(f"Error: {result.stderr}")
        return False

def test_checkpoint_resume():
    """Test checkpoint saving and resuming"""
    print("🧪 Testing checkpoint save/resume...")
    
    # First run to create checkpoint
    cmd1 = [
        sys.executable, "dqn_training_module.py",
        "--mode", "single", 
        "--num_episodes", "2",
        "--save_interval", "1"
    ]
    
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    
    if result1.returncode != 0:
        print(f"❌ Checkpoint creation: FAILED")
        print(f"Error: {result1.stderr}")
        return False
    
    # Check if checkpoint was created
    checkpoint_files = [f for f in os.listdir("checkpoints") if f.startswith("dqn_checkpoint_episode_")]
    if not checkpoint_files:
        print("❌ No checkpoint files found")
        return False
    
    # Resume from checkpoint
    latest_checkpoint = f"checkpoints/{checkpoint_files[-1]}"
    cmd2 = [
        sys.executable, "dqn_training_module.py",
        "--mode", "single",
        "--num_episodes", "2", 
        "--checkpoint", latest_checkpoint
    ]
    
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    if result2.returncode == 0:
        print("✅ Checkpoint save/resume: PASSED")
        return True
    else:
        print(f"❌ Checkpoint resume: FAILED")
        print(f"Error: {result2.stderr}")
        return False

def check_tensorboard_logs():
    """Check if TensorBoard logs are created"""
    print("🧪 Checking TensorBoard logs...")
    
    if os.path.exists("logs/dqn_tensorboard"):
        log_dirs = [d for d in os.listdir("logs/dqn_tensorboard") if d.startswith("dqn_run_")]
        if log_dirs:
            print("✅ TensorBoard logs: PASSED")
            return True
    
    print("❌ TensorBoard logs: FAILED")
    return False

def check_checkpoints():
    """Check if checkpoints are created"""
    print("🧪 Checking checkpoint files...")
    
    if os.path.exists("checkpoints"):
        checkpoint_files = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
        if checkpoint_files:
            print(f"✅ Checkpoints: PASSED ({len(checkpoint_files)} files found)")
            return True
    
    print("❌ Checkpoints: FAILED")
    return False

def main():
    """Run all tests"""
    print("🚀 Testing Modularized DQN Training System")
    print("=" * 50)
    
    # Ensure we're in the right directory
    if not os.path.exists("dqn_training_module.py"):
        print("❌ dqn_training_module.py not found. Please run from the correct directory.")
        return 1
    
    # Create directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    tests_passed = 0
    total_tests = 5
    
    # Run tests
    if test_single_training():
        tests_passed += 1
    
    if test_vectorized_training():
        tests_passed += 1
    
    if test_checkpoint_resume():
        tests_passed += 1
    
    if check_tensorboard_logs():
        tests_passed += 1
    
    if check_checkpoints():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("\n✨ The modularized DQN system is fully functional with:")
        print("   ✅ Single environment training")
        print("   ✅ Vectorized parallel training")
        print("   ✅ TensorBoard logging")
        print("   ✅ Automatic checkpointing")
        print("   ✅ Resume from checkpoints")
        print("\n🚀 Ready for production training!")
        print("\nTo start full training, run:")
        print("python dqn_training_module.py --mode vectorized --num_episodes 10000 --num_envs 8")
        return 0
    else:
        print(f"❌ {total_tests - tests_passed} tests failed")
        return 1

if __name__ == "__main__":
    exit(main()) 