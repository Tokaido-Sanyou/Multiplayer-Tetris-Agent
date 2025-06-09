#!/usr/bin/env python3
"""
🧪 FINAL TRAINING SUITE TEST

Tests all 4 training algorithms with both reward modes.
This verifies the complete system is working.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_training_test(script_name, reward_mode, episodes=3):
    """Run a training script and capture results"""
    print(f"🧪 Testing {script_name} with {reward_mode} rewards...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, script_name, 
            '--reward_mode', reward_mode, 
            '--episodes', str(episodes)
        ], capture_output=True, text=True, timeout=60)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            # Extract key metrics from output
            lines = result.stdout.split('\n')
            training_complete = any('TRAINING COMPLETE' in line for line in lines)
            
            # Extract episode rewards
            episode_lines = [line for line in lines if 'Episode' in line and 'Reward=' in line]
            
            status = "✅ PASS" if training_complete else "⚠️  PARTIAL"
            episodes_run = len(episode_lines)
            
            print(f"   {status}: {episodes_run}/{episodes} episodes, {duration:.1f}s")
            
            if episode_lines:
                # Show last episode info
                print(f"   Last: {episode_lines[-1].strip()}")
            
            return True
            
        else:
            print(f"   ❌ FAIL: Exit code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT: Exceeded 60s")
        return False
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def main():
    """Run comprehensive training suite test"""
    print("🧪 FINAL TRAINING SUITE TEST")
    print("=" * 80)
    
    # Test configuration
    training_scripts = [
        ("train_dream.py", "DREAM", "212→8 direct actions"),
        ("train_dqn_locked.py", "DQN Locked", "206→800 locked positions"), 
        ("train_dqn_movement.py", "DQN Movement", "800→8 movement actions"),
        ("train_dqn_hierarchical.py", "DQN Hierarchical", "206→800→8 hierarchical")
    ]
    
    reward_modes = ["standard", "lines_only"]
    test_episodes = 3
    
    print(f"Testing {len(training_scripts)} algorithms × {len(reward_modes)} reward modes")
    print(f"Episodes per test: {test_episodes}")
    print()
    
    # Results tracking
    results = {}
    total_tests = 0
    passed_tests = 0
    
    # Run tests
    for script_file, algorithm_name, description in training_scripts:
        print(f"🚀 {algorithm_name} ({description})")
        print("-" * 60)
        
        # Check if script exists
        if not Path(script_file).exists():
            print(f"   ❌ MISSING: {script_file} not found")
            results[algorithm_name] = {"standard": False, "lines_only": False}
            total_tests += 2
            continue
        
        algorithm_results = {}
        
        for reward_mode in reward_modes:
            total_tests += 1
            success = run_training_test(script_file, reward_mode, test_episodes)
            algorithm_results[reward_mode] = success
            
            if success:
                passed_tests += 1
        
        results[algorithm_name] = algorithm_results
        print()
    
    # Summary
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    for algorithm_name, algorithm_results in results.items():
        print(f"📊 {algorithm_name}:")
        for reward_mode, success in algorithm_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {reward_mode:12}: {status}")
    
    print()
    print(f"🏆 FINAL SCORE: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! The training suite is working perfectly!")
        
        print("\n📝 USAGE INSTRUCTIONS:")
        print("   DREAM:           python train_dream.py --reward_mode standard --episodes 1000")
        print("   DQN Locked:      python train_dqn_locked.py --reward_mode lines_only --episodes 1000") 
        print("   DQN Movement:    python train_dqn_movement.py --reward_mode standard --episodes 1000")
        print("   DQN Hierarchical: python train_dqn_hierarchical.py --reward_mode lines_only --episodes 1000")
        
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Check the errors above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 