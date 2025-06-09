#!/usr/bin/env python3
"""
Comprehensive training test for all agents
"""

import subprocess
import time
import os
from pathlib import Path

def test_trainer(trainer_script, episodes=10, timeout=120):
    """Test a trainer script with timeout"""
    print(f"\n🧪 TESTING {trainer_script}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run trainer with timeout
        result = subprocess.run(
            ["python", trainer_script, "--episodes", str(episodes)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {trainer_script} PASSED ({elapsed:.1f}s)")
            
            # Extract key metrics from output
            output = result.stdout
            lines = output.split('\n')
            
            # Find performance metrics
            episodes_with_lines = 0
            total_lines = 0
            final_reward = None
            
            for line in lines:
                if "Lines=" in line and "Lines=0" not in line:
                    episodes_with_lines += 1
                if "Total lines cleared:" in line:
                    try:
                        total_lines = int(line.split(":")[-1].strip())
                    except:
                        pass
                if "Mean reward:" in line:
                    try:
                        final_reward = float(line.split(":")[-1].strip())
                    except:
                        pass
            
            print(f"   📊 Total lines cleared: {total_lines}")
            print(f"   📊 Episodes with lines: {episodes_with_lines}")
            if final_reward is not None:
                print(f"   📊 Final mean reward: {final_reward:.2f}")
            
            return True, total_lines, episodes_with_lines, final_reward
            
        else:
            print(f"❌ {trainer_script} FAILED")
            print(f"   Error: {result.stderr}")
            return False, 0, 0, None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {trainer_script} TIMEOUT after {timeout}s")
        return False, 0, 0, None
    except Exception as e:
        print(f"💥 {trainer_script} EXCEPTION: {e}")
        return False, 0, 0, None

def check_tensorboard_logs():
    """Check if tensorboard logs are being created"""
    print(f"\n📊 CHECKING TENSORBOARD LOGS")
    print("=" * 60)
    
    expected_logs = [
        "logs/dqn_locked_standard/tensorboard",
        "logs/dqn_movement_standard/tensorboard", 
        "logs/dqn_hierarchical_standard/tensorboard",
        "logs/dream_fixed_complete/tensorboard"
    ]
    
    for log_path in expected_logs:
        if Path(log_path).exists():
            files = list(Path(log_path).glob("*.tfevents.*"))
            if files:
                print(f"✅ {log_path}: {len(files)} log files")
            else:
                print(f"⚠️  {log_path}: Directory exists but no log files")
        else:
            print(f"❌ {log_path}: Directory missing")

def main():
    print("🚀 COMPREHENSIVE TRAINING TEST")
    print("=" * 80)
    
    # Test all trainers
    trainers = [
        ("train_dqn_locked.py", 15),
        ("train_dqn_movement.py", 10), 
        ("train_dqn_hierarchical.py", 8),
        ("train_dream.py", 10)
    ]
    
    results = {}
    
    for trainer, episodes in trainers:
        success, lines, episodes_with_lines, reward = test_trainer(trainer, episodes, timeout=180)
        results[trainer] = {
            'success': success,
            'lines': lines, 
            'episodes_with_lines': episodes_with_lines,
            'reward': reward
        }
        time.sleep(2)  # Brief pause between tests
    
    # Check tensorboard logs
    check_tensorboard_logs()
    
    # Summary
    print(f"\n📈 FINAL SUMMARY")
    print("=" * 80)
    
    for trainer, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{trainer:25} {status:8} | Lines: {result['lines']:2} | Episodes w/ Lines: {result['episodes_with_lines']:2}")
        if result['reward'] is not None:
            print(f"{'':36} | Mean Reward: {result['reward']:8.2f}")
    
    # Overall assessment
    passing_trainers = sum(1 for r in results.values() if r['success'])
    total_lines = sum(r['lines'] for r in results.values() if r['success'])
    
    print(f"\n🎯 OVERALL RESULTS:")
    print(f"   Trainers Working: {passing_trainers}/4")
    print(f"   Total Lines Cleared: {total_lines}")
    
    if passing_trainers == 4 and total_lines > 0:
        print("🎉 ALL SYSTEMS WORKING WITH LINE CLEARING!")
    elif passing_trainers == 4:
        print("⚠️  ALL TRAINERS WORK BUT NO LINE CLEARING - STRATEGY ISSUE")
    else:
        print("🚨 SOME TRAINERS FAILING - IMPLEMENTATION ISSUES")

if __name__ == "__main__":
    main() 