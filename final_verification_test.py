#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE VERIFICATION - ALL TRAINERS
"""

import subprocess
import time

def run_comprehensive_test():
    """Run comprehensive test of all trainers with meaningful episode counts"""
    
    print("🎯 FINAL COMPREHENSIVE VERIFICATION")
    print("=" * 80)
    print("Testing all 4 trainers with sufficient episodes to see learning patterns")
    print()
    
    # Test configurations: (trainer, episodes, timeout)
    tests = [
        ("train_dqn_locked.py", 20, 120),
        ("train_dqn_movement.py", 15, 180), 
        ("train_dqn_hierarchical.py", 10, 300),
        ("train_dream.py", 25, 180)
    ]
    
    results = []
    
    for trainer, episodes, timeout in tests:
        print(f"\n🧪 TESTING {trainer} ({episodes} episodes, {timeout}s timeout)")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ["python", trainer, "--episodes", str(episodes)],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {trainer} COMPLETED in {elapsed:.1f}s")
                
                # Parse output for metrics
                output = result.stdout
                lines_cleared = 0
                final_reward = None
                episode_count = 0
                
                # Extract metrics
                for line in output.split('\n'):
                    if "Total lines cleared:" in line:
                        try:
                            lines_cleared = int(line.split(":")[-1].strip())
                        except:
                            pass
                    if "Mean reward:" in line:
                        try:
                            final_reward = float(line.split(":")[-1].strip())
                        except:
                            pass
                    if "Episode" in line and "Reward=" in line:
                        episode_count += 1
                
                results.append({
                    'trainer': trainer,
                    'success': True,
                    'lines': lines_cleared,
                    'episodes': episode_count,
                    'final_reward': final_reward,
                    'time': elapsed
                })
                
                print(f"   Lines cleared: {lines_cleared}")
                print(f"   Episodes completed: {episode_count}")
                if final_reward is not None:
                    print(f"   Final mean reward: {final_reward:.2f}")
                
            else:
                print(f"❌ {trainer} FAILED")
                print(f"   Error: {result.stderr[:200]}...")
                results.append({
                    'trainer': trainer,
                    'success': False,
                    'lines': 0,
                    'episodes': 0,
                    'final_reward': None,
                    'time': elapsed
                })
                
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"⏰ {trainer} TIMEOUT after {elapsed:.1f}s")
            results.append({
                'trainer': trainer,
                'success': False,
                'lines': 0,
                'episodes': 0,
                'final_reward': None,
                'time': elapsed
            })
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"💥 {trainer} EXCEPTION: {e}")
            results.append({
                'trainer': trainer,
                'success': False,
                'lines': 0,
                'episodes': 0,
                'final_reward': None,
                'time': elapsed
            })
    
    # Final summary
    print(f"\n🎯 FINAL VERIFICATION RESULTS")
    print("=" * 80)
    
    successful = 0
    total_lines = 0
    
    for result in results:
        trainer = result['trainer'].replace('.py', '')
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        lines = result['lines']
        episodes = result['episodes']
        reward = result['final_reward']
        time_taken = result['time']
        
        print(f"{trainer:20} {status:8} | Episodes: {episodes:2d} | Lines: {lines:2d} | Time: {time_taken:5.1f}s")
        if reward is not None:
            print(f"{'':31} | Mean Reward: {reward:8.2f}")
        
        if result['success']:
            successful += 1
            total_lines += lines
    
    print(f"\n📊 SUMMARY:")
    print(f"   Trainers working: {successful}/4")
    print(f"   Total lines cleared: {total_lines}")
    print(f"   TensorBoard logs created: {successful} directories")
    
    # Final assessment
    if successful == 4:
        if total_lines > 0:
            print(f"\n🎉 SUCCESS: All trainers work and can clear lines!")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: All trainers work but struggle with line clearing")
            print(f"    This suggests the agents need longer training or better exploration")
    else:
        print(f"\n🚨 FAILURE: Some trainers are not working properly")
        
    return results

if __name__ == "__main__":
    run_comprehensive_test() 