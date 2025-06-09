#!/usr/bin/env python3
"""
Compare Original vs Enhanced DREAM Exploration
"""

import subprocess
import time

def run_comparison_test():
    """Compare original vs enhanced DREAM training"""
    
    print("🔍 DREAM EXPLORATION COMPARISON")
    print("=" * 60)
    print("Testing Original vs Enhanced DREAM models")
    print("Focus: Episode lengths and exploration behavior")
    print()
    
    # Test configurations
    tests = [
        {
            'name': 'Original DREAM',
            'script': 'train_dream.py',
            'args': ['--episodes', '20', '--reward_mode', 'lines_only'],
            'expected': 'Short episodes (9-14 steps), early convergence'
        },
        {
            'name': 'Enhanced DREAM',
            'script': 'train_dream_enhanced_exploration.py', 
            'args': ['--episodes', '20', '--reward_mode', 'lines_only'],
            'expected': 'Longer episodes, sustained exploration'
        }
    ]
    
    results = []
    
    for test in tests:
        print(f"\n🧪 TESTING {test['name']}")
        print(f"   Expected: {test['expected']}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ['python', test['script']] + test['args'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                # Parse output for metrics
                output = result.stdout
                
                # Extract episode lengths
                episode_lengths = []
                lines_cleared = 0
                
                for line in output.split('\n'):
                    if 'Length=' in line:
                        try:
                            length_part = line.split('Length=')[1].split(',')[0].strip()
                            episode_lengths.append(int(length_part))
                        except:
                            pass
                    if 'Total lines cleared:' in line:
                        try:
                            lines_cleared = int(line.split(':')[-1].strip())
                        except:
                            pass
                
                avg_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
                max_length = max(episode_lengths) if episode_lengths else 0
                min_length = min(episode_lengths) if episode_lengths else 0
                
                print(f"✅ {test['name']} COMPLETED")
                print(f"   Average episode length: {avg_length:.1f} steps")
                print(f"   Episode length range: {min_length}-{max_length} steps")
                print(f"   Total lines cleared: {lines_cleared}")
                print(f"   Training time: {elapsed:.1f}s")
                
                results.append({
                    'name': test['name'],
                    'success': True,
                    'avg_length': avg_length,
                    'max_length': max_length,
                    'min_length': min_length,
                    'lines_cleared': lines_cleared,
                    'time': elapsed
                })
                
            else:
                print(f"❌ {test['name']} FAILED")
                print(f"   Error: {result.stderr[:200]}...")
                results.append({
                    'name': test['name'],
                    'success': False,
                    'avg_length': 0,
                    'max_length': 0,
                    'min_length': 0,
                    'lines_cleared': 0,
                    'time': elapsed
                })
                
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"⏰ {test['name']} TIMEOUT after {elapsed:.1f}s")
            results.append({
                'name': test['name'],
                'success': False,
                'avg_length': 0,
                'max_length': 0,
                'min_length': 0,
                'lines_cleared': 0,
                'time': elapsed
            })
    
    # Comparison summary
    print(f"\n📊 EXPLORATION COMPARISON RESULTS")
    print("=" * 60)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{result['name']:15} {status} | Avg: {result['avg_length']:5.1f} | Max: {result['max_length']:3d} | Lines: {result['lines_cleared']:2d}")
    
    # Analysis
    if len(results) == 2 and all(r['success'] for r in results):
        original = results[0]
        enhanced = results[1]
        
        length_improvement = enhanced['avg_length'] - original['avg_length']
        exploration_improvement = enhanced['max_length'] - original['max_length']
        
        print(f"\n🎯 ANALYSIS:")
        print(f"   Average length improvement: +{length_improvement:.1f} steps")
        print(f"   Max exploration improvement: +{exploration_improvement} steps")
        
        if length_improvement > 20:
            print(f"   🎉 SUCCESS: Enhanced exploration significantly improves episode length!")
        elif length_improvement > 5:
            print(f"   ✅ GOOD: Enhanced exploration shows improvement")
        else:
            print(f"   ⚠️  MARGINAL: Exploration enhancement needs more tuning")
    
    return results

if __name__ == "__main__":
    run_comparison_test() 