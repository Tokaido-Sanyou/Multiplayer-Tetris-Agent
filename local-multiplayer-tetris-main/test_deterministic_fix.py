#!/usr/bin/env python3
"""
Test script to validate the enhanced deterministic exploration
Now uses partially filled boards with line clearing potential instead of empty boards
"""

from localMultiplayerTetris.rl_utils.rnd_exploration import DeterministicTerminalExplorer
from localMultiplayerTetris.tetris_env import TetrisEnv
import numpy as np

def test_enhanced_deterministic():
    """Test the enhanced deterministic exploration with line clearing scenarios"""
    print('🧪 Testing Enhanced Deterministic Exploration with Line Clearing Scenarios')
    print('=' * 70)
    
    # Create test environment
    env = TetrisEnv()
    explorer = DeterministicTerminalExplorer(env)
    
    # Test with smaller sequence to see results quickly
    print('🎯 Generating terminal states with sequence length 2...')
    terminal_data = explorer.generate_all_terminal_states(sequence_length=2)
    
    if not terminal_data:
        print('❌ No terminal data generated')
        return
    
    # Analyze terminal values
    terminal_values = [d['terminal_value'] for d in terminal_data]
    
    print(f'\n📊 ENHANCED DETERMINISTIC TERMINAL VALUE ANALYSIS:')
    print(f'   • Total terminals: {len(terminal_values)}')
    print(f'   • Average terminal value: {np.mean(terminal_values):.2f} (vs RND: 109.72)')
    print(f'   • Std dev: {np.std(terminal_values):.2f}')
    print(f'   • Min value: {np.min(terminal_values):.2f}')
    print(f'   • Max value: {np.max(terminal_values):.2f}')
    print(f'   • Median value: {np.median(terminal_values):.2f}')
    
    # Count high-value terminals
    high_values = [v for v in terminal_values if v > 50]
    very_high_values = [v for v in terminal_values if v > 100]
    line_clear_values = [v for v in terminal_values if v > 150]
    
    print(f'   • Values > 50: {len(high_values)}/{len(terminal_values)} ({len(high_values)/len(terminal_values)*100:.1f}%)')
    print(f'   • Values > 100: {len(very_high_values)}/{len(terminal_values)} ({len(very_high_values)/len(terminal_values)*100:.1f}%)')
    print(f'   • Values > 150 (likely line clears): {len(line_clear_values)}/{len(terminal_values)} ({len(line_clear_values)/len(terminal_values)*100:.1f}%)')
    
    # Show scenario distribution
    scenario_counts = {}
    scenario_values = {}
    for d in terminal_data:
        sid = d.get('scenario_id', 'unknown')
        scenario_counts[sid] = scenario_counts.get(sid, 0) + 1
        if sid not in scenario_values:
            scenario_values[sid] = []
        scenario_values[sid].append(d['terminal_value'])
    
    print(f'\n📋 BY SCENARIO ANALYSIS:')
    scenario_names = ['Near-complete lines', 'Tetris well', 'Multiple partial', 'T-spin setup', 'Random mid-game']
    for sid, count in scenario_counts.items():
        if isinstance(sid, int) and sid < len(scenario_names):
            name = scenario_names[sid]
        else:
            name = f'Scenario {sid}'
        avg_val = np.mean(scenario_values[sid])
        max_val = np.max(scenario_values[sid])
        print(f'   • {name}: {count} terminals, avg={avg_val:.2f}, max={max_val:.2f}')
    
    # Show top 10 terminal values
    sorted_values = sorted(terminal_values, reverse=True)[:10]
    print(f'\n🏆 TOP 10 TERMINAL VALUES:')
    for i, val in enumerate(sorted_values, 1):
        print(f'   {i:2d}. {val:.2f}')
    
    # Comparison with previous results
    old_deterministic_avg = 9.64
    rnd_avg = 109.72
    new_avg = np.mean(terminal_values)
    
    print(f'\n🔍 COMPARISON ANALYSIS:')
    print(f'   • Old deterministic avg: {old_deterministic_avg:.2f}')
    print(f'   • NEW deterministic avg: {new_avg:.2f}')
    print(f'   • RND exploration avg: {rnd_avg:.2f}')
    print(f'   • Improvement factor: {new_avg/old_deterministic_avg:.1f}x')
    print(f'   • vs RND ratio: {new_avg/rnd_avg:.2f} ({new_avg/rnd_avg*100:.1f}%)')
    
    if new_avg > old_deterministic_avg * 5:
        print('   ✅ MAJOR IMPROVEMENT: 5x+ increase in terminal values!')
    elif new_avg > old_deterministic_avg * 2:
        print('   ✅ SIGNIFICANT IMPROVEMENT: 2x+ increase in terminal values!')
    elif new_avg > old_deterministic_avg:
        print('   ✅ IMPROVEMENT: Terminal values increased!')
    else:
        print('   ❌ NO IMPROVEMENT: Terminal values did not increase significantly')
    
    if new_avg > rnd_avg * 0.8:
        print('   🎯 EXCELLENT: Deterministic now reaches 80%+ of RND performance!')
    elif new_avg > rnd_avg * 0.5:
        print('   🎯 GOOD: Deterministic now reaches 50%+ of RND performance!')
    elif new_avg > rnd_avg * 0.2:
        print('   🎯 MODERATE: Deterministic now reaches 20%+ of RND performance!')
    else:
        print('   ⚠️  Still significantly below RND performance')

if __name__ == '__main__':
    test_enhanced_deterministic() 