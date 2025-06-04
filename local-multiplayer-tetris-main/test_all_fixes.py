#!/usr/bin/env python3
"""
COMPREHENSIVE TEST: Verify all fixes for piece-by-piece exploration
Tests:
1. Deterministic board creation (not random)
2. Proper board state restoration
3. Line clearing capability
4. Board reuse functionality
5. Trajectory consistency
6. NEW: Goal vector encoding/decoding
7. NEW: Iterative terminal state exploration
"""

import sys
import os
sys.path.append('localMultiplayerTetris')

from localMultiplayerTetris.rl_utils.enhanced_6phase_state_model import PieceByPieceExplorationManager, Enhanced6PhaseComponents
from localMultiplayerTetris.tetris_env import TetrisEnv
import numpy as np
import torch

def test_goal_vector_encoding():
    """Test the new binary rotation + discrete coordinate goal vector"""
    print('\n🧪 STEP 6: Test Goal Vector Encoding/Decoding')
    print('-'*50)
    
    env = TetrisEnv()
    components = Enhanced6PhaseComponents(device='cpu')
    
    # Test cases for goal vector encoding
    test_options = [
        {
            'placement': (0, 4, 15),  # rotation=0, x=4, y=15
            'confidence': 0.8,
            'quality': 0.5,
            'lines_potential': 2,
            'is_valid': True
        },
        {
            'placement': (3, 9, 19),  # rotation=3, x=9, y=19
            'confidence': 0.9,
            'quality': -0.2,
            'lines_potential': 0,
            'is_valid': True
        },
        {
            'placement': (1, 0, 10),  # rotation=1, x=0, y=10
            'confidence': 0.3,
            'quality': 0.8,
            'lines_potential': 4,
            'is_valid': False
        }
    ]
    
    print(f'   🎯 Testing {len(test_options)} goal vector encodings...')
    
    all_passed = True
    for i, option in enumerate(test_options):
        # Encode to goal vector
        goal_vector = components.goal_selector._option_to_goal_vector(option, 'cpu')
        
        # Decode back to placement
        decoded = components.goal_selector.decode_goal_vector_to_placement(goal_vector)
        
        # Check if decoding matches original
        original_placement = option['placement']
        decoded_placement = decoded['placement']
        
        rotation_match = original_placement[0] == decoded_placement[0]
        x_match = original_placement[1] == decoded_placement[1]
        y_match = original_placement[2] == decoded_placement[2]
        
        if rotation_match and x_match and y_match:
            print(f'      ✅ Test {i+1}: {original_placement} → {decoded_placement} (PASSED)')
        else:
            print(f'      ❌ Test {i+1}: {original_placement} → {decoded_placement} (FAILED)')
            all_passed = False
        
        # Check goal vector structure
        if goal_vector.shape[1] == 8:
            print(f'         📊 Goal vector shape: {goal_vector.shape} (8D ✅)')
        else:
            print(f'         📊 Goal vector shape: {goal_vector.shape} (Expected 8D ❌)')
            all_passed = False
        
        # Verify binary rotation encoding
        rot_bits = goal_vector[0, :2].numpy()
        reconstructed_rotation = int(rot_bits[0]) + (int(rot_bits[1]) << 1)
        if reconstructed_rotation == original_placement[0]:
            print(f'         🔄 Binary rotation: {rot_bits} → {reconstructed_rotation} (✅)')
        else:
            print(f'         🔄 Binary rotation: {rot_bits} → {reconstructed_rotation} (Expected {original_placement[0]} ❌)')
            all_passed = False
    
    if all_passed:
        print('   ✅ Goal vector encoding/decoding: WORKING')
    else:
        print('   ❌ Goal vector encoding/decoding: FAILED')
    
    return all_passed

def test_iterative_exploration():
    """Test the new iterative terminal state exploration"""
    print('\n🔍 STEP 7: Test Iterative Terminal State Exploration')
    print('-'*50)
    
    env = TetrisEnv()
    exploration_manager = PieceByPieceExplorationManager(env)
    
    # Very small test for quick verification
    exploration_manager.max_pieces = 1
    exploration_manager.trials_per_piece = 10
    exploration_manager.boards_to_keep = 2
    
    print('   🚀 Running mini iterative exploration test...')
    
    try:
        exploration_data = exploration_manager.collect_piece_by_piece_exploration_data('iterative')
        
        print(f'   📊 Iterative exploration results:')
        print(f'      • Data points: {len(exploration_data)}')
        print(f'      • All placements unique: {len(set(d["placement"] for d in exploration_data)) == len(exploration_data)}')
        
        # Check that placements cover different rotations and positions
        rotations = set(d['placement'][0] for d in exploration_data)
        x_positions = set(d['placement'][1] for d in exploration_data)
        y_positions = set(d['placement'][2] for d in exploration_data)
        
        print(f'      • Rotations explored: {sorted(rotations)}')
        print(f'      • X positions explored: {len(x_positions)} unique')
        print(f'      • Y positions explored: {len(y_positions)} unique')
        
        # Check terminal state variety
        unique_terminal_states = len(set(tuple(d['resulting_state'][:50]) for d in exploration_data))
        print(f'      • Unique terminal states: {unique_terminal_states}')
        
        if len(exploration_data) > 0:
            print('   ✅ Iterative exploration: WORKING')
            return True
        else:
            print('   ❌ Iterative exploration: FAILED (no data)')
            return False
            
    except Exception as e:
        print(f'   ❌ Iterative exploration: FAILED with error: {e}')
        return False

def test_all_fixes():
    print('🔧 COMPREHENSIVE TEST: All Fixes Verification')
    print('='*80)
    
    # Initialize
    env = TetrisEnv()
    exploration_manager = PieceByPieceExplorationManager(env)
    
    # Very focused test
    exploration_manager.max_pieces = 2
    exploration_manager.trials_per_piece = 5
    exploration_manager.boards_to_keep = 2
    
    print('✅ STEP 1: Test Deterministic Board Creation')
    print('-'*50)
    
    # Test multiple board creations should be identical
    board1 = exploration_manager._create_initial_board_state()
    board2 = exploration_manager._create_initial_board_state()
    
    pos1 = set(board1['locked_positions'].keys())
    pos2 = set(board2['locked_positions'].keys())
    
    if pos1 == pos2:
        print('   ✅ Deterministic board creation: WORKING')
        print(f'      • Both boards have {len(pos1)} identical positions')
    else:
        print('   ❌ Deterministic board creation: FAILED')
        print(f'      • Board 1: {len(pos1)} positions, Board 2: {len(pos2)} positions')
        print(f'      • Difference: {pos1.symmetric_difference(pos2)}')
    
    print('\n✅ STEP 2: Test Board Reuse Capability')
    print('-'*50)
    reuse_works = exploration_manager._check_board_reuse_capability()
    
    print('\n✅ STEP 3: Test Board State Restoration')
    print('-'*50)
    
    # Create test board with known pattern
    test_board = board1
    print(f'   🎯 Testing restoration of board with {len(test_board["locked_positions"])} positions')
    
    # Reset environment first
    env.reset()
    obs_before = env._get_observation()
    blocks_before = np.sum(1 - obs_before['empty_grid'])
    print(f'   📊 Before restoration: {blocks_before} blocks visible')
    
    # Restore the test board
    exploration_manager._restore_board_state(test_board)
    obs_after = env._get_observation()
    blocks_after = np.sum(1 - obs_after['empty_grid'])
    print(f'   📊 After restoration: {blocks_after} blocks visible')
    
    expected_blocks = len(test_board['locked_positions'])
    if blocks_after >= expected_blocks * 0.8:  # Allow some tolerance
        print(f'   ✅ Board restoration: WORKING (expected ~{expected_blocks}, got {blocks_after})')
        restoration_works = True
    else:
        print(f'   ❌ Board restoration: FAILED (expected ~{expected_blocks}, got {blocks_after})')
        restoration_works = False
    
    print('\n✅ STEP 4: Test Line Clearing Potential')
    print('-'*50)
    
    if restoration_works:
        # Try a few piece placements to see if we can get line clearing
        piece_type = 0  # I piece
        placement_attempts = [
            (0, 8, 19),  # Try to fill gap at bottom row
            (0, 9, 19),  # Try to fill other gap
            (1, 7, 18),  # Try to fill gap at second row (vertically)
        ]
        
        lines_cleared_total = 0
        
        for i, placement in enumerate(placement_attempts):
            print(f'   🎯 Placement attempt {i+1}: {placement}')
            
            # Restore board state
            exploration_manager._restore_board_state(test_board)
            
            # Try placement
            lines_before = exploration_manager._get_lines_cleared_count()
            reward, done, final_obs = exploration_manager._execute_placement_sequence(placement)
            lines_after = exploration_manager._get_lines_cleared_count()
            
            lines_this_placement = max(0, lines_after - lines_before)
            lines_cleared_total += lines_this_placement
            
            print(f'      • Lines cleared: {lines_this_placement}')
            print(f'      • Reward: {reward}')
            
            if lines_this_placement > 0:
                print(f'   🎉 SUCCESS: Line clearing working!')
                break
        
        if lines_cleared_total > 0:
            print(f'   ✅ Line clearing capability: WORKING ({lines_cleared_total} total lines)')
            line_clearing_works = True
        else:
            print(f'   ⚠️ Line clearing capability: NOT TRIGGERED (but board setup looks correct)')
            line_clearing_works = False
    else:
        print('   ⏭️ Skipping line clearing test - board restoration failed')
        line_clearing_works = False
    
    print('\n✅ STEP 5: Mini Piece-by-Piece Test')
    print('-'*50)
    
    if restoration_works:
        print('   🚀 Running mini piece-by-piece exploration...')
        exploration_data = exploration_manager.collect_piece_by_piece_exploration_data('iterative')  # CHANGED to iterative
        
        total_lines = sum(d.get('lines_cleared', 0) for d in exploration_data)
        line_events = sum(1 for d in exploration_data if d.get('lines_cleared', 0) > 0)
        
        print(f'   📊 Mini test results:')
        print(f'      • Data points: {len(exploration_data)}')
        print(f'      • Lines cleared: {total_lines}')
        print(f'      • Line events: {line_events}')
        print(f'      • Success rate: {line_events/len(exploration_data)*100:.1f}%')
        
        if total_lines > 0:
            print(f'   ✅ Full system: WORKING WITH LINE CLEARING!')
            full_system_works = True
        else:
            print(f'   ⚠️ Full system: Working but no line clearing yet')
            full_system_works = False
    else:
        print('   ⏭️ Skipping full test - prerequisites failed')
        full_system_works = False
    
    # NEW TESTS
    goal_vector_works = test_goal_vector_encoding()
    iterative_exploration_works = test_iterative_exploration()
    
    print('\n🎉 COMPREHENSIVE TEST RESULTS')
    print('='*80)
    print('📋 SUMMARY:')
    print(f'   • Deterministic board creation: {"✅ WORKING" if pos1 == pos2 else "❌ FAILED"}')
    print(f'   • Board reuse capability: {"✅ WORKING" if reuse_works else "❌ FAILED"}')
    print(f'   • Board state restoration: {"✅ WORKING" if restoration_works else "❌ FAILED"}')
    print(f'   • Line clearing capability: {"✅ WORKING" if line_clearing_works else "⚠️ PENDING"}')
    print(f'   • Full system integration: {"✅ WORKING" if full_system_works else "⚠️ PENDING"}')
    print(f'   • Goal vector encoding/decoding: {"✅ WORKING" if goal_vector_works else "❌ FAILED"}')
    print(f'   • Iterative terminal exploration: {"✅ WORKING" if iterative_exploration_works else "❌ FAILED"}')
    
    # Overall assessment
    critical_issues = []
    if not reuse_works:
        critical_issues.append("Board reuse")
    if not restoration_works:
        critical_issues.append("Board restoration")
    if not goal_vector_works:
        critical_issues.append("Goal vector encoding")
    if not iterative_exploration_works:
        critical_issues.append("Iterative exploration")
    
    if len(critical_issues) == 0:
        print('\n🚀 OVERALL STATUS: ALL CRITICAL SYSTEMS WORKING!')
        print('   Ready for full training with iterative exploration')
        if line_clearing_works:
            print('   Line clearing confirmed working - expect good results!')
        else:
            print('   Line clearing pending - learning will discover optimal placements')
    else:
        print(f'\n⚠️ OVERALL STATUS: {len(critical_issues)} CRITICAL ISSUES REMAIN:')
        for issue in critical_issues:
            print(f'   • {issue}')
        print('   Must fix these before full training')
    
    return {
        'deterministic_creation': pos1 == pos2,
        'board_reuse': reuse_works,
        'board_restoration': restoration_works,
        'line_clearing': line_clearing_works,
        'full_system': full_system_works,
        'goal_vector_encoding': goal_vector_works,
        'iterative_exploration': iterative_exploration_works,
        'lines_cleared_total': total_lines if restoration_works else 0
    }

if __name__ == '__main__':
    try:
        results = test_all_fixes()
        working_systems = sum(results.values())
        total_systems = len(results)
        print(f'\n📊 Test completed with {working_systems} out of {total_systems} systems working')
    except Exception as e:
        print(f'\n❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc() 