#!/usr/bin/env python3
"""
Debug Complete Exploration Pipeline
Trace through the actual exploration pipeline to see why no line clears
"""

import sys
import os
sys.path.append('localMultiplayerTetris')

import numpy as np
from localMultiplayerTetris.tetris_env import TetrisEnv
from localMultiplayerTetris.rl_utils.enhanced_6phase_state_model import PieceByPieceExplorationManager

def debug_complete_exploration_pipeline():
    """Debug the complete exploration pipeline step by step"""
    print("🔍 DEBUG: Complete Exploration Pipeline")
    print("=" * 60)
    
    # Initialize environment and exploration manager
    env = TetrisEnv(single_player=True, headless=True)
    exploration_manager = PieceByPieceExplorationManager(env)
    
    # Set to minimal for debugging
    exploration_manager.max_pieces = 2
    exploration_manager.boards_to_keep = 2
    
    print("✅ Environment and exploration manager initialized")
    print(f"   • Max pieces: {exploration_manager.max_pieces}")
    print(f"   • Boards to keep: {exploration_manager.boards_to_keep}")
    
    # Test initial board state creation
    print("\n🔧 Testing initial board state creation...")
    for i in range(5):
        initial_board = exploration_manager._create_strategic_initial_board_state()
        print(f"   Initial board {i+1}: {initial_board['board_id']}")
        print(f"      • Locked positions: {len(initial_board['locked_positions'])}")
        
        # Check if this board has near-complete rows
        locked_positions = initial_board['locked_positions']
        row_counts = {}
        for (x, y), color in locked_positions.items():
            if y not in row_counts:
                row_counts[y] = 0
            row_counts[y] += 1
        
        near_complete_rows = [y for y, count in row_counts.items() if count >= 7]
        print(f"      • Near-complete rows (7+ blocks): {near_complete_rows}")
    
    # Test single piece exploration on a manually created good board
    print("\n🎯 Testing single piece exploration on strategic board...")
    
    # Create a board with a nearly complete row
    strategic_board = {
        'board_id': 'debug_strategic',
        'locked_positions': {},
        'lines_cleared': 0,
        'generation': 0,
        'parent_reward': 0,
        'trajectory_lineage': []
    }
    
    # Fill bottom row with 8 blocks (gaps at x=4 and x=5)
    for x in range(10):
        if x not in [4, 5]:  # Leave gaps at x=4 and x=5
            strategic_board['locked_positions'][(x, 19)] = (128, 128, 128)
    
    print(f"   Created strategic board with {len(strategic_board['locked_positions'])} blocks")
    print(f"   Row 19 has 8/10 blocks (gaps at x=4,5)")
    
    # Test exploration on this board
    piece_type = 0  # I-piece
    print(f"\n🚀 Testing exploration with piece type {piece_type}...")
    
    exploration_data = exploration_manager._explore_board_iteratively_optimized(
        strategic_board, piece_type, 0
    )
    
    print(f"   Exploration results: {len(exploration_data)} placements")
    
    # Check for line clears in results
    line_clear_placements = [d for d in exploration_data if d.get('lines_cleared', 0) > 0]
    print(f"   Line clearing placements: {len(line_clear_placements)}")
    
    if line_clear_placements:
        for i, placement_data in enumerate(line_clear_placements[:3]):  # Show first 3
            print(f"      Placement {i+1}:")
            print(f"         • Placement: {placement_data.get('placement')}")
            print(f"         • Lines cleared: {placement_data.get('lines_cleared')}")
            print(f"         • Terminal reward: {placement_data.get('terminal_reward')}")
    else:
        print("   ❌ No line clearing placements found!")
        
        # Debug: check a few placements
        print("\n   🔍 Debugging first few placements:")
        for i, placement_data in enumerate(exploration_data[:5]):
            print(f"      Placement {i+1}:")
            print(f"         • Placement: {placement_data.get('placement')}")
            print(f"         • Lines cleared: {placement_data.get('lines_cleared')}")
            print(f"         • Terminal reward: {placement_data.get('terminal_reward')}")
    
    return len(line_clear_placements) > 0

def debug_board_state_restoration():
    """Debug board state restoration process"""
    print("\n🔍 DEBUG: Board State Restoration")
    print("=" * 60)
    
    env = TetrisEnv(single_player=True, headless=True)
    exploration_manager = PieceByPieceExplorationManager(env)
    
    # Create a test board state
    test_board = {
        'board_id': 'test_board',
        'locked_positions': {(0, 19): (255, 0, 0), (1, 19): (255, 0, 0)},
        'lines_cleared': 0,
        'generation': 0,
        'parent_reward': 0,
        'trajectory_lineage': []
    }
    
    print(f"Test board has {len(test_board['locked_positions'])} locked positions")
    
    # Test restoration
    exploration_manager._restore_board_state_silent(test_board)
    
    # Check if restoration worked
    if hasattr(env, 'player') and env.player:
        restored_count = len(env.player.locked_positions)
        print(f"After restoration: {restored_count} locked positions")
        print(f"Restoration success: {restored_count == len(test_board['locked_positions'])}")
        
        # Check specific positions
        for pos, color in test_board['locked_positions'].items():
            if pos in env.player.locked_positions:
                print(f"   Position {pos}: ✅ Restored")
            else:
                print(f"   Position {pos}: ❌ Missing")
    
    return True

def debug_valid_locked_positions():
    """Debug valid locked position generation"""
    print("\n🔍 DEBUG: Valid Locked Position Generation")
    print("=" * 60)
    
    env = TetrisEnv(single_player=True, headless=True)
    exploration_manager = PieceByPieceExplorationManager(env)
    
    # Test with different piece types
    for piece_type in range(3):  # Test first 3 piece types
        print(f"\n   Testing piece type {piece_type}:")
        
        valid_positions = exploration_manager._generate_valid_locked_positions(piece_type)
        print(f"      • Valid positions found: {len(valid_positions)}")
        
        # Check position variety
        rotations = set(pos[0] for pos in valid_positions)
        x_positions = set(pos[1] for pos in valid_positions)
        y_positions = set(pos[2] for pos in valid_positions)
        
        print(f"      • Rotations: {sorted(rotations)}")
        print(f"      • X range: {min(x_positions) if x_positions else 'N/A'}-{max(x_positions) if x_positions else 'N/A'}")
        print(f"      • Y range: {min(y_positions) if y_positions else 'N/A'}-{max(y_positions) if y_positions else 'N/A'}")

if __name__ == "__main__":
    print("🚀 Complete Exploration Pipeline Debug")
    print("=" * 60)
    
    # Debug board state restoration
    debug_board_state_restoration()
    
    # Debug valid position generation  
    debug_valid_locked_positions()
    
    # Debug complete exploration pipeline
    success = debug_complete_exploration_pipeline()
    
    print("\n🎯 PIPELINE DEBUG SUMMARY")
    print("=" * 60)
    if success:
        print("✅ Line clearing works in exploration pipeline!")
    else:
        print("❌ Line clearing still not working in pipeline")
        print("   Possible remaining issues:")
        print("   1. Board state restoration not working properly")
        print("   2. Valid position generation avoiding line-clearing positions")
        print("   3. Initial board states not being used correctly")
        print("   4. Piece types not matching expected pieces") 