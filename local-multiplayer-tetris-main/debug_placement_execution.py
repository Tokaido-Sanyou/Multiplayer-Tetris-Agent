#!/usr/bin/env python3
"""
Debug Placement Execution
Test our actual placement execution method to see why no line clears
"""

import sys
import os
sys.path.append('localMultiplayerTetris')

import numpy as np
from localMultiplayerTetris.tetris_env import TetrisEnv
from localMultiplayerTetris.rl_utils.enhanced_6phase_state_model import PieceByPieceExplorationManager

def test_placement_execution():
    """Test our actual placement execution method"""
    print("🔍 DEBUG: Testing Placement Execution Method")
    print("=" * 60)
    
    # Initialize environment and exploration manager
    env = TetrisEnv(single_player=True, headless=True)
    exploration_manager = PieceByPieceExplorationManager(env)
    
    # Reset environment
    obs = env.reset()
    print("✅ Environment and exploration manager initialized")
    
    # Manually create a nearly complete row
    if hasattr(env, 'player') and env.player:
        print("\n🔧 Creating nearly complete row (9/10 blocks filled)")
        
        # Fill bottom row except position x=4 (where pieces usually spawn)
        for x in range(10):
            if x != 4:  # Leave gap at x=4
                env.player.locked_positions[(x, 19)] = (128, 128, 128)  # Gray block
        
        print(f"   • Bottom row filled: 9/10 blocks (gap at x=4)")
        print(f"   • Locked positions: {len(env.player.locked_positions)}")
        
        # Test our actual placement execution method
        print("\n🎯 Testing _execute_placement_silent method")
        
        # Get current piece info
        current_piece_type = 0  # I-piece
        placement = (0, 4, 19)  # rotation=0, x=4, y=19 (should fill the gap)
        
        print(f"   • Testing placement: {placement}")
        print(f"   • Piece type: {current_piece_type}")
        
        # Track before state
        score_before = getattr(env.player, 'score', 0)
        locked_positions_before = len(env.player.locked_positions)
        print(f"   • Score before: {score_before}")
        print(f"   • Locked positions before: {locked_positions_before}")
        
        # Execute our placement method
        print("\n🚀 Executing _execute_placement_silent...")
        terminal_reward, lines_cleared, resulting_state, final_positions = exploration_manager._execute_placement_silent(placement)
        
        # Check results
        score_after = getattr(env.player, 'score', 0)
        locked_positions_after = len(env.player.locked_positions)
        
        print(f"\n📊 Results from _execute_placement_silent:")
        print(f"   • Terminal reward: {terminal_reward}")
        print(f"   • Lines cleared: {lines_cleared}")
        print(f"   • Score after: {score_after} (change: +{score_after - score_before})")
        print(f"   • Locked positions after: {locked_positions_after} (change: +{locked_positions_after - locked_positions_before})")
        print(f"   • Final positions count: {len(final_positions)}")
        
        if lines_cleared > 0:
            print("✅ PLACEMENT EXECUTION WORKS!")
            return True
        else:
            print("❌ No lines cleared - investigating action sequence...")
            
            # Test the action sequence generation
            test_action_sequence_generation(exploration_manager, placement)
            return False
    
    return False

def test_action_sequence_generation(exploration_manager, placement):
    """Test our action sequence generation"""
    print("\n🔍 DEBUG: Testing Action Sequence Generation")
    print("=" * 60)
    
    action_sequence = exploration_manager._placement_to_action_sequence(placement)
    
    print(f"Placement: {placement}")
    print(f"Generated action sequence ({len(action_sequence)} actions):")
    
    action_names = ['Left', 'Right', 'Down', 'Rotate', 'Rotate_CCW', 'HardDrop', 'Hold', 'NoOp']
    
    for i, action in enumerate(action_sequence):
        action_idx = np.argmax(action) if sum(action) > 0 else 7
        action_name = action_names[action_idx] if action_idx < len(action_names) else f"Unknown({action_idx})"
        print(f"   Action {i+1}: {action} -> {action_name}")
    
    # Test if this sequence makes sense
    rotation, x_pos, y_pos = placement
    print(f"\n🤔 Action sequence analysis:")
    print(f"   • Target rotation: {rotation} (should have {rotation} rotate actions)")
    print(f"   • Target x: {x_pos} (piece spawns at x=4, need to move {x_pos - 4})")
    print(f"   • Target y: {y_pos} (should end with hard drop)")
    
    # Count actions
    rotate_count = sum(1 for action in action_sequence if np.argmax(action) == 3)
    left_count = sum(1 for action in action_sequence if np.argmax(action) == 0)
    right_count = sum(1 for action in action_sequence if np.argmax(action) == 1)
    hard_drop_count = sum(1 for action in action_sequence if np.argmax(action) == 5)
    
    print(f"   • Rotate actions: {rotate_count} (expected: {rotation})")
    print(f"   • Left actions: {left_count}")
    print(f"   • Right actions: {right_count}")
    print(f"   • Hard drop actions: {hard_drop_count} (expected: 1)")
    
    if hard_drop_count == 0:
        print("❌ NO HARD DROP ACTION - This is likely the problem!")
    elif rotate_count != rotation:
        print(f"❌ Wrong rotation count - expected {rotation}, got {rotate_count}")
    else:
        print("✅ Action sequence looks correct")

def test_simple_hard_drop():
    """Test just a hard drop action on the setup"""
    print("\n🔍 DEBUG: Testing Simple Hard Drop")
    print("=" * 60)
    
    env = TetrisEnv(single_player=True, headless=True)
    obs = env.reset()
    
    # Setup nearly complete row
    if hasattr(env, 'player') and env.player:
        # Fill bottom row except x=4
        for x in range(10):
            if x != 4:
                env.player.locked_positions[(x, 19)] = (128, 128, 128)
        
        print("Set up nearly complete row (gap at x=4)")
        
        # Check current piece position
        if hasattr(env.player, 'current_piece'):
            print(f"Current piece: type={env.player.current_piece.shape}, pos=({env.player.current_piece.x}, {env.player.current_piece.y})")
            
            # Move piece to x=4 if needed
            while env.player.current_piece.x > 4:
                env.step([1, 0, 0, 0, 0, 0, 0, 0])  # Left
            while env.player.current_piece.x < 4:
                env.step([0, 1, 0, 0, 0, 0, 0, 0])  # Right
            
            print(f"Moved piece to x={env.player.current_piece.x}")
            
            # Now hard drop
            obs, reward, done, info = env.step([0, 0, 0, 0, 0, 1, 0, 0])  # Hard drop
            
            print(f"Hard drop result:")
            print(f"   • Reward: {reward}")
            print(f"   • Lines cleared: {info.get('lines_cleared', 0)}")
            print(f"   • Done: {done}")
            
            if info.get('lines_cleared', 0) > 0:
                print("✅ SIMPLE HARD DROP WORKS!")
                return True
    
    print("❌ Simple hard drop failed")
    return False

if __name__ == "__main__":
    print("🚀 Placement Execution Debug Script")
    print("=" * 60)
    
    # Test our placement execution method
    success1 = test_placement_execution()
    
    # Test simple hard drop 
    success2 = test_simple_hard_drop()
    
    print("\n🎯 SUMMARY")
    print("=" * 60)
    if success1:
        print("✅ Our placement execution method works!")
    elif success2:
        print("⚠️ Simple hard drop works, but our action sequence is wrong")
    else:
        print("❌ Both methods failed - need to investigate further")
        print("   Possible issues:")
        print("   1. Piece spawning position incorrect")
        print("   2. Action sequence generation is wrong")
        print("   3. Piece not falling to intended position")
        print("   4. Environment state not being restored correctly") 