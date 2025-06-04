#!/usr/bin/env python3
"""
Debug Line Clearing Detection
Test if our line clearing tracking is working with a simple scenario
"""

import sys
import os
sys.path.append('localMultiplayerTetris')

import numpy as np
from localMultiplayerTetris.tetris_env import TetrisEnv

def test_basic_line_clearing():
    """Test basic line clearing with manual setup"""
    print("🔍 DEBUG: Testing Basic Line Clearing Detection")
    print("=" * 60)
    
    # Initialize environment
    env = TetrisEnv(single_player=True, headless=True)
    obs = env.reset()
    
    print("✅ Environment initialized")
    print(f"   • Player exists: {hasattr(env, 'player') and env.player is not None}")
    print(f"   • Initial score: {getattr(env.player, 'score', 'N/A') if hasattr(env, 'player') else 'N/A'}")
    
    # Manually create a nearly complete row (9 out of 10 blocks filled)
    if hasattr(env, 'player') and env.player:
        print("\n🔧 Creating nearly complete row (9/10 blocks filled)")
        
        # Fill bottom row except position x=5
        for x in range(10):
            if x != 5:  # Leave gap at x=5
                env.player.locked_positions[(x, 19)] = (128, 128, 128)  # Gray block
        
        print(f"   • Bottom row filled: 9/10 blocks (gap at x=5)")
        print(f"   • Locked positions: {len(env.player.locked_positions)}")
        
        # Test a simple hard drop action
        print("\n🎯 Testing hard drop action (should complete the line)")
        
        # Track before state
        score_before = getattr(env.player, 'score', 0)
        lines_before = getattr(env.player, 'lines_cleared', 0)
        print(f"   • Score before: {score_before}")
        print(f"   • Lines before: {lines_before}")
        
        # Execute hard drop (action 5)
        print("\n🚀 Executing hard drop...")
        action = [0, 0, 0, 0, 0, 1, 0, 0]  # Hard drop
        obs, reward, done, info = env.step(action)
        
        # Check results
        score_after = getattr(env.player, 'score', 0)
        lines_after = getattr(env.player, 'lines_cleared', 0)
        lines_from_info = info.get('lines_cleared', 0)
        
        print(f"\n📊 Results:")
        print(f"   • Score after: {score_after} (change: +{score_after - score_before})")
        print(f"   • Lines after: {lines_after} (change: +{lines_after - lines_before})")
        print(f"   • Lines from info: {lines_from_info}")
        print(f"   • Reward: {reward}")
        print(f"   • Done: {done}")
        print(f"   • Full info: {info}")
        print(f"   • Locked positions after: {len(env.player.locked_positions)}")
        
        if lines_from_info > 0:
            print("✅ LINE CLEARING WORKS!")
            return True
        else:
            print("❌ No lines cleared - investigating...")
            
            # Check current piece position
            if hasattr(env.player, 'current_piece'):
                print(f"   • Current piece type: {env.player.current_piece.shape}")
                print(f"   • Current piece position: ({env.player.current_piece.x}, {env.player.current_piece.y})")
                print(f"   • Current piece rotation: {env.player.current_piece.rotation}")
            
            # Check if piece can actually complete the line
            return False
    
    return False

def test_environment_step_info():
    """Test what info is returned by env.step()"""
    print("\n🔍 DEBUG: Testing Environment Step Info")
    print("=" * 60)
    
    env = TetrisEnv(single_player=True, headless=True)
    obs = env.reset()
    
    print("Testing several step actions to see info structure...")
    
    for i in range(5):
        action = [0, 0, 0, 0, 0, 0, 0, 1]  # No-op
        obs, reward, done, info = env.step(action)
        
        print(f"Step {i+1}: reward={reward}, done={done}")
        print(f"   Info keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")
        print(f"   Info: {info}")
        print()
        
        if done:
            break

def test_lines_cleared_tracking():
    """Test different ways to track lines cleared"""
    print("\n🔍 DEBUG: Testing Lines Cleared Tracking Methods")
    print("=" * 60)
    
    env = TetrisEnv(single_player=True, headless=True)
    obs = env.reset()
    
    print("Checking available line tracking attributes:")
    
    # Check environment
    print(f"   • env.lines_cleared: {getattr(env, 'lines_cleared', 'Not found')}")
    
    # Check player
    if hasattr(env, 'player') and env.player:
        print(f"   • env.player.lines_cleared: {getattr(env.player, 'lines_cleared', 'Not found')}")
        print(f"   • env.player.score: {getattr(env.player, 'score', 'Not found')}")
        print(f"   • env.player attributes: {[attr for attr in dir(env.player) if not attr.startswith('_')]}")
    
    # Check game
    if hasattr(env, 'game') and env.game:
        print(f"   • env.game.lines_cleared: {getattr(env.game, 'lines_cleared', 'Not found')}")
        print(f"   • env.game attributes: {[attr for attr in dir(env.game) if not attr.startswith('_')]}")

if __name__ == "__main__":
    print("🚀 Line Clearing Debug Script")
    print("=" * 60)
    
    # Test basic environment info
    test_environment_step_info()
    
    # Test line tracking methods
    test_lines_cleared_tracking()
    
    # Test actual line clearing
    success = test_basic_line_clearing()
    
    print("\n🎯 SUMMARY")
    print("=" * 60)
    if success:
        print("✅ Line clearing detection is working!")
    else:
        print("❌ Line clearing detection has issues")
        print("   Possible causes:")
        print("   1. Action sequence not placing piece correctly")
        print("   2. Piece not in right position to complete line")
        print("   3. Environment not detecting line completion")
        print("   4. Info dict not being populated correctly") 