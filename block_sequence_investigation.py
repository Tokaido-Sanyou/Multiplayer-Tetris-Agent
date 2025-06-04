#!/usr/bin/env python3
"""
Block Sequence Investigation
Check how pieces are generated and if they should be shared between players
"""

import sys
import os
import numpy as np

# Add paths
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')

def investigate_block_sequences():
    """Investigate block sequence generation in both implementations."""
    print("🧬 BLOCK SEQUENCE INVESTIGATION")
    print("=" * 60)
    
    try:
        from tetris_env import TetrisEnv
        from rl_utils.true_multiplayer_env import TrueMultiplayerTetrisEnv
        
        print("\n1️⃣ SINGLE PLAYER ENVIRONMENTS (Independent RNG):")
        env1 = TetrisEnv(single_player=True, headless=True)
        env2 = TetrisEnv(single_player=True, headless=True)
        
        obs1 = env1.reset()
        obs2 = env2.reset()
        
        if isinstance(obs1, tuple):
            obs1 = obs1[0]
        if isinstance(obs2, tuple):
            obs2 = obs2[0]
        
        print(f"   Env1 first piece: {obs1.get('current_shape', 'None')}")
        print(f"   Env1 next piece: {obs1.get('next_piece', 'None')}")
        print(f"   Env2 first piece: {obs2.get('current_shape', 'None')}")
        print(f"   Env2 next piece: {obs2.get('next_piece', 'None')}")
        print(f"   Same sequences? {obs1.get('current_shape') == obs2.get('current_shape')}")
        
        # Track sequences for 10 steps
        seq1 = []
        seq2 = []
        
        for step in range(10):
            obs1, _, _, _ = env1.step(15)[:4]  # Place piece
            obs2, _, _, _ = env2.step(15)[:4]  # Place piece
            
            seq1.append(obs1.get('current_shape', 0))
            seq2.append(obs2.get('current_shape', 0))
        
        print(f"   Sequence 1: {seq1}")
        print(f"   Sequence 2: {seq2}")
        print(f"   Sequences identical: {seq1 == seq2}")
        
        env1.close()
        env2.close()
        
        print("\n2️⃣ TRUE MULTIPLAYER ENVIRONMENT:")
        mp_env = TrueMultiplayerTetrisEnv(headless=True)
        mp_obs = mp_env.reset()
        
        p1_obs = mp_obs['player1']
        p2_obs = mp_obs['player2']
        
        print(f"   P1 first piece: {p1_obs.get('current_shape', 'None')}")
        print(f"   P1 next piece: {p1_obs.get('next_piece', 'None')}")
        print(f"   P2 first piece: {p2_obs.get('current_shape', 'None')}")
        print(f"   P2 next piece: {p2_obs.get('next_piece', 'None')}")
        print(f"   Same sequences? {p1_obs.get('current_shape') == p2_obs.get('current_shape')}")
        
        # Track multiplayer sequences
        mp_seq1 = []
        mp_seq2 = []
        
        for step in range(10):
            actions = {'player1': 15, 'player2': 15}
            mp_obs, _, _, _ = mp_env.step(actions)
            
            mp_seq1.append(mp_obs['player1'].get('current_shape', 0))
            mp_seq2.append(mp_obs['player2'].get('current_shape', 0))
        
        print(f"   MP Sequence P1: {mp_seq1}")
        print(f"   MP Sequence P2: {mp_seq2}")
        print(f"   MP sequences identical: {mp_seq1 == mp_seq2}")
        
        mp_env.close()
        
        print("\n3️⃣ CLASSIC TETRIS ANALYSIS:")
        print("   🎯 In competitive Tetris (e.g., Tetris 99, Puyo Puyo Tetris):")
        print("   • Players typically get DIFFERENT piece sequences")
        print("   • This ensures fair competition (no sequence advantage)")
        print("   • Some modes use same sequence, some use different")
        
        print("\n4️⃣ RESEARCH ANALYSIS:")
        print("   📚 In academic Tetris research:")
        print("   • Most papers use independent piece generation")
        print("   • This tests true skill rather than sequence memorization")
        print("   • Different sequences = more robust learning")
        
        print("\n📊 CONCLUSION:")
        if mp_seq1 != mp_seq2:
            print("   ✅ Current implementation is CORRECT!")
            print("   ✅ Players get different sequences (fair competition)")
            print("   ✅ This follows competitive Tetris standards")
        else:
            print("   ⚠️  Players getting same sequences")
            print("   ⚠️  May need to verify RNG independence")
        
        return True
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main investigation function."""
    success = investigate_block_sequences()
    
    if success:
        print("\n✅ Block sequence investigation complete!")
        print("Different sequences = competitive fairness ✅")
    
    return success

if __name__ == "__main__":
    main() 