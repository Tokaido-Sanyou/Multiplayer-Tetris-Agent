#!/usr/bin/env python3
"""
Test script for the enhanced Sequential DeterministicTerminalExplorer
"""

from localMultiplayerTetris.rl_utils.rnd_exploration import DeterministicTerminalExplorer
from localMultiplayerTetris.tetris_env import TetrisEnv

def test_sequential_deterministic_explorer():
    print("🧪 Testing Sequential Deterministic Terminal Explorer")
    print("="*60)
    
    # Create environment
    env = TetrisEnv(single_player=True, headless=True)
    explorer = DeterministicTerminalExplorer(env)
    
    # Test with short sequence (FIXED: reduced from 3 to 2 for faster testing)
    print(f"📊 Generating sequential terminal states for 2 pieces...")
    terminals = explorer.generate_all_terminal_states(2)  # Very short sequence for testing
    
    print(f"\n✅ Sequential Chain Results:")
    print(f"   • Generated: {len(terminals)} terminal states (Variable batch size)")
    
    if len(terminals) == 0:
        print(f"   ⚠️  No terminal states generated - debugging validation...")
        env.close()
        return False
    
    print(f"   • All validated: {all(t.get('validated', False) for t in terminals)}")
    
    # Check chain depth distribution
    chain_depths = [t.get('chain_depth', 0) for t in terminals]
    depth_distribution = {}
    for depth in chain_depths:
        depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
    print(f"   • Chain depth distribution: {dict(sorted(depth_distribution.items()))}")
    
    # Check chain vs final terminals
    chain_terminals = [t for t in terminals if t.get('is_chain_terminal', False)]
    intermediate_states = [t for t in terminals if not t.get('is_chain_terminal', False)]
    print(f"   • Chain terminals: {len(chain_terminals)}")
    print(f"   • Intermediate states: {len(intermediate_states)}")
    
    # Check piece sequence in placement histories
    placement_histories = [t.get('placement_history', []) for t in terminals if t.get('placement_history')]
    if placement_histories:
        max_history_length = max(len(h) for h in placement_histories)
        avg_history_length = sum(len(h) for h in placement_histories) / len(placement_histories)
        print(f"   • Max placement history length: {max_history_length}")
        print(f"   • Average placement history length: {avg_history_length:.1f}")
        
        # Show example placement history
        if placement_histories:
            example_history = placement_histories[0][:2]  # First 2 placements (reduced from 3)
            print(f"   • Example placement sequence: {[f'{h[3]}({h[0]},{h[1]})' for h in example_history]}")
    
    # Check piece type distribution
    piece_types = [t.get('target_piece_type', 0) for t in terminals]
    piece_distribution = {}
    for piece_type in piece_types:
        piece_distribution[piece_type] = piece_distribution.get(piece_type, 0) + 1
    print(f"   • Piece type distribution: {dict(sorted(piece_distribution.items()))}")
    
    # Check placement coordinates range - FIXED: Handle empty lists
    placements = [t.get('placement', (0, 0, 0)) for t in terminals]
    
    if placements:
        rotations = [p[0] for p in placements]
        x_positions = [p[1] for p in placements]
        y_positions = [p[2] for p in placements]
        
        print(f"   • Rotation range: {min(rotations)} - {max(rotations)}")
        print(f"   • X position range: {min(x_positions)} - {max(x_positions)}")
        print(f"   • Y position range: {min(y_positions)} - {max(y_positions)}")
        
        # VALIDATION: Check if any coordinates are out of bounds
        invalid_rotations = [r for r in rotations if r < 0 or r > 3]
        invalid_x = [x for x in x_positions if x < 0 or x > 9]
        invalid_y = [y for y in y_positions if y < 0 or y > 19]
        
        if invalid_rotations:
            print(f"   ⚠️  WARNING: {len(invalid_rotations)} invalid rotations found: {set(invalid_rotations)}")
        if invalid_x:
            print(f"   ⚠️  WARNING: {len(invalid_x)} invalid x-positions found: {set(invalid_x)}")
        if invalid_y:
            print(f"   ⚠️  WARNING: {len(invalid_y)} invalid y-positions found: {set(invalid_y)}")
        
        if not invalid_rotations and not invalid_x and not invalid_y:
            print(f"   ✅ All placement coordinates are within valid bounds")
    else:
        print(f"   ⚠️  No placements to validate coordinates")
    
    # Check exploration method
    exploration_methods = [t.get('exploration_method', 'unknown') for t in terminals]
    unique_methods = set(exploration_methods)
    print(f"   • Exploration methods used: {unique_methods}")
    
    # Chain connectivity analysis
    if len(terminals) > 0:
        # Group by chain depth
        by_depth = {}
        for terminal in terminals:
            depth = terminal.get('chain_depth', 0)
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(terminal)
        
        print(f"   • States by chain depth:")
        for depth in sorted(by_depth.keys()):
            count = len(by_depth[depth])
            print(f"     - Depth {depth}: {count} states")
    
    env.close()
    print(f"\n🎯 Sequential test completed successfully!")
    print(f"   Variable batch size: {len(terminals)} terminal states generated")
    return len(terminals) > 0  # Success if any states generated

if __name__ == "__main__":
    test_sequential_deterministic_explorer() 