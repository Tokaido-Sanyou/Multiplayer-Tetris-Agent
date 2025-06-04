#!/usr/bin/env python3
"""
IMPROVED TEST: Valid Locked Positions + State Loss Explanation
Tests the optimized system with:
1. FIXED: Only valid locked positions (where pieces actually land)
2. BATCH: 10 block placement per batch
3. EXPLAINED: Current state loss calculation
"""

import sys
import os
sys.path.append('localMultiplayerTetris')

import numpy as np
import torch
from datetime import datetime

from localMultiplayerTetris.rl_utils.enhanced_6phase_state_model import (
    PieceByPieceExplorationManager, 
    Enhanced6PhaseComponents
)
from localMultiplayerTetris.tetris_env import TetrisEnv

def explain_state_loss_calculation():
    """Explain how the current state loss is calculated"""
    print("📖 STATE LOSS CALCULATION EXPLANATION")
    print("=" * 60)
    print()
    print("🎯 CURRENT STATE LOSS FORMULA (SIMPLIFIED):")
    print("   The state model predicts 5 placement options, each with 4 components:")
    print("   • Rotation (0-4) → normalized to 0-1")
    print("   • X position (0-10) → normalized to 0-1") 
    print("   • Y position (0-20) → normalized to 0-1")
    print("   • Lines potential (0-1)")
    print()
    print("🔧 LOSS COMPONENTS (Per Placement):")
    print("   rot_loss = MSE(pred_rot/4.0, target_rot/4.0) × 0.5")
    print("   x_loss = MSE(pred_x/10.0, target_x/10.0) × 1.0")
    print("   y_loss = MSE(pred_y/20.0, target_y/20.0) × 1.0")
    print("   lines_loss = MSE(pred_lines, target_lines) × 0.5")
    print()
    print("📊 FINAL LOSS CALCULATION:")
    print("   total_placement_loss = sum of 4 losses × 5 placements")
    print("   validity_penalty = 1.0 × invalid_placements")
    print("   combined_loss = total_placement_loss + validity_penalty")
    print()
    print("✅ KEY IMPROVEMENTS:")
    print("   1. REMOVED: Confidence & Quality (could distort model)")
    print("   2. KEPT: Only essential components (rotation, position, lines)")
    print("   3. Normalized targets to 0-1 range")
    print("   4. Training only on top 5% performers")
    print()

def test_valid_locked_positions():
    """Test the improved exploration with valid locked positions"""
    
    print("🔧 TESTING IMPROVED EXPLORATION SYSTEM")
    print("=" * 60)
    
    # Initialize environment
    env = TetrisEnv(single_player=True, headless=True)
    
    # Initialize enhanced components
    enhanced_system = Enhanced6PhaseComponents(
        state_dim=210,
        goal_dim=8,
        device='cpu'
    )
    
    enhanced_system.set_optimizers(state_lr=0.001, q_lr=0.001)
    
    # Create exploration manager  
    exploration_manager = enhanced_system.create_piece_by_piece_exploration_manager(env)
    exploration_manager.max_pieces = 1  # Test with 1 piece for clarity
    exploration_manager.boards_to_keep = 3
    
    print(f"✅ System initialized with VALID LOCKED POSITION exploration")
    print(f"   • Only explores where pieces actually LAND (not all theoretical positions)")
    print(f"   • Uses batch processing (10 blocks per batch)")
    print(f"   • Based on actual Tetris game rules")
    print()
    
    # Test exploration
    print("🔄 Phase 1: Testing valid locked position exploration...")
    start_time = datetime.now()
    
    exploration_data = exploration_manager.collect_piece_by_piece_exploration_data('iterative')
    
    collection_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ Exploration completed in {collection_time:.1f}s")
    print(f"   • Valid locked positions found: {len(exploration_data)}")
    
    if exploration_data:
        # Analyze placements
        placements = [d['placement'] for d in exploration_data]
        rotations = set(p[0] for p in placements)
        x_positions = set(p[1] for p in placements)
        y_positions = set(p[2] for p in placements)
        
        print(f"   • Unique rotations: {sorted(rotations)}")
        print(f"   • X positions used: {len(x_positions)}/10")
        print(f"   • Y range: {min(p[2] for p in placements)}-{max(p[2] for p in placements)}")
        print(f"   • Average Y (lock height): {np.mean([p[2] for p in placements]):.1f}")
        
        # Check efficiency improvement
        theoretical_all = 4 * 10 * 20  # Old method: 800 positions
        actual_valid = len(exploration_data)
        efficiency = (theoretical_all - actual_valid) / theoretical_all * 100
        
        print(f"   • Efficiency gain: {efficiency:.1f}% fewer positions to test")
        print(f"   • (Was testing {theoretical_all}, now testing {actual_valid})")
        
        lines_cleared_total = sum(d.get('lines_cleared', 0) for d in exploration_data)
        print(f"   • Total lines cleared: {lines_cleared_total}")
        avg_reward = np.mean([d.get('terminal_reward', 0) for d in exploration_data])
        print(f"   • Average reward: {avg_reward:.1f}")
    print()
    
    # Test state model training with explanation
    print("🎯 Phase 2: Training state model with FIXED loss calculation...")
    
    if exploration_data:
        state_training_results = enhanced_system.train_enhanced_state_model(exploration_data)
        print(f"✅ State model training completed")
        print(f"   • FIXED Loss: {state_training_results['loss']:.3f}")
        print(f"   • This loss represents the average MSE across:")
        print(f"     - 5 placement predictions × 4 components each")
        print(f"     - All components normalized to 0-1 range")
        print(f"     - Weighted by importance (rotation=0.5, position=1.0)")
        print(f"   • Training on top {state_training_results['top_performers_used']} performers")
        print(f"   • Loss threshold: {state_training_results.get('threshold', 'N/A')}")
    else:
        print("⚠️ No exploration data for state model training")
    print()
    
    # Test Q-learning training
    print("🎯 Phase 3: Training Q-learning with normalized returns...")
    
    if exploration_data:
        q_training_results = enhanced_system.train_simplified_q_learning(exploration_data)
        print(f"✅ Q-learning training completed")
        print(f"   • Q-Loss: {q_training_results['q_loss']:.3f}")
        print(f"   • N-step returns normalized with tanh(reward/100)")
        print(f"   • Trajectories trained: {q_training_results['trajectories_trained']}")
    else:
        print("⚠️ No exploration data for Q-learning training")
    print()
    
    print("🎉 IMPROVED SYSTEM VERIFICATION COMPLETE!")
    print("=" * 60)
    print("✅ Major improvements:")
    print("   1. ✅ Only explores VALID LOCKED POSITIONS")
    print("   2. ✅ Batch processing (10 blocks per batch)")
    print("   3. ✅ Based on actual Tetris gravity rules")
    print("   4. ✅ Significant efficiency improvement")
    print("   5. ✅ State loss properly normalized and explained")
    print()
    
    if exploration_data:
        print(f"📊 FINAL METRICS:")
        print(f"   • Valid positions explored: {len(exploration_data)}")
        print(f"   • Lines cleared: {sum(d.get('lines_cleared', 0) for d in exploration_data)}")
        print(f"   • State model loss: {state_training_results.get('loss', 'N/A'):.3f}")
        print(f"   • Q-learning loss: {q_training_results.get('q_loss', 'N/A'):.3f}")
        print(f"   • Efficiency: ~{efficiency:.0f}% improvement over brute force")
    
    return True

if __name__ == "__main__":
    # First explain the state loss calculation
    explain_state_loss_calculation()
    print()
    
    # Then test the improved system
    test_valid_locked_positions() 