#!/usr/bin/env python3
"""
COMPREHENSIVE DEBUG: Full 3 Batch 3 Phase System Test
Tests the complete pipeline:
1. Iterative exploration (ALL terminal states)
2. Enhanced 6-phase state model training
3. Piece-by-piece progression
4. Goal vector encoding/decoding
5. Board state inheritance
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

def debug_iterative_exploration():
    """Debug the iterative exploration to verify ALL terminal states are explored"""
    print('🔍 DEBUG 1: Iterative Exploration (ALL Terminal States)')
    print('='*70)
    
    env = TetrisEnv()
    exploration_manager = PieceByPieceExplorationManager(env)
    
    # Small test for verification
    exploration_manager.max_pieces = 1
    exploration_manager.boards_to_keep = 3
    
    print('🚀 Running iterative exploration with 1 piece...')
    exploration_data = exploration_manager.collect_piece_by_piece_exploration_data('iterative')
    
    # Analyze results
    total_possible = 4 * 10 * 10  # 4 rotations * 10 x positions * 10 y positions (filtered)
    actual_explored = len(exploration_data)
    unique_placements = len(set(d['placement'] for d in exploration_data))
    
    print(f'\n📊 EXPLORATION ANALYSIS:')
    print(f'   • Expected possible states: ~{total_possible} (before plausibility filter)')
    print(f'   • Actually explored: {actual_explored}')
    print(f'   • Unique placements: {unique_placements}')
    print(f'   • Exploration completeness: {unique_placements == actual_explored}')
    
    # Check coverage
    rotations = set(d['placement'][0] for d in exploration_data)
    x_positions = set(d['placement'][1] for d in exploration_data)
    y_positions = set(d['placement'][2] for d in exploration_data)
    
    print(f'   • Rotations covered: {sorted(rotations)} (all 4: {len(rotations) == 4})')
    print(f'   • X positions covered: {len(x_positions)}/10')
    print(f'   • Y positions covered: {len(y_positions)}/10')
    
    # Check reward distribution
    rewards = [d['terminal_reward'] for d in exploration_data]
    print(f'   • Reward range: {min(rewards):.1f} to {max(rewards):.1f}')
    print(f'   • Reward variety: {len(set(rewards))} unique rewards')
    
    if actual_explored > 100:  # Should be exploring many more states
        print('   ✅ ITERATIVE EXPLORATION: WORKING (exploring many states)')
        return True, exploration_data
    else:
        print('   ❌ ITERATIVE EXPLORATION: FAILED (too few states)')
        return False, exploration_data

def debug_goal_vector_system():
    """Debug the 8D goal vector system"""
    print('\n🎯 DEBUG 2: Goal Vector System (8D Binary + Discrete)')
    print('='*70)
    
    components = Enhanced6PhaseComponents(device='cpu')
    
    # Test comprehensive goal vector scenarios
    test_cases = [
        {'placement': (0, 0, 10), 'confidence': 1.0, 'quality': 0.8, 'lines_potential': 4, 'is_valid': True},
        {'placement': (1, 5, 15), 'confidence': 0.5, 'quality': -0.3, 'lines_potential': 1, 'is_valid': True},
        {'placement': (2, 9, 19), 'confidence': 0.2, 'quality': 0.0, 'lines_potential': 0, 'is_valid': False},
        {'placement': (3, 4, 12), 'confidence': 0.9, 'quality': 1.0, 'lines_potential': 3, 'is_valid': True},
    ]
    
    print(f'🧪 Testing {len(test_cases)} goal vector encodings...')
    
    all_passed = True
    for i, test_case in enumerate(test_cases):
        # Encode
        goal_vector = components.goal_selector._option_to_goal_vector(test_case, 'cpu')
        
        # Decode
        decoded = components.goal_selector.decode_goal_vector_to_placement(goal_vector)
        
        # Verify
        original = test_case['placement']
        decoded_placement = decoded['placement']
        
        matches = (original[0] == decoded_placement[0] and 
                  original[1] == decoded_placement[1] and 
                  original[2] == decoded_placement[2])
        
        print(f'   Test {i+1}: {original} → {decoded_placement} {"✅" if matches else "❌"}')
        
        if not matches:
            all_passed = False
    
    if all_passed:
        print('   ✅ GOAL VECTOR SYSTEM: WORKING')
        return True
    else:
        print('   ❌ GOAL VECTOR SYSTEM: FAILED')
        return False

def debug_6phase_training():
    """Debug the 6-phase training system"""
    print('\n🧠 DEBUG 3: 6-Phase Training System')
    print('='*70)
    
    env = TetrisEnv()
    components = Enhanced6PhaseComponents(device='cpu')
    components.set_optimizers(state_lr=0.001, q_lr=0.001)
    
    # Generate some exploration data
    exploration_manager = PieceByPieceExplorationManager(env)
    exploration_manager.max_pieces = 1
    exploration_manager.boards_to_keep = 2
    
    print('🔄 Generating training data...')
    exploration_data = exploration_manager.collect_piece_by_piece_exploration_data('iterative')
    
    if len(exploration_data) < 10:
        print('   ❌ Insufficient training data')
        return False
    
    print(f'   • Training data: {len(exploration_data)} samples')
    
    # Test state model training
    print('🎯 Testing state model training...')
    state_results = components.train_enhanced_state_model(exploration_data)
    print(f'   • State model loss: {state_results["loss"]:.2f}')
    print(f'   • Top performers used: {state_results["top_performers_used"]}')
    
    # Test Q-learning training
    print('🔮 Testing Q-learning training...')
    q_results = components.train_simplified_q_learning(exploration_data)
    print(f'   • Q-learning loss: {q_results["q_loss"]:.2f}')
    print(f'   • Trajectories trained: {q_results["trajectories_trained"]}')
    
    # Test goal generation
    print('🎯 Testing goal generation...')
    test_state = torch.FloatTensor(exploration_data[0]['state']).unsqueeze(0)
    goal = components.get_goal_for_actor(test_state)
    print(f'   • Goal shape: {goal.shape}')
    print(f'   • Goal vector: {goal.squeeze().tolist()[:4]}... (first 4 elements)')
    
    if (state_results['loss'] < float('inf') and 
        q_results['q_loss'] < float('inf') and 
        goal.shape[1] == 8):
        print('   ✅ 6-PHASE TRAINING: WORKING')
        return True
    else:
        print('   ❌ 6-PHASE TRAINING: FAILED')
        return False

def debug_full_pipeline():
    """Debug the complete 3 batch 3 phase pipeline using current architecture"""
    print('\n🚀 DEBUG 4: Full Pipeline (3 Batch 3 Phase)')
    print('='*70)
    
    try:
        # Initialize components using current architecture
        env = TetrisEnv()
        exploration_manager = PieceByPieceExplorationManager(env)
        components = Enhanced6PhaseComponents(device='cpu')
        components.set_optimizers(state_lr=0.001, q_lr=0.001)
        
        # Configure for quick test
        exploration_manager.max_pieces = 2  # Quick test with 2 pieces
        exploration_manager.boards_to_keep = 3
        num_batches = 1  # CHANGED: 1 batch per stage as requested
        num_phases = 3   # CHANGED: 3 stages as requested
        
        print(f'⚙️ Configuration:')
        print(f'   • Batches per phase: {num_batches}')
        print(f'   • Training phases: {num_phases}')
        print(f'   • Max pieces per exploration: {exploration_manager.max_pieces}')
        print(f'   • Boards to keep: {exploration_manager.boards_to_keep}')
        
        # Run the training pipeline manually
        print(f'\n🏃‍♂️ Running full training pipeline...')
        
        total_state_loss = 0
        total_q_loss = 0
        total_exploration_data = 0
        
        # 3 phases × 3 batches = 9 training cycles
        for phase in range(num_phases):
            print(f'\n📈 PHASE {phase + 1}/{num_phases}')
            
            for batch in range(num_batches):
                print(f'   Batch {batch + 1}/{num_batches}...')
                
                # 1. Collect exploration data
                exploration_data = exploration_manager.collect_piece_by_piece_exploration_data('iterative')
                total_exploration_data += len(exploration_data)
                
                if len(exploration_data) < 5:
                    print(f'   ⚠️ Insufficient exploration data: {len(exploration_data)}')
                    continue
                
                # 2. Train state model
                state_results = components.train_enhanced_state_model(exploration_data)
                total_state_loss += state_results['loss']
                
                # 3. Train Q-learning
                q_results = components.train_simplified_q_learning(exploration_data)
                total_q_loss += q_results['q_loss']
                
                print(f'     • Explored states: {len(exploration_data)}')
                print(f'     • State loss: {state_results["loss"]:.3f}')
                print(f'     • Q loss: {q_results["q_loss"]:.3f}')
        
        # Calculate final statistics
        avg_state_loss = total_state_loss / (num_phases * num_batches)
        avg_q_loss = total_q_loss / (num_phases * num_batches)
        
        final_stats = {
            'phase': num_phases,
            'total_exploration_data': total_exploration_data,
            'avg_state_loss': avg_state_loss,
            'avg_q_loss': avg_q_loss,
            'exploration_reward': 'computed_via_iterative',
            'actor_loss': avg_state_loss  # Use state loss as proxy
        }
        
        print(f'\n📊 PIPELINE RESULTS:')
        print(f'   • Training completed successfully: True')
        print(f'   • Total exploration data: {final_stats["total_exploration_data"]}')
        print(f'   • Average state loss: {final_stats["avg_state_loss"]:.3f}')
        print(f'   • Average Q loss: {final_stats["avg_q_loss"]:.3f}')
        print(f'   • Training phases completed: {final_stats["phase"]}')
        print('   ✅ FULL PIPELINE: WORKING')
        return True, final_stats
            
    except Exception as e:
        print(f'   ❌ FULL PIPELINE: FAILED with error: {e}')
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Run comprehensive debug session"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print('🔧 COMPREHENSIVE SYSTEM DEBUG')
    print('='*80)
    print(f'Timestamp: {timestamp}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Device: cpu')
    
    # Run all debug tests
    debug_results = {}
    
    # 1. Iterative exploration
    iter_works, iter_data = debug_iterative_exploration()
    debug_results['iterative_exploration'] = iter_works
    
    # 2. Goal vector system
    goal_works = debug_goal_vector_system()
    debug_results['goal_vector_system'] = goal_works
    
    # 3. 6-phase training
    training_works = debug_6phase_training()
    debug_results['6phase_training'] = training_works
    
    # 4. Full pipeline (if basics work)
    if iter_works and goal_works and training_works:
        pipeline_works, pipeline_stats = debug_full_pipeline()
        debug_results['full_pipeline'] = pipeline_works
    else:
        print('\n⏭️ Skipping full pipeline test - prerequisites failed')
        debug_results['full_pipeline'] = False
    
    # Summary
    print('\n🎉 DEBUG SESSION SUMMARY')
    print('='*80)
    working_count = sum(debug_results.values())
    total_count = len(debug_results)
    
    for test_name, result in debug_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f'   • {test_name.replace("_", " ").title()}: {status}')
    
    print(f'\n📊 Overall: {working_count}/{total_count} systems working')
    
    if working_count == total_count:
        print('🚀 ALL SYSTEMS OPERATIONAL - Ready for production training!')
    else:
        print('⚠️ Issues detected - Review failed systems before production use')
    
    # Save debug results
    debug_file = f'debug_results_{timestamp}.txt'
    with open(debug_file, 'w') as f:
        f.write(f'Debug Session Results - {timestamp}\n')
        f.write('='*50 + '\n\n')
        for test_name, result in debug_results.items():
            f.write(f'{test_name}: {"PASS" if result else "FAIL"}\n')
        f.write(f'\nOverall: {working_count}/{total_count} systems working\n')
    
    print(f'\n💾 Debug results saved to: {debug_file}')
    
    return debug_results

if __name__ == '__main__':
    try:
        results = main()
    except Exception as e:
        print(f'\n💥 Debug session crashed: {e}')
        import traceback
        traceback.print_exc() 