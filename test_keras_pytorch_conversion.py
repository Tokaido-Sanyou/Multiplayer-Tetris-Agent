#!/usr/bin/env python3
"""
Comprehensive test for Keras to PyTorch conversion and expert performance validation.
"""

import numpy as np
import torch
import os
import sys

# Add tetris-ai-master to path
sys.path.append('tetris-ai-master')

try:
    from pytorch_dqn import DQNAgent
    from tetris import Tetris
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_keras_conversion():
    """Test Keras to PyTorch conversion with detailed validation"""
    print("=" * 60)
    print("TESTING KERAS TO PYTORCH CONVERSION")
    print("=" * 60)
    
    keras_model_path = "tetris-ai-master/sample.keras"
    if not os.path.exists(keras_model_path):
        print(f"❌ Keras model not found: {keras_model_path}")
        return False
    
    try:
        # Load Keras model and convert to PyTorch
        print(f"\n🔄 Loading and converting Keras model: {keras_model_path}")
        agent = DQNAgent(state_size=4, modelFile=keras_model_path)
        
        print(f"\n✅ Model loaded successfully!")
        print(f"   Device: {agent.device}")
        print(f"   Model type: {type(agent.model)}")
        
        # Test with sample inputs
        print(f"\n🧪 Testing model with sample inputs...")
        test_cases = [
            [0, 0, 0, 0],      # All zeros
            [1, 1, 1, 1],      # All ones  
            [10, 5, 3, 15],    # Typical tetris features
            [20, 10, 8, 30],   # Higher values
            [0, 15, 2, 5],     # Mixed values
        ]
        
        predictions = []
        for i, test_input in enumerate(test_cases):
            state = np.array(test_input, dtype=np.float32)
            prediction = agent.predict_value(state.reshape(1, -1))
            predictions.append(prediction)
            print(f"   Input {i+1}: {test_input} -> Prediction: {prediction:.6f}")
        
        # Check that predictions are reasonable (not all same, not NaN/inf)
        predictions = np.array(predictions)
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            print("❌ Model produces NaN or Inf values!")
            return False
        
        if np.std(predictions) < 1e-6:
            print("❌ Model produces identical outputs for different inputs!")
            return False
        
        print(f"✅ Model predictions look reasonable (std: {np.std(predictions):.6f})")
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_expert_performance():
    """Test expert performance on original tetris-ai-master environment"""
    print("\n" + "=" * 60)
    print("TESTING EXPERT PERFORMANCE ON ORIGINAL ENVIRONMENT")
    print("=" * 60)
    
    keras_model_path = "tetris-ai-master/sample.keras"
    
    try:
        # Load the converted agent
        agent = DQNAgent(state_size=4, modelFile=keras_model_path)
        
        # Test on original Tetris environment
        num_games = 5
        scores = []
        
        for game_num in range(num_games):
            print(f"\n🎮 Playing game {game_num + 1}/{num_games}...")
            
            # Create new tetris game
            env = Tetris()
            current_state = env.reset()
            
            total_score = 0
            pieces_placed = 0
            max_pieces = 500  # Prevent infinite games
            
            while not env.game_over and pieces_placed < max_pieces:
                # Get all possible next states
                next_states = env.get_next_states()
                
                if not next_states:
                    print("   No valid moves available - game over")
                    break
                
                # Get best action using agent
                best_state = agent.best_state(list(next_states.values()))
                
                # Find corresponding action
                best_action = None
                for action, state in next_states.items():
                    if np.allclose(state, best_state, atol=1e-6):
                        best_action = action
                        break
                
                if best_action is None:
                    print("   Could not find matching action - using first available")
                    best_action = list(next_states.keys())[0]
                
                # Execute action
                reward, game_over = env.play(best_action[0], best_action[1])
                total_score += reward
                pieces_placed += 1
                
                if pieces_placed % 50 == 0:
                    print(f"   Pieces: {pieces_placed}, Score: {total_score}, Lines: {env.score // 10}")
            
            final_score = env.get_game_score()
            scores.append(final_score)
            
            print(f"   🏁 Game {game_num + 1} finished:")
            print(f"      Final Score: {final_score}")
            print(f"      Pieces Placed: {pieces_placed}")
            print(f"      Game Over: {env.game_over}")
        
        # Analyze results
        scores = np.array(scores)
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Games Played: {num_games}")
        print(f"   Average Score: {np.mean(scores):.1f}")
        print(f"   Best Score: {np.max(scores):.1f}")
        print(f"   Worst Score: {np.min(scores):.1f}")
        print(f"   Score Std: {np.std(scores):.1f}")
        
        # Check if performance is reasonable
        avg_score = np.mean(scores)
        if avg_score < 100:
            print(f"❌ Average score {avg_score:.1f} is very low - conversion may be incorrect")
            return False
        elif avg_score < 500:
            print(f"⚠️  Average score {avg_score:.1f} is below expected range (500+)")
            return False
        else:
            print(f"✅ Average score {avg_score:.1f} looks good!")
            return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_keras_pytorch_outputs():
    """Compare outputs between original Keras and converted PyTorch models"""
    print("\n" + "=" * 60)
    print("COMPARING KERAS VS PYTORCH MODEL OUTPUTS")
    print("=" * 60)
    
    keras_model_path = "tetris-ai-master/sample.keras"
    
    try:
        # Load PyTorch model
        pytorch_agent = DQNAgent(state_size=4, modelFile=keras_model_path)
        
        # Try to load Keras model directly for comparison
        try:
            import tensorflow as tf
            keras_model = tf.keras.models.load_model(keras_model_path)
            print("✅ Loaded both Keras and PyTorch models")
            
            # Test with several inputs
            test_inputs = [
                [0, 0, 0, 0],
                [1, 1, 1, 1],  
                [10, 5, 3, 15],
                [20, 10, 8, 30],
                [5, 2, 1, 8],
            ]
            
            print("\n🔍 Comparing outputs:")
            print("Input                  | Keras Output | PyTorch Output | Difference")
            print("-" * 70)
            
            max_diff = 0
            for test_input in test_inputs:
                # Keras prediction
                keras_input = np.array(test_input, dtype=np.float32).reshape(1, -1)
                keras_output = keras_model.predict(keras_input, verbose=0)[0][0]
                
                # PyTorch prediction  
                pytorch_output = pytorch_agent.predict_value(keras_input)
                
                # Compare
                diff = abs(keras_output - pytorch_output)
                max_diff = max(max_diff, diff)
                
                print(f"{str(test_input):20} | {keras_output:11.6f} | {pytorch_output:13.6f} | {diff:.6f}")
            
            print(f"\nMaximum difference: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                print("✅ Outputs match very closely - conversion is excellent!")
                return True
            elif max_diff < 1e-2:
                print("✅ Outputs match reasonably - conversion is good!")
                return True
            else:
                print("❌ Outputs differ significantly - conversion may have issues!")
                return False
                
        except ImportError:
            print("⚠️  TensorFlow not available - skipping direct comparison")
            return True
            
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 STARTING COMPREHENSIVE KERAS TO PYTORCH CONVERSION TESTS")
    
    results = {}
    
    # Test 1: Basic conversion
    results['conversion'] = test_keras_conversion()
    
    # Test 2: Expert performance
    results['performance'] = test_expert_performance()
    
    # Test 3: Direct comparison (if possible)
    results['comparison'] = compare_keras_pytorch_outputs()
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper():20}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 