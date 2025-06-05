#!/usr/bin/env python3
"""
Test the existing sample.pth PyTorch model directly.
"""

import torch
import numpy as np
from pytorch_dqn import DQNAgent
from tetris import Tetris

def test_pytorch_model():
    """Test the existing sample.pth model"""
    print("🧪 TESTING EXISTING PYTORCH MODEL")
    print("=" * 50)
    
    try:
        # Load the PyTorch model directly
        print("📂 Loading sample.pth...")
        agent = DQNAgent(state_size=4, modelFile="sample.pth")
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {agent.device}")
        print(f"   Model type: {type(agent.model)}")
        print(f"   Model parameters: {sum(p.numel() for p in agent.model.parameters())}")
        
        # Test with sample inputs
        print(f"\n🔍 Testing model predictions...")
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
        
        # Check prediction quality
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
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_game_performance():
    """Test model performance on actual Tetris games"""
    print("\n🎮 TESTING GAME PERFORMANCE")
    print("=" * 50)
    
    try:
        # Load agent
        agent = DQNAgent(state_size=4, modelFile="sample.pth")
        
        # Play several games
        num_games = 3
        scores = []
        
        for game_num in range(num_games):
            print(f"\n🕹️  Playing game {game_num + 1}/{num_games}...")
            
            env = Tetris()
            current_state = env.reset()
            
            total_score = 0
            pieces_placed = 0
            max_pieces = 200  # Prevent infinite games
            
            while not env.game_over and pieces_placed < max_pieces:
                # Get all possible next states
                next_states = env.get_next_states()
                
                if not next_states:
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
                    best_action = list(next_states.keys())[0]
                
                # Execute action
                reward, game_over = env.play(best_action[0], best_action[1])
                total_score += reward
                pieces_placed += 1
                
                if pieces_placed % 20 == 0:
                    print(f"   Pieces: {pieces_placed:3d}, Score: {total_score:5.0f}")
            
            final_score = env.get_game_score()
            scores.append(final_score)
            
            print(f"   🏁 Game {game_num + 1} result: {final_score} points, {pieces_placed} pieces")
        
        # Analyze results
        scores = np.array(scores)
        avg_score = np.mean(scores)
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Games: {num_games}")
        print(f"   Average Score: {avg_score:.1f}")
        print(f"   Best Score: {np.max(scores):.1f}")
        print(f"   Score Range: {np.min(scores):.1f} - {np.max(scores):.1f}")
        
        # Evaluate performance
        if avg_score < 50:
            print(f"❌ Performance is very poor (avg: {avg_score:.1f})")
            return False
        elif avg_score < 200:
            print(f"⚠️  Performance is below expectations (avg: {avg_score:.1f})")
            return False
        else:
            print(f"✅ Performance looks good! (avg: {avg_score:.1f})")
            return True
        
    except Exception as e:
        print(f"❌ Game performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 STARTING PYTORCH MODEL VALIDATION")
    
    # Test 1: Basic model functionality
    model_ok = test_pytorch_model()
    
    # Test 2: Game performance
    performance_ok = test_game_performance()
    
    # Final result
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Model Loading:     {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Game Performance:  {'✅ PASS' if performance_ok else '❌ FAIL'}")
    
    overall = model_ok and performance_ok
    print(f"\nOVERALL: {'✅ SUCCESS' if overall else '❌ ISSUES DETECTED'}")
    
    return overall

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 