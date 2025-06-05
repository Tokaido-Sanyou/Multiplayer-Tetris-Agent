#!/usr/bin/env python3
"""
Test and verify the PyTorch implementation against the original Keras model.
This script converts the sample.keras model to PyTorch and verifies that they
produce the same outputs for the same inputs.
"""

import os
import sys
import numpy as np
import torch
import random

def verify_implementation():
    """
    Verify the PyTorch implementation against the original Keras model.
    
    Steps:
    1. Convert sample.keras to PyTorch if not already converted
    2. Create test states
    3. Compare outputs from both models
    4. Verify best_state functionality
    """
    print("Testing PyTorch implementation against Keras model...")
    
    # Check if sample.keras exists
    keras_model_path = "sample.keras"
    pytorch_model_path = "sample.pth"
    
    if not os.path.exists(keras_model_path):
        print(f"Error: {keras_model_path} not found.")
        print("Make sure you have the sample.keras model file.")
        return
    
    # Convert to PyTorch if needed
    if not os.path.exists(pytorch_model_path):
        print(f"Converting {keras_model_path} to PyTorch...")
        from convert_keras_to_pytorch import convert_keras_to_pytorch
        success = convert_keras_to_pytorch(keras_model_path, pytorch_model_path)
        if not success:
            print("Conversion failed. Cannot continue testing.")
            return
    
    # Import DQN agents (both PyTorch and Keras if available)
    try:
        # Try to import Keras for comparison
        from keras.models import load_model
        keras_available = True
        keras_model = load_model(keras_model_path)
        print("Keras model loaded for comparison.")
    except ImportError:
        keras_available = False
        print("Keras not available, will test PyTorch implementation only.")
    
    # Import PyTorch agent
    from pytorch_dqn import PyTorchDQNAgent, DQNModel
    
    # Load PyTorch model
    pytorch_agent = PyTorchDQNAgent(4, model_file=pytorch_model_path)  # 4 = state size
    print(f"PyTorch model loaded from {pytorch_model_path}")
    
    # Create test states (4 features each)
    num_test_states = 10
    test_states = []
    for _ in range(num_test_states):
        # Generate random state with same range as actual game states
        state = np.array([
            random.randint(0, 5),      # lines_cleared
            random.randint(0, 50),     # holes
            random.randint(0, 100),    # bumpiness
            random.randint(0, 200)     # height
        ], dtype=np.float32)
        test_states.append(state)
    
    # Test predictions
    print("\nComparing predictions for test states:")
    total_diff = 0
    max_diff = 0
    
    for i, state in enumerate(test_states):
        # PyTorch prediction
        state_tensor = torch.FloatTensor(state.reshape(1, 4)).to(pytorch_agent.device)
        with torch.no_grad():
            pytorch_pred = pytorch_agent.model(state_tensor).item()
        
        # Keras prediction (if available)
        if keras_available:
            keras_pred = keras_model.predict(state.reshape(1, 4), verbose=0)[0][0]
            diff = abs(pytorch_pred - keras_pred)
            total_diff += diff
            max_diff = max(max_diff, diff)
            print(f"State {i+1}: PyTorch = {pytorch_pred:.6f}, Keras = {keras_pred:.6f}, Diff = {diff:.6f}")
        else:
            print(f"State {i+1}: PyTorch = {pytorch_pred:.6f}")
    
    if keras_available:
        avg_diff = total_diff / num_test_states
        print(f"\nAverage difference: {avg_diff:.6f}")
        print(f"Maximum difference: {max_diff:.6f}")
        
        # Acceptable threshold for differences (models will never be exactly the same due to implementation differences)
        threshold = 0.1
        if avg_diff < threshold:
            print("\n✅ PyTorch implementation matches Keras model (within acceptable tolerance).")
        else:
            print("\n❌ PyTorch implementation differs significantly from Keras model.")
    
    # Test best_state functionality
    print("\nTesting best_state functionality:")
    
    # Create a set of states
    states = {}
    for i in range(5):
        # Create distinct states
        state = tuple(np.random.rand(4) * np.array([5, 50, 100, 200]))
        states[state] = (i, i)  # dummy action
    
    # Get best state using PyTorch agent
    best_state = pytorch_agent.best_state(states.keys())
    
    # Verify that it picked the state with the highest value
    print(f"Best state selected: {best_state}")
    print(f"Value of best state: {pytorch_agent.predict_value(np.array(best_state).reshape(1, 4)):.6f}")
    
    # Verify other states have lower values
    all_values = []
    for state in states.keys():
        value = pytorch_agent.predict_value(np.array(state).reshape(1, 4))
        all_values.append((state, value))
    
    # Sort by value
    all_values.sort(key=lambda x: x[1], reverse=True)
    print("\nAll states by value:")
    for state, value in all_values:
        print(f"State: {state}, Value: {value:.6f}")
    
    # Final verification
    if all_values[0][0] == best_state:
        print("\n✅ best_state function correctly selected the highest-valued state.")
    else:
        print("\n❌ best_state function did not select the highest-valued state.")
    
    print("\nPyTorch implementation test complete!")

if __name__ == "__main__":
    verify_implementation() 