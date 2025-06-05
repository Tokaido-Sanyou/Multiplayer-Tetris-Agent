#!/usr/bin/env python3
"""
Run Tetris AI using PyTorch DQN model.
Same functionality as the original run_model.py but using PyTorch.
"""

import sys
import os

if len(sys.argv) < 2:
    exit("Missing model file path. Usage: python run_pytorch_model.py <model_file>")

# Import PyTorch DQN agent
from pytorch_dqn import PyTorchDQNAgent
from tetris import Tetris

def main():
    """Run Tetris with the PyTorch DQN model."""
    model_file = sys.argv[1]
    
    # Handle both .keras and .pth files
    if model_file.endswith('.keras') and not os.path.exists(model_file.replace('.keras', '.pth')):
        print(f"Converting Keras model {model_file} to PyTorch...")
        try:
            from convert_keras_to_pytorch import convert_keras_to_pytorch
            pytorch_model = model_file.replace('.keras', '.pth')
            success = convert_keras_to_pytorch(model_file, pytorch_model)
            if success:
                print(f"Using converted PyTorch model: {pytorch_model}")
                model_file = pytorch_model
        except Exception as e:
            print(f"Warning: Could not convert Keras model: {e}")
            print("Using direct loading mechanism instead.")
    
    # Initialize Tetris environment
    env = Tetris()
    
    # Initialize PyTorch DQN agent with the model
    agent = PyTorchDQNAgent(env.get_state_size(), model_file=model_file)
    print(f"Model loaded: {model_file}")
    print(f"Using device: {agent.device}")
    
    # Run the game
    done = False
    total_reward = 0
    
    while not done:
        # Get possible next states
        next_states = env.get_next_states()
        next_states_dict = {tuple(v): k for k, v in next_states.items()}
        
        # Get best action using the model
        best_state = agent.best_state(next_states_dict.keys())
        best_action = next_states_dict[best_state]
        
        # Execute the action
        reward, done = env.play(best_action[0], best_action[1], render=True)
        total_reward += reward
    
    print(f"Game over! Total reward: {total_reward}")

if __name__ == "__main__":
    main() 