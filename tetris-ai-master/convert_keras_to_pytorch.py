#!/usr/bin/env python3
"""
Convert Keras model to PyTorch model without TensorFlow dependency.
This script loads the weights directly from the Keras model file
and creates an equivalent PyTorch model.
"""

import numpy as np
import os
import sys
import h5py
import torch
from pytorch_dqn import DQNModel

def convert_keras_to_pytorch(keras_model_path, output_path):
    """
    Convert a Keras model to PyTorch without TF dependency.
    Uses h5py to directly read the weights from the Keras file.
    """
    print(f"Converting {keras_model_path} to PyTorch...")
    
    try:
        # Check if file exists
        if not os.path.exists(keras_model_path):
            print(f"Error: Keras model file {keras_model_path} not found.")
            return False
        
        # Read model structure from the Keras H5 file
        with h5py.File(keras_model_path, 'r') as f:
            # Extract model architecture
            layer_names = []
            for layer_name in f['model_weights']:
                if layer_name != '_layer_checkpoint_dependencies':
                    layer_names.append(layer_name)
            
            # Extract layer sizes from weights
            layer_sizes = []
            for layer_name in layer_names:
                if 'kernel:0' in f['model_weights'][layer_name]:
                    weights = np.array(f['model_weights'][layer_name]['kernel:0'])
                    layer_sizes.append(weights.shape)
            
            # Create PyTorch model with matching architecture
            input_dim = layer_sizes[0][0]  # Input dimension
            hidden_layers = [s[1] for s in layer_sizes[:-1]]  # Hidden layer sizes
            
            print(f"Model architecture: input={input_dim}, hidden={hidden_layers}")
            
            # Create PyTorch model
            model = DQNModel(input_dim, hidden_layers)
            
            # Load weights for each layer
            for i, layer_name in enumerate(layer_names):
                # Skip non-weights layers
                if 'kernel:0' not in f['model_weights'][layer_name]:
                    continue
                
                # Get weights and bias
                weights = np.array(f['model_weights'][layer_name]['kernel:0'])
                bias = np.array(f['model_weights'][layer_name]['bias:0'])
                
                # Keras weights are (input_dim, output_dim), PyTorch uses (output_dim, input_dim)
                weights = np.transpose(weights)
                
                # Convert to PyTorch tensors
                weights_tensor = torch.FloatTensor(weights)
                bias_tensor = torch.FloatTensor(bias)
                
                # Assign to PyTorch model
                model.layers[i].weight.data = weights_tensor
                model.layers[i].bias.data = bias_tensor
                
                print(f"Loaded layer {i}: {weights.shape}")
        
        # Save the PyTorch model
        torch.save(model, output_path)
        print(f"PyTorch model saved to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error converting model: {e}")
        return False

def main():
    """Main function for converting the model."""
    if len(sys.argv) < 3:
        print("Usage: python convert_keras_to_pytorch.py <keras_model_path> <output_path>")
        print("Example: python convert_keras_to_pytorch.py sample.keras sample.pth")
        return
    
    keras_model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    success = convert_keras_to_pytorch(keras_model_path, output_path)
    if success:
        print("Conversion successful!")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main() 