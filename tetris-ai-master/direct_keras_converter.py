#!/usr/bin/env python3
"""
Direct Keras to PyTorch converter that reads .keras/.h5 files without TensorFlow.
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import json

def load_keras_weights_directly(keras_file_path):
    """
    Load weights directly from Keras HDF5 file without TensorFlow.
    
    Args:
        keras_file_path: Path to .keras or .h5 file
        
    Returns:
        List of (weight_matrix, bias_vector) tuples for each layer
    """
    print(f"📂 Reading Keras file directly: {keras_file_path}")
    
    weights = []
    
    with h5py.File(keras_file_path, 'r') as f:
        print("📋 Keras file structure:")
        
        def print_structure(name, obj):
            print(f"  {name}: {type(obj)}")
            if isinstance(obj, h5py.Dataset):
                print(f"    Shape: {obj.shape}, dtype: {obj.dtype}")
        
        f.visititems(print_structure)
        
        # Try different possible structures
        model_weights_group = None
        
        # Common locations for weights in Keras files
        possible_paths = [
            'model_weights',
            'model',
            'sequential',
            'sequential_1',
            'weights'
        ]
        
        for path in possible_paths:
            if path in f:
                model_weights_group = f[path]
                print(f"✅ Found weights at: {path}")
                break
        
        if model_weights_group is None:
            # Try to find any group that contains layer weights
            for key in f.keys():
                if 'dense' in key.lower() or 'layer' in key.lower():
                    model_weights_group = f
                    print(f"✅ Using root level weights")
                    break
        
        if model_weights_group is None:
            raise ValueError("Could not find model weights in Keras file")
        
        # Extract weights from each dense layer
        layer_names = []
        for key in model_weights_group.keys():
            if 'dense' in key.lower():
                layer_names.append(key)
        
        layer_names.sort()  # Ensure consistent ordering
        print(f"🔍 Found {len(layer_names)} dense layers: {layer_names}")
        
        for layer_name in layer_names:
            layer_group = model_weights_group[layer_name]
            print(f"\n📋 Processing layer: {layer_name}")
            
            # Look for weight and bias datasets
            weight_data = None
            bias_data = None
            
            # Try different possible names for weights and biases
            for item_name in layer_group.keys():
                item = layer_group[item_name]
                print(f"  Item: {item_name}, type: {type(item)}")
                
                if isinstance(item, h5py.Group):
                    # Weights might be in a subgroup
                    for sub_name in item.keys():
                        sub_item = item[sub_name]
                        print(f"    Sub-item: {sub_name}, shape: {getattr(sub_item, 'shape', 'N/A')}")
                        
                        if 'kernel' in sub_name or 'weight' in sub_name:
                            weight_data = np.array(sub_item)
                        elif 'bias' in sub_name:
                            bias_data = np.array(sub_item)
                elif isinstance(item, h5py.Dataset):
                    print(f"    Dataset shape: {item.shape}")
                    
                    if 'kernel' in item_name or 'weight' in item_name:
                        weight_data = np.array(item)
                    elif 'bias' in item_name:
                        bias_data = np.array(item)
            
            if weight_data is not None and bias_data is not None:
                print(f"  ✅ Extracted weights: {weight_data.shape}, bias: {bias_data.shape}")
                weights.append((weight_data, bias_data))
            else:
                print(f"  ❌ Could not find weights/bias for layer {layer_name}")
                print(f"     Weight data: {weight_data is not None}")
                print(f"     Bias data: {bias_data is not None}")
    
    print(f"\n✅ Successfully extracted {len(weights)} layer weight sets")
    return weights

def create_pytorch_model_from_weights(layer_weights, activations=['relu', 'relu', 'linear']):
    """
    Create a PyTorch model from extracted Keras weights.
    
    Args:
        layer_weights: List of (weight_matrix, bias_vector) tuples
        activations: List of activation functions
        
    Returns:
        PyTorch model with loaded weights
    """
    print(f"\n🏗️  Building PyTorch model from {len(layer_weights)} layers")
    
    # Build model architecture
    layers = []
    
    for i, (weight_matrix, bias_vector) in enumerate(layer_weights):
        input_dim, output_dim = weight_matrix.shape
        print(f"  Layer {i}: {input_dim} -> {output_dim}")
        
        # Create linear layer
        linear = nn.Linear(input_dim, output_dim)
        
        # Load weights (Keras format is already input_dim x output_dim)
        # PyTorch expects output_dim x input_dim, so we need to transpose
        linear.weight.data = torch.FloatTensor(weight_matrix.T)
        linear.bias.data = torch.FloatTensor(bias_vector)
        
        layers.append(linear)
        
        # Add activation (except for last layer if linear)
        if i < len(activations) and not (i == len(layer_weights) - 1 and activations[i] == 'linear'):
            if activations[i] == 'relu':
                layers.append(nn.ReLU())
            elif activations[i] == 'tanh':
                layers.append(nn.Tanh())
            elif activations[i] == 'sigmoid':
                layers.append(nn.Sigmoid())
    
    model = nn.Sequential(*layers)
    print(f"✅ Created PyTorch model with {len(layers)} total layers")
    
    return model

def convert_keras_to_pytorch(keras_file_path, pytorch_file_path=None, test_conversion=True):
    """
    Convert a Keras model file to PyTorch format.
    
    Args:
        keras_file_path: Path to input .keras/.h5 file
        pytorch_file_path: Path to save converted .pth file (optional)
        test_conversion: Whether to test the converted model
        
    Returns:
        PyTorch model
    """
    print("🔄 CONVERTING KERAS MODEL TO PYTORCH")
    print("=" * 50)
    
    try:
        # Extract weights from Keras file
        layer_weights = load_keras_weights_directly(keras_file_path)
        
        if len(layer_weights) == 0:
            raise ValueError("No layer weights extracted from Keras file")
        
        # Create PyTorch model
        model = create_pytorch_model_from_weights(layer_weights)
        
        # Test the conversion
        if test_conversion:
            print(f"\n🧪 Testing converted model...")
            test_input = torch.randn(1, layer_weights[0][0].shape[0])  # Input size from first layer
            
            with torch.no_grad():
                output = model(test_input)
                print(f"  Input shape: {test_input.shape}")
                print(f"  Output shape: {output.shape}")
                print(f"  Output value: {output.item():.6f}")
                print("  ✅ Forward pass successful!")
        
        # Save if requested
        if pytorch_file_path:
            torch.save(model.state_dict(), pytorch_file_path)
            print(f"💾 Saved PyTorch model to: {pytorch_file_path}")
        
        print("✅ CONVERSION COMPLETED SUCCESSFULLY!")
        return model
        
    except Exception as e:
        print(f"❌ CONVERSION FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Test conversion
    keras_path = "sample.keras"
    pytorch_path = "sample_converted.pth"
    
    try:
        model = convert_keras_to_pytorch(keras_path, pytorch_path)
        print("🎉 Conversion test completed successfully!")
    except Exception as e:
        print(f"💥 Conversion test failed: {e}") 