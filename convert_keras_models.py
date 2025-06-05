#!/usr/bin/env python3
"""
Utility script to find and convert all Keras models in the project to PyTorch.
This helps prepare the project for running without TensorFlow dependencies.
"""

import os
import glob
import argparse
from typing import List, Tuple

def find_keras_models(root_dir: str = ".") -> List[str]:
    """Find all .keras and .h5 model files in the project."""
    keras_models = []
    
    # Find all .keras files
    for keras_file in glob.glob(f"{root_dir}/**/*.keras", recursive=True):
        keras_models.append(keras_file)
    
    # Find all .h5 files (alternative Keras format)
    for h5_file in glob.glob(f"{root_dir}/**/*.h5", recursive=True):
        keras_models.append(h5_file)
    
    return keras_models

def convert_models(keras_models: List[str], force: bool = False) -> List[Tuple[str, str, bool]]:
    """Convert all Keras models to PyTorch format."""
    results = []
    
    # Check if conversion module exists
    try:
        # Add tetris-ai-master to path
        import sys
        tetris_ai_path = os.path.join(os.path.dirname(__file__), 'tetris-ai-master')
        if os.path.exists(tetris_ai_path):
            sys.path.append(tetris_ai_path)
        
        # Import conversion function
        from convert_keras_to_pytorch import convert_keras_to_pytorch
        
        # Convert each model
        for keras_model in keras_models:
            # Define output path
            if keras_model.endswith('.keras'):
                pytorch_model = keras_model.replace('.keras', '.pth')
            else:  # .h5 file
                pytorch_model = keras_model.replace('.h5', '.pth')
            
            # Skip if already exists and not forcing overwrite
            if os.path.exists(pytorch_model) and not force:
                print(f"✓ PyTorch model already exists: {pytorch_model} (use --force to overwrite)")
                results.append((keras_model, pytorch_model, True))
                continue
            
            # Convert the model
            print(f"Converting {keras_model} to {pytorch_model}...")
            success = convert_keras_to_pytorch(keras_model, pytorch_model)
            
            if success:
                print(f"✓ Successfully converted: {pytorch_model}")
            else:
                print(f"✗ Failed to convert: {keras_model}")
            
            results.append((keras_model, pytorch_model, success))
    
    except ImportError as e:
        print(f"Error: Could not import conversion module: {e}")
        print("Make sure tetris-ai-master/convert_keras_to_pytorch.py exists")
    except Exception as e:
        print(f"Error during conversion: {e}")
    
    return results

def main():
    """Main function to find and convert all Keras models."""
    parser = argparse.ArgumentParser(description="Convert Keras models to PyTorch format")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing PyTorch models")
    parser.add_argument("--dir", default=".", help="Root directory to search for models")
    args = parser.parse_args()
    
    # Find all Keras models
    keras_models = find_keras_models(args.dir)
    
    if not keras_models:
        print("No Keras models found in the project.")
        return
    
    print(f"Found {len(keras_models)} Keras model(s):")
    for model in keras_models:
        print(f"  - {model}")
    
    # Convert models
    results = convert_models(keras_models, args.force)
    
    # Summarize results
    success_count = sum(1 for _, _, success in results if success)
    print(f"\nSummary: Converted {success_count}/{len(results)} models successfully")

if __name__ == "__main__":
    main() 