#!/usr/bin/env python3
"""
Analyze the checkpoint file to understand the exact architecture needed
"""

import torch

def analyze_checkpoint():
    """Analyze the saved checkpoint to understand the architecture"""
    checkpoint_path = 'local-multiplayer-tetris-main/localMultiplayerTetris/checkpoints/80k_bad.pt'
    
    print("🔍 Analyzing Checkpoint Architecture")
    print("=" * 60)
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Loaded checkpoint from: {checkpoint_path}")
        print(f"📁 Checkpoint keys: {list(ckpt.keys())}")
        
        if 'network' not in ckpt:
            print("❌ No 'network' key found in checkpoint!")
            return
        
        network = ckpt['network']
        print(f"\n🏗️  Network Architecture from Checkpoint:")
        print(f"📋 Total layers: {len(network)}")
        
        # Categorize layers
        feature_layers = []
        actor_layers = []
        critic_layers = []
        other_layers = []
        
        for key in sorted(network.keys()):
            shape = network[key].shape
            if key.startswith('feature_extractor'):
                feature_layers.append((key, shape))
            elif key.startswith('actor'):
                actor_layers.append((key, shape))
            elif key.startswith('critic'):
                critic_layers.append((key, shape))
            else:
                other_layers.append((key, shape))
        
        # Feature Extractor Analysis
        print(f"\n🧠 Feature Extractor ({len(feature_layers)} layers):")
        for key, shape in feature_layers:
            print(f"  {key}: {shape}")
        
        # Calculate expected feature dimension from checkpoint
        if 'feature_extractor.grid_conv.0.weight' in network:
            grid_channels = network['feature_extractor.grid_conv.0.weight'].shape[0]
            grid_feature_dim = grid_channels * 20 * 10  # channels * H * W
            print(f"  → Grid conv channels: {grid_channels}")
            print(f"  → Grid feature dim: {grid_feature_dim}")
        
        if 'feature_extractor.piece_embed.2.weight' in network:
            piece_embed_dim = network['feature_extractor.piece_embed.2.weight'].shape[0]
            print(f"  → Piece embed dim: {piece_embed_dim}")
            total_feature_dim = grid_feature_dim + piece_embed_dim
            print(f"  → Total feature dim: {total_feature_dim}")
        
        # Actor Analysis
        print(f"\n🎭 Actor Network ({len(actor_layers)} layers):")
        for key, shape in actor_layers:
            print(f"  {key}: {shape}")
        
        if actor_layers:
            first_layer_input = actor_layers[0][1][1]  # First layer input dimension
            last_layer_output = actor_layers[-1][1][0]  # Last layer output dimension
            print(f"  → Input dimension: {first_layer_input}")
            print(f"  → Output dimension (actions): {last_layer_output}")
            print(f"  → Total actor layers: {len([k for k, _ in actor_layers if 'weight' in k])}")
        
        # Critic Analysis
        print(f"\n🏛️  Critic Network ({len(critic_layers)} layers):")
        for key, shape in critic_layers:
            print(f"  {key}: {shape}")
        
        if critic_layers:
            first_layer_input = critic_layers[0][1][1]  # First layer input dimension
            last_layer_output = critic_layers[-1][1][0]  # Last layer output dimension
            print(f"  → Input dimension: {first_layer_input}")
            print(f"  → Output dimension: {last_layer_output}")
            print(f"  → Total critic layers: {len([k for k, _ in critic_layers if 'weight' in k])}")
        
        # Other layers
        if other_layers:
            print(f"\n❓ Other Layers ({len(other_layers)}):")
            for key, shape in other_layers:
                print(f"  {key}: {shape}")
        
        # Architecture Summary
        print(f"\n📊 Architecture Summary:")
        print(f"  Feature extractor → {total_feature_dim if 'total_feature_dim' in locals() else 'Unknown'} features")
        print(f"  Actor: {first_layer_input if 'first_layer_input' in locals() else 'Unknown'} → {last_layer_output if 'last_layer_output' in locals() else 'Unknown'}")
        print(f"  Critic: {first_layer_input if 'first_layer_input' in locals() else 'Unknown'} → 1")
        
        # Additional metadata
        if 'episode' in ckpt:
            print(f"  Training episode: {ckpt['episode']}")
        if 'epsilon' in ckpt:
            print(f"  Epsilon: {ckpt['epsilon']}")
        
    except Exception as e:
        print(f"❌ Error analyzing checkpoint: {e}")

def compare_with_current_architectures():
    """Compare checkpoint with current implementations"""
    print(f"\n🔄 Comparing with Current Implementations:")
    print("=" * 60)
    
    print("📝 Current implementations found:")
    print("  1. actor_critic.py - Original implementation")
    print("  2. checkpoint_compatible_actor_critic.py - Updated for checkpoint")
    print("  3. airl_fixed.py - AIRL implementation")
    
    print(f"\n⚠️  Architecture Mismatches:")
    print("  From error message:")
    print("    - Missing: actor.6.weight, actor.6.bias, actor.8.weight, actor.8.bias")
    print("    - Missing: critic.6.weight, critic.6.bias, critic.8.weight, critic.8.bias")
    print("    - Grid conv: Expected [2,1,3,3] but current [8,1,3,3]")
    print("    - Piece embed: Expected [16,7] but current [32,7]")
    print("    - Actor layers: Expected [128,416] but current [512,1632]")

if __name__ == "__main__":
    analyze_checkpoint()
    compare_with_current_architectures() 