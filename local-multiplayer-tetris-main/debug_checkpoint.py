import torch

# Load checkpoint
ckpt = torch.load('local-multiplayer-tetris-main/localMultiplayerTetris/checkpoints/80k_bad.pt', map_location='cpu')
print("Checkpoint keys:", list(ckpt.keys()))

network = ckpt['network']
print("\nNetwork architecture from checkpoint:")

# Feature extractor
print("Feature extractor grid conv:")
print(f"  grid_conv.0.weight: {network['feature_extractor.grid_conv.0.weight'].shape}")
print(f"  grid_conv.2.weight: {network['feature_extractor.grid_conv.2.weight'].shape}")

print("Feature extractor piece embed:")
print(f"  piece_embed.0.weight: {network['feature_extractor.piece_embed.0.weight'].shape}")
print(f"  piece_embed.2.weight: {network['feature_extractor.piece_embed.2.weight'].shape}")

# Actor
print("Actor layers:")
print(f"  actor.0.weight: {network['actor.0.weight'].shape}")
print(f"  actor.2.weight: {network['actor.2.weight'].shape}")
print(f"  actor.4.weight: {network['actor.4.weight'].shape}")

# Additional layers
extra_layers = [k for k in network.keys() if 'actor.6' in k or 'actor.8' in k]
if extra_layers:
    print("Extra actor layers:")
    for layer in extra_layers:
        print(f"  {layer}: {network[layer].shape}")

# Calculate feature dimension
grid_conv_out = network['feature_extractor.grid_conv.0.weight'].shape[0] * 20 * 10  # channels * h * w
piece_embed_out = network['feature_extractor.piece_embed.2.weight'].shape[0]
feature_dim = grid_conv_out + piece_embed_out
print(f"\nCalculated feature dimension: {feature_dim}")
print(f"  Grid conv output: {grid_conv_out} ({network['feature_extractor.grid_conv.0.weight'].shape[0]} channels)")
print(f"  Piece embed output: {piece_embed_out}")

# Check output dimension
output_dim = network['actor.4.weight'].shape[0]
print(f"Action space dimension: {output_dim}") 