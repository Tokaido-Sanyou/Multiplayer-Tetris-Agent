"""Utility to convert a *fully-connected* Keras model (saved with
``tf.keras.Model.save``) to an equivalent PyTorch ``state_dict``.

This is designed specifically for the legacy `tetris-ai-master` DQN model
which is *feed-forward only* and built via::

    Sequential([
        Dense(n_neurons[0], activation=activations[0]),
        Dense(n_neurons[1], activation=activations[1]),
        Dense(1,               activation=activations[2])
    ])

Typical usage
-------------
$ python -m localMultiplayerTetris.convert_keras_to_torch \
      /path/to/model.keras  \
      /path/to/output.pth

The script will:
1. Load the Keras model (with the custom `DTypePolicy` if needed).
2. Build an isomorphic PyTorch ``nn.Sequential``.
3. Transpose and copy weights / biases.
4. Save ``state_dict`` (and layer/activation metadata) into a ``.pth``
   that can be loaded via ``torch.load``.

Limitations
-----------
• Only supports **Dense** (fully-connected) layers.
• Expects activations among: relu, sigmoid, tanh, linear.
• Does **not** attempt to replicate optimiser / training state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import get_custom_objects

# Import custom DTypePolicy if the patched version exists ------------------
try:
    from dqn_agent import DTypePolicy  # type: ignore
except Exception:
    # Fallback: register a dummy alias so model loading still works.
    from tensorflow.keras.mixed_precision import Policy as _Policy

    class DTypePolicy(_Policy):
        pass

    get_custom_objects()["DTypePolicy"] = DTypePolicy  # type: ignore
else:
    get_custom_objects()["DTypePolicy"] = DTypePolicy  # type: ignore


ACTIVATION_MAP = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "linear": nn.Identity(),  # no-op
}


def build_torch_sequential(layer_sizes: List[int], activations: List[str]) -> nn.Sequential:
    """Create a torch ``nn.Sequential`` mirroring the Keras architecture."""
    layers: List[nn.Module] = []
    for in_dim, out_dim, act_name in zip(layer_sizes[:-1], layer_sizes[1:], activations):
        layers.append(nn.Linear(in_dim, out_dim))
        act = ACTIVATION_MAP.get(act_name.lower())
        if act is None:
            raise ValueError(f"Unsupported activation '{act_name}'. Supported: {list(ACTIVATION_MAP)}")
        # Skip final Identity if last layer is linear → keep output identical
        if not (act_name.lower() == "linear" and out_dim == layer_sizes[-1]):
            layers.append(act)
    return nn.Sequential(*layers)


def convert(keras_path: Path, torch_path: Path) -> None:
    # ------------------------------------------------------------------
    # 1. Load Keras model (no compilation)
    # ------------------------------------------------------------------
    model_keras = tf.keras.models.load_model(keras_path, compile=False)

    # Extract Dense layers & activations --------------------------------
    dense_layers = [l for l in model_keras.layers if isinstance(l, Dense)]
    if not dense_layers:
        raise RuntimeError("No Dense layers found – only fully-connected models are supported.")

    layer_sizes = [dense_layers[0].input_shape[-1]] + [l.units for l in dense_layers]
    activations = [l.activation.__name__ for l in dense_layers]

    # ------------------------------------------------------------------
    # 2. Build equivalent PyTorch model
    # ------------------------------------------------------------------
    model_torch = build_torch_sequential(layer_sizes, activations)

    # ------------------------------------------------------------------
    # 3. Transfer weights (kernel needs transposition)
    # ------------------------------------------------------------------
    torch_layers = [m for m in model_torch.modules() if isinstance(m, nn.Linear)]
    if len(torch_layers) != len(dense_layers):
        raise AssertionError("Layer count mismatch during transfer.")

    for k_layer, t_layer in zip(dense_layers, torch_layers):
        k_weight, k_bias = k_layer.get_weights()  # shapes (in, out) and (out,)
        t_layer.weight.data = torch.tensor(k_weight.T, dtype=torch.float32)
        t_layer.bias.data = torch.tensor(k_bias, dtype=torch.float32)

    model_torch.eval()

    # ------------------------------------------------------------------
    # 4. Save state_dict + meta
    # ------------------------------------------------------------------
    torch_path = torch_path.with_suffix(".pth")  # ensure .pth extension
    meta = {
        "layer_sizes": layer_sizes,
        "activations": activations,
    }
    torch.save({"state_dict": model_torch.state_dict(), "meta": meta}, torch_path)
    print(f"[✓] Saved PyTorch weights to {torch_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Convert a Keras Dense model to PyTorch.")
    p.add_argument("keras_model", type=Path, help="Path to .keras or .h5 file")
    p.add_argument("output", type=Path, help="Where to save converted .pth")
    return p.parse_args()


def main():
    args = _parse_args()
    convert(args.keras_model, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
