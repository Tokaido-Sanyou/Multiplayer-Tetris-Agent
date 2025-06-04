"""Evaluate a converted PyTorch DQN model inside the gym-style
``TetrisEnv``.

Usage
-----
$ python -m localMultiplayerTetris.eval_pytorch_dqn \
        --model tetris-ai-master/sample_torch.pth \
        --episodes 20 \
        --headless

Notes
-----
• Works with the ``.pth`` produced by ``convert_keras_to_torch.py``.
• The network is rebuilt with the same layer sizes/activations and the
  saved ``state_dict`` is loaded.
• ``dqn_adapter.enumerate_next_states`` is used at every step to brute-force
  legal placements; the model chooses the placement with the highest Q-value.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch

from .tetris_env import TetrisEnv
from .dqn_adapter import enumerate_next_states
from .convert_keras_to_torch import build_torch_sequential


def load_model(pth_file: Path, device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    """Rebuild and load the sequential DQN network from *pth_file*."""
    ckpt = torch.load(pth_file, map_location=device)
    meta = ckpt["meta"]
    state_dict = ckpt["state_dict"]
    model = build_torch_sequential(meta["layer_sizes"], meta["activations"])
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def evaluate(model: torch.nn.Module, episodes: int = 10, headless: bool = True) -> List[float]:
    device = next(model.parameters()).device
    env = TetrisEnv(single_player=True, headless=headless)
    episode_rewards: List[float] = []

    for ep in range(episodes):
        obs, _ = env.reset()
        # initial render when windowed
        if not headless:
            env.render()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            # Enumerate legal placements and their resulting 4-feature states
            mapping = enumerate_next_states(env)
            if not mapping:
                # No legal moves (should not normally happen) – resign
                break
            states = np.array(list(mapping.keys()), dtype=np.float32)
            with torch.no_grad():
                q_values = model(torch.from_numpy(states).to(device))
            best_idx = int(torch.argmax(q_values))
            best_action = list(mapping.values())[best_idx]

            obs, reward, terminated, truncated, _info = env.step(best_action)
            # render frame when windowed
            if not headless:
                env.render()
            total_reward += reward
            done = terminated or truncated
            step += 1
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}/{episodes}: reward={total_reward:.1f} steps={step}")

    env.close()
    avg = sum(episode_rewards) / len(episode_rewards)
    print(f"\nAverage reward over {episodes} episodes: {avg:.2f}")
    return episode_rewards


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate converted PyTorch DQN on TetrisEnv")
    p.add_argument("--model", type=Path, required=True, help="Path to .pth file from conversion script")
    p.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    p.add_argument("--cuda", action="store_true", help="Run on GPU if available")
    p.add_argument("--render", action="store_true", help="Render the game window (overrides --headless)")
    p.add_argument("--headless", action="store_true", help="Disable rendering")
    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model = load_model(args.model, device)
    # By default, render the game window; disable only if --headless set
    headless = args.headless
    # --render flag explicitly forces rendering
    if args.render:
        headless = False
    evaluate(model, episodes=args.episodes, headless=headless)


if __name__ == "__main__":  # pragma: no cover
    main()
