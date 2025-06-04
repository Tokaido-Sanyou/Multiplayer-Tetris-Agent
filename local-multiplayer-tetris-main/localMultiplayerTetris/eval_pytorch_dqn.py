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
from torch import nn

from .tetris_env import TetrisEnv
from .dqn_adapter import enumerate_next_states
from .utils import add_garbage_line
from .constants import s_width  # kept for potential future positioning

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


def _garbage_from_lines(cleared: int) -> int:
    """Translate *lines_cleared* to garbage lines sent to opponent."""
    return {1: 0, 2: 1, 3: 2, 4: 4}.get(cleared, 0)


# ---------------------------------------------------------------------------
# Two-player battle
# ---------------------------------------------------------------------------

def _mirror_b_to_a(env_a: TetrisEnv, env_b: TetrisEnv):
    """Copy Player B's visible state from *env_b* into *env_a*'s player2 so they render together."""
    env_a.game.player2.locked_positions = dict(env_b.player.locked_positions)
    env_a.game.player2.current_piece = env_b.player.current_piece
    env_a.game.player2.next_pieces = env_b.player.next_pieces
    env_a.game.player2.hold_piece = env_b.player.hold_piece
    env_a.game.player2.score = env_b.player.score

def evaluate_duo(model_a: torch.nn.Module, model_b: torch.nn.Module, *, episodes: int = 10, headless: bool = True):
    """Pit two DQN agents against each other, sending garbage lines."""
    device = next(model_a.parameters()).device
    assert device == next(model_b.parameters()).device

    results = []
    for ep in range(episodes):
        # Single render window via env_a; env_b is headless but updates logic
        env_a = TetrisEnv(single_player=False, headless=headless)
        env_b = TetrisEnv(single_player=False, headless=True)  # always headless

        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()

        done_a = done_b = False
        step = 0
        while True:
            # --- Agent A turn -------------------------------------------------
            if not done_a:
                mapping = enumerate_next_states(env_a)
                if not mapping:
                    done_a = True
                else:
                    states = np.array(list(mapping.keys()), dtype=np.float32)
                    with torch.no_grad():
                        q_vals = model_a(torch.from_numpy(states).to(device))
                    best_idx = int(torch.argmax(q_vals))
                    action = list(mapping.values())[best_idx]
                    _, _, term, trunc, info = env_a.step(action)
                    lines = info.get("lines_cleared", 0)
                    garbage = _garbage_from_lines(lines)
                    if garbage > 0 and not done_b:
                        add_garbage_line(env_b.player.locked_positions, garbage)
                    done_a = term or trunc

            # End episode immediately if Player A died
            if done_a:
                break

            # --- Agent B turn -------------------------------------------------
            if not done_b:
                mapping = enumerate_next_states(env_b)
                if not mapping:
                    done_b = True
                else:
                    states = np.array(list(mapping.keys()), dtype=np.float32)
                    with torch.no_grad():
                        q_vals = model_b(torch.from_numpy(states).to(device))
                    best_idx = int(torch.argmax(q_vals))
                    action = list(mapping.values())[best_idx]
                    _, _, term, trunc, info = env_b.step(action)
                    lines = info.get("lines_cleared", 0)
                    garbage = _garbage_from_lines(lines)
                    if garbage > 0 and not done_a:
                        add_garbage_line(env_a.player.locked_positions, garbage)
                    done_b = term or trunc

            # End episode immediately if Player B died
            if done_b:
                break

            # Mirror env_b to env_a so both boards appear side-by-side
            _mirror_b_to_a(env_a, env_b)
            # Optionally render single window
            if not headless:
                env_a.render()
            step += 1

        score_a = env_a.player.score
        score_b = env_b.player.score
        results.append((score_a, score_b))
        print(f"Episode {ep+1}: A_score={score_a}  B_score={score_b}  steps={step}")

        env_a.close()
        env_b.close()

    avg_a = sum(s for s, _ in results) / episodes
    avg_b = sum(s for _, s in results) / episodes
    print(f"\nAverage  AgentA: {avg_a:.1f}   AgentB: {avg_b:.1f}")
    return results


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
    p.add_argument("--two_player", action="store_true", help="Play two agents against each other")
    p.add_argument("--model2", type=Path, help="Optional second model; defaults to --model")
    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # By default, render the game window; disable only if --headless set
    headless = args.headless
    if args.render:
        headless = False

    if args.two_player:
        model_a = load_model(args.model, device)
        model_b = load_model(args.model2 or args.model, device)
        evaluate_duo(model_a, model_b, episodes=args.episodes, headless=headless)
    else:
        model = load_model(args.model, device)
        evaluate(model, episodes=args.episodes, headless=headless)


if __name__ == "__main__":  # pragma: no cover
    main()
