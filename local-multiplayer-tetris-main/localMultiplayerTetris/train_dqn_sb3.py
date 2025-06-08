"""Train a Deep Q-Network (DQN) agent on the single-player `TetrisEnv`.

This script uses Stable-Baselines3's DQN implementation with a simple
multi-layer perceptron (MLP) policy that operates on a **flattened 206-dim
float vector** observation (20×10 board + 6 scalars).

Usage
-----
$ python -m localMultiplayerTetris.train_dqn_sb3 \
        --timesteps 1_000_000 \
        --out weights/dqn_new \
        --headless            # run without rendering

TensorBoard logs are written to ``--tb_dir``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter  # pylint: disable=import-error

from .tetris_env import TetrisEnv

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _dict_to_vec(obs: dict[str, np.ndarray | int]) -> np.ndarray:  # 206-dim vec
    """Convert Dict observation from ``TetrisEnv`` to flat float32 vector."""
    grid = np.asarray(obs["grid"], dtype=np.float32).flatten() / 2.0  # scale 0-1
    scalars = np.array(
        [
            obs["next_piece"],
            obs["hold_piece"],
            obs["current_shape"],
            obs["current_rotation"],
            obs["current_x"],
            obs["current_y"],
        ],
        dtype=np.float32,
    )
    return np.concatenate([grid, scalars])


class VectorObsWrapper(gym.ObservationWrapper):
    """Wrap `TetrisEnv` to return flat 206-dim observations compatible with SB3."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(206,), dtype=np.float32
        )
        # SB3 expects Gymnasium space types for both obs and actions.
        self.action_space = gym.spaces.Discrete(env.action_space.n)

    def observation(self, obs):  # type: ignore[override]
        return _dict_to_vec(obs)


def make_env(headless: bool = True) -> gym.Env:
    """Create the wrapped single-player environment."""
    base_env = TetrisEnv(single_player=True, headless=headless)
    return VectorObsWrapper(base_env)


# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------

def train_dqn(timesteps: int, out_dir: Path, tb_dir: Path | None, headless: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if tb_dir is None:
        tb_dir = out_dir / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Vectorised env (single env) – required for SB3 DQN
    vec_env = DummyVecEnv([lambda: make_env(headless=headless)])

    # Model configuration – tweak as desired
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,
        batch_size=1024,
        buffer_size=200_000,
        learning_starts=10_000,
        target_update_interval=5_000,
        train_freq=4,
        gamma=0.995,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        tensorboard_log=str(tb_dir),
        verbose=1,
        device="auto",
    )

    # Optional custom TensorBoard writer (for extra metrics)
    writer: SummaryWriter | None = SummaryWriter(log_dir=str(tb_dir / "custom"))

    # Train
    model.learn(total_timesteps=timesteps)

    # Save model weights (PyTorch format)
    save_path = out_dir / "dqn_tetris.pth"
    torch.save(model.policy.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

    if writer is not None:
        writer.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train a DQN agent on TetrisEnv")
    p.add_argument("--timesteps", type=int, default=1_000_000, help="Total environment steps to train")
    p.add_argument("--out", type=Path, default=Path("weights/dqn_new"), help="Output directory for checkpoints")
    p.add_argument("--tb_dir", type=Path, default=None, help="TensorBoard log directory (default: OUT/tb_logs)")
    p.add_argument("--render", action="store_true", help="Render the game window during training")
    return p.parse_args()


def main():  # pragma: no cover
    args = _parse_args()
    headless = not args.render
    train_dqn(args.timesteps, args.out, args.tb_dir, headless=headless)


if __name__ == "__main__":  # pragma: no cover
    main()
