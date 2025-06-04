"""Train an AIRL agent in Tetris using expert demonstrations.

Requires `imitation` and `stable-baselines3`.
Assumes trajectories collected with `collect_expert.py`.

Usage
-----
$ python -m localMultiplayerTetris.airl.train_airl \
        --demos expert_demos.npz \
        --timesteps 500_000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from imitation.algorithms.adversarial import airl
from imitation.data.types import TrajectoryWithRew
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.rewards.reward_nets import BasicRewardNet
from ..tetris_env import TetrisEnv
from datetime import datetime

# Try to import TensorBoard SummaryWriter gracefully
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
except Exception:
    SummaryWriter = None  # type: ignore


def load_demos(path: Path):
    data = np.load(path, allow_pickle=True)
    return list(data["trajectories"],)


def make_env(headless=True):
    # Ensure Gymnasium compliance
    class _Wrapper(TetrisEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.metadata = {"render_modes": ["human"], "render_fps": 10}

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                super().reset_seed(seed)
            return super().reset()

    return _Wrapper(headless=headless)


def train_airl(demo_path: Path, timesteps: int, out_dir: Path, tb_dir: Path | None = None):
    demonstrations: list[TrajectoryWithRew] = load_demos(demo_path)

    # Vectorized environment for SB3/imitation compatibility
    vec_env = DummyVecEnv([lambda: make_env(headless=True)])

    # TensorBoard directory
    if tb_dir is None:
        tb_dir = out_dir / "tb_logs"

    # Generator (learner) policy with PPO
    learner = PPO(
        "MultiInputPolicy",
        vec_env,
        batch_size=4096,
        n_steps=2048,
        learning_rate=3e-4,
        gamma=0.995,
        verbose=1,
        device="auto",
        tensorboard_log=str(tb_dir),
    )

    # Reward network
    reward_net = BasicRewardNet(vec_env.observation_space, vec_env.action_space)

    # SummaryWriter for custom logs (discriminator losses)
    writer = None
    if SummaryWriter is not None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=str(tb_dir / run_name))

    # AIRL trainer
    airl_trainer = airl.AIRL(
        demonstrations=demonstrations,
        demo_batch_size=256,
        venv=vec_env,
        gen_algo=learner,
        reward_net=reward_net,
    )

    total_steps = 0
    iter_steps = 10000
    while total_steps < timesteps:
        learner.learn(total_timesteps=iter_steps, reset_num_timesteps=False)

        # Train discriminator and optionally update reward net
        disc_stats = airl_trainer.train_disc()
        disc_loss = 0.0
        if isinstance(disc_stats, dict):
            disc_loss = float(disc_stats.get("disc_loss", disc_stats.get("loss", 0)))
        if hasattr(airl_trainer, "update_reward_net"):
            airl_trainer.update_reward_net()
        total_steps += iter_steps
        if total_steps % 100000 == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            learner.save(out_dir / f"ppo_generator_{total_steps}.zip")
            torch.save(airl_trainer.reward_net.state_dict(), out_dir / f"reward_net_{total_steps}.pth")
            print(f"Checkpoint saved @ {total_steps} timesteps")

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("airl/disc_loss", disc_loss, total_steps)
            writer.add_scalar("airl/total_steps", total_steps, total_steps)
            writer.flush()

    if writer is not None:
        writer.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=Path, required=True)
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--out", type=Path, default=Path("airl_checkpoints"))
    p.add_argument("--tb_dir", type=Path, default=None, help="TensorBoard log directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_airl(args.demos, args.timesteps, args.out, args.tb_dir)
