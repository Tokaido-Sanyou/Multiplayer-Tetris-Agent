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
import dataclasses
import numpy as np
import torch
from datetime import datetime

from gym.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms.adversarial import airl
from imitation.data.types import TrajectoryWithRew
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.logger import configure

from ..tetris_env import TetrisEnv


def load_demos(path: Path) -> list[TrajectoryWithRew]:
    data = np.load(path, allow_pickle=True)
    trajs = list(data["trajectories"])
    fixed: list[TrajectoryWithRew] = []
    for traj in trajs:
        term = traj.terminal
        if not isinstance(term, (bool, np.bool_)):
            arr = np.asarray(term)
            term_scalar = bool(arr.item()) if arr.size == 1 else bool(arr[-1])
            traj = dataclasses.replace(traj, terminal=term_scalar)  # type: ignore[arg-type]
        fixed.append(traj)
    return fixed


def make_env(headless: bool = True):
    class _Wrapper(TetrisEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.metadata = {"render_modes": ["human"], "render_fps": 10}

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                super().reset_seed(seed)
            return super().reset()

    return _Wrapper(headless=headless)


def train_airl(
    demo_path: Path,
    timesteps: int,
    out_dir: Path,
    tb_dir: Path | None = None,
):
    # Load and fix expert demos
    demonstrations = load_demos(demo_path)

    # FlattenObservation for the generator AND expert data
    flat_wrapper = FlattenObservation(make_env(headless=True))
    # manually flatten each expert trajectory’s obs
    flattened = []
    for traj in demonstrations:
        flat_obs = np.stack([flat_wrapper.observation(o) for o in traj.obs], axis=0)
        flattened.append(dataclasses.replace(traj, obs=flat_obs))
    demonstrations = flattened

    # VecEnv for PPO, with flat vectors
    venv = DummyVecEnv([lambda: FlattenObservation(make_env(headless=True))])

    # TensorBoard setup
    if tb_dir is None:
        tb_dir = out_dir / "tb_logs"
    logger = configure(folder=str(tb_dir), format_strs=["stdout", "tensorboard"])

    # 1) Generator: PPO with extra entropy & more epochs
    learner = PPO(
        "MlpPolicy",
        venv,
        batch_size=4096,
        n_steps=2048,
        learning_rate=3e-4,
        ent_coef=0.01,         # encourage exploration :contentReference[oaicite:3]{index=3}
        n_epochs=15,           # more policy-gradient epochs per batch
        gamma=0.995,
        verbose=1,
        device="auto",
        tensorboard_log=str(tb_dir),
    )

    # 2) Reward net for the AIRL discriminator
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # 3) AIRL trainer: fewer, slower discriminator updates
    airl_trainer = airl.AIRL(
        demonstrations=demonstrations,
        demo_batch_size=2048,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        log_dir=str(tb_dir),
        init_tensorboard=True,
        init_tensorboard_graph=True,
        custom_logger=logger,
        allow_variable_horizon=True,
        n_disc_updates_per_round=1,          # only 1 disc update per round :contentReference[oaicite:4]{index=4}
        disc_opt_kwargs={"lr": 1e-6},        # disc LR ~1e-6, ~3 orders lower :contentReference[oaicite:5]{index=5}
    )

    # 4) Run the full adversarial loop
    airl_trainer.train(total_timesteps=timesteps)

    # Save final checkpoints
    out_dir.mkdir(parents=True, exist_ok=True)
    learner.save(out_dir / "ppo_generator_final.zip")
    torch.save(airl_trainer.reward_net.state_dict(), out_dir / "reward_net_final.pth")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=Path, required=True)
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--out", type=Path, default=Path("airl_checkpoints"))
    p.add_argument(
        "--tb_dir",
        type=Path,
        default=None,
        help="TensorBoard log directory",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_airl(args.demos, args.timesteps, args.out, args.tb_dir)
