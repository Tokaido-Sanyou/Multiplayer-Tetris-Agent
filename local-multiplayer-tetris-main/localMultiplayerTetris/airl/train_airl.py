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

from gym.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

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
    # 1) Load and fix expert demos
    demos = load_demos(demo_path)

    # 2) FlattenObservation wrapper for expert data
    flat_wrapper = FlattenObservation(make_env(headless=True))
    flattened: list[TrajectoryWithRew] = []
    for traj in demos:
        # stack flattened obs per timestep → array of shape (T+1, obs_dim)
        flat_obs = np.stack([flat_wrapper.observation(o) for o in traj.obs], axis=0)
        flattened.append(dataclasses.replace(traj, obs=flat_obs))
    demos = flattened

    # 3) Build the generator VecEnv with FlattenObservation + reward normalization
    raw_venv = DummyVecEnv([lambda: FlattenObservation(make_env(headless=True))])
    venv = VecNormalize(raw_venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 4) TensorBoard directory & logger
    if tb_dir is None:
        tb_dir = out_dir / "tb_logs"
    logger = configure(folder=str(tb_dir), format_strs=["stdout", "tensorboard"])

    # 5) PPO generator with stability & exploration tweaks
    learner = PPO(
        "MlpPolicy",
        venv,
        batch_size=4096,
        n_steps=2048,
        learning_rate=1e-4,
        ent_coef=0.01,
        vf_coef=0.25,
        clip_range=0.2,
        clip_range_vf=0.2,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        n_epochs=15,
        gamma=0.995,
        verbose=1,
        device="auto",
        tensorboard_log=str(tb_dir),
    )

    # 6) AIRL reward network (discriminator)
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # 7) AIRL trainer: slow & regularize discriminator
    airl_trainer = airl.AIRL(
        demonstrations=demos,
        demo_batch_size=2048,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        log_dir=str(tb_dir),
        init_tensorboard=True,
        init_tensorboard_graph=True,
        custom_logger=logger,
        allow_variable_horizon=True,
        n_disc_updates_per_round=1,
        disc_opt_kwargs={
            "lr": 1e-6,
            "weight_decay": 1e-5,
        },
        reward_net_kwargs={
            "grad_norm_clipping": 0.5,
        },
    )

    # 8) CheckpointCallback: save PPO generator every 100k steps
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = CheckpointCallback(
        save_freq=100_000 // learner.n_steps,
        save_path=str(out_dir),
        name_prefix="ppo_generator",
    )

    # 9) Warm up generator under initial reward for 100k steps
    warmup = 100_000
    airl_trainer.train_gen(
        total_timesteps=warmup,
        callback=ckpt_cb,
    )

    # 10) Run the full AIRL loop for the remaining timesteps
    airl_trainer.train(
        total_timesteps=timesteps - warmup,
        callback=ckpt_cb,
    )

    # 11) Final save of both generator and reward-net
    learner.save(out_dir / "ppo_generator_final.zip")
    torch.save(
        airl_trainer.reward_net.state_dict(),
        out_dir / "reward_net_final.pth",
    )


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
