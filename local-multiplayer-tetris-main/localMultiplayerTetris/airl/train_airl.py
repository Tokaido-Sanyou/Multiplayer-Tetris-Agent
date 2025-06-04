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
import gymnasium as gym

# Try to import TensorBoard SummaryWriter gracefully
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
except Exception:
    SummaryWriter = None  # type: ignore


def load_demos(path: Path):
    """Load demonstrations and sanitize terminal field to be bool.

    Older collections stored `terminal` as an array; convert to a single bool.
    """
    data = np.load(path, allow_pickle=True)
    trajs = list(data["trajectories"],)

    # Helper to flatten observation dict -> 206 dim vector
    def _dict_to_vec(obs: dict[str, np.ndarray | int]) -> np.ndarray:
        grid = np.asarray(obs["grid"], dtype=np.float32).flatten() / 2.0
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

    cleaned: list[TrajectoryWithRew] = []
    for traj in trajs:
        term = traj.terminal
        if not isinstance(term, (bool, np.bool_)):
            # Expect last value indicates episode end
            term_bool = bool(np.asarray(term).flatten()[-1])
            obs_vecs = np.stack([_dict_to_vec(o) for o in traj.obs])
            traj = TrajectoryWithRew(
                obs=obs_vecs,
                acts=traj.acts,
                infos=None,
                terminal=term_bool,
                rews=traj.rews,
            )
        else:
            # Already bool terminal, but ensure obs vectors
            if traj.obs.ndim == 1 or traj.obs.dtype == object:
                obs_vecs = np.stack([_dict_to_vec(o) for o in traj.obs])
                traj = TrajectoryWithRew(
                    obs=obs_vecs,
                    acts=traj.acts,
                    infos=None,
                    terminal=traj.terminal,
                    rews=traj.rews,
                )
        cleaned.append(traj)
    return cleaned


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

    env = _Wrapper(headless=headless)

    # Custom flatten wrapper to convert Dict obs -> vector (206 dims)
    class VectorObsWrapper(gym.ObservationWrapper):
        def __init__(self, env_inner):
            super().__init__(env_inner)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(206,), dtype=np.float32
            )
            # Convert action space to gymnasium type
            self.action_space = gym.spaces.Discrete(env_inner.action_space.n)

        def observation(self, obs):
            grid = obs["grid"].astype(np.float32).flatten() / 2.0  # scale to 0-1
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

    return VectorObsWrapper(env)


def train_airl(demo_path: Path, timesteps: int, out_dir: Path, tb_dir: Path | None = None):
    demonstrations: list[TrajectoryWithRew] = load_demos(demo_path)

    # Vectorized environment for SB3/imitation compatibility
    vec_env = DummyVecEnv([lambda: make_env(headless=True)])

    # TensorBoard directory
    if tb_dir is None:
        tb_dir = out_dir / "tb_logs"

    # Generator (learner) policy with PPO
    learner = PPO(
        "MlpPolicy",
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
        allow_variable_horizon=True,
    )

    total_steps = 0
    iter_steps = 10000
    while total_steps < timesteps:
        # Train generator (policy) and collect fresh samples
        airl_trainer.train_gen(iter_steps)

        # Train discriminator using latest samples
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
            _rw = getattr(airl_trainer, "reward_net", getattr(airl_trainer, "_reward_net", None))
            if _rw is not None:
                torch.save(_rw.state_dict(), out_dir / f"reward_net_{total_steps}.pth")
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
