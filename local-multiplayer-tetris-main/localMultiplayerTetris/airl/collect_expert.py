"""Collect expert demonstrations from a pre-trained DQN agent.

The script saves trajectories in the format expected by `imitation`.
Usage
-----
$ python -m localMultiplayerTetris.airl.collect_expert \
        --model weights/dqn_mod.pth \
        --output demos_expert.npz \
        --episodes 200
"""
from __future__ import annotations

import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List
import numpy as np
import torch
from imitation.data.types import TrajectoryWithRew

from ..tetris_env import TetrisEnv
from ..dqn_adapter import enumerate_next_states
from ..eval_pytorch_dqn import build_torch_sequential  # util to rebuild network
import multiprocessing as mp
from functools import partial


def load_model(path: Path, device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    """Load saved PyTorch model along with architecture metadata."""
    data = torch.load(path, map_location=device)
    meta = data["meta"]
    model = build_torch_sequential(meta["layer_sizes"], meta["activations"])
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model.to(device)


def dqn_action(model: torch.nn.Module, env: TetrisEnv, device: torch.device) -> int:
    mapping = enumerate_next_states(env)
    states = np.array(list(mapping.keys()), dtype=np.float32)
    with torch.no_grad():
        q_vals = model(torch.from_numpy(states).to(device))
    best_idx = int(torch.argmax(q_vals))
    return list(mapping.values())[best_idx]


def _collect_episodes(model_path: Path, episodes: int, device_str: str, headless: bool, max_steps: int = 2500):
    """Worker: collect *episodes* trajectories and return list[TrajectoryWithRew]."""
    device = torch.device(device_str)
    model = load_model(model_path, device)
    env = TetrisEnv(headless=headless, max_steps=max_steps)

    trajectories: List[TrajectoryWithRew] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        traj_obs, traj_acts, traj_next_obs, traj_rews = [], [], [], []
        done = False
        while not done:
            act = dqn_action(model, env, device)
            next_obs, rew, term, trunc, _ = env.step(act)
            traj_obs.append(obs)
            traj_acts.append(act)
            traj_next_obs.append(next_obs)
            traj_rews.append(rew)
            done_flag = term or trunc
            obs = next_obs
            done = done_flag

        # Add final observation after episode ends so len(obs)=len(acts)+1
        traj_obs.append(obs)

        trajectories.append(
            TrajectoryWithRew(
                obs=np.asarray(traj_obs, dtype=object),
                acts=np.asarray(traj_acts, dtype=np.int64),
                infos=None,
                terminal=True,
                rews=np.asarray(traj_rews, dtype=np.float32),
            )
        )
    env.close()
    return trajectories


def collect_trajectories_parallel(model_path: Path, episodes: int, output: Path, device: torch.device, workers: int = 4):
    """Spawn *workers* processes to gather trajectories in parallel."""
    if workers <= 1:
        # Single process with progress bar
        pbar = tqdm(total=episodes, desc="Collecting")
        all_traj = []
        for _ in range(episodes):
            trajs = _collect_episodes(model_path, 1, str(device), True, 150)
            all_traj.extend(trajs)
            pbar.update(1)
        pbar.close()
    else:
        chunks = [episodes // workers] * workers
        for i in range(episodes % workers):
            chunks[i] += 1

        with mp.get_context("spawn").Pool(processes=workers) as pool:
            func = partial(_collect_episodes, model_path, device_str=str(device), headless=True, max_steps=150)
            pbar = tqdm(total=episodes, desc="Collecting")
            all_traj = []
            for sub in pool.imap_unordered(func, chunks):
                all_traj.extend(sub)
                pbar.update(len(sub))
            pbar.close()

    np.savez_compressed(output, trajectories=all_traj)
    print(f"Saved {len(all_traj)} trajectories → {output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="Path to expert DQN .pth file")
    p.add_argument("--output", type=Path, default=Path("expert_demos.npz"))
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel processes")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    collect_trajectories_parallel(args.model, args.episodes, args.output, device, args.workers)


if __name__ == "__main__":
    main()
