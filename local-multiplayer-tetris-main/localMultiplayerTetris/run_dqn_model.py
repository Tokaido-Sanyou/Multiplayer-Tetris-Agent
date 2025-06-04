"""Run the pre-trained DQN model from *tetris-ai-master* inside the
modern Gym environment.

Usage
-----
$ python -m localMultiplayerTetris.run_dqn_model /path/to/model.keras

If no path is given it will try ``tetris-ai-master/sample.keras`` relative
to the repo root.
"""
from __future__ import annotations

import sys
import os
import argparse
import importlib.util

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects

from dqn_agent import DQNAgent, DTypePolicy  # type: ignore

# Workaround for legacy DTypePolicy in saved models
get_custom_objects()['DTypePolicy'] = DTypePolicy

# Ensure tetris-ai-master is on the path ---------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)  # .../local-multiplayer-tetris-main
TAI_DIR = os.path.join(PROJECT_DIR, "tetris-ai-master")

# Fallback: if not found try one level higher (for editable installs)
if not os.path.isdir(TAI_DIR):
    TAI_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "tetris-ai-master")

if os.path.isdir(TAI_DIR) and TAI_DIR not in sys.path:
    sys.path.append(TAI_DIR)

from .tetris_env import TetrisEnv
from .dqn_adapter import enumerate_next_states


def main():
    parser = argparse.ArgumentParser(description="Play a pre-trained DQN model inside modern TetrisEnv")
    parser.add_argument("model", nargs="?", default=os.path.join(TAI_DIR, "sample.keras"), help="Path to .keras model file")
    parser.add_argument("--headless", action="store_true", help="Disable rendering for maximum speed")
    parser.add_argument("--render_every", type=int, default=0, help="Render every N moves (0 = never, 1 = every move)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    # ------------------------------------------------------------------
    env = TetrisEnv(single_player=True, headless=args.headless)
    # Init agent and load pretrained Keras model bypassing internal loader
    agent = DQNAgent(state_size=4)
    agent.model = tf.keras.models.load_model(args.model, compile=False)

    obs, _ = env.reset()
    done = False
    step_counter = 0
    try:
        while not done:
            next_states = enumerate_next_states(env)

            # Vectorised selection for speed --------------------------------
            state_array = np.array(list(next_states.keys()))
            values = agent.model.predict(state_array, verbose=0).flatten()
            best_idx = int(np.argmax(values))
            action = list(next_states.values())[best_idx]

            # Fallback if predict fails
            # if np.isnan(values).any():
            #     best_state = agent.best_state(next_states.keys())
            #     action = next_states[best_state]

            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            if not args.headless and (args.render_every == 1 or (args.render_every and step_counter % args.render_every == 0)):
                env.render()
            step_counter += 1
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
