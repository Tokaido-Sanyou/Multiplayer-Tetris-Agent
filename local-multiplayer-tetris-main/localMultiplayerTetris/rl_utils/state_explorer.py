"""
Exploratory state-action transition collector for Tetris Env.
"""
import pickle
from collections import defaultdict

def is_invalid(info):
    # Identify invalid transitions based on env.info flags
    return info.get('invalid_placement', False) or info.get('spawn_collision', False)

class StateExplorer:
    def __init__(self, save_dir='exploration_data'):
        self.save_dir = save_dir
        # Each entry: (initial_obs, action, next_obs, valid)
        self.initial_transitions = []

    def collect_initial_transitions(self, env):
        """
        For each action in the action space, reset env, record initial observation,
        perform action, and record resulting observation and validity.
        """
        self.initial_transitions.clear()
        for action in range(env.action_space.n):
            obs = env.reset()
            next_obs, reward, done, info = env.step(action)
            valid = not is_invalid(info)
            # store (initial_obs, action, next_obs, validity)
            self.initial_transitions.append((obs, action, next_obs, valid))
        return self.initial_transitions

    def save(self, filename):
        """
        Persist collected transitions to disk as a pickle file.
        """
        out = {
            'initial_transitions': self.initial_transitions
        }
        with open(f"{self.save_dir}/{filename}", 'wb') as f:
            pickle.dump(out, f)

    def summary(self):
        """
        Returns a summary dict of valid vs invalid counts per action.
        """
        counts = defaultdict(lambda: {'valid': 0, 'invalid': 0})
        # Unpack (obs, action, next_obs, valid)
        for _, action, _, valid in self.initial_transitions:
            if valid:
                counts[action]['valid'] += 1
            else:
                counts[action]['invalid'] += 1
        return counts
