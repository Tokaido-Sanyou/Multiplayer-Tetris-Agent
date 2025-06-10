#!/usr/bin/env python3
"""
Debug script to test TetrisEnv random interactions: ensures lines can clear under random policy.
"""
from envs.tetris_env import TetrisEnv

def main(n_episodes=1000, max_steps=1000):
    env = TetrisEnv(reward_mode='lines_only', headless=True)
    total_actions = 0
    total_piece_placed = 0
    total_lines_cleared = 0
    total_game_overs = 0

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            total_actions += 1
            if info.get('piece_placed', False):
                total_piece_placed += 1
            total_lines_cleared += info.get('lines_cleared', 0)
            if info.get('game_over', False):
                total_game_overs += 1
            state = next_state
            steps += 1

    print(f"Random policy over {n_episodes} episodes:")
    print(f"  Total actions taken: {total_actions}")
    print(f"  Pieces placed: {total_piece_placed} ({total_piece_placed/total_actions*100 if total_actions>0 else 0:.2f}% of actions)")
    print(f"  Total lines cleared: {total_lines_cleared}")
    print(f"  Game overs: {total_game_overs} ({total_game_overs/n_episodes*100:.2f}% of episodes)")

if __name__ == '__main__':
    main(1000, 1000) 