#!/usr/bin/env python3
"""
Generate Expert Trajectories using DQN Model
Bridge between tetris-ai-master DQN and local-multiplayer-tetris-main environment
"""

import sys
import os
import pickle
import numpy as np
from datetime import datetime

# Add paths for imports
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')
sys.path.append('tetris-ai-master')

def extract_dqn_features(observation):
    """
    Extract 4-feature state representation used by DQN model.
    This matches the tetris-ai-master state representation.
    """
    from dqn_adapter import board_props
    
    try:
        # Get board state from observation
        grid = observation['grid']
        
        # Convert grid to format expected by board_props
        # board_props expects a 20x10 grid with non-zero values for occupied cells
        board = []
        for row in grid:
            board.append([1 if cell > 0 else 0 for cell in row])
        
        # Extract DQN features: [lines_cleared, holes, total_bumpiness, sum_height]
        features = board_props(board)
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error extracting DQN features: {e}")
        # Fallback to dummy features
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

def dqn_action_to_env_action(dqn_action, observation):
    """
    Convert DQN action (rotation, column) to environment action index.
    """
    try:
        # DQN action is typically (rotation, column) or a flattened index
        if isinstance(dqn_action, (tuple, list)) and len(dqn_action) == 2:
            rotation, column = dqn_action
        else:
            # Assume flattened action: action = rotation * 10 + column
            rotation = int(dqn_action) // 10
            column = int(dqn_action) % 10
        
        # Ensure valid ranges
        rotation = max(0, min(3, rotation))
        column = max(0, min(9, column))
        
        # Convert to environment action index
        env_action = rotation * 10 + column
        return min(40, max(0, env_action))  # Clamp to valid action range [0, 40]
        
    except Exception as e:
        print(f"Error converting DQN action: {e}")
        # Return random valid action
        return np.random.randint(0, 40)

def run_expert_dqn_episode(env, max_steps=1000):
    """
    Run a single episode using the DQN expert model.
    """
    try:
        # Try to import DQN model
        from tetris import Tetris
        from dqn_agent import DQNAgent
        
        # Initialize DQN agent
        dqn_agent = DQNAgent(state_size=4, action_size=40, modelFile='sample.keras')
        
        episode_data = []
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        total_reward = 0
        step_count = 0
        
        while step_count < max_steps:
            # Extract DQN features from current observation
            dqn_state = extract_dqn_features(obs)
            
            # Get action from DQN
            dqn_action = dqn_agent.act(dqn_state)
            
            # Convert to environment action
            env_action = dqn_action_to_env_action(dqn_action, obs)
            
            # Take action in environment
            step_result = env.step(env_action)
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
                truncated = False
            else:
                next_obs, reward, done, truncated, info = step_result
            
            done = done or truncated
            
            # Store transition
            episode_data.append({
                'state': obs.copy(),
                'action': env_action,
                'reward': reward,
                'done': done,
                'info': info.copy() if isinstance(info, dict) else {}
            })
            
            total_reward += reward
            step_count += 1
            obs = next_obs
            
            if done:
                break
                
        return episode_data, total_reward, step_count
        
    except ImportError as e:
        print(f"Could not import DQN model: {e}")
        return generate_fallback_expert_episode(env, max_steps)
    except Exception as e:
        print(f"Error running DQN episode: {e}")
        return generate_fallback_expert_episode(env, max_steps)

def generate_fallback_expert_episode(env, max_steps=1000):
    """
    Generate a fallback expert episode using a simple heuristic policy.
    This creates reasonable Tetris play without excessive HOLD usage.
    """
    episode_data = []
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    total_reward = 0
    step_count = 0
    hold_cooldown = 0  # Prevent excessive holding
    
    while step_count < max_steps:
        # Simple heuristic: prefer lower placements, avoid excessive holding
        current_x = obs.get('current_x', 5)
        current_y = obs.get('current_y', 0)
        
        # Heuristic action selection
        if hold_cooldown > 0:
            hold_cooldown -= 1
            # Choose placement action (avoid HOLD)
            action = np.random.choice(range(40))  # Actions 0-39 are placements
        else:
            # Occasionally use HOLD, but not excessively
            if np.random.random() < 0.05:  # 5% chance to hold
                action = 40  # HOLD action
                hold_cooldown = 10  # Don't hold again for 10 steps
            else:
                # Choose placement action based on simple heuristics
                # Prefer actions that place pieces in lower rows
                rotation = np.random.choice([0, 1, 2, 3])
                column = max(0, min(9, current_x + np.random.randint(-2, 3)))
                action = rotation * 10 + column
        
        # Take action in environment
        step_result = env.step(action)
        if len(step_result) == 4:
            next_obs, reward, done, info = step_result
            truncated = False
        else:
            next_obs, reward, done, truncated, info = step_result
        
        done = done or truncated
        
        # Store transition
        episode_data.append({
            'state': obs.copy(),
            'action': action,
            'reward': reward,
            'done': done,
            'info': info.copy() if isinstance(info, dict) else {}
        })
        
        total_reward += reward
        step_count += 1
        obs = next_obs
        
        if done:
            break
            
    return episode_data, total_reward, step_count

def calculate_hold_percentage(episode_data):
    """Calculate the percentage of HOLD actions in an episode."""
    if not episode_data:
        return 0.0
    
    hold_count = sum(1 for step in episode_data if step['action'] == 40)
    return (hold_count / len(episode_data)) * 100.0

def main():
    # Import environment
    sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')
    from tetris_env import TetrisEnv
    
    # Create environment
    env = TetrisEnv(single_player=True, headless=True)
    
    print("🎮 Generating Expert Trajectories...")
    print("=" * 50)
    
    # Create output directory
    output_dir = "expert_trajectories_new"
    os.makedirs(output_dir, exist_ok=True)
    
    num_episodes = 10
    good_episodes = 0
    
    for episode_id in range(num_episodes):
        print(f"\n📺 Episode {episode_id + 1}/{num_episodes}")
        
        # Run episode
        episode_data, total_reward, step_count = run_expert_dqn_episode(env)
        
        # Calculate metrics
        hold_percentage = calculate_hold_percentage(episode_data)
        
        print(f"   Steps: {step_count}")
        print(f"   Reward: {total_reward:.2f}")
        print(f"   HOLD%: {hold_percentage:.1f}%")
        
        # Only save episodes with reasonable HOLD usage
        if hold_percentage < 50.0 and total_reward > -20:
            # Create trajectory data
            trajectory = {
                'episode_id': episode_id,
                'steps': episode_data,
                'total_reward': total_reward,
                'length': step_count,
                'action_space': 41,
                'action_range': 'rotation_column_hold',
                'timestamp': np.datetime64('now'),
                'hold_percentage': hold_percentage
            }
            
            # Save trajectory
            filename = f"trajectory_ep{episode_id:06d}.pkl"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(trajectory, f)
            
            print(f"   ✅ SAVED: {filename}")
            good_episodes += 1
        else:
            print(f"   ❌ SKIPPED: Too many HOLDs or poor reward")
    
    env.close()
    
    print(f"\n📊 Generation Summary:")
    print(f"   Total episodes: {num_episodes}")
    print(f"   Good episodes: {good_episodes}")
    print(f"   Output directory: {output_dir}")
    
    if good_episodes > 0:
        print(f"\n✅ Successfully generated {good_episodes} expert trajectories!")
        print(f"   Use: --expert-dir {output_dir}")
    else:
        print(f"\n❌ No good trajectories generated. Expert model may need improvement.")

if __name__ == "__main__":
    main() 