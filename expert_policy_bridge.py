#!/usr/bin/env python3
"""
Expert Policy Bridge
Load sample.keras directly and generate expert trajectories for local-multiplayer-tetris-main
"""

import sys
import os
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf

# Add paths
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')

def load_expert_keras_model():
    """Load the pre-trained keras model directly."""
    model_path = 'tetris-ai-master/sample.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"🔧 Loading keras model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    return model

def calculate_board_props(grid):
    """
    Calculate the 4-feature state representation used by tetris-ai-master DQN.
    This replicates the _get_board_props method from tetris.py
    
    Args:
        grid: 20x10 numpy array (0=empty, >0=filled)
    
    Returns:
        [lines_cleared, holes, total_bumpiness, sum_height]
    """
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20
    
    # Convert grid to tetris-ai-master format (0=empty, 1=block)
    board = [[1 if cell > 0 else 0 for cell in row] for row in grid]
    
    # 1. Clear lines (simulate line clearing)
    lines_to_clear = [i for i, row in enumerate(board) if sum(row) == BOARD_WIDTH]
    lines_cleared = len(lines_to_clear)
    
    # Remove cleared lines and add empty lines at top
    if lines_to_clear:
        board = [row for i, row in enumerate(board) if i not in lines_to_clear]
        for _ in lines_to_clear:
            board.insert(0, [0 for _ in range(BOARD_WIDTH)])
    
    # 2. Count holes (empty cells below filled cells)
    holes = 0
    for col in range(BOARD_WIDTH):
        found_block = False
        for row in range(BOARD_HEIGHT):
            if board[row][col] == 1:
                found_block = True
            elif found_block and board[row][col] == 0:
                holes += 1
    
    # 3. Calculate bumpiness (height differences)
    heights = []
    for col in range(BOARD_WIDTH):
        height = 0
        for row in range(BOARD_HEIGHT):
            if board[row][col] == 1:
                height = BOARD_HEIGHT - row
                break
        heights.append(height)
    
    total_bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(BOARD_WIDTH-1))
    
    # 4. Calculate sum height
    sum_height = sum(heights)
    
    return np.array([lines_cleared, holes, total_bumpiness, sum_height], dtype=np.float32)

def get_next_states_from_env_obs(observation):
    """
    Generate all possible next states from local-multiplayer-tetris-main observation.
    This replicates the get_next_states() logic from tetris.py but works with our env.
    
    Args:
        observation: Dict from TetrisEnv observation
        
    Returns:
        Dict mapping (x, rotation) -> state_vector
    """
    grid = observation['grid']
    current_shape = observation['current_shape'] - 1  # Convert to 0-based indexing
    
    # Tetris piece definitions (from tetris-ai-master)
    TETROMINOS = {
        0: {  # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: {  # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: {  # L
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: {  # J
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: {  # Z
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: {  # S
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: {  # O
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }
    
    if current_shape < 0 or current_shape >= len(TETROMINOS):
        return {}
    
    states = {}
    piece_id = current_shape
    
    # Determine rotations for this piece
    if piece_id == 6:  # O piece
        rotations = [0]
    elif piece_id == 0:  # I piece
        rotations = [0, 90]
    else:
        rotations = [0, 90, 180, 270]
    
    # For each rotation
    for rotation in rotations:
        if rotation not in TETROMINOS[piece_id]:
            continue
            
        piece_blocks = TETROMINOS[piece_id][rotation]
        min_x = min(p[0] for p in piece_blocks)
        max_x = max(p[0] for p in piece_blocks)
        
        # For each valid column position
        for x in range(-min_x, 10 - max_x):
            # Simulate dropping the piece
            y = 0
            while True:
                # Check collision
                collision = False
                for block_x, block_y in piece_blocks:
                    board_x = x + block_x
                    board_y = y + block_y
                    
                    if (board_x < 0 or board_x >= 10 or 
                        board_y >= 20 or 
                        (board_y >= 0 and grid[board_y][board_x] > 0)):
                        collision = True
                        break
                
                if collision:
                    y -= 1
                    break
                y += 1
            
            # If valid placement
            if y >= 0:
                # Create hypothetical board with piece placed
                new_grid = grid.copy()
                for block_x, block_y in piece_blocks:
                    board_x = x + block_x
                    board_y = y + block_y
                    if 0 <= board_y < 20 and 0 <= board_x < 10:
                        new_grid[board_y][board_x] = 1  # Mark as filled
                
                # Calculate state features
                state_vector = calculate_board_props(new_grid)
                states[(x, rotation)] = state_vector
    
    return states

def dqn_policy_action(model, observation):
    """
    Use the DQN model to select the best action for the given observation.
    
    Args:
        model: Loaded keras model
        observation: TetrisEnv observation dict
        
    Returns:
        action: Action index for local-multiplayer-tetris-main (0-40)
    """
    # Get all possible next states
    next_states = get_next_states_from_env_obs(observation)
    
    if not next_states:
        # Fallback: HOLD action
        return 40
    
    # Evaluate each state with the DQN model
    best_action = None
    best_value = float('-inf')
    
    for (x, rotation), state_vector in next_states.items():
        # Predict value for this state
        state_input = state_vector.reshape(1, -1)  # Shape: (1, 4)
        value = model.predict(state_input, verbose=0)[0][0]
        
        if value > best_value:
            best_value = value
            best_action = (x, rotation)
    
    if best_action is None:
        return 40  # HOLD
    
    # Convert (x, rotation) to action index
    x, rotation = best_action
    
    # Map rotation degrees to rotation index
    rotation_map = {0: 0, 90: 1, 180: 2, 270: 3}
    rotation_idx = rotation_map.get(rotation, 0)
    
    # Action index = rotation * 10 + column
    action = rotation_idx * 10 + x
    
    # Clamp to valid range
    return min(39, max(0, action))

def run_expert_episode(env, model, max_steps=1000):
    """
    Run a single episode using the DQN expert policy.
    
    Args:
        env: TetrisEnv instance
        model: Loaded keras model
        max_steps: Maximum steps per episode
        
    Returns:
        episode_data, total_reward, step_count
    """
    episode_data = []
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    total_reward = 0
    step_count = 0
    
    while step_count < max_steps:
        # Get action from DQN policy
        action = dqn_policy_action(model, obs)
        
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
            'info': info.copy() if isinstance(info, dict) else {},
            'next_state': next_obs.copy() if not done else None
        })
        
        total_reward += reward
        step_count += 1
        obs = next_obs
        
        if done:
            break
    
    return episode_data, total_reward, step_count

def main():
    """Generate expert trajectories using the pre-trained DQN policy."""
    print("🏆 EXPERT POLICY BRIDGE - DQN TO LOCAL MULTIPLAYER")
    print("=" * 70)
    
    try:
        # Load the expert model
        model = load_expert_keras_model()
        
        # Import environment
        from tetris_env import TetrisEnv
        env = TetrisEnv(single_player=True, headless=True)
        
        # Create output directory
        output_dir = "expert_trajectories_dqn_bridge"
        os.makedirs(output_dir, exist_ok=True)
        
        num_episodes = 15
        successful_episodes = 0
        
        print(f"\n🎮 Generating {num_episodes} expert episodes...")
        
        for episode_id in range(num_episodes):
            print(f"\n📺 Episode {episode_id + 1}/{num_episodes}")
            
            # Run episode with DQN policy
            episode_data, total_reward, step_count = run_expert_episode(env, model)
            
            # Calculate metrics
            actions = [step['action'] for step in episode_data]
            hold_count = sum(1 for a in actions if a == 40)
            hold_percentage = (hold_count / len(actions)) * 100 if actions else 0
            
            print(f"   Steps: {step_count}")
            print(f"   Reward: {total_reward:.1f}")
            print(f"   HOLD%: {hold_percentage:.1f}%")
            
            # Quality criteria
            if total_reward >= -50 and hold_percentage <= 50:
                # Create trajectory
                trajectory = {
                    'episode_id': episode_id,
                    'steps': episode_data,
                    'total_reward': total_reward,
                    'length': step_count,
                    'action_space': 41,
                    'state_space': 207,
                    'policy_type': 'dqn_expert_bridge',
                    'timestamp': datetime.now().isoformat(),
                    'hold_percentage': hold_percentage
                }
                
                # Save trajectory
                filename = f"dqn_expert_ep{episode_id:03d}_r{total_reward:.0f}.pkl"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(trajectory, f)
                
                print(f"   ✅ SAVED: {filename}")
                successful_episodes += 1
            else:
                print(f"   ❌ SKIPPED: Poor quality")
        
        env.close()
        
        print(f"\n📊 SUMMARY:")
        print(f"   Episodes saved: {successful_episodes}/{num_episodes}")
        print(f"   Output directory: {output_dir}")
        
        if successful_episodes > 0:
            print("   ✅ SUCCESS: Expert trajectories generated!")
            return True
        else:
            print("   ❌ FAILED: No valid episodes generated")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Ready for AIRL training with DQN expert trajectories!") 