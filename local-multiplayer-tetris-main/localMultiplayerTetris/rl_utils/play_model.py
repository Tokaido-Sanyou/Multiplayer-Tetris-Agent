"""
Play Trained Model Script - Load and visualize a trained Tetris agent
"""
import argparse
import logging
import torch
import time
import os
import sys
import numpy as np

# Handle both direct execution and module import
try:
    from ..tetris_env import TetrisEnv
    from .state_model import StateModel
    from .actor_critic import ActorCriticAgent
    from .future_reward_predictor import FutureRewardPredictor
    from .unified_trainer import TrainingConfig
except ImportError:
    # Direct execution - add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tetris_env import TetrisEnv
    from rl_utils.state_model import StateModel
    from rl_utils.actor_critic import ActorCriticAgent
    from rl_utils.future_reward_predictor import FutureRewardPredictor
    from rl_utils.unified_trainer import TrainingConfig

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize models
    state_model = StateModel(state_dim=1817).to(device)
    future_reward_predictor = FutureRewardPredictor(state_dim=1817, action_dim=8).to(device)
    actor_critic = ActorCriticAgent(
        state_dim=1817,
        action_dim=8,
        state_model=state_model
    )
    
    # Load state dicts
    if 'state_model' in checkpoint:
        state_model.load_state_dict(checkpoint['state_model'])
    if 'future_reward_predictor' in checkpoint:
        future_reward_predictor.load_state_dict(checkpoint['future_reward_predictor'])
    if 'actor_critic' in checkpoint:
        actor_critic.network.load_state_dict(checkpoint['actor_critic'])
    
    # Set to evaluation mode
    state_model.eval()
    future_reward_predictor.eval()
    actor_critic.network.eval()
    
    # Disable exploration
    actor_critic.epsilon = 0.0
    
    return actor_critic, state_model, future_reward_predictor

def obs_to_state_vector(obs):
    """Convert multi-channel observation dict to flattened state vector"""
    # Flatten all the grids
    piece_grids_flat = obs['piece_grids'].flatten()  # 7*20*10 = 1400
    current_piece_flat = obs['current_piece_grid'].flatten()  # 20*10 = 200
    empty_grid_flat = obs['empty_grid'].flatten()  # 20*10 = 200
    
    # Get one-hot encodings and metadata
    next_piece = obs['next_piece']  # 7 values
    hold_piece = obs['hold_piece']  # 7 values
    metadata = np.array([
        obs['current_rotation'],
        obs['current_x'], 
        obs['current_y']
    ])  # 3 values
    
    # Concatenate all components: 1400 + 200 + 200 + 7 + 7 + 3 = 1817
    return np.concatenate([
        piece_grids_flat,
        current_piece_flat, 
        empty_grid_flat,
        next_piece,
        hold_piece,
        metadata
    ])

def play_game(actor_critic, env, max_episodes=5, max_steps=2000, step_delay=0.1):
    """Play games with the trained model"""
    
    total_rewards = []
    total_steps = []
    
    for episode in range(max_episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"\n=== Episode {episode + 1}/{max_episodes} ===")
        
        while not done and steps < max_steps:
            # Convert observation to state vector
            state = obs_to_state_vector(obs)
            
            # Select action using trained policy
            action = actor_critic.select_action(state)
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            # Render the game
            env.render()
            
            # Add delay to make it watchable
            time.sleep(step_delay)
            
            # Print game info periodically
            if steps % 100 == 0:
                print(f"  Step {steps}: Score={info.get('score', 0)}, Lines={info.get('lines_cleared', 0)}, Reward={episode_reward:.2f}")
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        print(f"Episode {episode + 1} completed:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Final Score: {info.get('score', 0)}")
        print(f"  Reason: {'Game Over' if done else 'Max Steps Reached'}")
    
    # Print summary statistics
    print(f"\n=== Summary ({max_episodes} episodes) ===")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"Best Reward: {np.max(total_rewards):.2f}")
    print(f"Worst Reward: {np.min(total_rewards):.2f}")

def main():
    parser = argparse.ArgumentParser(description="Play trained Tetris model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to play')
    parser.add_argument('--max_steps', type=int, default=2000, help='Maximum steps per episode')
    parser.add_argument('--step_delay', type=float, default=0.1, help='Delay between steps (seconds)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(args.device)
    
    try:
        # Load trained model
        print(f"Loading model from {args.checkpoint}...")
        actor_critic, state_model, future_reward_predictor = load_model(args.checkpoint, device)
        print("Model loaded successfully!")
        
        # Initialize environment with visualization enabled
        env = TetrisEnv(single_player=True, headless=False)
        print("Environment initialized with visualization enabled")
        
        # Play games
        play_game(
            actor_critic=actor_critic,
            env=env,
            max_episodes=args.episodes,
            max_steps=args.max_steps,
            step_delay=args.step_delay
        )
        
    except Exception as e:
        logging.error(f"Error during model playback: {e}")
        raise
    finally:
        if 'env' in locals():
            env.close()

if __name__ == '__main__':
    main() 