import os
import time
import torch
import numpy as np
import argparse
from localMultiplayerTetris.rl_utils.checkpoint_compatible_actor_critic import CheckpointCompatibleAgent
from localMultiplayerTetris.rl_utils.train import preprocess_state
from localMultiplayerTetris.rl_utils.trajectory_collector import ExpertTrajectoryCollector
from localMultiplayerTetris.tetris_env import TetrisEnv


def collect_expert_trajectories(checkpoint_path: str, num_episodes: int = 100, 
                               visualize: bool = False, min_reward_threshold: float = None,
                               action_space: int = 41):
    """
    Collect expert trajectories from a trained agent
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        num_episodes: Number of episodes to collect
        visualize: Whether to render the game during collection
        min_reward_threshold: Minimum reward threshold for expert dataset
        action_space: Target action space (8 or 41)
    """
    # Create environment
    env = TetrisEnv(single_player=True, headless=not visualize)
    
    # Initialize compatible agent for loading checkpoint
    state_dim = 207  # 20x10 grid + 7 scalar features
    original_action_dim = 41   # Original model's action space
    agent = CheckpointCompatibleAgent(state_dim, original_action_dim)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)
    print(f"Running on device: {agent.device}")
    
    # Initialize trajectory collector with action conversion if needed
    convert_actions = (action_space == 8)
    trajectory_collector = ExpertTrajectoryCollector(
        save_dir="expert_trajectories",
        max_traj_length=5000,
        convert_actions=convert_actions,
        target_action_space=action_space
    )
    
    # Set expert metadata
    trajectory_collector.set_expert_metadata(
        model_path=checkpoint_path,
        performance_metrics={}  # Will be filled during collection
    )
    
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    print(f"Starting trajectory collection for {num_episodes} episodes...")
    print(f"Original model action space: {original_action_dim}")
    print(f"Target action space: {action_space}")
    print(f"Action conversion: {convert_actions}")
    
    for episode in range(num_episodes):
        # Properly unpack reset return value
        obs, info = env.reset()
        state = preprocess_state(obs)
        done = False
        episode_reward = 0
        episode_length = 0
        episode_score = 0
        
        while not done:
            # Select action using greedy policy (no exploration)
            action = agent.select_action(state, deterministic=True)
            
            # Validate original action is in range
            if not (0 <= action < original_action_dim):
                print(f"Warning: Agent produced action {action} outside range 0-{original_action_dim-1}")
                action = min(max(0, action), original_action_dim - 1)
            
            # Step environment with original action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_obs)
            
            # Add step to trajectory (conversion happens inside)
            trajectory_collector.add_step(obs, action, reward, done, info)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            episode_score += info.get('score', 0)
            
            # Render if visualization enabled
            if visualize:
                env.render()
                time.sleep(0.05)  # Slow down for visibility
            
            # Update state
            state = next_state
            obs = next_obs
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(episode_score)
        
        print(f"Episode {episode + 1}/{num_episodes} - "
              f"Reward: {episode_reward:.2f}, Score: {episode_score}, Length: {episode_length}")
        
        # Optional: Early stopping if performance is too low
        if len(episode_rewards) >= 10:
            recent_avg = np.mean(episode_rewards[-10:])
            if recent_avg < -50:  # Adjust threshold as needed
                print(f"Warning: Recent average reward ({recent_avg:.2f}) is low")
    
    # Calculate performance metrics
    performance_metrics = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_score': np.mean(episode_scores),
        'avg_length': np.mean(episode_lengths),
        'total_episodes': num_episodes,
        'original_action_space': original_action_dim,
        'target_action_space': action_space
    }
    
    print("\nCollection Statistics:")
    for key, value in performance_metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Update expert metadata with performance metrics
    trajectory_collector.expert_metadata['performance_metrics'] = performance_metrics
    
    # Save expert dataset
    dataset_path = trajectory_collector.save_expert_dataset(min_reward_threshold)
    
    print(f"\nExpert trajectories saved to: {dataset_path}")
    print("Trajectory collection completed!")
    
    return dataset_path, performance_metrics


def main():
    parser = argparse.ArgumentParser(description='Collect expert trajectories from trained Tetris agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of episodes to collect (default: 100)')
    parser.add_argument('--visualize', action='store_true',
                        help='Render the game during collection')
    parser.add_argument('--min-reward', type=float, default=None,
                        help='Minimum reward threshold for expert dataset')
    parser.add_argument('--auto-checkpoint', action='store_true',
                        help='Automatically find the latest checkpoint')
    parser.add_argument('--action-space', type=int, choices=[8, 41], default=41,
                        help='Target action space for trajectories (default: 41)')
    
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    
    # Auto-find latest checkpoint if requested
    if args.auto_checkpoint:
        # Check multiple possible checkpoint directories
        ckpt_dirs = ['checkpoints', 'localMultiplayerTetris/checkpoints', '.']
        checkpoint_path = None
        
        for ckpt_dir in ckpt_dirs:
            if os.path.exists(ckpt_dir):
                files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
                if files:
                    checkpoint_path = max(files, key=os.path.getmtime)
                    print(f"Auto-selected checkpoint: {checkpoint_path}")
                    break
        
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoints found in any standard directory")
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Collect trajectories
    collect_expert_trajectories(
        checkpoint_path=checkpoint_path,
        num_episodes=args.num_episodes,
        visualize=args.visualize,
        min_reward_threshold=args.min_reward,
        action_space=args.action_space
    )


if __name__ == '__main__':
    main() 