#!/usr/bin/env python3
"""
Enhanced DQN Training Script with Parallel Support

This script provides both single and vectorized environment training for the DQN agent.
Features:
- TensorBoard logging with comprehensive metrics
- Automatic checkpointing every 1000 episodes
- Parallel environment support for faster training
- Configurable hyperparameters
- Resume training from checkpoints
"""

import os
import sys
import argparse
import numpy as np
import torch
import logging
from datetime import datetime
from gym.vector import AsyncVectorEnv, SyncVectorEnv

# Import from parent package
from ..tetris_env import TetrisEnv
from .dqn_new import DQNAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dqn_training.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def sanitize_space_random(space):
    """
    Recursively clear the ``np_random`` attribute from any Gym ``Space``.
    
    This fixes issues with deepcopy in vectorized environments.
    """
    from gym import spaces
    
    if hasattr(space, "_np_random"):
        space._np_random = None
    
    try:
        if hasattr(space, "np_random"):
            space.np_random = None
    except Exception:
        space.__dict__.pop("np_random", None)
    
    if isinstance(space, spaces.Dict):
        for sub in space.spaces.values():
            sanitize_space_random(sub)
    elif isinstance(space, (spaces.Tuple, getattr(spaces, "Sequence", tuple()))):
        for sub in space.spaces:
            sanitize_space_random(sub)

def make_env(env_id, seed, headless=True):
    """Factory function for creating TetrisEnv instances."""
    def _init():
        env = TetrisEnv(single_player=True, headless=headless)
        env.seed(seed + env_id)
        
        # Sanitize spaces for vectorized environments
        sanitize_space_random(env.observation_space)
        sanitize_space_random(env.action_space)
        
        return env
    return _init

def preprocess_obs_dict(obs_dict):
    """
    Convert observation dictionary to state vector format (207,).
    
    Args:
        obs_dict: Dictionary with keys 'grid', 'next_piece', 'hold_piece', etc.
        
    Returns:
        numpy array of shape (207,)
    """
    # Flatten grid (20x10 -> 200)
    grid_flat = obs_dict['grid'].flatten().astype(np.float32)
    
    # Extract metadata (7 values)
    metadata = np.array([
        obs_dict['next_piece'],
        obs_dict['hold_piece'],
        obs_dict['current_shape'],
        obs_dict['current_rotation'],
        obs_dict['current_x'],
        obs_dict['current_y'],
        obs_dict['can_hold']
    ]).astype(np.float32)
    
    # Combine into single state vector
    state = np.concatenate([grid_flat, metadata])
    return state

def preprocess_obs_batch(batched_obs_dict):
    """
    Convert batched observation dictionary to batch of state vectors.
    
    Args:
        batched_obs_dict: Dictionary with batched observations
        
    Returns:
        numpy array of shape (batch_size, 207)
    """
    batch_size = batched_obs_dict['grid'].shape[0]
    states = []
    
    for i in range(batch_size):
        obs_dict = {key: value[i] for key, value in batched_obs_dict.items()}
        state = preprocess_obs_dict(obs_dict)
        states.append(state)
    
    return np.array(states)

def train_single_env(agent, num_episodes=10000, eval_interval=1000, checkpoint_path=None):
    """
    Train DQN agent on single environment.
    
    Args:
        agent: DQNAgent instance
        num_episodes: Number of episodes to train
        eval_interval: Episodes between evaluations
        checkpoint_path: Path to load checkpoint from (optional)
    """
    logger.info(f"Starting single environment training for {num_episodes} episodes")
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Create environment
    env = TetrisEnv(single_player=True, headless=True)
    
    episode_start = agent.episodes_done
    for episode in range(episode_start, episode_start + num_episodes):
        obs_dict, info = env.reset()
        state = preprocess_obs_dict(obs_dict)
        
        episode_reward = 0
        episode_length = 0
        episode_info = {'score': 0, 'lines_cleared': 0, 'level': 1}
        
        done = False
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_obs_dict, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            next_state = preprocess_obs_dict(next_obs_dict)
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            
            # Update state and tracking
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Update episode info
            episode_info['score'] = step_info.get('score', episode_info['score'])
            episode_info['lines_cleared'] += step_info.get('lines_cleared', 0)
            episode_info['level'] = step_info.get('level', episode_info['level'])
        
        # Log episode
        agent.log_episode(episode_reward, episode_length, episode_info)
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            eval_score = evaluate_agent(agent, num_episodes=10)
            logger.info(f"Episode {episode + 1}: Evaluation score: {eval_score:.2f}")
    
    env.close()
    logger.info("Training completed")

def train_vectorized_env(agent, num_envs=4, num_episodes=10000, eval_interval=1000, 
                        update_frequency=4, checkpoint_path=None):
    """
    Train DQN agent using vectorized environments for parallel data collection.
    
    Args:
        agent: DQNAgent instance
        num_envs: Number of parallel environments
        num_episodes: Total number of episodes to train
        eval_interval: Episodes between evaluations
        update_frequency: Number of environment steps between agent updates
        checkpoint_path: Path to load checkpoint from (optional)
    """
    logger.info(f"Starting vectorized training with {num_envs} environments for {num_episodes} episodes")
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Create vectorized environments
    # Use SyncVectorEnv for easier debugging, AsyncVectorEnv for better performance
    envs = SyncVectorEnv([make_env(i, seed=i + 1000, headless=True) for i in range(num_envs)])
    
    # Initialize environments
    obs_dicts = envs.reset()
    states = preprocess_obs_batch(obs_dicts)
    
    episode_rewards = [0] * num_envs
    episode_lengths = [0] * num_envs
    episode_infos = [{'score': 0, 'lines_cleared': 0, 'level': 1} for _ in range(num_envs)]
    completed_episodes = agent.episodes_done
    
    step_count = 0
    
    while completed_episodes < num_episodes:
        # Select actions for all environments
        actions = agent.select_actions_batch(states)
        
        # Take steps in all environments
        next_obs_dicts, rewards, dones, infos = envs.step(actions)
        next_states = preprocess_obs_batch(next_obs_dicts)
        
        # Store experiences and update tracking
        for i in range(num_envs):
            # Store experience
            agent.memory.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
            
            # Update episode tracking
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1
            
            # Update episode info
            if infos[i]:
                episode_infos[i]['score'] = infos[i].get('score', episode_infos[i]['score'])
                episode_infos[i]['lines_cleared'] += infos[i].get('lines_cleared', 0)
                episode_infos[i]['level'] = infos[i].get('level', episode_infos[i]['level'])
            
            # Handle episode completion
            if dones[i]:
                # Log completed episode
                agent.log_episode(episode_rewards[i], episode_lengths[i], episode_infos[i])
                completed_episodes += 1
                
                # Reset tracking for this environment
                episode_rewards[i] = 0
                episode_lengths[i] = 0
                episode_infos[i] = {'score': 0, 'lines_cleared': 0, 'level': 1}
                
                # Periodic evaluation
                if completed_episodes % eval_interval == 0:
                    eval_score = evaluate_agent(agent, num_episodes=10)
                    logger.info(f"Episode {completed_episodes}: Evaluation score: {eval_score:.2f}")
        
        # Update agent
        step_count += 1
        if step_count % update_frequency == 0:
            loss = agent.train_batch_update(num_updates=update_frequency)
        
        # Update states
        states = next_states
    
    envs.close()
    logger.info("Vectorized training completed")

def evaluate_agent(agent, num_episodes=10):
    """
    Evaluate agent performance on clean environment.
    
    Args:
        agent: DQNAgent instance
        num_episodes: Number of evaluation episodes
        
    Returns:
        Average score over evaluation episodes
    """
    env = TetrisEnv(single_player=True, headless=True)
    scores = []
    
    for _ in range(num_episodes):
        obs_dict, _ = env.reset()
        state = preprocess_obs_dict(obs_dict)
        score = 0
        done = False
        
        while not done:
            # Use greedy action selection (no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
            
            next_obs_dict, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = preprocess_obs_dict(next_obs_dict)
            score = info.get('score', score)
        
        scores.append(score)
    
    env.close()
    return np.mean(scores)

def main():
    parser = argparse.ArgumentParser(description='Train DQN agent on Tetris')
    parser.add_argument('--mode', choices=['single', 'vectorized'], default='vectorized',
                       help='Training mode: single environment or vectorized')
    parser.add_argument('--num_episodes', type=int, default=10000,
                       help='Number of episodes to train')
    parser.add_argument('--num_envs', type=int, default=4,
                       help='Number of parallel environments (vectorized mode only)')
    parser.add_argument('--eval_interval', type=int, default=1000,
                       help='Episodes between evaluations')
    parser.add_argument('--update_frequency', type=int, default=4,
                       help='Environment steps between agent updates (vectorized mode)')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='Initial epsilon for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                       help='Final epsilon for exploration')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                       help='Epsilon decay rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--buffer_size', type=int, default=100000,
                       help='Replay buffer size')
    parser.add_argument('--target_update', type=int, default=10,
                       help='Steps between target network updates')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Episodes between checkpoints')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create agent
    agent = DQNAgent(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=device,
        save_interval=args.save_interval
    )
    
    try:
        if args.mode == 'single':
            train_single_env(
                agent=agent,
                num_episodes=args.num_episodes,
                eval_interval=args.eval_interval,
                checkpoint_path=args.checkpoint
            )
        else:  # vectorized
            train_vectorized_env(
                agent=agent,
                num_envs=args.num_envs,
                num_episodes=args.num_episodes,
                eval_interval=args.eval_interval,
                update_frequency=args.update_frequency,
                checkpoint_path=args.checkpoint
            )
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Save final model and close resources
        final_save_path = f"checkpoints/dqn_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        agent.save(final_save_path)
        agent.close()
        logger.info(f"Final model saved to: {final_save_path}")

if __name__ == "__main__":
    main() 