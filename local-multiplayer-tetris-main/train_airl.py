#!/usr/bin/env python3
"""
AIRL Training Script for Tetris Agent
"""
import os
import argparse
import logging
import torch
import numpy as np
from typing import Dict
from torch.utils.tensorboard import SummaryWriter

from localMultiplayerTetris.tetris_env import TetrisEnv
from localMultiplayerTetris.rl_utils.airl import AIRLAgent, AIRLConfig
from localMultiplayerTetris.rl_utils.train import preprocess_state


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('airl_training.log'),
            logging.StreamHandler()
        ]
    )


def train_airl_agent(config: AIRLConfig, args: argparse.Namespace):
    """
    Main AIRL training loop
    
    Args:
        config: AIRL configuration
        args: Command line arguments
    """
    # Setup logging and tensorboard
    writer = SummaryWriter(log_dir='logs/airl_tensorboard')
    logger = logging.getLogger(__name__)
    
    # Create environment
    env = TetrisEnv(single_player=True, headless=not args.visualize)
    logger.info("Environment created")
    
    # Initialize AIRL agent
    agent = AIRLAgent(config)
    
    # Load expert data
    try:
        agent.load_expert_data(config.expert_data_path)
    except FileNotFoundError:
        logger.error(f"Expert data not found at {config.expert_data_path}")
        logger.error("Please run replay_agent_with_trajectories.py first to generate expert data")
        return
    
    # Load pretrained policy if specified
    if args.pretrained_policy:
        if os.path.exists(args.pretrained_policy):
            logger.info(f"Loading pretrained policy from {args.pretrained_policy}")
            agent.load(args.pretrained_policy)
        else:
            logger.warning(f"Pretrained policy not found: {args.pretrained_policy}")
    
    # Create save directory
    os.makedirs('airl_checkpoints', exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    logger.info(f"Starting AIRL training for {config.max_episodes} episodes")
    logger.info(f"Device: {config.device}")
    logger.info(f"Discriminator parameters: {agent.count_parameters(agent.discriminator)}")
    logger.info(f"Policy parameters: {agent.count_parameters(agent.policy)}")
    
    global_step = 0
    
    for episode in range(config.max_episodes):
        # Reset environment
        obs = env.reset()
        state = preprocess_state(obs)
        done = False
        episode_reward = 0
        episode_length = 0
        episode_score = 0
        
        while not done:
            # Select action
            action = agent.select_action(state, deterministic=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_obs)
            
            # Add experience to buffer
            agent.add_experience(state, action, reward, next_state, done)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            episode_score += info.get('score', 0)
            global_step += 1
            
            # Render if enabled
            if args.visualize:
                env.render()
            
            # Update networks
            if len(agent.policy_buffer) >= config.batch_size:
                # Update discriminator
                disc_stats = agent.update_discriminator()
                if disc_stats:
                    for key, value in disc_stats.items():
                        writer.add_scalar(f'Discriminator/{key}', value, global_step)
                
                # Update policy
                policy_stats = agent.update_policy()
                if policy_stats:
                    for key, value in policy_stats.items():
                        writer.add_scalar(f'Policy/{key}', value, global_step)
            
            # Update state
            state = next_state
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(episode_score)
        
        # Log episode results
        if episode % config.log_interval == 0:
            recent_rewards = episode_rewards[-config.log_interval:]
            recent_scores = episode_scores[-config.log_interval:]
            recent_lengths = episode_lengths[-config.log_interval:]
            
            logger.info(f"Episode {episode + 1}/{config.max_episodes}")
            logger.info(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
            logger.info(f"  Avg Score: {np.mean(recent_scores):.2f}")
            logger.info(f"  Avg Length: {np.mean(recent_lengths):.2f}")
            logger.info(f"  Episode Reward: {episode_reward:.2f}")
            logger.info(f"  Episode Score: {episode_score}")
            
            # Log training statistics
            stats = agent.get_stats()
            for key, value in stats.items():
                logger.info(f"  {key}: {value:.4f}")
        
        # TensorBoard logging
        writer.add_scalar('Episode/Reward', episode_reward, episode + 1)
        writer.add_scalar('Episode/Score', episode_score, episode + 1)
        writer.add_scalar('Episode/Length', episode_length, episode + 1)
        
        if len(episode_rewards) >= 100:
            writer.add_scalar('Episode/Reward_100', np.mean(episode_rewards[-100:]), episode + 1)
            writer.add_scalar('Episode/Score_100', np.mean(episode_scores[-100:]), episode + 1)
        
        # Save checkpoint
        if (episode + 1) % config.save_interval == 0:
            checkpoint_path = f'airl_checkpoints/airl_episode_{episode + 1}.pt'
            agent.save(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Evaluation
        if (episode + 1) % config.eval_interval == 0:
            eval_rewards, eval_scores = evaluate_agent(agent, env, num_episodes=5)
            logger.info(f"Evaluation after episode {episode + 1}:")
            logger.info(f"  Eval Avg Reward: {np.mean(eval_rewards):.2f}")
            logger.info(f"  Eval Avg Score: {np.mean(eval_scores):.2f}")
            
            writer.add_scalar('Eval/Reward', np.mean(eval_rewards), episode + 1)
            writer.add_scalar('Eval/Score', np.mean(eval_scores), episode + 1)
    
    # Final save
    final_checkpoint = 'airl_checkpoints/airl_final.pt'
    agent.save(final_checkpoint)
    logger.info(f"Training completed. Final checkpoint: {final_checkpoint}")
    
    # Training summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total episodes: {config.max_episodes}")
    logger.info(f"Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    logger.info(f"Final avg score (last 100): {np.mean(episode_scores[-100:]):.2f}")
    logger.info(f"Best episode reward: {max(episode_rewards):.2f}")
    logger.info(f"Best episode score: {max(episode_scores)}")
    
    # Close environment and writer
    env.close()
    writer.close()


def evaluate_agent(agent: AIRLAgent, env: TetrisEnv, num_episodes: int = 5):
    """
    Evaluate agent performance
    
    Args:
        agent: AIRL agent to evaluate
        env: Environment for evaluation
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Tuple of (rewards, scores)
    """
    eval_rewards = []
    eval_scores = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        state = preprocess_state(obs)
        done = False
        episode_reward = 0
        episode_score = 0
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_score += info.get('score', 0)
            state = preprocess_state(next_obs)
        
        eval_rewards.append(episode_reward)
        eval_scores.append(episode_score)
    
    return eval_rewards, eval_scores


def create_config_from_args(args: argparse.Namespace) -> AIRLConfig:
    """Create AIRL config from command line arguments"""
    config = AIRLConfig()
    
    # Update config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr_discriminator:
        config.learning_rate_discriminator = args.lr_discriminator
    if args.lr_policy:
        config.learning_rate_policy = args.lr_policy
    if args.max_episodes:
        config.max_episodes = args.max_episodes
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    if args.feature_dim:
        config.feature_dim = args.feature_dim
    if args.expert_data_path:
        config.expert_data_path = args.expert_data_path
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train AIRL agent for Tetris')
    
    # Training parameters
    parser.add_argument('--max-episodes', type=int, default=5000,
                        help='Maximum number of training episodes')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr-discriminator', type=float, default=1e-4,
                        help='Learning rate for discriminator')
    parser.add_argument('--lr-policy', type=float, default=1e-4,
                        help='Learning rate for policy')
    
    # Network architecture
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for policy network')
    parser.add_argument('--feature-dim', type=int, default=64,
                        help='Feature dimension for discriminator')
    
    # Data and checkpoints
    parser.add_argument('--expert-data-path', type=str, 
                        default='expert_trajectories/expert_dataset.pkl',
                        help='Path to expert trajectory data')
    parser.add_argument('--pretrained-policy', type=str, default=None,
                        help='Path to pretrained policy checkpoint')
    
    # Training options
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize training (slower)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create config
    config = create_config_from_args(args)
    
    # Start training
    train_airl_agent(config, args)


if __name__ == '__main__':
    main() 