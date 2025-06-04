#!/usr/bin/env python3
"""
Dual Agent Training Script for Competitive Tetris
"""
import os
import argparse
import logging
import torch
import numpy as np
from typing import Dict, Tuple, List
from torch.utils.tensorboard import SummaryWriter

from localMultiplayerTetris.dual_agent_env import DualAgentTetrisEnv
from localMultiplayerTetris.rl_utils.airl import AIRLAgent, AIRLConfig
from localMultiplayerTetris.rl_utils.trajectory_collector import TrajectoryCollector
from localMultiplayerTetris.rl_utils.train import preprocess_state


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dual_agent_training.log'),
            logging.StreamHandler()
        ]
    )


def preprocess_dual_state(obs_dict: Dict) -> np.ndarray:
    """
    Preprocess dual agent observation to state vector
    
    Args:
        obs_dict: Observation dictionary from dual agent environment
        
    Returns:
        Preprocessed state vector (227 dimensions)
        Standard state (207) + opponent info (20)
    """
    # Standard tetris state (207 dimensions)
    grid = obs_dict['grid'].flatten()  # 200
    scalars = np.array([
        obs_dict['next_piece'],
        obs_dict['hold_piece'], 
        obs_dict['current_shape'],
        obs_dict['current_rotation'],
        obs_dict['current_x'],
        obs_dict['current_y'],
        obs_dict['can_hold']
    ])  # 7
    
    # Opponent information (20 dimensions)
    opponent_grid_summary = np.sum(obs_dict['opponent_grid'], axis=1)  # Column heights (10)
    opponent_info = np.array([
        obs_dict['opponent_score'] / 1000.0,  # Normalized opponent score
        obs_dict['score_diff'] / 1000.0,      # Normalized score difference
        # Add more opponent features as needed
        np.sum(obs_dict['opponent_grid']) / 200.0,  # Opponent grid fullness
        np.max(opponent_grid_summary),              # Opponent max height
        np.std(opponent_grid_summary),              # Opponent height variance
        np.sum(opponent_grid_summary > 15),         # Opponent danger zones
        np.sum(opponent_grid_summary > 18),         # Opponent critical zones
        1.0 if np.max(opponent_grid_summary) > 16 else 0.0,  # Opponent in danger
        1.0 if np.max(opponent_grid_summary) > 18 else 0.0,  # Opponent critical
        0.0  # Padding for future features
    ])  # 10
    
    return np.concatenate([grid, scalars, opponent_grid_summary, opponent_info])


class DualAgentConfig(AIRLConfig):
    """Extended config for dual agent training"""
    
    def __init__(self):
        super().__init__()
        self.state_dim = 227  # Extended state with opponent info
        self.self_play_episodes = 100  # Episodes before opponent switching
        self.curriculum_episodes = 1000  # Episodes for curriculum learning
        self.save_trajectories = True  # Save trajectories during training


def train_dual_agents(config: DualAgentConfig, args: argparse.Namespace):
    """
    Train two agents to compete against each other
    
    Args:
        config: Dual agent configuration
        args: Command line arguments
    """
    # Setup logging and tensorboard
    writer = SummaryWriter(log_dir='logs/dual_agent_tensorboard')
    logger = logging.getLogger(__name__)
    
    # Create dual agent environment
    env = DualAgentTetrisEnv(headless=not args.visualize, max_steps=5000)
    logger.info("Dual agent environment created")
    
    # Initialize two agents
    agent1 = AIRLAgent(config)
    agent2 = AIRLAgent(config)
    
    # Load pretrained models if specified
    if args.pretrained_agent1:
        if os.path.exists(args.pretrained_agent1):
            logger.info(f"Loading pretrained agent1 from {args.pretrained_agent1}")
            agent1.load(args.pretrained_agent1)
        else:
            logger.warning(f"Pretrained agent1 not found: {args.pretrained_agent1}")
    
    if args.pretrained_agent2:
        if os.path.exists(args.pretrained_agent2):
            logger.info(f"Loading pretrained agent2 from {args.pretrained_agent2}")
            agent2.load(args.pretrained_agent2)
        else:
            logger.warning(f"Pretrained agent2 not found: {args.pretrained_agent2}")
    
    # Create save directories
    os.makedirs('dual_agent_checkpoints', exist_ok=True)
    
    # Initialize trajectory collectors
    trajectory_collector1 = None
    trajectory_collector2 = None
    if config.save_trajectories:
        trajectory_collector1 = TrajectoryCollector("trajectories_agent1")
        trajectory_collector2 = TrajectoryCollector("trajectories_agent2")
    
    # Training metrics
    agent1_rewards = []
    agent2_rewards = []
    agent1_wins = 0
    agent2_wins = 0
    ties = 0
    
    logger.info(f"Starting dual agent training for {config.max_episodes} episodes")
    logger.info(f"Device: {config.device}")
    logger.info(f"Agent1 parameters: {agent1.count_parameters(agent1.policy)}")
    logger.info(f"Agent2 parameters: {agent2.count_parameters(agent2.policy)}")
    
    global_step = 0
    
    for episode in range(config.max_episodes):
        # Reset environment
        obs1, obs2 = env.reset()
        state1 = preprocess_dual_state(obs1)
        state2 = preprocess_dual_state(obs2)
        done = False
        
        episode_reward1 = 0
        episode_reward2 = 0
        episode_length = 0
        
        while not done:
            # Select actions
            action1 = agent1.select_action(state1, deterministic=False)
            action2 = agent2.select_action(state2, deterministic=False)
            
            # Step environment
            (next_obs1, next_obs2), (reward1, reward2), (done1, done2), info = env.step((action1, action2))
            done = done1 or done2
            
            next_state1 = preprocess_dual_state(next_obs1)
            next_state2 = preprocess_dual_state(next_obs2)
            
            # Add experiences to buffers
            agent1.add_experience(state1, action1, reward1, next_state1, done)
            agent2.add_experience(state2, action2, reward2, next_state2, done)
            
            # Add to trajectory collectors
            if config.save_trajectories:
                if trajectory_collector1:
                    trajectory_collector1.add_step(obs1, action1, reward1, done, info.get('player1', {}))
                if trajectory_collector2:
                    trajectory_collector2.add_step(obs2, action2, reward2, done, info.get('player2', {}))
            
            # Update metrics
            episode_reward1 += reward1
            episode_reward2 += reward2
            episode_length += 1
            global_step += 1
            
            # Render if enabled
            if args.visualize:
                env.render()
            
            # Update networks (alternating to prevent correlation)
            if len(agent1.policy_buffer) >= config.batch_size and global_step % 2 == 0:
                # Update agent1
                agent1_stats = agent1.update_policy()
                if agent1_stats:
                    for key, value in agent1_stats.items():
                        writer.add_scalar(f'Agent1/{key}', value, global_step)
            
            if len(agent2.policy_buffer) >= config.batch_size and global_step % 2 == 1:
                # Update agent2
                agent2_stats = agent2.update_policy()
                if agent2_stats:
                    for key, value in agent2_stats.items():
                        writer.add_scalar(f'Agent2/{key}', value, global_step)
            
            # Update states
            state1 = next_state1
            state2 = next_state2
            obs1 = next_obs1
            obs2 = next_obs2
        
        # Determine winner
        winner = env.get_winner()
        if winner == 1:
            agent1_wins += 1
        elif winner == 2:
            agent2_wins += 1
        else:
            ties += 1
        
        # Store episode statistics
        agent1_rewards.append(episode_reward1)
        agent2_rewards.append(episode_reward2)
        
        # Log episode results
        if episode % config.log_interval == 0:
            recent_rewards1 = agent1_rewards[-config.log_interval:]
            recent_rewards2 = agent2_rewards[-config.log_interval:]
            
            logger.info(f"Episode {episode + 1}/{config.max_episodes}")
            logger.info(f"  Agent1 Avg Reward: {np.mean(recent_rewards1):.2f}")
            logger.info(f"  Agent2 Avg Reward: {np.mean(recent_rewards2):.2f}")
            logger.info(f"  Winner: {'Agent1' if winner == 1 else ('Agent2' if winner == 2 else 'Tie')}")
            logger.info(f"  Win Rates: Agent1={agent1_wins/(episode+1):.3f}, Agent2={agent2_wins/(episode+1):.3f}, Ties={ties/(episode+1):.3f}")
            logger.info(f"  Episode Length: {episode_length}")
        
        # TensorBoard logging
        writer.add_scalar('Episode/Reward_Agent1', episode_reward1, episode + 1)
        writer.add_scalar('Episode/Reward_Agent2', episode_reward2, episode + 1)
        writer.add_scalar('Episode/Length', episode_length, episode + 1)
        writer.add_scalar('Episode/Winner', winner, episode + 1)
        writer.add_scalar('Winrate/Agent1', agent1_wins / (episode + 1), episode + 1)
        writer.add_scalar('Winrate/Agent2', agent2_wins / (episode + 1), episode + 1)
        writer.add_scalar('Winrate/Ties', ties / (episode + 1), episode + 1)
        
        # Save checkpoints
        if (episode + 1) % config.save_interval == 0:
            agent1_path = f'dual_agent_checkpoints/agent1_episode_{episode + 1}.pt'
            agent2_path = f'dual_agent_checkpoints/agent2_episode_{episode + 1}.pt'
            agent1.save(agent1_path)
            agent2.save(agent2_path)
            logger.info(f"Checkpoints saved: {agent1_path}, {agent2_path}")
        
        # Evaluation (agents vs each other)
        if (episode + 1) % config.eval_interval == 0:
            eval_results = evaluate_dual_agents(agent1, agent2, env, num_episodes=10)
            logger.info(f"Evaluation after episode {episode + 1}:")
            logger.info(f"  Agent1 win rate: {eval_results['agent1_winrate']:.3f}")
            logger.info(f"  Agent2 win rate: {eval_results['agent2_winrate']:.3f}")
            logger.info(f"  Tie rate: {eval_results['tie_rate']:.3f}")
            logger.info(f"  Avg game length: {eval_results['avg_length']:.1f}")
            
            writer.add_scalar('Eval/Agent1_Winrate', eval_results['agent1_winrate'], episode + 1)
            writer.add_scalar('Eval/Agent2_Winrate', eval_results['agent2_winrate'], episode + 1)
            writer.add_scalar('Eval/Tie_Rate', eval_results['tie_rate'], episode + 1)
            writer.add_scalar('Eval/Avg_Length', eval_results['avg_length'], episode + 1)
        
        # Self-play curriculum: occasionally train against copies
        if (episode + 1) % config.self_play_episodes == 0:
            logger.info("Curriculum learning: agents will play against copies for diversity")
            # This could involve creating copies of the current agents for variety
    
    # Final saves
    agent1.save('dual_agent_checkpoints/agent1_final.pt')
    agent2.save('dual_agent_checkpoints/agent2_final.pt')
    
    # Training summary
    logger.info("\n" + "="*50)
    logger.info("DUAL AGENT TRAINING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total episodes: {config.max_episodes}")
    logger.info(f"Agent1 final win rate: {agent1_wins/config.max_episodes:.3f}")
    logger.info(f"Agent2 final win rate: {agent2_wins/config.max_episodes:.3f}")
    logger.info(f"Tie rate: {ties/config.max_episodes:.3f}")
    logger.info(f"Agent1 avg reward (last 100): {np.mean(agent1_rewards[-100:]):.2f}")
    logger.info(f"Agent2 avg reward (last 100): {np.mean(agent2_rewards[-100:]):.2f}")
    
    # Close environment and writer
    env.close()
    writer.close()


def evaluate_dual_agents(agent1: AIRLAgent, agent2: AIRLAgent, env: DualAgentTetrisEnv, num_episodes: int = 10) -> Dict:
    """
    Evaluate two agents against each other
    
    Args:
        agent1: First agent
        agent2: Second agent
        env: Dual agent environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary with evaluation results
    """
    agent1_wins = 0
    agent2_wins = 0
    ties = 0
    total_lengths = []
    
    for _ in range(num_episodes):
        obs1, obs2 = env.reset()
        state1 = preprocess_dual_state(obs1)
        state2 = preprocess_dual_state(obs2)
        done = False
        episode_length = 0
        
        while not done:
            action1 = agent1.select_action(state1, deterministic=True)
            action2 = agent2.select_action(state2, deterministic=True)
            
            (next_obs1, next_obs2), (reward1, reward2), (done1, done2), info = env.step((action1, action2))
            done = done1 or done2
            
            state1 = preprocess_dual_state(next_obs1)
            state2 = preprocess_dual_state(next_obs2)
            episode_length += 1
        
        winner = env.get_winner()
        if winner == 1:
            agent1_wins += 1
        elif winner == 2:
            agent2_wins += 1
        else:
            ties += 1
        
        total_lengths.append(episode_length)
    
    return {
        'agent1_winrate': agent1_wins / num_episodes,
        'agent2_winrate': agent2_wins / num_episodes,
        'tie_rate': ties / num_episodes,
        'avg_length': np.mean(total_lengths)
    }


def main():
    parser = argparse.ArgumentParser(description='Train dual agents for competitive Tetris')
    
    # Training parameters
    parser.add_argument('--max-episodes', type=int, default=10000,
                        help='Maximum number of training episodes')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr-policy', type=float, default=1e-4,
                        help='Learning rate for policy')
    
    # Network architecture
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for networks')
    
    # Pretrained models
    parser.add_argument('--pretrained-agent1', type=str, default=None,
                        help='Path to pretrained agent1 checkpoint')
    parser.add_argument('--pretrained-agent2', type=str, default=None,
                        help='Path to pretrained agent2 checkpoint')
    
    # Training options
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize training (slower)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--save-trajectories', action='store_true',
                        help='Save trajectories during training')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create config
    config = DualAgentConfig()
    if args.max_episodes:
        config.max_episodes = args.max_episodes
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr_policy:
        config.learning_rate_policy = args.lr_policy
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    config.save_trajectories = args.save_trajectories
    
    # Start training
    train_dual_agents(config, args)


if __name__ == '__main__':
    main() 