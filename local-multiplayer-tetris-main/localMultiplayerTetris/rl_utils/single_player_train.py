import os
import numpy as np
import torch
import logging
from ..tetris_env import TetrisEnv
from .dqn_agent import DQNAgent
from .train import preprocess_state, evaluate_agent
import pygame
import cProfile
import pstats
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def train_single_player(num_episodes=1000, save_interval=100, eval_interval=50, visualize=True, checkpoint=None, no_eval=False, verbose=False, top_k=3):
    """
    Train an agent as player 1 in the Tetris environment using DQN
    
    Args:
        num_episodes: Number of episodes to train for
        save_interval: Save model every N episodes
        eval_interval: Evaluate agent every N episodes
        visualize: Whether to render the environment during training
        checkpoint: Path to checkpoint file to load
        no_eval: Whether to disable evaluation during training
        verbose: Enable per-step logging
        top_k: Number of top actions to sample from during action selection
    """
    # Set up TensorBoard writer with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join('logs', 'tensorboard', f'dqn_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    try:
        # Create environment; show window if visualize=True
        env = TetrisEnv(single_player=True, headless=not visualize)
        
        # Create agent
        state_dim = 202  # 20x10 grid + next_piece + hold_piece scalars
        action_dim = 8   # 8 possible actions, including no-op (ID 7)
        agent = DQNAgent(state_dim, action_dim, top_k_ac=top_k)
        if checkpoint:
            try:
                agent.load(checkpoint)
                logging.info(f"Loaded checkpoint from {checkpoint}")
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
        
        # Create directories for saving
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        episode_lines = []
        episode_scores = []
        episode_max_levels = []
        q_losses = []
        episode_rnd_losses = []
        episode_intrinsic_rewards = []
        
        for episode in range(num_episodes):
            try:
                # Initialize observation and state
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                episode_lines_cleared = 0
                episode_score = 0
                episode_max_level = 0
                episode_q_loss = 0
                episode_rnd_loss = 0
                episode_intrinsic_reward = 0
                train_steps = 0
                episode_pieces_placed = 0
                last_piece_count = 0
                no_piece_steps = 0
                
                while not done:
                    try:
                        # Select and perform action
                        action = agent.select_action(obs)
                        step_result = env.step(action)
                        
                        # Handle potential error returns from env.step()
                        if isinstance(step_result, (int, float)):
                            logging.error(f"Environment step returned {step_result} instead of state tuple")
                            done = True
                            break
                            
                        next_obs, reward, done, info = step_result
                        
                        # Validate state structure
                        if not isinstance(next_obs, dict) or 'grid' not in next_obs:
                            logging.error(f"Invalid state structure: {next_obs}")
                            done = True
                            break
                        
                        # Render if visualization enabled
                        if visualize:
                            env.render()
                        
                        # Update metrics
                        episode_reward += reward
                        episode_length += 1
                        episode_lines_cleared += info.get('lines_cleared', 0)
                        episode_score += info.get('score', 0)
                        episode_max_level = max(episode_max_level, info.get('level', 0))
                        
                        # Track piece placement
                        try:
                            current_piece_count = np.sum(next_obs['grid'] > 0)
                            piece_placed = current_piece_count > last_piece_count
                            if piece_placed:
                                episode_pieces_placed += 1
                                no_piece_steps = 0
                            else:
                                no_piece_steps += 1
                            last_piece_count = current_piece_count
                        except Exception as e:
                            logging.error(f"Error tracking piece placement: {e}")
                            done = True
                            break
                        
                        # Store transition in replay buffer
                        try:
                            agent.memory.push(obs, action, reward, next_obs, done, info)
                        except Exception as e:
                            logging.error(f"Error storing transition: {e}")
                            done = True
                            break
                        
                        # Train agent
                        try:
                            if piece_placed:
                                # Train multiple times when a piece is placed
                                for _ in range(3):
                                    train_result = agent.train()
                                    if train_result is not None:
                                        q_loss, rnd_loss = train_result
                                        episode_q_loss += q_loss
                                        episode_rnd_loss += rnd_loss
                                        train_steps += 1
                                        writer.add_scalar('Step/QLoss', q_loss, episode * env.max_steps + episode_length)
                                        writer.add_scalar('Step/RNDLoss', rnd_loss, episode * env.max_steps + episode_length)
                            else:
                                # Train once when no piece is placed
                                train_result = agent.train()
                                if train_result is not None:
                                    q_loss, rnd_loss = train_result
                                    episode_q_loss += q_loss
                                    episode_rnd_loss += rnd_loss
                                    train_steps += 1
                                    writer.add_scalar('Step/QLoss', q_loss, episode * env.max_steps + episode_length)
                                    writer.add_scalar('Step/RNDLoss', rnd_loss, episode * env.max_steps + episode_length)
                        except Exception as e:
                            logging.error(f"Error during training: {str(e)}")
                            # Don't break here, continue with episode
                        
                        # Update observation
                        obs = next_obs
                        
                        # Log episode details
                        if verbose:
                            logging.info(f"Action taken: {action}")
                            logging.info(f"Reward received: {reward}")
                            logging.info(f"Game state: {info}")
                            logging.info(f"Pieces placed: {episode_pieces_placed}")
                        
                        # Get intrinsic reward
                        try:
                            state_tensor = torch.FloatTensor(np.concatenate([
                                obs['grid'].flatten(),
                                [obs['next_piece']],
                                [obs['hold_piece']]
                            ])).unsqueeze(0).to(agent.device)
                            intrinsic_reward = agent.rnd.compute_intrinsic_reward(state_tensor).item()
                            episode_intrinsic_reward += intrinsic_reward
                            
                            # Log step-level metrics
                            writer.add_scalar('Step/IntrinsicReward', intrinsic_reward, episode * env.max_steps + episode_length)
                            writer.add_scalar('Step/PiecesPlaced', episode_pieces_placed, episode * env.max_steps + episode_length)
                        except Exception as e:
                            logging.error(f"Error computing intrinsic reward: {e}")
                            # Don't break here, continue with episode
                        
                        # Early termination if no pieces placed after too many steps
                        if no_piece_steps > 30:  # Reduced from 50 to 30
                            logging.warning(f"Terminating episode {episode + 1} early - no pieces placed for {no_piece_steps} steps")
                            done = True
                        
                    except Exception as e:
                        logging.error(f"Error during episode {episode + 1} step: {str(e)}")
                        done = True
                        break
                
                # Update exploration rate
                agent.update_epsilon()
                
                # Calculate average loss
                if train_steps > 0:
                    episode_q_loss /= train_steps
                    episode_rnd_loss /= train_steps
                
                # Store episode metrics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_lines.append(episode_lines_cleared)
                episode_scores.append(episode_score)
                episode_max_levels.append(episode_max_level)
                q_losses.append(episode_q_loss)
                episode_rnd_losses.append(episode_rnd_loss)
                episode_intrinsic_rewards.append(episode_intrinsic_reward)
                
                # Log episode summary to TensorBoard
                writer.add_scalar('Episode/Reward', episode_reward, episode+1)
                writer.add_scalar('Episode/Length', episode_length, episode+1)
                writer.add_scalar('Episode/Lines', episode_lines_cleared, episode+1)
                writer.add_scalar('Episode/Score', episode_score, episode+1)
                writer.add_scalar('Episode/PiecesPlaced', episode_pieces_placed, episode+1)
                writer.add_scalar('Episode/MaxLevel', episode_max_level, episode+1)
                writer.add_scalar('Episode/Epsilon', agent.epsilon, episode+1)
                if train_steps > 0:
                    writer.add_scalar('Episode/AvgQLoss', episode_q_loss, episode+1)
                    writer.add_scalar('Episode/AvgRNDLoss', episode_rnd_loss, episode+1)
                writer.add_scalar('Episode/IntrinsicReward', episode_intrinsic_reward, episode+1)
                
                # Log episode summary to console
                logging.info(f"Episode {episode + 1}/{num_episodes}")
                logging.info(f"Player 1 Reward: {episode_reward:.2f}")
                logging.info(f"Length: {episode_length}")
                logging.info(f"Lines Cleared: {episode_lines_cleared}")
                logging.info(f"Score: {episode_score}")
                logging.info(f"Pieces Placed: {episode_pieces_placed}")
                logging.info(f"Max Level: {episode_max_level}")
                logging.info(f"Epsilon: {agent.epsilon:.3f}")
                if train_steps > 0:
                    logging.info(f"Q-Loss: {episode_q_loss:.4f}")
                    logging.info(f"RND-Loss: {episode_rnd_loss:.4f}")
                logging.info(f"Intrinsic Reward: {episode_intrinsic_reward:.3f}")
                logging.info("")
                
                # Save model checkpoint
                if (episode + 1) % save_interval == 0:
                    try:
                        agent.save(f'checkpoints/dqn_episode_{episode + 1}.pt')
                        logging.info(f"Saved checkpoint at episode {episode + 1}")
                    except Exception as e:
                        logging.error(f"Error saving checkpoint: {str(e)}")
                
                # Evaluate agent
                if not no_eval and (episode + 1) % eval_interval == 0:
                    try:
                        eval_reward = evaluate_agent(env, agent)
                        writer.add_scalar('Evaluation/Reward', eval_reward, episode+1)
                        logging.info(f"Evaluation Reward: {eval_reward:.2f}")
                        logging.info("")
                    except Exception as e:
                        logging.error(f"Error during evaluation: {str(e)}")
                
            except Exception as e:
                logging.error(f"Error during episode {episode + 1}: {str(e)}")
                continue
        
        # Save final model and metrics
        try:
            agent.save('checkpoints/dqn_final.pt')
            # Save metrics as numpy arrays
            metrics = {
                'rewards': np.array(episode_rewards),
                'lengths': np.array(episode_lengths),
                'lines': np.array(episode_lines),
                'scores': np.array(episode_scores),
                'max_levels': np.array(episode_max_levels),
                'q_losses': np.array(q_losses),
                'rnd_losses': np.array(episode_rnd_losses),
                'intrinsic_rewards': np.array(episode_intrinsic_rewards)
            }
            np.save(os.path.join(log_dir, 'training_metrics.npy'), metrics)
            logging.info("Training completed successfully")
        except Exception as e:
            logging.error(f"Error saving final model and metrics: {str(e)}")
        
    except Exception as e:
        logging.error(f"Fatal error during training: {str(e)}")
    finally:
        env.close()
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='Enable GUI visualization')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to load')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--eval-interval', type=int, default=50, help='Evaluate agent every N episodes')
    parser.add_argument('--no-eval', action='store_true', help='Disable evaluation during training')
    parser.add_argument('--verbose', action='store_true', help='Enable per-step logging')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top actions to sample from')
    args = parser.parse_args()

    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    try:
        train_single_player(
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            visualize=args.visualize,
            checkpoint=args.checkpoint,
            no_eval=args.no_eval,
            verbose=args.verbose,
            top_k=args.top_k
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        if profiler:
            profiler.disable()
            ps = pstats.Stats(profiler, stream=sys.stdout).strip_dirs().sort_stats('cumtime')
            ps.print_stats(20)