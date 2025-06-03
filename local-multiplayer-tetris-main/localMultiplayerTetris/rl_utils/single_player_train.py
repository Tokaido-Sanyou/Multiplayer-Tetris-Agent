import os
import numpy as np
import torch
import logging
from ..tetris_env import TetrisEnv
from .actor_critic import ActorCriticAgent
from .train import preprocess_state, evaluate_agent
import pygame
import cProfile
import pstats
import sys
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

# Helper to configure root logger with console output each run
def _configure_logging(verbose: bool=False):
    """Configure root logger; force reconfiguration each run."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log', mode='w'),
            logging.StreamHandler()
        ],
        force=True  # Python>=3.8: override previous config
    )

# Call early so rest of module uses it
_configure_logging(verbose='--verbose' in sys.argv)

def train_single_player(num_episodes=10000, save_interval=100, eval_interval=50, visualize=True, checkpoint=None, no_eval=False, verbose=False):
    """
    Train an agent as player 1 in the Tetris environment
    
    Args:
        num_episodes: Number of episodes to train for
        save_interval: Save model every N episodes
        eval_interval: Evaluate agent every N episodes
        visualize: Whether to render the environment during training
        checkpoint: Path to checkpoint file to load
        no_eval: Whether to disable evaluation during training
        verbose: Enable per-step logging
    """
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir='logs/tensorboard')
    try:
        # Create environment; show window if visualize=True
        env = TetrisEnv(single_player=True, headless=not visualize)
        
        # Create agent
        state_dim = 207  # 20x10 grid + next_piece + hold_piece + current_shape + current_rotation + current_x + current_y + can_hold
        action_dim = 41   # 40 placements + 1 hold
        agent = ActorCriticAgent(
            state_dim,
            action_dim,
            gamma_start=0.9,
            gamma_end=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            schedule_episodes=num_episodes
        )
        print("Using device:", agent.device)
        if checkpoint:
            try:
                agent.load(checkpoint)
                logging.info(f"Loaded checkpoint from {checkpoint}")
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
        
        # Create directories for saving
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        episode_lines = []
        episode_scores = []
        episode_max_levels = []
        actor_losses = []
        critic_losses = []
        
        for episode in range(num_episodes):
            episode_pieces_placed = 0
            try:
                # Initialize observation and state
                obs, _ = env.reset()  # Unpack the (obs, info) tuple
                state = preprocess_state(obs)
                done = False
                episode_reward = 0
                episode_length = 0
                episode_lines_cleared = 0
                episode_score = 0
                episode_max_level = 0
                episode_actor_loss = 0
                episode_critic_loss = 0
                train_steps = 0
                
                while not done:
                    try:
                        # Select and perform action
                        action = agent.select_action(state)
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        next_state = preprocess_state(next_obs)
                        
                        # Render if visualization enabled
                        if visualize:
                            env.render()
                        
                        # Update metrics
                        episode_reward += reward
                        episode_length += 1
                        episode_lines_cleared += info.get('lines_cleared', 0)
                        episode_score += info.get('score', 0)
                        episode_max_level = max(episode_max_level, info.get('level', 0))
                        episode_pieces_placed += int(info.get('piece_placed', False))
                        
                        # Store transition in replay buffer
                        agent.memory.push(obs, action, reward, next_obs, done, info)
                        
                        # Train agent
                        losses = agent.train()
                        if losses is not None:
                            actor_loss, critic_loss = losses
                            episode_actor_loss += actor_loss
                            episode_critic_loss += critic_loss
                            train_steps += 1
                        
                        # Update state and observation
                        state = next_state
                        obs = next_obs
                        
                        # Log episode details
                        if verbose:
                            logging.info(f"Action taken: {action}")
                            logging.info(f"Reward received: {reward}")
                            logging.info(f"Game state: {info}")
                        
                    except Exception as e:
                        logging.error(f"Error during episode {episode + 1} step: {str(e)}")
                        break
                
                # Debug end-of-episode reason
                if info.get('invalid_move', False):
                    reason = 'invalid placement'
                elif terminated:
                    reason = 'game over (blocks above grid)'
                elif truncated:
                    reason = 'max steps reached'
                else:
                    reason = 'unknown'
                print(f"Episode {episode+1} ended: {reason}", flush=True)
                # Update exploration / gamma schedules
                agent.update_schedules(episode+1)
                
                # Calculate average losses
                if train_steps > 0:
                    episode_actor_loss /= train_steps
                    episode_critic_loss /= train_steps
                
                # Store episode metrics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_lines.append(episode_lines_cleared)
                episode_scores.append(episode_score)
                episode_max_levels.append(episode_max_level)
                actor_losses.append(episode_actor_loss)
                critic_losses.append(episode_critic_loss)
                
                # Log episode summary
                logging.info(f"Episode {episode + 1}/{num_episodes}")
                logging.info(f"Reward: {episode_reward:.2f}")
                logging.info(f"Score: {episode_score}")
                logging.info(f"Lines Cleared: {episode_lines_cleared}")
                logging.info(f"Max Level: {episode_max_level}")
                logging.info(f"Episode Length: {episode_length}")
                logging.info(f"Pieces Placed: {episode_pieces_placed}")
                logging.info(f"Epsilon: {agent.epsilon:.3f}")
                logging.info(f"Gamma: {agent.gamma:.3f}")
                if train_steps > 0:
                    logging.info(f"Actor Loss: {episode_actor_loss:.4f}")
                    logging.info(f"Critic Loss: {episode_critic_loss:.4f}")
                logging.info("")
                
                # Log to TensorBoard
                writer.add_scalar('Episode/Reward', episode_reward, episode + 1)
                writer.add_scalar('Episode/Score', episode_score, episode + 1)
                writer.add_scalar('Episode/Lines', episode_lines_cleared, episode + 1)
                writer.add_scalar('Episode/MaxLevel', episode_max_level, episode + 1)
                writer.add_scalar('Episode/Length', episode_length, episode + 1)
                writer.add_scalar('Episode/PiecesPlaced', episode_pieces_placed, episode + 1)
                writer.add_scalar('Episode/Epsilon', agent.epsilon, episode + 1)
                writer.add_scalar('Episode/Gamma', agent.gamma, episode + 1)
                if train_steps > 0:
                    writer.add_scalar('Loss/Actor', episode_actor_loss, episode + 1)
                    writer.add_scalar('Loss/Critic', episode_critic_loss, episode + 1)
                
                # Print episode summary
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"Score: {episode_score}")
                print(f"Lines Cleared: {episode_lines_cleared}")
                print(f"Max Level: {episode_max_level}")
                print(f"Episode Length: {episode_length}")
                print(f"Pieces Placed: {episode_pieces_placed}")
                print(f"Epsilon: {agent.epsilon:.3f}")
                print(f"Gamma: {agent.gamma:.3f}")
                if train_steps > 0:
                    print(f"Actor Loss: {episode_actor_loss:.4f}")
                    print(f"Critic Loss: {episode_critic_loss:.4f}")
                print()
                
                # Save model checkpoint
                if (episode + 1) % save_interval == 0:
                    try:
                        agent.save(f'checkpoints/actor_critic_episode_{episode + 1}.pt')
                        logging.info(f"Saved checkpoint at episode {episode + 1}")
                    except Exception as e:
                        logging.error(f"Error saving checkpoint: {e}")
                
                # Evaluate agent
                if (episode + 1) % eval_interval == 0 and not no_eval:
                    try:
                        logging.info(f"Evaluating agent at episode {episode + 1}...")
                        # evaluate_agent now returns (rewards, scores, lines)
                        eval_rewards, eval_scores, eval_lines = evaluate_agent(env, agent)
                        r_avg, r_std, r_max = np.mean(eval_rewards), np.std(eval_rewards), np.max(eval_rewards)
                        s_avg, s_std, s_max = np.mean(eval_scores), np.std(eval_scores), np.max(eval_scores)
                        l_max = np.max(eval_lines)
                        # log to console
                        logging.info(f"Eval Rewards   – avg={r_avg:.2f}, std={r_std:.2f}, max={r_max:.2f}")
                        logging.info(f"Eval Scores    – avg={s_avg:.2f}, std={s_std:.2f}, max={s_max:.2f}")
                        logging.info(f"Eval Lines Cleared – max={l_max}")
                        # log to TensorBoard
                        writer.add_scalar('Eval/AvgReward', r_avg, episode + 1)
                        writer.add_scalar('Eval/StdReward', r_std, episode + 1)
                        writer.add_scalar('Eval/MaxReward', r_max, episode + 1)
                        writer.add_scalar('Eval/AvgScore', s_avg, episode + 1)
                        writer.add_scalar('Eval/StdScore', s_std, episode + 1)
                        writer.add_scalar('Eval/MaxScore', s_max, episode + 1)
                        writer.add_scalar('Eval/MaxLines', l_max, episode + 1)
                        print(f"Eval Rewards   – avg={r_avg:.2f}, std={r_std:.2f}, max={r_max:.2f}")
                        print(f"Eval Scores    – avg={s_avg:.2f}, std={s_std:.2f}, max={s_max:.2f}")
                        print(f"Eval Lines Cleared – max={l_max}")
                        logging.info("")
                    except Exception as e:
                        logging.error(f"Error during evaluation: {str(e)}")
                
            except Exception as e:
                logging.error(f"Error during episode {episode + 1}: {str(e)}")
                continue
        
        # Save final model and metrics
        try:
            agent.save('checkpoints/actor_critic_final.pt')
            np.save('logs/training_metrics.npy', {
                'rewards': episode_rewards,
                'lengths': episode_lengths,
                'lines': episode_lines,
                'scores': episode_scores,
                'max_levels': episode_max_levels,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses
            })
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
    parser.add_argument('--eval', action='store_true', help='Run evaluation only (no training, no exploration, no gradients)')
    parser.add_argument('--visualize', action='store_true', help='Enable GUI visualization')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to load')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--eval-interval', type=int, default=50, help='Evaluate agent every N episodes')
    parser.add_argument('--no-eval', action='store_true', help='Disable evaluation during training')
    parser.add_argument('--verbose', action='store_true', help='Enable per-step logging')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    args = parser.parse_args()

    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    try:
        if args.eval:
            # Evaluation-only mode: no training, no exploration, no gradients
            from .train import preprocess_state
            env = TetrisEnv(single_player=True, headless=not args.visualize)
            state_dim = 207
            action_dim = 41
            agent = ActorCriticAgent(
                state_dim,
                action_dim,
                gamma_start=0.9,
                gamma_end=0.99,
                epsilon_start=0.0,
                epsilon_end=0.0,
                schedule_episodes=1
            )
            if args.checkpoint:
                agent.load(args.checkpoint)
            agent.epsilon = 0.0
            num_eval_episodes = args.episodes
            rewards = []
            scores = []
            lines = []
            for ep in range(num_eval_episodes):
                obs, _ = env.reset()
                state = preprocess_state(obs)
                done = False
                total_reward = 0
                total_score = 0
                total_lines = 0
                steps = 0
                with torch.no_grad():
                    while not done:
                        action = agent.select_action(state)
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        state = preprocess_state(next_obs)
                        total_reward += reward
                        total_score += info.get('score', 0)
                        total_lines += info.get('lines_cleared', 0)
                        steps += 1
                        if args.visualize:
                            env.render()
                rewards.append(total_reward)
                scores.append(total_score)
                lines.append(total_lines)
                print(f"Eval Episode {ep+1}/{num_eval_episodes} | Reward: {total_reward:.2f} | Score: {total_score} | Lines: {total_lines} | Steps: {steps}")
            print(f"\nEval Results over {num_eval_episodes} episodes:")
            print(f"Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"Avg Score: {np.mean(scores):.2f}")
            print(f"Avg Lines: {np.mean(lines):.2f}")
            env.close()
        else:
            train_single_player(
                num_episodes=args.episodes,
                save_interval=args.save_interval,
                eval_interval=args.eval_interval,
                visualize=args.visualize,
                checkpoint=args.checkpoint,
                no_eval=args.no_eval,
                verbose=args.verbose
            )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        if profiler:
            profiler.disable()
            ps = pstats.Stats(profiler, stream=sys.stdout).strip_dirs().sort_stats('cumtime')
            ps.print_stats(20)