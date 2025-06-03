import os
import numpy as np
import torch
import logging
import pygame
import cProfile
import pstats
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter

# Handle both direct execution and module import
try:
    from ..tetris_env import TetrisEnv
    from .state_model import StateModel
    from .hierarchical_agent import HierarchicalAgent
except ImportError:
    # Direct execution - imports without relative paths
    from .tetris_env import TetrisEnv
    from state_model import StateModel
    from hierarchical_agent import HierarchicalAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def preprocess_state(state):
    # flatten grid and piece metadata into vector (206 dims): grid(200)+cur_metadata(6)
    grid = state['grid'].flatten()
    # current piece metadata assumed appended in state dict
    cur_meta = np.array([state['cur_shape'], state['cur_rot'], state['cur_x'], state['cur_y'], state['next_piece'], state['hold_piece']])
    return np.concatenate([grid, cur_meta])

def train_single_player(num_episodes=1000, save_interval=100, eval_interval=50,
                        visualize=True, checkpoint=None, no_eval=False, verbose=False):
    """
    Train an agent as player 1 in the Tetris environment
    
    Args:
        num_episodes: Number of episodes to train for
        save_interval: Save model every N episodes
        visualize: Whether to render the environment during training
        checkpoint: Path to checkpoint file to load
        no_eval: Disable evaluation during training
        verbose: Enable per-step logging
    """
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir='logs/tensorboard')
    try:
        # Create environment; show window if visualize=True
        env = TetrisEnv(single_player=True, headless=not visualize)
        
        # Initialize state prediction model and hierarchical agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_model = StateModel(state_dim=206).to(device)
        agent = HierarchicalAgent(state_model, device=device)
        if checkpoint:
            try:
                # load state_model weights
                state_model.load_state_dict(torch.load(checkpoint, map_location=device))
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
                obs = env.reset()
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
                        next_obs, reward, done, info = env.step(action)
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
                        
                        # Store transition in replay buffer (use raw observation dict)
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
                
                # Update exploration rate
                agent.update_epsilon()
                
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
                logging.info(f"Player 1 Reward: {episode_reward:.2f}")
                logging.info(f"Length: {episode_length}")
                logging.info(f"Lines Cleared: {episode_lines_cleared}")
                logging.info(f"Score: {episode_score}")
                logging.info(f"Pieces Placed: {episode_pieces_placed}")
                logging.info(f"Max Level: {episode_max_level}")
                logging.info(f"Epsilon: {agent.epsilon:.3f}")
                if train_steps > 0:
                    logging.info(f"Actor Loss: {episode_actor_loss:.4f}")
                    logging.info(f"Critic Loss: {episode_critic_loss:.4f}")
                logging.info("")
                
                # Log to TensorBoard
                writer.add_scalar('Train/Reward', episode_reward, episode+1)
                writer.add_scalar('Train/Length', episode_length, episode+1)
                writer.add_scalar('Train/Lines', episode_lines_cleared, episode+1)
                writer.add_scalar('Train/Score', episode_score, episode+1)
                writer.add_scalar('Train/PiecesPlaced', episode_pieces_placed, episode+1)
                writer.add_scalar('Train/MaxLevel', episode_max_level, episode+1)
                writer.add_scalar('Train/Epsilon', agent.epsilon, episode+1)
                if train_steps > 0:
                    writer.add_scalar('Train/ActorLoss', episode_actor_loss, episode+1)
                    writer.add_scalar('Train/CriticLoss', episode_critic_loss, episode+1)
                
                # Save model checkpoint
                if (episode + 1) % save_interval == 0:
                    try:
                        agent.save(f'checkpoints/actor_critic_episode_{episode + 1}.pt')
                        logging.info(f"Saved checkpoint at episode {episode + 1}")
                    except Exception as e:
                        logging.error(f"Error saving checkpoint: {str(e)}")
                
                # evaluation deprecated in single_player_train
                
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
        train_single_player(
            num_episodes=args.episodes,
            save_interval=args.save_interval,
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