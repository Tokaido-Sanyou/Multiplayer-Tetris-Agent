import os
import numpy as np
import torch
import logging
from ..tetris_env import TetrisEnv
from .dqn_new import DQNAgent
import pygame
import sys
from torch.utils.tensorboard import SummaryWriter

def preprocess_state(state):
    """
    Preprocess state dictionary into a flat array
    
    State Structure (from tetris_env.py):
    - grid: 20x10 matrix (0 for empty, 1 for locked, 2 for current piece)
    - next_piece: scalar ID (0-7)
    - hold_piece: scalar ID (0-7)
    
    Returns:
        Flattened array of shape (207,) containing:
        - First 200 values: Flattened grid (20x10)
        - Next value: next_piece
        - hold_piece
        - current_shape
        - current_rotation
        - current_x
        - current_y
        - can_hold
    """
    # Validate grid shape
    grid = state['grid']
    if grid.shape != (20, 10):
        raise ValueError(f"Grid should be 20x10, got shape {grid.shape}")
    
    # Convert grid to float32 and flatten
    grid_flat = grid.astype(np.float32).flatten()
    if grid_flat.shape[0] != 200:
        raise ValueError(f"Flattened grid should have 200 values, got {grid_flat.shape[0]}")
    
    # Create metadata array with validation
    metadata = np.array([
        float(state.get('next_piece', 0)),
        float(state.get('hold_piece', 0)),
        float(state.get('current_shape', 0)),
        float(state.get('current_rotation', 0)),
        float(state.get('current_x', 0)),
        float(state.get('current_y', 0)),
        float(state.get('can_hold', 1))
    ], dtype=np.float32)  # Explicitly convert to float32
    
    # Concatenate and validate final shape
    state_array = np.concatenate([grid_flat, metadata])
    if state_array.shape[0] != 207:
        raise ValueError(f"State array should have 207 values, got {state_array.shape[0]}")
    
    return state_array

def evaluate_agent(env, agent, num_episodes=5):
    """
    Evaluate agent performance over several episodes
    
    Args:
        env: Tetris environment
        agent: DQN agent
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Tuple of (rewards, scores, lines) lists
    """
    rewards = []
    scores = []
    lines = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = preprocess_state(obs)
        done = False
        episode_reward = 0
        episode_score = 0
        episode_lines = 0
        
        while not done:
            # Select action (no exploration during evaluation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action = agent.policy_net(state_tensor).max(1)[1].item()
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = preprocess_state(obs)
            
            # Update metrics
            episode_reward += reward
            episode_score += info.get('score', 0)
            episode_lines += info.get('lines_cleared', 0)
        
        rewards.append(episode_reward)
        scores.append(episode_score)
        lines.append(episode_lines)
    
    return rewards, scores, lines

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
        # Create environment
        env = TetrisEnv(single_player=True, headless=not visualize)
        
        # Create agent
        agent = DQNAgent(
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            target_update=10,
            buffer_size=100000,
            batch_size=64
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
        dqn_losses = []
        
        for episode in range(num_episodes):
            try:
                # Initialize observation and state
                obs, _ = env.reset()
                state = preprocess_state(obs)
                done = False
                episode_reward = 0
                episode_length = 0
                episode_lines_cleared = 0
                episode_score = 0
                episode_max_level = 0
                episode_loss = 0
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
                        
                        # Store experience
                        agent.memory.push(state, action, reward, next_state, done)
                        
                        # Train agent
                        loss = agent.train_step()
                        if loss is not None:
                            episode_loss += loss
                            train_steps += 1
                        
                        # Update state
                        state = next_state
                        
                        # Log episode details
                        if verbose:
                            logging.info(f"Action taken: {action}")
                            logging.info(f"Reward received: {reward}")
                            logging.info(f"Game state: {info}")
                        
                    except Exception as e:
                        logging.error(f"Error during episode {episode + 1} step: {str(e)}")
                        break
                
                # Calculate average loss
                if train_steps > 0:
                    episode_loss /= train_steps
                
                # Store episode metrics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_lines.append(episode_lines_cleared)
                episode_scores.append(episode_score)
                episode_max_levels.append(episode_max_level)
                dqn_losses.append(episode_loss)
                
                # Log episode summary
                logging.info(f"Episode {episode + 1}/{num_episodes}")
                logging.info(f"Reward: {episode_reward:.2f}")
                logging.info(f"Score: {episode_score}")
                logging.info(f"Lines Cleared: {episode_lines_cleared}")
                logging.info(f"Max Level: {episode_max_level}")
                logging.info(f"Episode Length: {episode_length}")
                logging.info(f"Epsilon: {agent.epsilon:.3f}")
                logging.info(f"Loss: {episode_loss:.4f}")
                logging.info("")
                
                # Log to TensorBoard
                writer.add_scalar('Episode/Reward', episode_reward, episode + 1)
                writer.add_scalar('Episode/Score', episode_score, episode + 1)
                writer.add_scalar('Episode/Lines', episode_lines_cleared, episode + 1)
                writer.add_scalar('Episode/MaxLevel', episode_max_level, episode + 1)
                writer.add_scalar('Episode/Length', episode_length, episode + 1)
                writer.add_scalar('Episode/Epsilon', agent.epsilon, episode + 1)
                writer.add_scalar('Episode/Loss', episode_loss, episode + 1)
                
                # Save model checkpoint
                if (episode + 1) % save_interval == 0:
                    try:
                        agent.save(f'checkpoints/dqn_checkpoint_{episode + 1}.pt')
                        logging.info(f"Saved checkpoint at episode {episode + 1}")
                    except Exception as e:
                        logging.error(f"Error saving checkpoint: {e}")
                
                # Evaluate agent
                if (episode + 1) % eval_interval == 0 and not no_eval:
                    try:
                        logging.info(f"Evaluating agent at episode {episode + 1}...")
                        eval_rewards, eval_scores, eval_lines = evaluate_agent(env, agent)
                        r_avg, r_std, r_max = np.mean(eval_rewards), np.std(eval_rewards), np.max(eval_rewards)
                        s_avg, s_std, s_max = np.mean(eval_scores), np.std(eval_scores), np.max(eval_scores)
                        l_max = np.max(eval_lines)
                        
                        # Log evaluation metrics
                        writer.add_scalar('Eval/AvgReward', r_avg, episode + 1)
                        writer.add_scalar('Eval/StdReward', r_std, episode + 1)
                        writer.add_scalar('Eval/MaxReward', r_max, episode + 1)
                        writer.add_scalar('Eval/AvgScore', s_avg, episode + 1)
                        writer.add_scalar('Eval/StdScore', s_std, episode + 1)
                        writer.add_scalar('Eval/MaxScore', s_max, episode + 1)
                        writer.add_scalar('Eval/MaxLines', l_max, episode + 1)
                        
                        logging.info(f"Eval Rewards   – avg={r_avg:.2f}, std={r_std:.2f}, max={r_max:.2f}")
                        logging.info(f"Eval Scores    – avg={s_avg:.2f}, std={s_std:.2f}, max={s_max:.2f}")
                        logging.info(f"Eval Lines Cleared – max={l_max}")
                        logging.info("")
                    except Exception as e:
                        logging.error(f"Error during evaluation: {str(e)}")
                
            except Exception as e:
                logging.error(f"Error during episode {episode + 1}: {str(e)}")
                continue
        
        # Save final model and metrics
        try:
            agent.save('checkpoints/dqn_final.pt')
            np.save('logs/training_metrics.npy', {
                'rewards': episode_rewards,
                'lengths': episode_lengths,
                'lines': episode_lines,
                'scores': episode_scores,
                'max_levels': episode_max_levels,
                'losses': dqn_losses
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--eval-interval', type=int, default=50, help='Evaluate agent every N episodes')
    parser.add_argument('--visualize', action='store_true', help='Enable GUI visualization')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to load')
    parser.add_argument('--no-eval', action='store_true', help='Disable evaluation during training')
    parser.add_argument('--verbose', action='store_true', help='Enable per-step logging')
    args = parser.parse_args()
    
    train_single_player(
        num_episodes=args.episodes,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        visualize=args.visualize,
        checkpoint=args.checkpoint,
        no_eval=args.no_eval,
        verbose=args.verbose
    )