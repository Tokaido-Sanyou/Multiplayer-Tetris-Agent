import os
import numpy as np
import torch
import logging
from ..tetris_env import TetrisEnv
from .actor_critic import ActorCriticAgent
from .replay_buffer import ReplayBuffer
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
        - First 200 values: Flattened grid
        - Next value: next_piece
        - hold_piece
        - current_shape
        - current_rotation
        - current_x
        - current_y
        - can_hold
    """
    # Flatten grid and append metadata scalars
    grid_flat = state['grid'].flatten().astype(np.float32)  # 200 values
    metadata = np.array([
        state['next_piece'],
        state['hold_piece'],
        state.get('current_shape', 0),
        state.get('current_rotation', 0),
        state.get('current_x', 0),
        state.get('current_y', 0),
        state.get('can_hold', 1)
    ], dtype=np.float32)  # 7 values
    return np.concatenate([grid_flat, metadata])

def train_actor_critic(env, agent, num_episodes, save_interval=100, eval_interval=50):
    """
    Train Actor-Critic agent on Tetris environment
    
    Args:
        env: TetrisEnv instance
        agent: ActorCriticAgent instance
        num_episodes: Number of episodes to train for
        save_interval: Save model every N episodes
        eval_interval: Evaluate agent every N episodes
    
    Action Space:
    - Flattened placement index = rotation*10 + column (4×10=40 actions)
    
    Reward Structure (from tetris_env.py):
    - +100 per line cleared
    - -100 for game over
    - -0.1 per step (to encourage faster play)
    """
    # Create directories for saving and TensorBoard logs
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    tb_log_dir = os.path.join('logs', 'tensorboard')
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_lines = []
    episode_scores = []
    episode_max_levels = []
    actor_losses = []
    critic_losses = []
    
    for episode in range(num_episodes):
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
            # Select and perform action
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_obs)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            episode_lines_cleared += info.get('lines_cleared', 0)
            episode_score += info.get('score', 0)
            episode_max_level = max(episode_max_level, info.get('level', 0))
            
            # Store transition in replay buffer (use raw observation dictionaries)
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
        # TensorBoard logging
        writer.add_scalar('Episode/Reward', episode_reward, episode + 1)
        writer.add_scalar('Episode/Length', episode_length, episode + 1)
        writer.add_scalar('Episode/LinesCleared', episode_lines_cleared, episode + 1)
        writer.add_scalar('Episode/Score', episode_score, episode + 1)
        writer.add_scalar('Episode/MaxLevel', episode_max_level, episode + 1)
        writer.flush()


        # Print episode summary
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Length: {episode_length}")
        print(f"Lines Cleared: {episode_lines_cleared}")
        print(f"Score: {episode_score}")
        print(f"Max Level: {episode_max_level}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        if train_steps > 0:
            print(f"Actor Loss: {episode_actor_loss:.4f}")
            print(f"Critic Loss: {episode_critic_loss:.4f}")
        print()
        
        # Save model checkpoint
        if (episode + 1) % save_interval == 0:
            agent.save(f'checkpoints/actor_critic_episode_{episode + 1}.pt')
        
        # Evaluate agent and log statistics
        if (episode + 1) % eval_interval == 0:
            # get per-episode metrics: reward, score, lines
            eval_rewards, eval_scores, eval_lines = evaluate_agent(env, agent)
            r_avg, r_std, r_max = np.mean(eval_rewards), np.std(eval_rewards), np.max(eval_rewards)
            s_avg, s_std, s_max = np.mean(eval_scores), np.std(eval_scores), np.max(eval_scores)
            l_max = np.max(eval_lines)
            # print
            print(f"Eval Rewards   – avg={r_avg:.2f}, std={r_std:.2f}, max={r_max:.2f}")
            print(f"Eval Scores    – avg={s_avg:.2f}, std={s_std:.2f}, max={s_max:.2f}")
            print(f"Eval Lines Cleared – max={l_max}")
            # TensorBoard scalars
            writer.add_scalar('Eval/AvgReward', r_avg, episode + 1)
            writer.add_scalar('Eval/StdReward', r_std, episode + 1)
            writer.add_scalar('Eval/MaxReward', r_max, episode + 1)
            writer.add_scalar('Eval/AvgScore', s_avg, episode + 1)
            writer.add_scalar('Eval/StdScore', s_std, episode + 1)
            writer.add_scalar('Eval/MaxScore', s_max, episode + 1)
            writer.add_scalar('Eval/MaxLines', l_max, episode + 1)
            writer.add_scalar('Episode/Epsilon', agent.epsilon, episode + 1)
            if train_steps > 0:
                writer.add_scalar('Loss/Actor', episode_actor_loss, episode + 1)
                writer.add_scalar('Loss/Critic', episode_critic_loss, episode + 1)
            print()
    
    # Save final model and metrics
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
    # Close TensorBoard writer
    writer.close()

def evaluate_agent(env, agent, num_episodes=5):
    """
    Evaluate agent performance over several episodes
    Args:
        env: TetrisEnv instance
        agent: ActorCriticAgent instance
        num_episodes: Number of episodes to evaluate over
    Returns:
        Tuple of (rewards, scores, lines_cleared) lists
    """
    rewards = []
    scores = []
    lines = []
    
    # Store original epsilon and set to 0 for deterministic evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    try:
        for i in range(num_episodes):
            obs, _ = env.reset()
            state = preprocess_state(obs)
            done = False
            total_reward = 0
            total_score = 0
            total_lines = 0
            
            while not done:
                with torch.no_grad():
                    action = agent.select_action(state)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    state = preprocess_state(next_obs)
                    total_reward += reward
                    total_score += info.get('score', 0)
                    total_lines += info.get('lines_cleared', 0)
            
            rewards.append(total_reward)
            scores.append(total_score)
            lines.append(total_lines)
    finally:
        # Restore original epsilon
        agent.epsilon = original_epsilon
    
    return rewards, scores, lines

if __name__ == '__main__':
    # Create environment and agent
    env = TetrisEnv()
    state_dim = 207  # 200 grid + 7 metadata incl can_hold
    action_dim = 41  # 40 placement + hold
    agent = ActorCriticAgent(state_dim, action_dim)
    
    # Train agent
    train_actor_critic(env, agent, num_episodes=1000)