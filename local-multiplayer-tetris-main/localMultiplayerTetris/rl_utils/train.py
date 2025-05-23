import os
import numpy as np
import torch
from ..tetris_env import TetrisEnv
from .actor_critic import ActorCriticAgent
from .replay_buffer import ReplayBuffer

def preprocess_state(state):
    """
    Preprocess state dictionary into a flat array
    
    State Structure (from tetris_env.py):
    - grid: 20x10 matrix (0 for empty, 1 for locked, 2 for current piece)
    - next_piece: scalar ID (0-7)
    - hold_piece: scalar ID (0-7)
    
    Returns:
        Flattened array of shape (202,) containing:
        - First 200 values: Flattened grid
        - Next value: next_piece
        - Last value: hold_piece
    """
    grid = state['grid'].flatten()
    next_piece = np.array([state['next_piece']])
    hold_piece = np.array([state['hold_piece']])
    return np.concatenate([grid, next_piece, hold_piece])

def train_actor_critic(env, agent, num_episodes, save_interval=100, eval_interval=50):
    """
    Train Actor-Critic agent on Tetris environment
    
    Args:
        env: TetrisEnv instance
        agent: ActorCriticAgent instance
        num_episodes: Number of episodes to train for
        save_interval: Save model every N episodes
        eval_interval: Evaluate agent every N episodes
    
    Action Space (from tetris_env.py):
    - 0: Move Left
    - 1: Move Right
    - 2: Move Down
    - 3: Rotate Clockwise
    - 4: Rotate Counter-clockwise
    - 5: Hard Drop
    - 6: Hold Piece
    
    Reward Structure (from tetris_env.py):
    - +100 per line cleared
    - -100 for game over
    - -0.1 per step (to encourage faster play)
    """
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
        state = env.reset()
        state = preprocess_state(state)
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
            next_state, reward, done, info = env.step(action)
            next_state = preprocess_state(next_state)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            episode_lines_cleared += info.get('lines_cleared', 0)
            episode_score += info.get('score', 0)
            episode_max_level = max(episode_max_level, info.get('level', 0))
            
            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state, done, info)
            
            # Train agent
            losses = agent.train()
            if losses is not None:
                actor_loss, critic_loss = losses
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                train_steps += 1
            
            # Update state
            state = next_state
        
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
        
        # Evaluate agent
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent)
            print(f"Evaluation Reward: {eval_reward:.2f}")
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

def evaluate_agent(env, agent, num_episodes=10):
    """
    Evaluate agent's performance
    
    Args:
        env: TetrisEnv instance
        agent: ActorCriticAgent instance
        num_episodes: Number of episodes to evaluate
    
    Returns:
        Average reward over evaluation episodes
    """
    eval_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        state = preprocess_state(state)
        done = False
        episode_reward = 0
        
        while not done:
            # Select action without exploration
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action_probs, _ = agent.network(state_tensor)
                action = action_probs.argmax().item()
            
            # Perform action
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            
            # Update metrics
            episode_reward += reward
            state = next_state
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)

if __name__ == '__main__':
    # Create environment and agent
    env = TetrisEnv()
    state_dim = 202  # 20x10 grid + 2 scalars
    action_dim = 7   # 7 possible actions
    agent = ActorCriticAgent(state_dim, action_dim)
    
    # Train agent
    train_actor_critic(env, agent, num_episodes=1000) 