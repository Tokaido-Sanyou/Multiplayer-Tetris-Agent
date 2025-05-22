import torch
import numpy as np
import os
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent

def preprocess_state(state):
    """
    Preprocess state dictionary into a flat tensor
    
    State Structure (from tetris_env.py):
    - grid: 20x10 matrix (0 for empty, 1-7 for different piece colors)
    - current_piece: 4x4 matrix (0 for empty, 1 for filled)
    - next_piece: 4x4 matrix (0 for empty, 1 for filled)
    - hold_piece: 4x4 matrix (0 for empty, 1 for filled)
    
    Returns:
        Flattened tensor of shape (248,) containing:
        - First 200 values: Flattened grid
        - Next 16 values: Flattened current piece
        - Next 16 values: Flattened next piece
        - Last 16 values: Flattened hold piece
    """
    grid = state['grid'].flatten()
    current_piece = state['current_piece'].flatten()
    next_piece = state['next_piece'].flatten()
    hold_piece = state['hold_piece'].flatten()
    
    return np.concatenate([grid, current_piece, next_piece, hold_piece])

def train_dqn(env, agent, num_episodes, save_interval=100, eval_interval=50):
    """
    Train DQN agent on Tetris environment
    
    Args:
        env: TetrisEnv instance
        agent: DQNAgent instance
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
    
    for episode in range(num_episodes):
        state = env.reset()
        state = preprocess_state(state)
        done = False
        episode_reward = 0
        episode_length = 0
        episode_lines_cleared = 0
        episode_score = 0
        episode_max_level = 0
        
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
            loss = agent.train()
            
            # Update state
            state = next_state
            
            # Update target network
            if episode_length % agent.target_update == 0:
                agent.update_target_network()
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_lines.append(episode_lines_cleared)
        episode_scores.append(episode_score)
        episode_max_levels.append(episode_max_level)
        
        # Print episode summary
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Length: {episode_length}")
        print(f"Lines Cleared: {episode_lines_cleared}")
        print(f"Score: {episode_score}")
        print(f"Max Level: {episode_max_level}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        if loss is not None:
            print(f"Loss: {loss:.4f}")
        print()
        
        # Save model checkpoint
        if (episode + 1) % save_interval == 0:
            agent.save(f'checkpoints/dqn_episode_{episode + 1}.pt')
        
        # Evaluate agent
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent)
            print(f"Evaluation Reward: {eval_reward:.2f}")
            print()
    
    # Save final model and metrics
    agent.save('checkpoints/dqn_final.pt')
    np.save('logs/training_metrics.npy', {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'lines': episode_lines,
        'scores': episode_scores,
        'max_levels': episode_max_levels
    })

def evaluate_agent(env, agent, num_episodes=10):
    """
    Evaluate agent's performance
    
    Args:
        env: TetrisEnv instance
        agent: DQNAgent instance
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
                action = agent.policy_net(state_tensor).argmax().item()
            
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
    state_dim = 248  # 20x10 grid + 3x(4x4) pieces
    action_dim = 7   # 7 possible actions
    agent = DQNAgent(state_dim, action_dim)
    
    # Train agent
    train_dqn(env, agent, num_episodes=1000) 