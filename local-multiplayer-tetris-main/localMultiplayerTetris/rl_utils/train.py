import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from ..tetris_env import TetrisEnv
from .actor_critic import ActorCriticAgent
from .replay_buffer import ReplayBuffer
from torch.distributions import Categorical
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

class Discriminator(nn.Module):
    """
    Discriminator network for MA-AIRL
    Estimates the probability that a state-action pair comes from a particular agent
    """
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state, action):
        """
        Forward pass through discriminator
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
        Returns:
            Probability that state-action pair comes from expert
        """
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class ParallelTetrisEnv:
    """
    Manages multiple parallel Tetris environments
    """
    def __init__(self, num_envs, single_player=True, headless=True):
        self.envs = [TetrisEnv(single_player=single_player, headless=headless) for _ in range(num_envs)]
        self.num_envs = num_envs
        
    def reset(self):
        """Reset all environments"""
        return [env.reset() for env in self.envs]
    
    def step(self, actions):
        """Step all environments with respective actions"""
        results = []
        for env, action in zip(self.envs, actions):
            next_state, reward, done, info = env.step(action)
            results.append((next_state, reward, done, info))
        return zip(*results)
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()

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

def train_mairl(num_agents=4, num_episodes=1000, save_interval=100, eval_interval=50):
    """
    Train multiple agents using MA-AIRL
    
    Args:
        num_agents: Number of agents to train simultaneously
        num_episodes: Number of episodes to train for
        save_interval: Save models every N episodes
        eval_interval: Evaluate agents every N episodes
    """
    # Create directories
    os.makedirs('checkpoints/mairl', exist_ok=True)
    os.makedirs('logs/mairl', exist_ok=True)
    
    # Initialize parallel environments
    parallel_env = ParallelTetrisEnv(num_agents)
    
    # Initialize agents
    state_dim = 202  # 20x10 grid + next_piece + hold_piece
    action_dim = 8   # 8 possible actions
    agents = [ActorCriticAgent(state_dim, action_dim) for _ in range(num_agents)]
    
    # Initialize discriminators
    discriminators = [Discriminator(state_dim, action_dim) for _ in range(num_agents)]
    disc_optimizers = [optim.Adam(d.parameters(), lr=3e-4) for d in discriminators]
    
    # Training metrics
    metrics = {i: {
        'rewards': [], 'lengths': [], 'lines': [],
        'scores': [], 'levels': [], 'actor_losses': [],
        'critic_losses': [], 'disc_losses': []
    } for i in range(num_agents)}
    
    # Experience buffers for each agent
    buffers = [ReplayBuffer(100000) for _ in range(num_agents)]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for agent in agents:
        agent.to(device)
    for disc in discriminators:
        disc.to(device)
    
    for episode in range(num_episodes):
        # Reset environments
        states = parallel_env.reset()
        states = [preprocess_state(s) for s in states]
        dones = [False] * num_agents
        episode_data = {i: {
            'reward': 0, 'length': 0, 'lines': 0,
            'score': 0, 'level': 0, 'actor_loss': 0,
            'critic_loss': 0, 'disc_loss': 0, 'steps': 0
        } for i in range(num_agents)}
        
        while not all(dones):
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(agents):
                if not dones[i]:
                    action = agent.select_action(states[i])
                    actions.append(action)
                else:
                    actions.append(0)  # Dummy action for done environments
            
            # Step environments
            next_states, rewards, new_dones, infos = parallel_env.step(actions)
            next_states = [preprocess_state(s) for s in next_states]
            
            # Update discriminators and get intrinsic rewards
            intrinsic_rewards = []
            for i in range(num_agents):
                if not dones[i]:
                    # Convert state-action to tensor
                    state_tensor = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
                    action_onehot = torch.zeros(1, action_dim).to(device)
                    action_onehot[0, actions[i]] = 1
                    
                    # Get discriminator predictions
                    disc_pred = discriminators[i](state_tensor, action_onehot)
                    
                    # Calculate intrinsic reward
                    intrinsic_reward = torch.log(disc_pred + 1e-8).item()
                    
                    # Update discriminator
                    # Sample other agent transitions as negative examples
                    neg_indices = [j for j in range(num_agents) if j != i]
                    neg_idx = np.random.choice(neg_indices)
                    neg_state = torch.FloatTensor(states[neg_idx]).unsqueeze(0).to(device)
                    neg_action = torch.zeros(1, action_dim).to(device)
                    neg_action[0, actions[neg_idx]] = 1
                    
                    # Train discriminator
                    disc_optimizers[i].zero_grad()
                    pos_pred = discriminators[i](state_tensor, action_onehot)
                    neg_pred = discriminators[i](neg_state, neg_action)
                    
                    disc_loss = -(torch.log(pos_pred + 1e-8) + torch.log(1 - neg_pred + 1e-8)).mean()
                    disc_loss.backward()
                    disc_optimizers[i].step()
                    
                    episode_data[i]['disc_loss'] += disc_loss.item()
                    intrinsic_rewards.append(intrinsic_reward)
                else:
                    intrinsic_rewards.append(0)
            
            # Update agents with combined rewards
            for i in range(num_agents):
                if not dones[i]:
                    combined_reward = rewards[i] + 0.1 * intrinsic_rewards[i]
                    
                    # Store transition
                    buffers[i].push(
                        states[i], actions[i], combined_reward,
                        next_states[i], new_dones[i], infos[i]
                    )
                    
                    # Train agent
                    if len(buffers[i]) >= 128:  # Minimum batch size
                        losses = agents[i].train()
                        if losses is not None:
                            actor_loss, critic_loss = losses
                            episode_data[i]['actor_loss'] += actor_loss
                            episode_data[i]['critic_loss'] += critic_loss
                            episode_data[i]['steps'] += 1
                    
                    # Update metrics
                    episode_data[i]['reward'] += rewards[i]
                    episode_data[i]['length'] += 1
                    episode_data[i]['lines'] += infos[i].get('lines_cleared', 0)
                    episode_data[i]['score'] += infos[i].get('score', 0)
                    episode_data[i]['level'] = max(
                        episode_data[i]['level'],
                        infos[i].get('level', 0)
                    )
            
            # Update states and dones
            states = next_states
            dones = new_dones
        
        # Update exploration rates
        for agent in agents:
            agent.update_epsilon()
        
        # Log episode results
        for i in range(num_agents):
            steps = max(1, episode_data[i]['steps'])
            metrics[i]['rewards'].append(episode_data[i]['reward'])
            metrics[i]['lengths'].append(episode_data[i]['length'])
            metrics[i]['lines'].append(episode_data[i]['lines'])
            metrics[i]['scores'].append(episode_data[i]['score'])
            metrics[i]['levels'].append(episode_data[i]['level'])
            metrics[i]['actor_losses'].append(episode_data[i]['actor_loss'] / steps)
            metrics[i]['critic_losses'].append(episode_data[i]['critic_loss'] / steps)
            metrics[i]['disc_losses'].append(episode_data[i]['disc_loss'] / steps)
            
            print(f"Agent {i} - Episode {episode + 1}/{num_episodes}")
            print(f"Reward: {episode_data[i]['reward']:.2f}")
            print(f"Length: {episode_data[i]['length']}")
            print(f"Lines: {episode_data[i]['lines']}")
            print(f"Score: {episode_data[i]['score']}")
            print(f"Level: {episode_data[i]['level']}")
            print(f"Epsilon: {agents[i].epsilon:.3f}")
            print()
        
        # Save checkpoints
        if (episode + 1) % save_interval == 0:
            for i in range(num_agents):
                agents[i].save(f'checkpoints/mairl/agent_{i}_episode_{episode + 1}.pt')
                torch.save(discriminators[i].state_dict(),
                         f'checkpoints/mairl/discriminator_{i}_episode_{episode + 1}.pt')
        
        # Evaluate agents
        if (episode + 1) % eval_interval == 0:
            eval_rewards = evaluate_mairl_agents(agents)
            for i, reward in enumerate(eval_rewards):
                print(f"Agent {i} Evaluation Reward: {reward:.2f}")
            print()
    
    # Save final models and metrics
    for i in range(num_agents):
        agents[i].save(f'checkpoints/mairl/agent_{i}_final.pt')
        torch.save(discriminators[i].state_dict(),
                  f'checkpoints/mairl/discriminator_{i}_final.pt')
        np.save(f'logs/mairl/agent_{i}_metrics.npy', metrics[i])
    
    parallel_env.close()

def evaluate_mairl_agents(agents, num_episodes=5):
    """
    Evaluate all agents
    
    Args:
        agents: List of trained agents
        num_episodes: Number of evaluation episodes per agent
    
    Returns:
        List of average rewards for each agent
    """
    eval_rewards = []
    
    for agent in agents:
        env = TetrisEnv(single_player=True, headless=True)
        agent_rewards = []
        
        for _ in range(num_episodes):
            state = env.reset()
            state = preprocess_state(state)
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    action_probs, _ = agent.network(state_tensor)
                    action = action_probs.argmax().item()
                
                next_state, reward, done, _ = env.step(action)
                next_state = preprocess_state(next_state)
                episode_reward += reward
                state = next_state
            
            agent_rewards.append(episode_reward)
        
        eval_rewards.append(np.mean(agent_rewards))
        env.close()
    
    return eval_rewards

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Train agents
    train_mairl(num_agents=4, num_episodes=1000) 