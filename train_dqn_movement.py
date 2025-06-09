#!/usr/bin/env python3
"""
🎮 DQN MOVEMENT AGENT TRAINING

Movement agent that takes 800 outputs from locked agent and converts to movement actions.
This is part of the hierarchical DQN structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import argparse
from collections import deque
from typing import Dict, Any, Tuple

from envs.tetris_env import TetrisEnv

class DQNMovementAgent:
    """Movement agent that processes locked position outputs into movement actions"""
    
    def __init__(self,
                 input_dim: int = 800,      # Takes 800 outputs from locked agent
                 num_actions: int = 8,      # 8 movement actions
                 hidden_dim: int = 512,
                 device: str = 'cuda',
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 50000,
                 buffer_size: int = 100000,
                 batch_size: int = 32):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Build networks
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=buffer_size)
        
        self.step_count = 0
        self.target_update_freq = 1000
        
        print(f"🎮 Movement Agent initialized:")
        print(f"   Input: {input_dim} (from locked agent)")
        print(f"   Output: {num_actions} movement actions")
        print(f"   Parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def _build_network(self):
        """Build the movement network"""
        class MovementNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_actions):
                super().__init__()
                
                # Process 800 locked position Q-values
                self.fc1 = nn.Linear(input_dim, hidden_dim)     # 800 → 512
                self.fc2 = nn.Linear(hidden_dim, hidden_dim//2) # 512 → 256
                self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4) # 256 → 128
                self.fc4 = nn.Linear(hidden_dim//4, num_actions)   # 128 → 8
                
                self.dropout = nn.Dropout(0.2)
                self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
                self.batch_norm2 = nn.BatchNorm1d(hidden_dim//2)
                
                # Initialize weights
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                # Process locked position Q-values into movement actions
                x = F.relu(self.batch_norm1(self.fc1(x)))  # 800 → 512
                x = self.dropout(x)
                x = F.relu(self.batch_norm2(self.fc2(x)))  # 512 → 256
                x = self.dropout(x)
                x = F.relu(self.fc3(x))                    # 256 → 128
                x = self.fc4(x)                            # 128 → 8
                return x
        
        return MovementNetwork(self.input_dim, self.hidden_dim, self.num_actions)
    
    def select_action(self, locked_q_values: np.ndarray, training: bool = True) -> int:
        """Select movement action based on locked position Q-values"""
        if not isinstance(locked_q_values, np.ndarray):
            locked_q_values = np.array(locked_q_values, dtype=np.float32)
        
        assert locked_q_values.shape[0] == 800, f"Expected 800 locked Q-values, got {locked_q_values.shape[0]}"
        
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        
        # Get Q-values from movement network
        state_tensor = torch.FloatTensor(locked_q_values).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def store_experience(self, locked_q_values: np.ndarray, action: int, 
                        reward: float, next_locked_q_values: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((locked_q_values, action, reward, next_locked_q_values, done))
    
    def update(self) -> Dict[str, float]:
        """Update the movement network"""
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0}
        
        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        self.q_network.train()
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (Double DQN)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, 
                             self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay)
        
        return {'loss': loss.item()}
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.q_network.parameters())

class DQNMovementTrainer:
    """Trainer for the movement agent"""
    
    def __init__(self, reward_mode='standard', episodes=1000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reward_mode = reward_mode
        self.episodes = episodes
        
        # Create environment
        self.env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='direct',  # Movement agent uses direct actions
            reward_mode=reward_mode
        )
        
        # Initialize movement agent
        self.agent = DQNMovementAgent(
            input_dim=800,
            num_actions=8,
            device=str(self.device),
            learning_rate=0.0001,
            epsilon_decay=episodes * 10
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.lines_cleared = []
        self.losses = []
        
        print(f"🎮 Movement Trainer Initialized:")
        print(f"   Device: {self.device}")
        print(f"   Reward mode: {reward_mode}")
        print(f"   Episodes: {episodes}")
    
    def simulate_locked_agent_output(self, obs: np.ndarray) -> np.ndarray:
        """Simulate locked agent Q-values (placeholder until hierarchical integration)"""
        # Generate realistic Q-values for 800 locked positions
        # In practice, this would come from the actual locked agent
        q_values = np.random.randn(800) * 0.1  # Small random values
        
        # Add some structure based on observation
        board_state = obs[:200] if len(obs) >= 200 else obs
        board_fill = np.mean(board_state)
        
        # Simulate preference for lower positions when board is fuller
        for i in range(800):
            y = (i // 40) % 20  # Extract y coordinate
            q_values[i] += (20 - y) * board_fill * 0.01
        
        return q_values
    
    def train(self):
        """Main training loop"""
        print(f"\n🎮 STARTING MOVEMENT AGENT TRAINING ({self.episodes} episodes)")
        print("=" * 70)
        
        start_time = time.time()
        total_lines = 0
        
        for episode in range(self.episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_lines = 0
            
            # Simulate locked agent output for first step
            prev_locked_q = self.simulate_locked_agent_output(obs)
            
            for step in range(500):  # Max steps per episode
                # Get movement action from agent
                action = self.agent.select_action(prev_locked_q, training=True)
                
                # Environment step
                next_obs, reward, done, info = self.env.step(action)
                
                # Track episode stats
                if 'lines_cleared' in info:
                    episode_lines += info['lines_cleared']
                
                # Simulate next locked agent output
                next_locked_q = self.simulate_locked_agent_output(next_obs)
                
                # Store experience and update
                self.agent.store_experience(prev_locked_q, action, reward, next_locked_q, done)
                loss_dict = self.agent.update()
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                    
                obs = next_obs
                prev_locked_q = next_locked_q
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.lines_cleared.append(episode_lines)
            if loss_dict and 'loss' in loss_dict:
                self.losses.append(loss_dict['loss'])
            
            total_lines += episode_lines
            
            # Logging
            if episode % 50 == 0 or episode < 5:
                avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
                
                print(f"Episode {episode:4d}: "
                      f"Reward={episode_reward:7.2f}, "
                      f"Length={episode_length:3d}, "
                      f"Lines={episode_lines:1d}, "
                      f"TotalLines={total_lines:3d}, "
                      f"ε={self.agent.epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        print("=" * 70)
        print(f"🎉 MOVEMENT AGENT TRAINING COMPLETE!")
        print(f"   Total time: {training_time:.1f}s")
        print(f"   Episodes: {self.episodes}")
        print(f"   Total lines cleared: {total_lines}")
        print(f"   Mean reward: {np.mean(self.episode_rewards):.2f}")
        print(f"   Final epsilon: {self.agent.epsilon:.3f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'lines_cleared': self.lines_cleared,
            'training_time': training_time,
            'total_lines': total_lines
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.env.close()
        except:
            pass

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='DQN Movement Agent Training')
    parser.add_argument('--reward_mode', choices=['standard', 'lines_only'], 
                       default='standard', help='Reward mode')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    args = parser.parse_args()
    
    print("🎮 DQN MOVEMENT AGENT TRAINING")
    print("=" * 80)
    
    trainer = DQNMovementTrainer(reward_mode=args.reward_mode, episodes=args.episodes)
    
    try:
        results = trainer.train()
        print(f"\n✅ Training completed successfully!")
        print(f"   Total lines cleared: {results['total_lines']}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 