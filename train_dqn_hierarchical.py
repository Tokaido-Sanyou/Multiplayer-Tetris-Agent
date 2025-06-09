#!/usr/bin/env python3
"""
🏗️ HIERARCHICAL DQN TRAINING

Combines locked state DQN (212 inputs → 800 outputs) with movement DQN (800 inputs → 8 outputs).
This is the complete hierarchical structure.
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path

from agents.dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent
from train_dqn_movement import DQNMovementAgent
from envs.tetris_env import TetrisEnv

class HierarchicalDQNTrainer:
    """Hierarchical DQN trainer combining locked and movement agents"""
    
    def __init__(self, reward_mode='standard', episodes=1000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reward_mode = reward_mode
        self.episodes = episodes
        
        # Create environment
        self.env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='direct',  # Final actions are direct movements
            reward_mode=reward_mode
        )
        
        # Initialize locked state agent (212 → 800)
        self.locked_agent = RedesignedLockedStateDQNAgent(
            input_dim=212,  # Padded observation
            num_actions=800,  # Locked positions
            device=str(self.device),
            learning_rate=0.0001,
            epsilon_start=0.9 if reward_mode == 'standard' else 0.95,
            epsilon_end=0.01 if reward_mode == 'standard' else 0.05,
            epsilon_decay=episodes * 10,
            reward_mode=reward_mode
        )
        
        # Initialize movement agent (800 → 8)
        self.movement_agent = DQNMovementAgent(
            input_dim=800,    # Takes locked agent Q-values
            num_actions=8,    # Movement actions
            device=str(self.device),
            learning_rate=0.0001,
            epsilon_start=0.8,
            epsilon_end=0.01,
            epsilon_decay=episodes * 5  # Faster decay for movement
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.lines_cleared = []
        self.locked_losses = []
        self.movement_losses = []
        
        print(f"🏗️  Hierarchical DQN Trainer Initialized:")
        print(f"   Device: {self.device}")
        print(f"   Reward mode: {reward_mode}")
        print(f"   Episodes: {episodes}")
        print(f"   Locked Agent: {self.locked_agent.get_parameter_count():,} parameters (212 → 800)")
        print(f"   Movement Agent: {self.movement_agent.get_parameter_count():,} parameters (800 → 8)")
        print(f"   Total parameters: {self.locked_agent.get_parameter_count() + self.movement_agent.get_parameter_count():,}")
    
    def pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """Pad observation from 206 to 212 dimensions for locked agent"""
        if obs.shape[0] == 206:
            return np.concatenate([obs, np.zeros(6)], axis=0)
        return obs
    
    def get_hierarchical_action(self, obs: np.ndarray, training: bool = True) -> tuple:
        """Get action using hierarchical approach"""
        # Step 1: Pad observation for locked agent
        padded_obs = self.pad_observation(obs)
        
        # Step 2: Get Q-values from locked agent (don't actually select action)
        padded_obs_tensor = torch.FloatTensor(padded_obs).unsqueeze(0).to(self.device)
        
        self.locked_agent.q_network.eval()
        with torch.no_grad():
            locked_q_values = self.locked_agent.q_network(padded_obs_tensor).squeeze(0).cpu().numpy()
        
        # Step 3: Use locked Q-values to get movement action
        movement_action = self.movement_agent.select_action(locked_q_values, training=training)
        
        return movement_action, locked_q_values
    
    def train(self):
        """Main hierarchical training loop"""
        print(f"\n🏗️  STARTING HIERARCHICAL DQN TRAINING ({self.episodes} episodes)")
        print("=" * 70)
        
        start_time = time.time()
        total_lines = 0
        
        for episode in range(self.episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_lines = 0
            
            # Get initial hierarchical action
            prev_action, prev_locked_q = self.get_hierarchical_action(obs, training=True)
            prev_padded_obs = self.pad_observation(obs)
            
            for step in range(500):  # Max steps per episode
                # Execute movement action in environment
                next_obs, reward, done, info = self.env.step(prev_action)
                
                # Track episode stats
                if 'lines_cleared' in info:
                    episode_lines += info['lines_cleared']
                
                # Get next hierarchical action
                if not done:
                    next_action, next_locked_q = self.get_hierarchical_action(next_obs, training=True)
                    next_padded_obs = self.pad_observation(next_obs)
                else:
                    next_action, next_locked_q = 0, np.zeros(800)
                    next_padded_obs = np.zeros(212)
                
                # Train locked agent (using padded observations)
                self.locked_agent.store_experience(prev_padded_obs, prev_action, reward, next_padded_obs, done)
                locked_loss_dict = self.locked_agent.update(prev_padded_obs, prev_action, reward, next_padded_obs, done)
                
                # Train movement agent (using locked Q-values)
                self.movement_agent.store_experience(prev_locked_q, prev_action, reward, next_locked_q, done)
                movement_loss_dict = self.movement_agent.update()
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                
                # Update for next iteration
                obs = next_obs
                prev_action = next_action
                prev_locked_q = next_locked_q
                prev_padded_obs = next_padded_obs
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.lines_cleared.append(episode_lines)
            
            if locked_loss_dict and 'loss' in locked_loss_dict:
                self.locked_losses.append(locked_loss_dict['loss'])
            if movement_loss_dict and 'loss' in movement_loss_dict:
                self.movement_losses.append(movement_loss_dict['loss'])
            
            total_lines += episode_lines
            
            # Logging
            if episode % 50 == 0 or episode < 5:
                avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
                recent_lines = sum(self.lines_cleared[-50:]) if len(self.lines_cleared) >= 50 else sum(self.lines_cleared)
                
                print(f"Episode {episode:4d}: "
                      f"Reward={episode_reward:7.2f}, "
                      f"Length={episode_length:3d}, "
                      f"Lines={episode_lines:1d}, "
                      f"TotalLines={total_lines:3d}, "
                      f"LockedE={self.locked_agent.epsilon:.3f}, "
                      f"MoveE={self.movement_agent.epsilon:.3f}, "
                      f"Recent50Lines={recent_lines:2d}")
        
        training_time = time.time() - start_time
        
        print("=" * 70)
        print(f"🎉 HIERARCHICAL DQN TRAINING COMPLETE!")
        print(f"   Total time: {training_time:.1f}s")
        print(f"   Episodes: {self.episodes}")
        print(f"   Total lines cleared: {total_lines}")
        print(f"   Mean reward: {np.mean(self.episode_rewards):.2f}")
        print(f"   Final locked epsilon: {self.locked_agent.epsilon:.3f}")
        print(f"   Final movement epsilon: {self.movement_agent.epsilon:.3f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'lines_cleared': self.lines_cleared,
            'training_time': training_time,
            'total_lines': total_lines,
            'locked_losses': self.locked_losses,
            'movement_losses': self.movement_losses
        }
    
    def save_models(self, filepath: str):
        """Save both agent models"""
        save_path = Path(filepath)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.locked_agent.save_checkpoint(str(save_path / "locked_agent.pth"))
        
        # Save movement agent
        torch.save({
            'q_network_state_dict': self.movement_agent.q_network.state_dict(),
            'target_network_state_dict': self.movement_agent.target_network.state_dict(),
            'optimizer_state_dict': self.movement_agent.optimizer.state_dict(),
            'epsilon': self.movement_agent.epsilon,
            'step_count': self.movement_agent.step_count
        }, str(save_path / "movement_agent.pth"))
        
        print(f"✅ Models saved to {save_path}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.env.close()
        except:
            pass

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Hierarchical DQN Training')
    parser.add_argument('--reward_mode', choices=['standard', 'lines_only'], 
                       default='standard', help='Reward mode')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--save_path', type=str, default='models/hierarchical_dqn', 
                       help='Path to save models')
    args = parser.parse_args()
    
    print("🏗️  HIERARCHICAL DQN TRAINING")
    print("=" * 80)
    
    trainer = HierarchicalDQNTrainer(reward_mode=args.reward_mode, episodes=args.episodes)
    
    try:
        results = trainer.train()
        
        # Save models
        trainer.save_models(args.save_path)
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Total lines cleared: {results['total_lines']}")
        print(f"   Models saved to: {args.save_path}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        trainer.save_models(args.save_path + "_interrupted")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 