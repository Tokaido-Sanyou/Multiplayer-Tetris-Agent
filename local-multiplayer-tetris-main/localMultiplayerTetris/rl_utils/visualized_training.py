#!/usr/bin/env python3
"""
Visualized Multiplayer AIRL Training
Real-time visualization of competitive training between two agents
"""

import sys
import os
import torch
import numpy as np
import time
import pygame
import logging
from collections import deque

# Import compatibility
try:
    from .multiplayer_airl import MultiplayerAIRLTrainer
    from .actor_critic import ActorCritic
    from .true_multiplayer_env import TrueMultiplayerTetrisEnv
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from multiplayer_airl import MultiplayerAIRLTrainer
    from actor_critic import ActorCritic
    from true_multiplayer_env import TrueMultiplayerTetrisEnv

# Environment import
try:
    from ..tetris_env import TetrisEnv
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tetris_env import TetrisEnv

class VisualizedMultiplayerTrainer(MultiplayerAIRLTrainer):
    """Multiplayer AIRL trainer with visualization capabilities."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Override environment with visualization enabled
        self.env.close()  # Close the headless environment
        self.env = TrueMultiplayerTetrisEnv(headless=False)
        
        # Visualization settings
        self.render_delay = config.get('render_delay', 0.1)  # Delay between moves
        self.show_metrics = config.get('show_metrics', True)
        self.save_screenshots = config.get('save_screenshots', False)
        
        # Training visualization metrics
        self.episode_rewards = {'player1': [], 'player2': []}
        self.episode_lengths = []
        self.win_history = deque(maxlen=50)  # Last 50 games
        
        self.logger.info("Initialized visualized multiplayer trainer")
    
    def run_visualized_episode(self, max_steps=1000, episode_num=0):
        """Run a competitive episode with visualization."""
        self.logger.info(f"🎮 Starting visualized episode {episode_num + 1}")
        
        # Reset environment
        observations = self.env.reset()
        
        # Always get proper multiplayer observations
        obs_p1 = observations['player1']
        obs_p2 = observations['player2']
        
        episode_data = []
        step_count = 0
        episode_rewards = {'player1': 0, 'player2': 0}
        
        print(f"\n🎯 Episode {episode_num + 1} - Competitive Training")
        print("=" * 50)
        
        while step_count < max_steps:
            # Extract features for both players
            state_p1 = self._extract_features(obs_p1)
            state_p2 = self._extract_features(obs_p2)
            
            state_p1_tensor = torch.FloatTensor(state_p1).unsqueeze(0).to(self.device)
            state_p2_tensor = torch.FloatTensor(state_p2).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get action probabilities and sample actions
                action_probs_p1, _ = self.policy_p1(state_p1_tensor)
                action_p1 = torch.multinomial(action_probs_p1, 1).item()
                
                action_probs_p2, _ = self.policy_p2(state_p2_tensor)
                action_p2 = torch.multinomial(action_probs_p2, 1).item()
            
            # Display action choices
            if step_count % 10 == 0:  # Every 10 steps
                print(f"Step {step_count:3d}: P1 action={action_p1:2d}, P2 action={action_p2:2d}")
            
            # Always use multiplayer actions
            actions = {'player1': action_p1, 'player2': action_p2}
                
            # Take actions in environment
            step_result = self.env.step(actions)
            next_observations, rewards, done, info = step_result
            
            # Always get proper multiplayer rewards
            reward_p1 = rewards['player1']
            reward_p2 = rewards['player2']
            
            episode_rewards['player1'] += reward_p1
            episode_rewards['player2'] += reward_p2
            
            # Render the game
            try:
                self.env.render()
                time.sleep(self.render_delay)  # Control rendering speed
            except:
                pass  # Continue if rendering fails
            
            # Always get proper multiplayer observations
            next_obs_p1 = next_observations['player1']
            next_obs_p2 = next_observations['player2']
            
            # Determine game outcome
            game_outcome = 'draw'
            if done:
                winner = info.get('winner')
                if winner == 'player1':
                    game_outcome = 'player1_wins'
                    self.metrics['player1_wins'] += 1
                    print(f"🏆 Player 1 WINS! (Steps: {step_count})")
                elif winner == 'player2':
                    game_outcome = 'player2_wins'
                    self.metrics['player2_wins'] += 1
                    print(f"🏆 Player 2 WINS! (Steps: {step_count})")
                else:
                    game_outcome = 'draw'
                    self.metrics['draws'] += 1
                    print(f"🤝 DRAW! (Steps: {step_count})")
            
            # Store step data
            episode_data.append({
                'states': (state_p1, state_p2),
                'actions': (action_p1, action_p2),
                'rewards': (reward_p1, reward_p2),
                'done': done,
                'game_outcome': game_outcome
            })
            
            # Update observations
            obs_p1, obs_p2 = next_obs_p1, next_obs_p2
            step_count += 1
            
            if done:
                break
        
        # Update metrics
        self.metrics['total_games'] += 1
        self.episode_rewards['player1'].append(episode_rewards['player1'])
        self.episode_rewards['player2'].append(episode_rewards['player2'])
        self.episode_lengths.append(step_count)
        self.win_history.append(game_outcome)
        
        # Display episode summary
        print(f"\n📊 Episode {episode_num + 1} Summary:")
        print(f"   Duration: {step_count} steps")
        print(f"   Rewards: P1={episode_rewards['player1']:.2f}, P2={episode_rewards['player2']:.2f}")
        print(f"   Outcome: {game_outcome}")
        
        if self.show_metrics:
            self._display_training_metrics()
        
        return episode_data
    
    def _display_training_metrics(self):
        """Display comprehensive training metrics."""
        if self.metrics['total_games'] == 0:
            return
        
        print(f"\n📈 Training Progress (Last {len(self.win_history)} games):")
        
        # Win rates
        recent_p1_wins = sum(1 for outcome in self.win_history if outcome == 'player1_wins')
        recent_p2_wins = sum(1 for outcome in self.win_history if outcome == 'player2_wins')
        recent_draws = sum(1 for outcome in self.win_history if outcome == 'draw')
        total_recent = len(self.win_history)
        
        print(f"   Win Rates: P1={recent_p1_wins/total_recent*100:.1f}%, "
              f"P2={recent_p2_wins/total_recent*100:.1f}%, "
              f"Draw={recent_draws/total_recent*100:.1f}%")
        
        # Average rewards
        if self.episode_rewards['player1']:
            avg_p1_reward = np.mean(self.episode_rewards['player1'][-10:])  # Last 10 episodes
            avg_p2_reward = np.mean(self.episode_rewards['player2'][-10:])
            print(f"   Avg Rewards (last 10): P1={avg_p1_reward:.2f}, P2={avg_p2_reward:.2f}")
        
        # Episode lengths
        if self.episode_lengths:
            avg_length = np.mean(self.episode_lengths[-10:])
            print(f"   Avg Episode Length: {avg_length:.1f} steps")
    
    def train_with_visualization(self, num_episodes=10):
        """Main training loop with visualization."""
        self.logger.info(f"🎬 Starting visualized training for {num_episodes} episodes")
        
        print("\n" + "="*80)
        print("🎬 VISUALIZED MULTIPLAYER AIRL TRAINING")
        print("="*80)
        print(f"Training {num_episodes} competitive episodes with real-time visualization")
        print("Watch the agents learn to play Tetris competitively!")
        print("=" * 80)
        
        try:
            for episode in range(num_episodes):
                print(f"\n🎮 Starting Episode {episode + 1}/{num_episodes}")
                
                # Run visualized episode
                episode_data = self.run_visualized_episode(
                    max_steps=500,  # Shorter episodes for visualization
                    episode_num=episode
                )
                
                # Small delay between episodes
                if episode < num_episodes - 1:
                    print("\n⏸️ Preparing next episode...")
                    time.sleep(2)
            
            # Final summary
            self._print_final_training_summary()
            
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
            self._print_final_training_summary()
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            print(f"\n❌ Training error: {e}")
        finally:
            try:
                self.env.close()
            except:
                pass
    
    def _print_final_training_summary(self):
        """Print comprehensive training summary."""
        print("\n" + "="*80)
        print("🏁 VISUALIZED TRAINING COMPLETE")
        print("="*80)
        
        total_games = self.metrics['total_games']
        if total_games == 0:
            print("No games completed.")
            return
        
        # Overall statistics
        p1_wins = self.metrics['player1_wins']
        p2_wins = self.metrics['player2_wins']
        draws = self.metrics['draws']
        
        print(f"📊 Overall Results ({total_games} games):")
        print(f"   Player 1: {p1_wins} wins ({p1_wins/total_games*100:.1f}%)")
        print(f"   Player 2: {p2_wins} wins ({p2_wins/total_games*100:.1f}%)")
        print(f"   Draws: {draws} ({draws/total_games*100:.1f}%)")
        
        # Performance metrics
        if self.episode_rewards['player1']:
            avg_p1_reward = np.mean(self.episode_rewards['player1'])
            avg_p2_reward = np.mean(self.episode_rewards['player2'])
            print(f"\n💰 Average Rewards:")
            print(f"   Player 1: {avg_p1_reward:.2f}")
            print(f"   Player 2: {avg_p2_reward:.2f}")
        
        if self.episode_lengths:
            avg_length = np.mean(self.episode_lengths)
            print(f"\n⏱️ Average Episode Length: {avg_length:.1f} steps")
        
        # Learning trends
        if len(self.episode_rewards['player1']) >= 2:
            early_avg_p1 = np.mean(self.episode_rewards['player1'][:len(self.episode_rewards['player1'])//2])
            late_avg_p1 = np.mean(self.episode_rewards['player1'][len(self.episode_rewards['player1'])//2:])
            improvement_p1 = late_avg_p1 - early_avg_p1
            
            early_avg_p2 = np.mean(self.episode_rewards['player2'][:len(self.episode_rewards['player2'])//2])
            late_avg_p2 = np.mean(self.episode_rewards['player2'][len(self.episode_rewards['player2'])//2:])
            improvement_p2 = late_avg_p2 - early_avg_p2
            
            print(f"\n📈 Learning Progress:")
            print(f"   Player 1 improvement: {improvement_p1:+.2f}")
            print(f"   Player 2 improvement: {improvement_p2:+.2f}")
        
        print("\n🎮 Training completed successfully!")
        print("="*80)

def create_visualization_demo():
    """Create a demonstration of visualized competitive training."""
    print("🎬 MULTIPLAYER TETRIS AIRL - VISUALIZATION DEMO")
    print("Real-time competitive training between two AIRL agents")
    print("=" * 60)
    
    # Configuration for visualization
    config = {
        'device': 'cpu',  # Use CPU for stable visualization
        'render_delay': 0.2,  # 200ms delay between moves for visibility
        'show_metrics': True,
        'save_screenshots': False
    }
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create visualized trainer
        trainer = VisualizedMultiplayerTrainer(config)
        
        # Run visualized training
        trainer.train_with_visualization(num_episodes=5)
        
    except Exception as e:
        print(f"❌ Visualization demo failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function for visualized training."""
    create_visualization_demo()

if __name__ == "__main__":
    main() 