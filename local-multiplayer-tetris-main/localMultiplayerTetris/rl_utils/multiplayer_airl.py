#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiplayer AIRL Implementation
Competitive training with two AIRL agents
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import logging
from collections import deque

# Import compatibility
try:
    from .actor_critic import ActorCritic
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from actor_critic import ActorCritic

# Environment import
try:
    from ..tetris_env import TetrisEnv
    from .true_multiplayer_env import TrueMultiplayerTetrisEnv
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tetris_env import TetrisEnv
    from true_multiplayer_env import TrueMultiplayerTetrisEnv

class MultiplayerAIRLTrainer:
    """Simplified Multiplayer AIRL trainer for competitive dynamics."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Initialize TRUE multiplayer environment
        self.env = TrueMultiplayerTetrisEnv(headless=True)
        
        # State and action dimensions
        self.state_dim = 207
        self.action_dim = 41
        
        # Initialize policies for both players
        self.policy_p1 = ActorCritic(
            input_dim=self.state_dim, 
            output_dim=self.action_dim
        ).to(self.device)
        
        self.policy_p2 = ActorCritic(
            input_dim=self.state_dim, 
            output_dim=self.action_dim
        ).to(self.device)
        
        # Training metrics
        self.metrics = {
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'total_games': 0,
            'player1_rewards': deque(maxlen=100),
            'player2_rewards': deque(maxlen=100)
        }
        
        self.logger = logging.getLogger('MultiplayerAIRL')
        self.logger.info(f"Initialized Multiplayer AIRL with device: {self.device}")
    
    def _extract_features(self, observation):
        """Extract features from multiplayer observation."""
        if isinstance(observation, dict):
            grid = observation.get('grid', np.zeros((20, 10)))
            features = []
            features.extend(np.array(grid).flatten())  # 200
            features.append(observation.get('current_shape', 0))
            features.append(observation.get('current_rotation', 0))
            features.append(observation.get('current_x', 5))
            features.append(observation.get('current_y', 0))
            features.append(observation.get('next_piece', 0))
            features.append(observation.get('hold_piece', -1))
            features.append(observation.get('can_hold', 1))
            return np.array(features, dtype=np.float32)
        else:
            return np.zeros(self.state_dim, dtype=np.float32)
    
    def run_competitive_episode(self, max_steps=1000):
        """Run a competitive episode between two agents."""
        observations = self.env.reset()
        
        # Now we always get proper multiplayer observations
        obs_p1 = observations['player1']
        obs_p2 = observations['player2']
        
        episode_data = []
        step_count = 0
        
        while step_count < max_steps:
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
            
            # Always use multiplayer actions
            actions = {'player1': action_p1, 'player2': action_p2}
                
            step_result = self.env.step(actions)
            next_observations, rewards, done, info = step_result
            
            # Always get proper multiplayer results
            next_obs_p1 = next_observations['player1']
            next_obs_p2 = next_observations['player2']
            reward_p1 = rewards['player1']
            reward_p2 = rewards['player2']
            
            game_outcome = 'draw'
            if done:
                winner = info.get('winner')
                if winner == 'player1':
                    game_outcome = 'player1_wins'
                    self.metrics['player1_wins'] += 1
                elif winner == 'player2':
                    game_outcome = 'player2_wins'
                    self.metrics['player2_wins'] += 1
                else:
                    game_outcome = 'draw'
                    self.metrics['draws'] += 1
            
            episode_data.append({
                'states': (state_p1, state_p2),
                'actions': (action_p1, action_p2),
                'rewards': (reward_p1, reward_p2),
                'done': done,
                'game_outcome': game_outcome
            })
            
            obs_p1, obs_p2 = next_obs_p1, next_obs_p2
            step_count += 1
            
            if done:
                break
        
        self.metrics['total_games'] += 1
        self.metrics['player1_rewards'].append(sum(step['rewards'][0] for step in episode_data))
        self.metrics['player2_rewards'].append(sum(step['rewards'][1] for step in episode_data))
        
        return episode_data
    
    def train(self, num_iterations=100):
        """Main training loop for competitive play."""
        self.logger.info(f"Starting Multiplayer AIRL training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            episode_data = self.run_competitive_episode()
            
            if iteration % 10 == 0:
                self._log_progress(iteration)
        
        self.logger.info("Training completed!")
        self._print_final_stats()
    
    def _log_progress(self, iteration):
        """Log training progress."""
        if self.metrics['total_games'] == 0:
            return
            
        p1_winrate = self.metrics['player1_wins'] / self.metrics['total_games']
        p2_winrate = self.metrics['player2_wins'] / self.metrics['total_games']
        draw_rate = self.metrics['draws'] / self.metrics['total_games']
        
        avg_p1_reward = np.mean(self.metrics['player1_rewards']) if self.metrics['player1_rewards'] else 0
        avg_p2_reward = np.mean(self.metrics['player2_rewards']) if self.metrics['player2_rewards'] else 0
        
        self.logger.info(f"Iteration {iteration}")
        self.logger.info(f"  Win rates - P1: {p1_winrate:.3f}, P2: {p2_winrate:.3f}, Draw: {draw_rate:.3f}")
        self.logger.info(f"  Avg rewards - P1: {avg_p1_reward:.2f}, P2: {avg_p2_reward:.2f}")
    
    def _print_final_stats(self):
        """Print final training statistics."""
        print("\n" + "="*60)
        print("🏆 MULTIPLAYER AIRL TRAINING COMPLETE")
        print("="*60)
        print(f"Total games played: {self.metrics['total_games']}")
        
        if self.metrics['total_games'] > 0:
            p1_win_pct = self.metrics['player1_wins']/self.metrics['total_games']*100
            p2_win_pct = self.metrics['player2_wins']/self.metrics['total_games']*100
            draw_pct = self.metrics['draws']/self.metrics['total_games']*100
            
            print(f"Player 1 wins: {self.metrics['player1_wins']} ({p1_win_pct:.1f}%)")
            print(f"Player 2 wins: {self.metrics['player2_wins']} ({p2_win_pct:.1f}%)")
            print(f"Draws: {self.metrics['draws']} ({draw_pct:.1f}%)")
        
        if self.metrics['player1_rewards']:
            print(f"Average P1 reward: {np.mean(self.metrics['player1_rewards']):.2f}")
        if self.metrics['player2_rewards']:
            print(f"Average P2 reward: {np.mean(self.metrics['player2_rewards']):.2f}")

def main():
    """Main function for multiplayer AIRL training."""
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    trainer = MultiplayerAIRLTrainer(config)
    trainer.train(num_iterations=50)

if __name__ == "__main__":
    main() 