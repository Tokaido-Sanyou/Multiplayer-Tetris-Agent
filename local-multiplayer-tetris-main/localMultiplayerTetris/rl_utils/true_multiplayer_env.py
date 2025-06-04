#!/usr/bin/env python3
"""
True Multiplayer Tetris Environment
Wrapper that provides genuine multiplayer functionality
"""

import numpy as np
import torch
import logging
from typing import Dict, Tuple, Any
import copy

# Import compatibility
try:
    from ..tetris_env import TetrisEnv
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tetris_env import TetrisEnv

class TrueMultiplayerTetrisEnv:
    """
    True multiplayer Tetris environment.
    Manages two separate TetrisEnv instances for genuine competitive play.
    """
    
    def __init__(self, headless=True):
        self.headless = headless
        
        # Create two separate environments for each player
        self.env_p1 = TetrisEnv(single_player=True, headless=True)  # Player 1 environment
        self.env_p2 = TetrisEnv(single_player=True, headless=True)  # Player 2 environment
        
        # For visualization, create a shared display environment
        if not headless:
            self.display_env = TetrisEnv(single_player=False, headless=False)
        else:
            self.display_env = None
        
        # Game state tracking
        self.episode_steps = 0
        self.max_steps = 1000
        
        # Player state tracking
        self.player1_alive = True
        self.player2_alive = True
        
        # Performance metrics
        self.player1_score = 0
        self.player2_score = 0
        self.player1_lines = 0
        self.player2_lines = 0
        
        self.logger = logging.getLogger('TrueMultiplayerTetris')
        self.logger.info("Initialized True Multiplayer Tetris Environment")
    
    def reset(self) -> Dict[str, Dict]:
        """Reset both player environments and return initial observations."""
        # Reset individual environments
        obs_p1 = self.env_p1.reset()
        obs_p2 = self.env_p2.reset()
        
        # Handle tuple format
        if isinstance(obs_p1, tuple):
            obs_p1 = obs_p1[0]
        if isinstance(obs_p2, tuple):
            obs_p2 = obs_p2[0]
        
        # Reset display environment if visualization enabled
        if self.display_env:
            self.display_env.reset()
        
        # Reset game state
        self.episode_steps = 0
        self.player1_alive = True
        self.player2_alive = True
        self.player1_score = 0
        self.player2_score = 0
        self.player1_lines = 0
        self.player2_lines = 0
        
        # Return multiplayer observations
        return {
            'player1': obs_p1,
            'player2': obs_p2
        }
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Execute actions for both players simultaneously.
        
        Args:
            actions: Dict with 'player1' and 'player2' keys containing action integers
            
        Returns:
            observations: Dict with player observations
            rewards: Dict with player rewards  
            done: Boolean indicating if episode is complete
            info: Dict with game information
        """
        self.episode_steps += 1
        
        # Extract actions
        action_p1 = actions.get('player1', 0)
        action_p2 = actions.get('player2', 0)
        
        # Step each environment independently
        rewards = {'player1': 0, 'player2': 0}
        observations = {}
        info = {'player1': {}, 'player2': {}, 'winner': None}
        
        # Player 1 step
        if self.player1_alive:
            try:
                step_result_p1 = self.env_p1.step(action_p1)
                if len(step_result_p1) == 4:
                    obs_p1, reward_p1, done_p1, info_p1 = step_result_p1
                else:
                    obs_p1, reward_p1, done_p1, truncated_p1, info_p1 = step_result_p1
                    done_p1 = done_p1 or truncated_p1
                
                observations['player1'] = obs_p1
                rewards['player1'] = reward_p1
                info['player1'] = info_p1
                
                # Track player 1 performance
                self.player1_score = info_p1.get('score', self.player1_score)
                self.player1_lines += info_p1.get('lines_cleared', 0)
                
                if done_p1:
                    self.player1_alive = False
                    self.logger.info(f"Player 1 eliminated at step {self.episode_steps}")
                    
            except Exception as e:
                self.logger.error(f"Player 1 step failed: {e}")
                self.player1_alive = False
                observations['player1'] = self._get_empty_observation()
                rewards['player1'] = -50  # Heavy penalty for crash
        else:
            observations['player1'] = self._get_empty_observation()
            rewards['player1'] = 0
        
        # Player 2 step  
        if self.player2_alive:
            try:
                step_result_p2 = self.env_p2.step(action_p2)
                if len(step_result_p2) == 4:
                    obs_p2, reward_p2, done_p2, info_p2 = step_result_p2
                else:
                    obs_p2, reward_p2, done_p2, truncated_p2, info_p2 = step_result_p2
                    done_p2 = done_p2 or truncated_p2
                
                observations['player2'] = obs_p2
                rewards['player2'] = reward_p2
                info['player2'] = info_p2
                
                # Track player 2 performance
                self.player2_score = info_p2.get('score', self.player2_score)
                self.player2_lines += info_p2.get('lines_cleared', 0)
                
                if done_p2:
                    self.player2_alive = False
                    self.logger.info(f"Player 2 eliminated at step {self.episode_steps}")
                    
            except Exception as e:
                self.logger.error(f"Player 2 step failed: {e}")
                self.player2_alive = False
                observations['player2'] = self._get_empty_observation()
                rewards['player2'] = -50  # Heavy penalty for crash
        else:
            observations['player2'] = self._get_empty_observation()
            rewards['player2'] = 0
        
        # Determine episode completion and winner
        episode_done = self._check_episode_complete()
        winner = self._determine_winner()
        info['winner'] = winner
        
        # Add competitive rewards
        rewards = self._add_competitive_rewards(rewards, winner)
        
        # Update visualization if enabled
        if self.display_env and not self.headless:
            self._update_visualization()
        
        return observations, rewards, episode_done, info
    
    def _check_episode_complete(self) -> bool:
        """Check if the episode should end."""
        # Episode ends if both players are eliminated or max steps reached
        if not self.player1_alive and not self.player2_alive:
            return True
        if self.episode_steps >= self.max_steps:
            return True
        # Episode also ends if one player is significantly ahead (optional rule)
        return False
    
    def _determine_winner(self) -> str:
        """Determine the winner based on current game state."""
        if not self.player1_alive and not self.player2_alive:
            # Both eliminated - winner is who lasted longer or scored higher
            if self.player1_score > self.player2_score:
                return 'player1'
            elif self.player2_score > self.player1_score:
                return 'player2'
            else:
                return 'draw'
        elif not self.player1_alive:
            return 'player2'  # Player 2 wins by survival
        elif not self.player2_alive:
            return 'player1'  # Player 1 wins by survival
        else:
            return None  # Game still ongoing
    
    def _add_competitive_rewards(self, rewards: Dict, winner: str) -> Dict:
        """Add competitive rewards based on relative performance."""
        # Base rewards are already computed by individual environments
        
        # Add survival bonus
        if self.player1_alive:
            rewards['player1'] += 0.1
        if self.player2_alive:
            rewards['player2'] += 0.1
        
        # Add relative performance rewards
        score_diff = self.player1_score - self.player2_score
        line_diff = self.player1_lines - self.player2_lines
        
        # Reward score advantage (small bonus for being ahead)
        if score_diff > 0:
            rewards['player1'] += min(score_diff * 0.001, 2.0)  # Cap at +2
            rewards['player2'] += max(-score_diff * 0.001, -2.0)  # Cap at -2
        elif score_diff < 0:
            rewards['player2'] += min(-score_diff * 0.001, 2.0)
            rewards['player1'] += max(score_diff * 0.001, -2.0)
        
        # Winning bonus
        if winner == 'player1':
            rewards['player1'] += 10.0
            rewards['player2'] -= 5.0
        elif winner == 'player2':
            rewards['player2'] += 10.0
            rewards['player1'] -= 5.0
        
        return rewards
    
    def _get_empty_observation(self) -> Dict:
        """Return an empty observation for eliminated players."""
        return {
            'grid': np.zeros((20, 10), dtype=np.int8),
            'next_piece': 0,
            'hold_piece': 0,
            'current_shape': 0,
            'current_rotation': 0,
            'current_x': 5,
            'current_y': 0,
            'can_hold': 0
        }
    
    def _update_visualization(self):
        """Update the visualization display with both players' states."""
        if self.display_env:
            try:
                # Copy states from individual environments to display environment
                if self.player1_alive and hasattr(self.env_p1, 'player'):
                    self.display_env.game.player1.locked_positions = copy.deepcopy(self.env_p1.player.locked_positions)
                    self.display_env.game.player1.current_piece = copy.deepcopy(self.env_p1.player.current_piece)
                    self.display_env.game.player1.score = self.player1_score
                
                if self.player2_alive and hasattr(self.env_p2, 'player'):
                    self.display_env.game.player2.locked_positions = copy.deepcopy(self.env_p2.player.locked_positions)
                    self.display_env.game.player2.current_piece = copy.deepcopy(self.env_p2.player.current_piece)
                    self.display_env.game.player2.score = self.player2_score
                
                # Render the combined state
                self.display_env.render()
            except Exception as e:
                self.logger.warning(f"Visualization update failed: {e}")
    
    def render(self):
        """Render the game state."""
        if self.display_env:
            self._update_visualization()
    
    def close(self):
        """Close all environments."""
        self.env_p1.close()
        self.env_p2.close()
        if self.display_env:
            self.display_env.close()
    
    def get_metrics(self) -> Dict:
        """Get current game metrics."""
        return {
            'episode_steps': self.episode_steps,
            'player1_alive': self.player1_alive,
            'player2_alive': self.player2_alive,
            'player1_score': self.player1_score,
            'player2_score': self.player2_score,
            'player1_lines': self.player1_lines,
            'player2_lines': self.player2_lines
        }

# Test function
def test_true_multiplayer():
    """Test the true multiplayer environment."""
    print("🧪 Testing True Multiplayer Environment")
    print("=" * 50)
    
    env = TrueMultiplayerTetrisEnv(headless=True)
    
    # Test reset
    obs = env.reset()
    print(f"✅ Reset successful: {list(obs.keys())}")
    print(f"   Player 1 obs keys: {list(obs['player1'].keys())}")
    print(f"   Player 2 obs keys: {list(obs['player2'].keys())}")
    
    # Test step
    actions = {'player1': 15, 'player2': 25}
    obs, rewards, done, info = env.step(actions)
    
    print(f"✅ Step successful:")
    print(f"   Rewards: P1={rewards['player1']:.2f}, P2={rewards['player2']:.2f}")
    print(f"   Done: {done}")
    print(f"   Winner: {info.get('winner', 'None')}")
    
    env.close()
    print("✅ True multiplayer environment working!")
    
    return True

if __name__ == "__main__":
    test_true_multiplayer() 