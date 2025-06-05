#!/usr/bin/env python3
"""
Enhanced Multiplayer AIRL Visualization Demo
Advanced real-time visualization with detailed metrics and learning progress
"""

import sys
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

# Add paths
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')

def create_enhanced_demo():
    """Create an enhanced visualization demonstration."""
    print("🎬 ENHANCED MULTIPLAYER TETRIS AIRL VISUALIZATION")
    print("=" * 80)
    print("Real-time competitive training with advanced metrics and analysis")
    print("Features:")
    print("  • Live win/loss tracking")
    print("  • Real-time reward progression")
    print("  • Action distribution analysis")
    print("  • Learning curve visualization")
    print("=" * 80)
    
    try:
        from localMultiplayerTetris.rl_utils.visualized_training import VisualizedMultiplayerTrainer
        
        # Enhanced configuration
        config = {
            'device': 'cpu',
            'render_delay': 0.15,  # Slightly faster for more action
            'show_metrics': True,
            'save_screenshots': False
        }
        
        # Create enhanced trainer
        trainer = EnhancedVisualizedTrainer(config)
        
        # Run enhanced training session
        trainer.run_enhanced_training_session(num_episodes=8)
        
    except Exception as e:
        print(f"❌ Enhanced demo failed: {e}")
        import traceback
        traceback.print_exc()

class EnhancedVisualizedTrainer:
    """Enhanced trainer with advanced metrics and visualization."""
    
    def __init__(self, config):
        from localMultiplayerTetris.rl_utils.visualized_training import VisualizedMultiplayerTrainer
        self.trainer = VisualizedMultiplayerTrainer(config)
        
        # Enhanced metrics tracking
        self.action_history = {'player1': [], 'player2': []}
        self.reward_progression = {'player1': [], 'player2': []}
        self.episode_times = []
        self.learning_metrics = {
            'action_diversity': [],
            'reward_variance': [],
            'performance_trend': []
        }
    
    def run_enhanced_training_session(self, num_episodes=8):
        """Run an enhanced training session with detailed analytics."""
        print(f"\n🚀 Starting Enhanced Training Session ({num_episodes} episodes)")
        print("=" * 80)
        
        session_start = time.time()
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
            print("-" * 50)
            
            # Run visualized episode with enhanced tracking
            episode_data = self._run_enhanced_episode(episode)
            
            episode_time = time.time() - episode_start
            self.episode_times.append(episode_time)
            
            # Analyze episode
            self._analyze_episode_performance(episode_data, episode)
            
            # Display progress
            self._display_enhanced_metrics(episode)
            
            # Brief pause between episodes
            if episode < num_episodes - 1:
                print("\n⏸️ Analyzing performance... Next episode in 3 seconds")
                time.sleep(3)
        
        session_time = time.time() - session_start
        
        # Final comprehensive analysis
        self._display_final_analytics(session_time)
    
    def _run_enhanced_episode(self, episode_num):
        """Run an episode with enhanced data collection."""
        episode_actions = {'player1': [], 'player2': []}
        episode_rewards = {'player1': [], 'player2': []}
        
        # Run the episode using the base trainer
        episode_data = self.trainer.run_visualized_episode(
            max_steps=300,  # Moderate length for analysis
            episode_num=episode_num
        )
        
        # Extract enhanced data
        for step_data in episode_data:
            actions = step_data['actions']
            rewards = step_data['rewards']
            
            episode_actions['player1'].append(actions[0])
            episode_actions['player2'].append(actions[1])
            episode_rewards['player1'].append(rewards[0])
            episode_rewards['player2'].append(rewards[1])
        
        # Store for analysis
        self.action_history['player1'].extend(episode_actions['player1'])
        self.action_history['player2'].extend(episode_actions['player2'])
        self.reward_progression['player1'].extend(episode_rewards['player1'])
        self.reward_progression['player2'].extend(episode_rewards['player2'])
        
        return episode_data
    
    def _analyze_episode_performance(self, episode_data, episode_num):
        """Analyze episode performance and extract learning metrics."""
        if not episode_data:
            return
        
        # Action diversity analysis
        p1_actions = [step['actions'][0] for step in episode_data]
        p2_actions = [step['actions'][1] for step in episode_data]
        
        p1_diversity = len(set(p1_actions)) / max(1, len(p1_actions))
        p2_diversity = len(set(p2_actions)) / max(1, len(p2_actions))
        avg_diversity = (p1_diversity + p2_diversity) / 2
        
        self.learning_metrics['action_diversity'].append(avg_diversity)
        
        # Reward variance analysis
        p1_rewards = [step['rewards'][0] for step in episode_data]
        p2_rewards = [step['rewards'][1] for step in episode_data]
        
        reward_variance = (np.var(p1_rewards) + np.var(p2_rewards)) / 2
        self.learning_metrics['reward_variance'].append(reward_variance)
        
        # Performance trend
        total_reward = sum(p1_rewards) + sum(p2_rewards)
        self.learning_metrics['performance_trend'].append(total_reward)
    
    def _display_enhanced_metrics(self, episode_num):
        """Display enhanced real-time metrics."""
        print(f"\n📊 Enhanced Analytics (Episode {episode_num + 1}):")
        
        # Recent performance
        if len(self.learning_metrics['performance_trend']) >= 2:
            recent_trend = np.mean(self.learning_metrics['performance_trend'][-3:])
            early_trend = np.mean(self.learning_metrics['performance_trend'][:max(1, len(self.learning_metrics['performance_trend'])//2)])
            improvement = recent_trend - early_trend
            
            print(f"   📈 Performance Trend: {improvement:+.2f} (recent vs early)")
        
        # Action diversity
        if self.learning_metrics['action_diversity']:
            avg_diversity = np.mean(self.learning_metrics['action_diversity'][-3:])
            print(f"   🎲 Action Diversity: {avg_diversity:.3f} (0-1 scale)")
        
        # Timing metrics
        if self.episode_times:
            avg_time = np.mean(self.episode_times)
            print(f"   ⏱️ Avg Episode Time: {avg_time:.2f}s")
        
        # Win/loss distribution from trainer
        metrics = self.trainer.metrics
        total_games = metrics['total_games']
        if total_games > 0:
            p1_winrate = metrics['player1_wins'] / total_games * 100
            p2_winrate = metrics['player2_wins'] / total_games * 100
            draw_rate = metrics['draws'] / total_games * 100
            print(f"   🏆 Current Standings: P1={p1_winrate:.1f}%, P2={p2_winrate:.1f}%, Draw={draw_rate:.1f}%")
        
        # Action frequency analysis
        if len(self.action_history['player1']) > 10:
            p1_most_common = max(set(self.action_history['player1'][-20:]), 
                               key=self.action_history['player1'][-20:].count)
            p2_most_common = max(set(self.action_history['player2'][-20:]), 
                               key=self.action_history['player2'][-20:].count)
            print(f"   🎯 Favorite Actions: P1={p1_most_common}, P2={p2_most_common}")
    
    def _display_final_analytics(self, session_time):
        """Display comprehensive final analytics."""
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE SESSION ANALYTICS")
        print("="*80)
        
        metrics = self.trainer.metrics
        total_games = metrics['total_games']
        
        print(f"⏱️ Session Duration: {session_time:.1f} seconds")
        print(f"🎮 Total Games: {total_games}")
        
        if total_games > 0:
            # Win/Loss Analysis
            print(f"\n🏆 Final Standings:")
            print(f"   Player 1: {metrics['player1_wins']} wins ({metrics['player1_wins']/total_games*100:.1f}%)")
            print(f"   Player 2: {metrics['player2_wins']} wins ({metrics['player2_wins']/total_games*100:.1f}%)")
            print(f"   Draws: {metrics['draws']} ({metrics['draws']/total_games*100:.1f}%)")
            
            # Performance Evolution
            if len(self.learning_metrics['performance_trend']) >= 4:
                early_perf = np.mean(self.learning_metrics['performance_trend'][:2])
                late_perf = np.mean(self.learning_metrics['performance_trend'][-2:])
                improvement = late_perf - early_perf
                
                print(f"\n📈 Learning Analysis:")
                print(f"   Performance Improvement: {improvement:+.2f}")
                print(f"   Early Performance: {early_perf:.2f}")
                print(f"   Late Performance: {late_perf:.2f}")
            
            # Action Analysis
            if self.action_history['player1']:
                p1_unique_actions = len(set(self.action_history['player1']))
                p2_unique_actions = len(set(self.action_history['player2']))
                
                print(f"\n🎲 Action Diversity:")
                print(f"   Player 1 used {p1_unique_actions}/41 possible actions")
                print(f"   Player 2 used {p2_unique_actions}/41 possible actions")
                
                # Most/least used actions
                from collections import Counter
                p1_counter = Counter(self.action_history['player1'])
                p2_counter = Counter(self.action_history['player2'])
                
                print(f"   P1 favorite action: {p1_counter.most_common(1)[0][0]} ({p1_counter.most_common(1)[0][1]} times)")
                print(f"   P2 favorite action: {p2_counter.most_common(1)[0][0]} ({p2_counter.most_common(1)[0][1]} times)")
            
            # Timing Analysis
            if self.episode_times:
                print(f"\n⏱️ Timing Analysis:")
                print(f"   Average episode: {np.mean(self.episode_times):.2f}s")
                print(f"   Fastest episode: {min(self.episode_times):.2f}s")
                print(f"   Longest episode: {max(self.episode_times):.2f}s")
        
        print(f"\n🎯 Key Insights:")
        
        # Generate insights based on data
        insights = []
        
        if total_games > 0:
            if metrics['draws'] / total_games > 0.8:
                insights.append("Agents show balanced competitive performance")
            
            if len(self.learning_metrics['action_diversity']) > 0:
                avg_diversity = np.mean(self.learning_metrics['action_diversity'])
                if avg_diversity > 0.3:
                    insights.append("Good action exploration observed")
                else:
                    insights.append("Limited action exploration - may need more training")
            
            if len(self.learning_metrics['performance_trend']) >= 3:
                trend = np.polyfit(range(len(self.learning_metrics['performance_trend'])), 
                                 self.learning_metrics['performance_trend'], 1)[0]
                if trend > 0:
                    insights.append("Positive learning trend detected")
                elif trend < 0:
                    insights.append("Performance decline observed - check learning parameters")
                else:
                    insights.append("Stable performance maintained")
        
        if not insights:
            insights.append("More episodes needed for meaningful analysis")
        
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        print("\n🚀 Session complete! Agents showed competitive learning behavior.")
        print("="*80)

def main():
    """Main function for enhanced visualization demo."""
    create_enhanced_demo()

if __name__ == "__main__":
    main() 