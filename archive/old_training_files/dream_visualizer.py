#!/usr/bin/env python3
"""
DREAM Training Visualizer

Provides real-time visualization of DREAM training including:
- Agent performance metrics
- World model accuracy
- Imagination vs reality comparisons
- Batch update progress
- Game state visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from collections import deque
import time
import threading
import queue

class DREAMVisualizer:
    """Real-time visualization for DREAM training"""
    
    def __init__(self, enable_plots=True, enable_game_viz=True, max_history=1000):
        self.enable_plots = enable_plots
        self.enable_game_viz = enable_game_viz
        self.max_history = max_history
        
        # Data storage
        self.episode_rewards = deque(maxlen=max_history)
        self.episode_lengths = deque(maxlen=max_history)
        self.world_losses = deque(maxlen=max_history)
        self.actor_losses = deque(maxlen=max_history)
        self.learning_rates = {'world': deque(maxlen=max_history), 'actor': deque(maxlen=max_history)}
        self.imagination_accuracy = deque(maxlen=max_history)
        self.batch_metrics = {'episodes': [], 'avg_reward': [], 'world_loss': [], 'actor_loss': []}
        
        # Current game state for visualization
        self.current_board = None
        self.current_piece = None
        self.current_score = 0
        self.current_lines = 0
        
        # Threading for non-blocking updates
        self.update_queue = queue.Queue()
        self.running = True
        
        if self.enable_plots:
            self.setup_plots()
    
    def setup_plots(self):
        """Setup matplotlib plots for real-time visualization"""
        plt.ion()  # Interactive mode
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('DREAM Training Visualization', fontsize=16)
        
        # Episode rewards
        self.ax_rewards = self.axes[0, 0]
        self.ax_rewards.set_title('Episode Rewards')
        self.ax_rewards.set_xlabel('Episode')
        self.ax_rewards.set_ylabel('Reward')
        self.reward_line, = self.ax_rewards.plot([], [], 'b-', alpha=0.7)
        self.reward_avg_line, = self.ax_rewards.plot([], [], 'r-', linewidth=2, label='Average (50 ep)')
        self.ax_rewards.legend()
        self.ax_rewards.grid(True, alpha=0.3)
        
        # Episode lengths
        self.ax_lengths = self.axes[0, 1]
        self.ax_lengths.set_title('Episode Lengths')
        self.ax_lengths.set_xlabel('Episode')
        self.ax_lengths.set_ylabel('Steps')
        self.length_line, = self.ax_lengths.plot([], [], 'g-', alpha=0.7)
        self.ax_lengths.grid(True, alpha=0.3)
        
        # World model loss
        self.ax_world_loss = self.axes[0, 2]
        self.ax_world_loss.set_title('World Model Loss')
        self.ax_world_loss.set_xlabel('Episode')
        self.ax_world_loss.set_ylabel('Loss')
        self.world_loss_line, = self.ax_world_loss.plot([], [], 'purple', alpha=0.7)
        self.ax_world_loss.grid(True, alpha=0.3)
        
        # Actor loss
        self.ax_actor_loss = self.axes[1, 0]
        self.ax_actor_loss.set_title('Actor-Critic Loss')
        self.ax_actor_loss.set_xlabel('Episode')
        self.ax_actor_loss.set_ylabel('Loss')
        self.actor_loss_line, = self.ax_actor_loss.plot([], [], 'orange', alpha=0.7)
        self.ax_actor_loss.grid(True, alpha=0.3)
        
        # Learning rates
        self.ax_lr = self.axes[1, 1]
        self.ax_lr.set_title('Learning Rates')
        self.ax_lr.set_xlabel('Episode')
        self.ax_lr.set_ylabel('Learning Rate')
        self.world_lr_line, = self.ax_lr.plot([], [], 'purple', label='World Model', alpha=0.7)
        self.actor_lr_line, = self.ax_lr.plot([], [], 'orange', label='Actor-Critic', alpha=0.7)
        self.ax_lr.legend()
        self.ax_lr.set_yscale('log')
        self.ax_lr.grid(True, alpha=0.3)
        
        # Batch performance
        self.ax_batch = self.axes[1, 2]
        self.ax_batch.set_title('Batch Performance')
        self.ax_batch.set_xlabel('Batch')
        self.ax_batch.set_ylabel('Average Reward')
        self.batch_line, = self.ax_batch.plot([], [], 'red', linewidth=2, label='Batch Avg')
        self.ax_batch.legend()
        self.ax_batch.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def update_episode_data(self, episode, reward, length, world_loss, actor_loss, world_lr, actor_lr):
        """Update data for a single episode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.world_losses.append(world_loss)
        self.actor_losses.append(actor_loss)
        self.learning_rates['world'].append(world_lr)
        self.learning_rates['actor'].append(actor_lr)
        
        if self.enable_plots:
            self.update_plots()
    
    def update_batch_data(self, batch_idx, episodes, avg_reward, world_loss, actor_loss):
        """Update data for batch completion"""
        self.batch_metrics['episodes'].append(episodes)
        self.batch_metrics['avg_reward'].append(avg_reward)
        self.batch_metrics['world_loss'].append(world_loss)
        self.batch_metrics['actor_loss'].append(actor_loss)
        
        if self.enable_plots:
            self.update_batch_plot()
    
    def update_game_state(self, board, piece, score, lines):
        """Update current game state for visualization"""
        self.current_board = board.copy() if board is not None else None
        self.current_piece = piece
        self.current_score = score
        self.current_lines = lines
    
    def update_imagination_accuracy(self, accuracy):
        """Update imagination vs reality accuracy"""
        self.imagination_accuracy.append(accuracy)
    
    def update_plots(self):
        """Update all real-time plots"""
        if not self.enable_plots or len(self.episode_rewards) == 0:
            return
        
        try:
            episodes = list(range(len(self.episode_rewards)))
            
            # Update episode rewards
            self.reward_line.set_data(episodes, list(self.episode_rewards))
            if len(self.episode_rewards) >= 50:
                # Calculate moving average
                rewards = list(self.episode_rewards)
                avg_rewards = []
                for i in range(49, len(rewards)):
                    avg_rewards.append(np.mean(rewards[i-49:i+1]))
                avg_episodes = episodes[49:]
                self.reward_avg_line.set_data(avg_episodes, avg_rewards)
            
            # Update episode lengths
            self.length_line.set_data(episodes, list(self.episode_lengths))
            
            # Update world model loss
            self.world_loss_line.set_data(episodes, list(self.world_losses))
            
            # Update actor loss
            self.actor_loss_line.set_data(episodes, list(self.actor_losses))
            
            # Update learning rates
            self.world_lr_line.set_data(episodes, list(self.learning_rates['world']))
            self.actor_lr_line.set_data(episodes, list(self.learning_rates['actor']))
            
            # Auto-scale axes
            for ax in [self.ax_rewards, self.ax_lengths, self.ax_world_loss, self.ax_actor_loss, self.ax_lr]:
                ax.relim()
                ax.autoscale_view()
            
            # Update display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"⚠️ Visualization update error: {e}")
    
    def update_batch_plot(self):
        """Update batch performance plot"""
        if not self.enable_plots or len(self.batch_metrics['avg_reward']) == 0:
            return
        
        try:
            batch_indices = list(range(len(self.batch_metrics['avg_reward'])))
            self.batch_line.set_data(batch_indices, self.batch_metrics['avg_reward'])
            
            self.ax_batch.relim()
            self.ax_batch.autoscale_view()
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"⚠️ Batch plot update error: {e}")
    
    def visualize_game_state(self, save_path=None):
        """Create a snapshot visualization of current game state"""
        if not self.enable_game_viz or self.current_board is None:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        
        # Draw the Tetris board
        board = self.current_board
        rows, cols = board.shape
        
        # Create colored representation
        color_map = {
            0: 'black',      # Empty
            1: 'cyan',       # I piece
            2: 'blue',       # J piece 
            3: 'orange',     # L piece
            4: 'yellow',     # O piece
            5: 'green',      # S piece
            6: 'purple',     # T piece
            7: 'red'         # Z piece
        }
        
        for i in range(rows):
            for j in range(cols):
                color = color_map.get(board[i, j], 'gray')
                rect = Rectangle((j, rows-i-1), 1, 1, facecolor=color, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
        
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.set_title(f'Tetris Game State\nScore: {self.current_score} | Lines: {self.current_lines}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        if len(self.episode_rewards) == 0:
            return "No data available for performance report."
        
        report = []
        report.append("=" * 60)
        report.append("DREAM TRAINING PERFORMANCE REPORT")
        report.append("=" * 60)
        
        # Episode statistics
        total_episodes = len(self.episode_rewards)
        avg_reward = np.mean(self.episode_rewards)
        best_reward = np.max(self.episode_rewards)
        worst_reward = np.min(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        
        report.append(f"Episodes Completed: {total_episodes}")
        report.append(f"Average Reward: {avg_reward:.2f}")
        report.append(f"Best Reward: {best_reward:.2f}")
        report.append(f"Worst Reward: {worst_reward:.2f}")
        report.append(f"Average Episode Length: {avg_length:.1f} steps")
        
        # Learning progress
        if len(self.episode_rewards) >= 50:
            recent_avg = np.mean(list(self.episode_rewards)[-50:])
            early_avg = np.mean(list(self.episode_rewards)[:50])
            improvement = recent_avg - early_avg
            report.append(f"Learning Progress (last 50 vs first 50): {improvement:+.2f}")
        
        # Loss statistics
        if self.world_losses:
            avg_world_loss = np.mean(self.world_losses)
            final_world_loss = self.world_losses[-1]
            report.append(f"Average World Model Loss: {avg_world_loss:.4f}")
            report.append(f"Final World Model Loss: {final_world_loss:.4f}")
        
        if self.actor_losses:
            avg_actor_loss = np.mean(self.actor_losses)
            final_actor_loss = self.actor_losses[-1]
            report.append(f"Average Actor-Critic Loss: {avg_actor_loss:.4f}")
            report.append(f"Final Actor-Critic Loss: {final_actor_loss:.4f}")
        
        # Batch statistics
        if self.batch_metrics['avg_reward']:
            batch_count = len(self.batch_metrics['avg_reward'])
            best_batch_reward = np.max(self.batch_metrics['avg_reward'])
            report.append(f"Batches Completed: {batch_count}")
            report.append(f"Best Batch Average Reward: {best_batch_reward:.2f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_training_data(self, filepath):
        """Save all training data to file"""
        data = {
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'world_losses': list(self.world_losses),
            'actor_losses': list(self.actor_losses),
            'learning_rates': {
                'world': list(self.learning_rates['world']),
                'actor': list(self.learning_rates['actor'])
            },
            'batch_metrics': self.batch_metrics,
            'imagination_accuracy': list(self.imagination_accuracy)
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Training data saved to {filepath}")
    
    def visualize_agent_demo(self, demo_result, save_path=None):
        """Visualize the agent demonstration results"""
        if not demo_result:
            print("⚠️ No demo result to visualize")
            return
            
        try:
            # Create demo visualization figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Agent Demo - Episode {demo_result.get('episode', 1)}", fontsize=16)
            
            # Show final board state (if available)
            if demo_result.get('game_states') and len(demo_result['game_states']) > 0:
                final_board = demo_result['game_states'][-1]
                im = axes[0, 0].imshow(final_board, cmap='viridis', aspect='auto')
                axes[0, 0].set_title(f"Final Board State\nScore: {demo_result.get('final_score', 0)}")
                axes[0, 0].set_xlabel("Column")
                axes[0, 0].set_ylabel("Row")
                plt.colorbar(im, ax=axes[0, 0])
            else:
                axes[0, 0].text(0.5, 0.5, 'No Board State\nAvailable', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title("Board State")
            
            # Show performance metrics
            metrics = ['Total Reward', 'Final Score', 'Lines Cleared', 'Steps']
            values = [
                demo_result.get('total_reward', 0),
                demo_result.get('final_score', 0),
                demo_result.get('lines_cleared', 0),
                demo_result.get('steps', 0)
            ]
            bars = axes[0, 1].bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
            axes[0, 1].set_title("Demo Performance Metrics")
            axes[0, 1].set_ylabel("Value")
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1f}', ha='center', va='bottom')
            
            # Show action distribution
            if demo_result.get('actions'):
                action_counts = {}
                for action in demo_result['actions']:
                    # Convert action to string to avoid comparison issues
                    action_key = str(action)
                    action_counts[action_key] = action_counts.get(action_key, 0) + 1
                
                if action_counts:
                    actions, counts = zip(*sorted(action_counts.items(), key=lambda x: x[0]))
                    # Convert back to numeric if possible for proper ordering
                    try:
                        numeric_actions = [float(a) if '.' in a else int(a) for a in actions]
                        sorted_pairs = sorted(zip(numeric_actions, counts))
                        actions, counts = zip(*sorted_pairs)
                    except (ValueError, TypeError):
                        # Keep as strings if conversion fails
                        pass
                    
                    axes[1, 0].bar(range(len(actions)), counts, color='purple', alpha=0.7)
                    axes[1, 0].set_title("Action Distribution")
                    axes[1, 0].set_xlabel("Action")
                    axes[1, 0].set_ylabel("Count")
                    axes[1, 0].set_xticks(range(len(actions)))
                    axes[1, 0].set_xticklabels([str(a) for a in actions], rotation=45)
            else:
                axes[1, 0].text(0.5, 0.5, 'No Action Data\nAvailable', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title("Action Distribution")
            
            # Show step progression (simple timeline)
            steps = list(range(demo_result.get('steps', 1)))
            axes[1, 1].plot(steps, [1] * len(steps), 'g-', linewidth=2, label='Game Active')
            if len(steps) > 0:
                axes[1, 1].scatter([len(steps)-1], [1], color='red', s=100, label='Game End')
            axes[1, 1].set_title(f"Game Progression ({demo_result.get('steps', 0)} steps)")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Game State")
            axes[1, 1].set_ylim(0.5, 1.5)
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"📸 Demo visualization saved to: {save_path}")
            
            plt.show(block=False)
            plt.pause(2.0)  # Show for 2 seconds
            
        except Exception as e:
            print(f"⚠️ Demo visualization failed: {e}")
            import traceback
            traceback.print_exc()


    def update_agent_evaluation(self, eval_result, batch_idx):
        """Update dashboard with agent evaluation results instead of pop-up"""
        if not eval_result:
            print("⚠️ No evaluation result to display")
            return
            
        try:
            # Store evaluation metrics for dashboard display
            eval_metrics = {
                'batch_idx': batch_idx,
                'total_reward': eval_result.get('total_reward', 0),
                'final_score': eval_result.get('final_score', 0),
                'lines_cleared': eval_result.get('lines_cleared', 0),
                'steps': eval_result.get('steps', 0),
                'actions': eval_result.get('actions', [])
            }
            
            # Add to batch metrics for display
            if not hasattr(self, 'agent_evaluations'):
                self.agent_evaluations = []
            self.agent_evaluations.append(eval_metrics)
            
            # Update the dashboard plots with evaluation data
            self.update_evaluation_plots()
            
            print(f"📊 Agent evaluation integrated into dashboard for batch {batch_idx}")
            print(f"   Reward: {eval_metrics['total_reward']:.2f}, Score: {eval_metrics['final_score']}, Steps: {eval_metrics['steps']}")
            
        except Exception as e:
            print(f"⚠️ Dashboard evaluation update failed: {e}")
    
    def update_evaluation_plots(self):
        """Update dashboard plots with agent evaluation data"""
        if not hasattr(self, 'agent_evaluations') or not self.agent_evaluations:
            return
            
        try:
            # Get evaluation data
            eval_data = self.agent_evaluations
            batch_indices = [e['batch_idx'] for e in eval_data]
            eval_rewards = [e['total_reward'] for e in eval_data]
            eval_scores = [e['final_score'] for e in eval_data]
            eval_steps = [e['steps'] for e in eval_data]
            
            # Update batch performance plot with evaluation overlay
            if hasattr(self, 'ax_batch') and self.enable_plots:
                # Add evaluation line to batch plot
                if not hasattr(self, 'eval_line'):
                    self.eval_line, = self.ax_batch.plot([], [], 'g--', linewidth=2, label='Agent Evaluation', alpha=0.8)
                    self.ax_batch.legend()
                
                self.eval_line.set_data(batch_indices, eval_rewards)
                
                # Update axis limits
                self.ax_batch.relim()
                self.ax_batch.autoscale_view()
                
                # Update display
                if hasattr(self, 'fig'):
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
            
            print(f"📈 Dashboard updated with {len(eval_data)} evaluation points")
            
        except Exception as e:
            print(f"⚠️ Evaluation plot update failed: {e}")

    def close(self):
        """Close visualization resources"""
        self.running = False
        if self.enable_plots:
            plt.close('all')
            plt.ioff()

class BatchTracker:
    """Tracks batch training progress"""
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.current_batch_episodes = []
        self.current_batch_rewards = []
        self.current_batch_losses = {'world': [], 'actor': []}
        self.batch_count = 0
    
    def add_episode(self, episode, reward, world_loss, actor_loss):
        """Add episode to current batch"""
        self.current_batch_episodes.append(episode)
        self.current_batch_rewards.append(reward)
        self.current_batch_losses['world'].append(world_loss)
        self.current_batch_losses['actor'].append(actor_loss)
    
    def is_batch_complete(self):
        """Check if current batch is complete"""
        return len(self.current_batch_episodes) >= self.batch_size
    
    def finalize_batch(self):
        """Finalize current batch and return metrics"""
        if not self.is_batch_complete():
            return None
        
        batch_metrics = {
            'batch_idx': self.batch_count,
            'episodes': list(self.current_batch_episodes),
            'avg_reward': np.mean(self.current_batch_rewards),
            'avg_world_loss': np.mean(self.current_batch_losses['world']),
            'avg_actor_loss': np.mean(self.current_batch_losses['actor']),
            'total_episodes': len(self.current_batch_episodes)
        }
        
        # Reset for next batch
        self.current_batch_episodes = []
        self.current_batch_rewards = []
        self.current_batch_losses = {'world': [], 'actor': []}
        self.batch_count += 1
        
        return batch_metrics 