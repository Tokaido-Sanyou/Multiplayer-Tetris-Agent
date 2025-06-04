"""Evaluate PPO models against baseline DQN agents in the Tetris environment.

Usage
-----
$ python -m localMultiplayerTetris.eval_ppo_vs_dqn \
        --ppo-model weights/100k/ppo_generator_100000/model.pth \
        --reward-net weights/100k/reward_net_100000.pth \
        --dqn-model tetris-ai-master/sample_torch.pth \
        --episodes 20 \
        --runs 5

Notes
-----
• Evaluates PPO models against baseline DQN agents
• Logs detailed metrics and statistics to TensorBoard
• Supports multiple evaluation runs for robust comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from .tetris_env import TetrisEnv
from .dqn_adapter import enumerate_next_states
from .utils import add_garbage_line
from .ppo_agent import PPOAgent  # Assuming this exists
from .reward_model import RewardModel  # Assuming this exists

def _mirror_b_to_a(env_a: TetrisEnv, env_b: TetrisEnv) -> None:
    """Mirror player 2's board to player 1's environment for side-by-side display."""
    env_a.game.player2.locked_positions = env_b.player.locked_positions.copy()
    env_a.game.player2.current_piece = env_b.player.current_piece
    env_a.game.player2.next_pieces = env_b.player.next_pieces.copy()
    env_a.game.player2.hold_piece = env_b.player.hold_piece
    env_a.game.player2.score = env_b.player.score
    # Lines cleared is tracked through score, no need to sync lines attribute

def preprocess_state(state: Dict) -> torch.Tensor:
    """Convert state dictionary to tensor format expected by PPO agent.
    
    Returns a 206-dimensional tensor:
    - First 200 values: Flattened grid
    - Next 6 values: next_piece, hold_piece, current_shape, rotation, x, y
    """
    # Flatten grid and append metadata scalars
    grid = torch.FloatTensor(state['grid'].flatten())
    # Scalars: next, hold, curr_shape, rotation, x, y (no can_hold)
    next_piece = torch.FloatTensor([state.get('next_piece', 0)])
    hold_piece = torch.FloatTensor([state.get('hold_piece', 0)])
    curr_shape = torch.FloatTensor([state.get('current_shape', 0)])
    curr_rot = torch.FloatTensor([state.get('current_rotation', 0)])
    curr_x = torch.FloatTensor([state.get('current_x', 0)])
    curr_y = torch.FloatTensor([state.get('current_y', 0)])
    return torch.cat([grid, next_piece, hold_piece, curr_shape, curr_rot, curr_x, curr_y])

def load_ppo_model(model_path: Path, reward_net_path: Path, device: torch.device) -> Tuple[PPOAgent, RewardModel]:
    """Load PPO model and reward network."""
    # Load reward network
    reward_net = RewardModel()  # Adjust based on your actual model
    reward_net.load_state_dict(torch.load(reward_net_path, map_location=device))
    reward_net.eval()
    reward_net.to(device)
    
    # Load PPO agent with correct state dimension
    ppo_agent = PPOAgent(state_dim=206)  # Explicitly set state_dim to match saved model
    ppo_agent.load_state_dict(torch.load(model_path, map_location=device))
    ppo_agent.eval()
    ppo_agent.to(device)
    
    return ppo_agent, reward_net

def load_dqn_model(pth_file: Path, device: torch.device) -> nn.Module:
    """Load DQN model from checkpoint."""
    from .eval_pytorch_dqn import load_model  # Reuse existing loader
    return load_model(pth_file, device)

class ProgressTracker:
    """Tracks and visualizes progress metrics for a single match."""
    
    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        # Initialize data storage
        self.steps = []
        self.ppo_scores = []
        self.dqn_scores = []
        self.ppo_lines = []
        self.dqn_lines = []
        self.ppo_garbage = []
        self.dqn_garbage = []
        
        # Create figure and subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.suptitle('PPO vs DQN Performance Comparison')
        
        # Setup subplots
        self.ax1.set_title('Scores Over Time')
        self.ax1.set_xlabel('Steps')
        self.ax1.set_ylabel('Score')
        
        self.ax2.set_title('Lines Cleared Over Time')
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Lines')
        
        self.ax3.set_title('Garbage Lines Sent')
        self.ax3.set_xlabel('Steps')
        self.ax3.set_ylabel('Garbage Lines')
    
    def update(self, step: int, metrics: Dict) -> None:
        """Update progress data and plot."""
        self.steps.append(step)
        self.ppo_scores.append(metrics['ppo_score'])
        self.dqn_scores.append(metrics['dqn_score'])
        self.ppo_lines.append(metrics['ppo_lines'])
        self.dqn_lines.append(metrics['dqn_lines'])
        self.ppo_garbage.append(metrics['ppo_garbage_sent'])
        self.dqn_garbage.append(metrics['dqn_garbage_sent'])
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot scores
        self.ax1.plot(self.steps, self.ppo_scores, label='PPO', color='blue')
        self.ax1.plot(self.steps, self.dqn_scores, label='DQN', color='red')
        self.ax1.set_title('Scores Over Time')
        self.ax1.set_xlabel('Steps')
        self.ax1.set_ylabel('Score')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Plot lines
        self.ax2.plot(self.steps, self.ppo_lines, label='PPO', color='blue')
        self.ax2.plot(self.steps, self.dqn_lines, label='DQN', color='red')
        self.ax2.set_title('Lines Cleared Over Time')
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Lines')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Plot garbage
        self.ax3.plot(self.steps, self.ppo_garbage, label='PPO', color='blue')
        self.ax3.plot(self.steps, self.dqn_garbage, label='DQN', color='red')
        self.ax3.set_title('Garbage Lines Sent')
        self.ax3.set_xlabel('Steps')
        self.ax3.set_ylabel('Garbage Lines')
        self.ax3.legend()
        self.ax3.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        self.writer.add_figure('Progress/Combined_Metrics', self.fig, step)

def evaluate_match(ppo_agent: PPOAgent, reward_net: RewardModel, dqn_model: nn.Module, 
                  run_id: int, match_id: int, writer: SummaryWriter, 
                  headless: bool = True, delay: float = 0.1,
                  track_progress: bool = False) -> Dict:
    """Run a single match between PPO and DQN agents.
    
    Args:
        track_progress: If True, logs step-wise metrics to TensorBoard for performance tracking.
    """
    device = next(ppo_agent.parameters()).device
    
    # Create environments
    env_ppo = TetrisEnv(single_player=False, headless=headless)
    env_dqn = TetrisEnv(single_player=False, headless=True)  # Always headless
    
    obs_ppo, _ = env_ppo.reset()
    obs_dqn, _ = env_dqn.reset()
    
    metrics = {
        'ppo_score': 0,
        'dqn_score': 0,
        'ppo_lines': 0,
        'dqn_lines': 0,
        'steps': 0,
        'ppo_garbage_sent': 0,
        'dqn_garbage_sent': 0,
        'winner': None
    }
    
    done_ppo = done_dqn = False
    step = 0
    global_step = step + (match_id * 1000) + (run_id * 100000)  # Offset steps for each match/run
    
    def _garbage_from_lines(cleared: int) -> int:
        return {1: 0, 2: 1, 3: 2, 4: 4}.get(cleared, 0)
    
    while True:
        # PPO agent turn
        if not done_ppo:
            # Get PPO action
            with torch.no_grad():
                state_tensor = preprocess_state(obs_ppo).to(device)
                ppo_action = ppo_agent.act(state_tensor)  # Now passing tensor
            
            # Take action
            obs_ppo, _, term, trunc, info = env_ppo.step(ppo_action)
            metrics['ppo_score'] = info.get('score', metrics['ppo_score'])
            lines = info.get('lines_cleared', 0)
            metrics['ppo_lines'] += lines
            
            # Handle garbage lines
            garbage = _garbage_from_lines(lines)
            metrics['ppo_garbage_sent'] += garbage
            if garbage > 0 and not done_dqn:
                add_garbage_line(env_dqn.player.locked_positions, garbage)
            
            # Track progress
            if track_progress:
                writer.add_scalars('Progress/Scores', {
                    'PPO': metrics['ppo_score'],
                    'DQN': metrics['dqn_score'],
                    'Delta': metrics['ppo_score'] - metrics['dqn_score']
                }, global_step)
                
                writer.add_scalars('Progress/Lines', {
                    'PPO': metrics['ppo_lines'],
                    'DQN': metrics['dqn_lines'],
                    'Delta': metrics['ppo_lines'] - metrics['dqn_lines']
                }, global_step)
                
                writer.add_scalars('Progress/Garbage', {
                    'PPO': metrics['ppo_garbage_sent'],
                    'DQN': metrics['dqn_garbage_sent'],
                    'Delta': metrics['ppo_garbage_sent'] - metrics['dqn_garbage_sent']
                }, global_step)
            
            done_ppo = term or trunc
        
        if done_ppo:
            break
            
        # DQN agent turn
        if not done_dqn:
            # Get DQN action using existing logic
            mapping = enumerate_next_states(env_dqn)
            if not mapping:
                done_dqn = True
            else:
                states = np.array(list(mapping.keys()), dtype=np.float32)
                with torch.no_grad():
                    q_vals = dqn_model(torch.from_numpy(states).to(device))
                best_idx = int(torch.argmax(q_vals))
                dqn_action = list(mapping.values())[best_idx]
                
                # Take action
                obs_dqn, _, term, trunc, info = env_dqn.step(dqn_action)
                metrics['dqn_score'] = info.get('score', metrics['dqn_score'])
                lines = info.get('lines_cleared', 0)
                metrics['dqn_lines'] += lines
                
                # Handle garbage lines
                garbage = _garbage_from_lines(lines)
                metrics['dqn_garbage_sent'] += garbage
                if garbage > 0 and not done_ppo:
                    add_garbage_line(env_ppo.player.locked_positions, garbage)
                
                # Track progress
                if track_progress:
                    writer.add_scalars('Progress/Scores', {
                        'PPO': metrics['ppo_score'],
                        'DQN': metrics['dqn_score'],
                        'Delta': metrics['ppo_score'] - metrics['dqn_score']
                    }, global_step)
                    
                    writer.add_scalars('Progress/Lines', {
                        'PPO': metrics['ppo_lines'],
                        'DQN': metrics['dqn_lines'],
                        'Delta': metrics['ppo_lines'] - metrics['dqn_lines']
                    }, global_step)
                    
                    writer.add_scalars('Progress/Garbage', {
                        'PPO': metrics['ppo_garbage_sent'],
                        'DQN': metrics['dqn_garbage_sent'],
                        'Delta': metrics['ppo_garbage_sent'] - metrics['dqn_garbage_sent']
                    }, global_step)
                
                done_dqn = term or trunc
        
        if done_dqn:
            break
            
        # Mirror DQN's board to PPO's environment for display
        _mirror_b_to_a(env_ppo, env_dqn)
        
        # Update visualization
        if not headless:
            env_ppo.render()
            time.sleep(delay)  # Add delay between steps
        
        step += 1
        global_step = step + (match_id * 1000) + (run_id * 100000)  # Update global step
    
    metrics['steps'] = step
    metrics['winner'] = 'PPO' if metrics['ppo_score'] > metrics['dqn_score'] else 'DQN'
    
    # Log match metrics
    writer.add_scalar(f'Runs/Run_{run_id}/Matches/Match_{match_id}/PPO_Score', metrics['ppo_score'], 0)
    writer.add_scalar(f'Runs/Run_{run_id}/Matches/Match_{match_id}/DQN_Score', metrics['dqn_score'], 0)
    writer.add_scalar(f'Runs/Run_{run_id}/Matches/Match_{match_id}/PPO_Lines', metrics['ppo_lines'], 0)
    writer.add_scalar(f'Runs/Run_{run_id}/Matches/Match_{match_id}/DQN_Lines', metrics['dqn_lines'], 0)
    writer.add_scalar(f'Runs/Run_{run_id}/Matches/Match_{match_id}/Steps', metrics['steps'], 0)
    writer.add_scalar(f'Runs/Run_{run_id}/Matches/Match_{match_id}/PPO_Garbage', metrics['ppo_garbage_sent'], 0)
    writer.add_scalar(f'Runs/Run_{run_id}/Matches/Match_{match_id}/DQN_Garbage', metrics['dqn_garbage_sent'], 0)
    
    env_ppo.close()
    env_dqn.close()
    
    return metrics

def evaluate_multiple_runs(ppo_agent: PPOAgent, reward_net: RewardModel, dqn_model: nn.Module,
                         episodes: int = 10, runs: int = 5, headless: bool = True,
                         delay: float = 0.1, track_progress: bool = False) -> None:
    """Run multiple evaluation runs and log aggregate statistics."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'logs/ppo_vs_dqn_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)
    
    all_metrics = []
    
    for run in range(runs):
        print(f"\nStarting evaluation run {run + 1}/{runs}")
        run_metrics = []
        
        for episode in range(episodes):
            print(f"  Match {episode + 1}/{episodes}")
            metrics = evaluate_match(ppo_agent, reward_net, dqn_model, run, episode, writer, 
                                  headless, delay, track_progress)
            run_metrics.append(metrics)
            
            # Print match summary
            print(f"    PPO Score: {metrics['ppo_score']} Lines: {metrics['ppo_lines']}")
            print(f"    DQN Score: {metrics['dqn_score']} Lines: {metrics['dqn_lines']}")
            print(f"    Winner: {metrics['winner']}")
        
        all_metrics.extend(run_metrics)
        
        # Calculate and log run summary
        ppo_wins = sum(1 for m in run_metrics if m['winner'] == 'PPO')
        run_avg_ppo_score = np.mean([m['ppo_score'] for m in run_metrics])
        run_avg_dqn_score = np.mean([m['dqn_score'] for m in run_metrics])
        
        writer.add_scalar('Summary/RunAverages/PPO_Score', run_avg_ppo_score, run)
        writer.add_scalar('Summary/RunAverages/DQN_Score', run_avg_dqn_score, run)
        writer.add_scalar('Summary/RunAverages/PPO_WinRate', ppo_wins/episodes, run)
    
    # Calculate aggregate statistics
    all_ppo_scores = [m['ppo_score'] for m in all_metrics]
    all_dqn_scores = [m['dqn_score'] for m in all_metrics]
    all_ppo_lines = [m['ppo_lines'] for m in all_metrics]
    all_dqn_lines = [m['dqn_lines'] for m in all_metrics]
    total_ppo_wins = sum(1 for m in all_metrics if m['winner'] == 'PPO')
    
    # Log aggregate statistics
    writer.add_scalar('Aggregate/PPO_Score/Mean', np.mean(all_ppo_scores), 0)
    writer.add_scalar('Aggregate/PPO_Score/Std', np.std(all_ppo_scores), 0)
    writer.add_scalar('Aggregate/DQN_Score/Mean', np.mean(all_dqn_scores), 0)
    writer.add_scalar('Aggregate/DQN_Score/Std', np.std(all_dqn_scores), 0)
    writer.add_scalar('Aggregate/PPO_Lines/Mean', np.mean(all_ppo_lines), 0)
    writer.add_scalar('Aggregate/DQN_Lines/Mean', np.mean(all_dqn_lines), 0)
    writer.add_scalar('Aggregate/PPO_WinRate', total_ppo_wins/(runs*episodes), 0)
    
    # Add histograms
    writer.add_histogram('Distributions/PPO_Scores', np.array(all_ppo_scores), 0)
    writer.add_histogram('Distributions/DQN_Scores', np.array(all_dqn_scores), 0)
    writer.add_histogram('Distributions/PPO_Lines', np.array(all_ppo_lines), 0)
    writer.add_histogram('Distributions/DQN_Lines', np.array(all_dqn_lines), 0)
    
    writer.close()
    
    # Print final summary
    print(f"\n=== Evaluation Summary ({runs} runs, {episodes} matches each) ===")
    print(f"PPO Score:     {np.mean(all_ppo_scores):.1f} ± {np.std(all_ppo_scores):.1f}")
    print(f"DQN Score:     {np.mean(all_dqn_scores):.1f} ± {np.std(all_dqn_scores):.1f}")
    print(f"PPO Lines:     {np.mean(all_ppo_lines):.1f} ± {np.std(all_ppo_lines):.1f}")
    print(f"DQN Lines:     {np.mean(all_dqn_lines):.1f} ± {np.std(all_dqn_lines):.1f}")
    print(f"PPO Win Rate:  {total_ppo_wins/(runs*episodes)*100:.1f}%")
    print(f"\nTensorBoard logs: {log_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO vs DQN in Tetris")
    parser.add_argument("--ppo-model", type=Path, required=True, help="Path to PPO model checkpoint")
    parser.add_argument("--reward-net", type=Path, required=True, help="Path to reward network checkpoint")
    parser.add_argument("--dqn-model", type=Path, required=True, help="Path to DQN model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of matches per run")
    parser.add_argument("--runs", type=int, default=5, help="Number of evaluation runs")
    parser.add_argument("--cuda", action="store_true", help="Run on GPU if available")
    parser.add_argument("--render", action="store_true", help="Render the game window")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps when rendering (seconds)")
    parser.add_argument("--track-progress", action="store_true", help="Track and visualize step-wise performance metrics for first match")
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    # Load models
    ppo_agent, reward_net = load_ppo_model(args.ppo_model, args.reward_net, device)
    dqn_model = load_dqn_model(args.dqn_model, device)
    
    # Run evaluation
    evaluate_multiple_runs(
        ppo_agent, reward_net, dqn_model,
        episodes=args.episodes,
        runs=args.runs,
        headless=not args.render,
        delay=args.delay,
        track_progress=args.track_progress
    )

if __name__ == "__main__":
    main() 