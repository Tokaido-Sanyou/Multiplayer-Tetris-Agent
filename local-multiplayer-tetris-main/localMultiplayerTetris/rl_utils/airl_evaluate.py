#!/usr/bin/env python3
"""
AIRL Model Evaluation Script
Evaluate trained AIRL models and compare with expert and baseline performance
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Module-based imports with fallbacks
try:
    from localMultiplayerTetris.tetris_env import TetrisEnv
    from localMultiplayerTetris.rl_utils.airl_agent import AIRLAgent, Discriminator
    from localMultiplayerTetris.rl_utils.actor_critic import ActorCritic
    from localMultiplayerTetris.rl_utils.expert_loader import ExpertTrajectoryLoader
    from localMultiplayerTetris.dqn_adapter import enumerate_next_states, board_props
except ImportError:
    # Add more paths to sys.path for imports
    grandparent_dir = os.path.dirname(os.path.dirname(parent_dir))
    if grandparent_dir not in sys.path:
        sys.path.append(grandparent_dir)
    
    try:
        from tetris_env import TetrisEnv
        from airl_agent import AIRLAgent, Discriminator
        from actor_critic import ActorCritic
        from expert_loader import ExpertTrajectoryLoader
        from dqn_adapter import enumerate_next_states, board_props
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure you're running from the correct directory")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        sys.exit(1)

class AIRLEvaluator:
    """
    Comprehensive evaluator for AIRL-trained Tetris agents.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.env = TetrisEnv(single_player=True, headless=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Store evaluation results
        self.results = {}
        
    def _extract_features(self, observation: Dict) -> np.ndarray:
        """Extract features from TetrisEnv observation."""
        grid = observation['grid'].flatten()  # 20*10 = 200 features
        next_piece = np.array([observation['next_piece']])  # 1 feature
        hold_piece = np.array([observation['hold_piece']])  # 1 feature
        current_shape = np.array([observation['current_shape']])  # 1 feature
        current_rotation = np.array([observation['current_rotation']])  # 1 feature
        current_x = np.array([observation['current_x']])  # 1 feature
        current_y = np.array([observation['current_y']])  # 1 feature
        can_hold = np.array([observation['can_hold']])  # 1 feature
        
        features = np.concatenate([
            grid, next_piece, hold_piece, current_shape, 
            current_rotation, current_x, current_y, can_hold
        ]).astype(np.float32)
        
        return features
    
    def load_airl_model(self, checkpoint_path: str) -> AIRLAgent:
        """Load trained AIRL model from checkpoint."""
        self.logger.info(f"Loading AIRL model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Get dimensions from checkpoint
        sample_obs = self.env.reset()
        # Handle gym API version differences
        if isinstance(sample_obs, tuple):
            sample_obs = sample_obs[0]
        state_dim = self._extract_features(sample_obs).shape[0]
        action_dim = self.env.action_space.n
        
        # Initialize policy network
        policy = ActorCritic(
            input_dim=state_dim,
            output_dim=action_dim
        )
        
        # Initialize AIRL agent
        airl_agent = AIRLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            policy_network=policy,
            device=self.device
        )
        
        # Load state dictionaries
        agent_state = checkpoint['airl_agent_state']
        airl_agent.discriminator.load_state_dict(agent_state['discriminator_state_dict'])
        airl_agent.policy.load_state_dict(agent_state['policy_state_dict'])
        
        self.logger.info("AIRL model loaded successfully")
        return airl_agent
    
    def load_expert_model(self) -> 'DQNExpertAdapter':
        """Load expert DQN model adapter."""
        try:
            from tetris_ai_master.dqn_agent import DQNAgent
            from tetris_ai_master.tetris import Tetris
            
            expert_dqn = DQNAgent(state_size=4, modelFile='tetris-ai-master/sample.keras')
            return DQNExpertAdapter(expert_dqn)
        except ImportError:
            self.logger.warning("Could not load expert DQN model")
            return None
    
    def evaluate_agent(self, 
                      agent,
                      num_episodes: int = 100,
                      agent_name: str = "Agent",
                      deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate an agent over multiple episodes.
        
        Args:
            agent: Agent to evaluate (must have select_action method)
            num_episodes: Number of episodes to run
            agent_name: Name for logging
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating {agent_name} over {num_episodes} episodes")
        
        scores = []
        episode_lengths = []
        lines_cleared_list = []
        max_heights = []
        action_distributions = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            # Handle gym API version differences
            if isinstance(obs, tuple):
                obs = obs[0]
            state = self._extract_features(obs)
            done = False
            episode_length = 0
            total_reward = 0
            lines_cleared = 0
            episode_actions = []
            max_height_episode = 0
            
            while not done:
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(state, deterministic=deterministic)
                else:
                    # For expert adapter or other agents
                    action = agent.get_action(obs)
                
                step_result = self.env.step(action)
                # Handle gym API version differences
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                    truncated = False
                else:
                    next_obs, reward, done, truncated, info = step_result
                
                # Combine done and truncated
                done = done or truncated
                
                # Track metrics
                state = self._extract_features(next_obs)
                total_reward += reward
                episode_length += 1
                episode_actions.append(action)
                
                # Track max height
                grid = next_obs['grid']
                heights = [next((r for r in range(20) if grid[r][c] > 0), 20) for c in range(10)]
                max_height_episode = max(max_height_episode, max(heights))
                
                # Lines cleared (if available in info)
                if 'lines_cleared' in info:
                    lines_cleared += info['lines_cleared']
            
            scores.append(total_reward)
            episode_lengths.append(episode_length)
            lines_cleared_list.append(lines_cleared)
            max_heights.append(max_height_episode)
            action_distributions.append(episode_actions)
            
            if (episode + 1) % 20 == 0:
                self.logger.info(f"  Episode {episode + 1}/{num_episodes} - Score: {total_reward:.2f}")
        
        # Calculate action distribution statistics
        all_actions = [a for episode_actions in action_distributions for a in episode_actions]
        action_counts = {}
        for action in all_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate metrics
        metrics = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'median_score': np.median(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'mean_lines_cleared': np.mean(lines_cleared_list),
            'std_lines_cleared': np.std(lines_cleared_list),
            'mean_max_height': np.mean(max_heights),
            'std_max_height': np.std(max_heights),
            'action_distribution': action_counts,
            'hold_percentage': action_counts.get(40, 0) / len(all_actions) * 100 if all_actions else 0
        }
        
        self.logger.info(f"{agent_name} Results:")
        self.logger.info(f"  Score: {metrics['mean_score']:.2f} ± {metrics['std_score']:.2f}")
        self.logger.info(f"  Episode Length: {metrics['mean_episode_length']:.1f} ± {metrics['std_episode_length']:.1f}")
        self.logger.info(f"  Lines Cleared: {metrics['mean_lines_cleared']:.1f} ± {metrics['std_lines_cleared']:.1f}")
        self.logger.info(f"  Max Height: {metrics['mean_max_height']:.1f} ± {metrics['std_max_height']:.1f}")
        self.logger.info(f"  HOLD Usage: {metrics['hold_percentage']:.1f}%")
        
        return metrics
    
    def evaluate_discriminator(self, 
                              airl_agent: AIRLAgent,
                              expert_loader: ExpertTrajectoryLoader,
                              num_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate discriminator performance on expert vs learner data.
        
        Args:
            airl_agent: Trained AIRL agent
            expert_loader: Expert trajectory loader
            num_samples: Number of samples to test
            
        Returns:
            Discriminator evaluation metrics
        """
        self.logger.info("Evaluating discriminator performance")
        
        # Get expert samples
        expert_states, expert_actions = expert_loader.get_state_action_pairs(num_samples, self.device)
        
        # Generate learner samples
        learner_states = []
        learner_actions = []
        
        for _ in range(num_samples):
            obs = self.env.reset()
            # Handle gym API version differences
            if isinstance(obs, tuple):
                obs = obs[0]
            state = self._extract_features(obs)
            action = airl_agent.select_action(state, deterministic=False)
            
            learner_states.append(state)
            
            # Convert to one-hot
            action_onehot = np.zeros(self.env.action_space.n)
            action_onehot[action] = 1.0
            learner_actions.append(action_onehot)
        
        learner_states = torch.FloatTensor(learner_states).to(self.device)
        learner_actions = torch.FloatTensor(learner_actions).to(self.device)
        
        # Get discriminator predictions
        with torch.no_grad():
            expert_logits = airl_agent.discriminator(expert_states, expert_actions)
            learner_logits = airl_agent.discriminator(learner_states, learner_actions)
            
            expert_probs = torch.sigmoid(expert_logits)
            learner_probs = torch.sigmoid(learner_logits)
        
        # Calculate metrics
        expert_accuracy = (expert_probs > 0.5).float().mean().item()
        learner_accuracy = (learner_probs < 0.5).float().mean().item()
        overall_accuracy = (expert_accuracy + learner_accuracy) / 2
        
        metrics = {
            'expert_accuracy': expert_accuracy,
            'learner_accuracy': learner_accuracy,
            'overall_accuracy': overall_accuracy,
            'expert_mean_prob': expert_probs.mean().item(),
            'learner_mean_prob': learner_probs.mean().item(),
            'expert_std_prob': expert_probs.std().item(),
            'learner_std_prob': learner_probs.std().item()
        }
        
        self.logger.info("Discriminator Results:")
        self.logger.info(f"  Expert Accuracy: {metrics['expert_accuracy']:.3f}")
        self.logger.info(f"  Learner Accuracy: {metrics['learner_accuracy']:.3f}")
        self.logger.info(f"  Overall Accuracy: {metrics['overall_accuracy']:.3f}")
        
        return metrics
    
    def compare_agents(self, 
                      agents: Dict[str, any], 
                      num_episodes: int = 100) -> Dict[str, Dict]:
        """
        Compare multiple agents side by side.
        
        Args:
            agents: Dictionary mapping agent names to agent objects
            num_episodes: Number of episodes per agent
            
        Returns:
            Dictionary with results for each agent
        """
        self.logger.info(f"Comparing {len(agents)} agents")
        
        results = {}
        for name, agent in agents.items():
            results[name] = self.evaluate_agent(agent, num_episodes, name)
        
        # Create comparison table
        self.logger.info("\n" + "="*80)
        self.logger.info("AGENT COMPARISON")
        self.logger.info("="*80)
        
        metrics_to_compare = ['mean_score', 'mean_episode_length', 'mean_lines_cleared', 'hold_percentage']
        
        for metric in metrics_to_compare:
            self.logger.info(f"\n{metric.replace('_', ' ').title()}:")
            for name in agents.keys():
                value = results[name][metric]
                self.logger.info(f"  {name:15s}: {value:8.2f}")
        
        return results
    
    def create_visualizations(self, 
                            results: Dict[str, Dict], 
                            output_dir: str = "evaluation_plots"):
        """
        Create visualization plots for evaluation results.
        
        Args:
            results: Results dictionary from compare_agents
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Score comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AIRL Agent Evaluation Results', fontsize=16)
        
        # Score comparison
        agents = list(results.keys())
        scores = [results[agent]['mean_score'] for agent in agents]
        score_stds = [results[agent]['std_score'] for agent in agents]
        
        axes[0, 0].bar(agents, scores, yerr=score_stds, capsize=5)
        axes[0, 0].set_title('Average Score Comparison')
        axes[0, 0].set_ylabel('Average Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Episode length comparison
        lengths = [results[agent]['mean_episode_length'] for agent in agents]
        length_stds = [results[agent]['std_episode_length'] for agent in agents]
        
        axes[0, 1].bar(agents, lengths, yerr=length_stds, capsize=5)
        axes[0, 1].set_title('Average Episode Length')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Lines cleared comparison
        lines = [results[agent]['mean_lines_cleared'] for agent in agents]
        line_stds = [results[agent]['std_lines_cleared'] for agent in agents]
        
        axes[1, 0].bar(agents, lines, yerr=line_stds, capsize=5)
        axes[1, 0].set_title('Average Lines Cleared')
        axes[1, 0].set_ylabel('Lines')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # HOLD usage comparison
        hold_percentages = [results[agent]['hold_percentage'] for agent in agents]
        
        axes[1, 1].bar(agents, hold_percentages)
        axes[1, 1].set_title('HOLD Action Usage')
        axes[1, 1].set_ylabel('Percentage (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'agent_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Action distribution heatmap
        if len(agents) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare action distribution data
            action_data = []
            for agent in agents:
                action_dist = results[agent]['action_distribution']
                # Normalize to percentages
                total_actions = sum(action_dist.values())
                normalized_dist = {k: (v/total_actions)*100 for k, v in action_dist.items()}
                action_data.append([normalized_dist.get(i, 0) for i in range(41)])
            
            action_df = pd.DataFrame(action_data, index=agents, columns=[f'Action {i}' for i in range(41)])
            
            sns.heatmap(action_df, annot=False, cmap='viridis', ax=ax)
            ax.set_title('Action Distribution Heatmap (%)')
            ax.set_xlabel('Actions')
            ax.set_ylabel('Agents')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'action_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Visualizations saved to {output_dir}")
    
    def run_comprehensive_evaluation(self, 
                                   airl_checkpoint: str,
                                   expert_trajectory_dir: str,
                                   num_episodes: int = 100,
                                   output_dir: str = "evaluation_results") -> Dict:
        """
        Run comprehensive evaluation comparing AIRL agent with baselines.
        
        Args:
            airl_checkpoint: Path to AIRL model checkpoint
            expert_trajectory_dir: Path to expert trajectories
            num_episodes: Number of episodes per evaluation
            output_dir: Output directory for results
            
        Returns:
            Complete evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load AIRL agent
        airl_agent = self.load_airl_model(airl_checkpoint)
        
        # Load expert trajectory loader for discriminator evaluation
        expert_loader = ExpertTrajectoryLoader(
            trajectory_dir=expert_trajectory_dir,
            state_feature_extractor=self._extract_features
        )
        expert_loader.load_trajectories()
        
        # Agents to compare
        agents = {
            'AIRL_Agent': airl_agent,
            'Random_Agent': RandomAgent(self.env.action_space.n)
        }
        
        # Try to load expert agent
        expert_agent = self.load_expert_model()
        if expert_agent:
            agents['Expert_DQN'] = expert_agent
        
        # Run comparisons
        results = self.compare_agents(agents, num_episodes)
        
        # Evaluate discriminator
        if expert_loader.transitions:
            discriminator_results = self.evaluate_discriminator(airl_agent, expert_loader)
            results['discriminator'] = discriminator_results
        
        # Create visualizations
        self.create_visualizations(results, os.path.join(output_dir, 'plots'))
        
        # Save results to JSON
        results_file = os.path.join(output_dir, f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive evaluation completed. Results saved to {output_dir}")
        return results

class RandomAgent:
    """Random baseline agent for comparison."""
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        return np.random.randint(0, self.action_dim)

class DQNExpertAdapter:
    """Adapter for the expert DQN model."""
    
    def __init__(self, dqn_agent):
        self.dqn_agent = dqn_agent
        self.tetris_game = None
    
    def get_action(self, observation: Dict) -> int:
        # This would need to be implemented to bridge between
        # TetrisEnv observation and DQN action selection
        # For now, return random action
        return np.random.randint(0, 41)

def main():
    parser = argparse.ArgumentParser(description='Evaluate AIRL Tetris Agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to AIRL model checkpoint')
    parser.add_argument('--expert-dir', type=str, default='../../../expert_trajectories',
                       help='Directory containing expert trajectories')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes for evaluation')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create evaluator and run evaluation
    evaluator = AIRLEvaluator(device=device)
    results = evaluator.run_comprehensive_evaluation(
        airl_checkpoint=args.checkpoint,
        expert_trajectory_dir=args.expert_dir,
        num_episodes=args.episodes,
        output_dir=args.output_dir
    )
    
    print("\nEvaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    # Need pandas for heatmap
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available, some visualizations may fail")
        pd = None
    
    main() 