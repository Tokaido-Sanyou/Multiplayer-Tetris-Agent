"""
Unified Training Script: Orchestrates exploration and exploitation phases
Implements the 6-phase algorithm with batch processing and episode rollouts
"""
import os
import sys
import argparse
import logging
import torch
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Handle both direct execution and module import
try:
    from ..tetris_env import TetrisEnv
    from ..config import TetrisConfig  # Import centralized config
    from .rnd_exploration import RNDExplorationActor  # NEW: Use RND exploration
    from .state_model import StateModel
    from .actor_critic import ActorCriticAgent
    from .future_reward_predictor import FutureRewardPredictor
    from .replay_buffer import ReplayBuffer
except ImportError:
    # Direct execution - add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tetris_env import TetrisEnv
    from config import TetrisConfig  # Import centralized config
    from rl_utils.rnd_exploration import RNDExplorationActor  # NEW: Use RND exploration
    from rl_utils.state_model import StateModel
    from rl_utils.actor_critic import ActorCriticAgent
    from rl_utils.future_reward_predictor import FutureRewardPredictor
    from rl_utils.replay_buffer import ReplayBuffer

class UnifiedTrainer:
    """
    Unified trainer that implements the 6-phase exploration-exploitation algorithm
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Get centralized network config
        self.tetris_config = TetrisConfig()
        
        # Initialize environment - headless by default, enable rendering only when needed
        self.env = TetrisEnv(single_player=True, headless=True)
        
        # Initialize models with centralized dimensions
        self.state_model = StateModel(state_dim=self.tetris_config.STATE_DIM).to(self.device)
        self.future_reward_predictor = FutureRewardPredictor(
            state_dim=self.tetris_config.STATE_DIM, 
            action_dim=self.tetris_config.ACTION_DIM
        ).to(self.device)
        self.actor_critic = ActorCriticAgent(
            state_dim=self.tetris_config.STATE_DIM, 
            action_dim=self.tetris_config.ACTION_DIM, 
            clip_ratio=config.clip_ratio,
            state_model=self.state_model
        )
        
        # Initialize exploration actors with mode selection
        self.exploration_mode = getattr(config, 'exploration_mode', 'rnd')  # 'rnd', 'random', 'deterministic'
        
        if self.exploration_mode == 'rnd':
            from .rnd_exploration import RNDExplorationActor
            self.exploration_actor = RNDExplorationActor(self.env)
        elif self.exploration_mode == 'random':
            from .rnd_exploration import TrueRandomExplorer
            self.exploration_actor = TrueRandomExplorer(self.env)
        elif self.exploration_mode == 'deterministic':
            from .rnd_exploration import DeterministicTerminalExplorer
            self.exploration_actor = DeterministicTerminalExplorer(self.env)
        else:
            # Default to RND if unknown mode
            from .rnd_exploration import RNDExplorationActor
            self.exploration_actor = RNDExplorationActor(self.env)
            self.exploration_mode = 'rnd'
        
        print(f"🔧 Exploration mode: {self.exploration_mode.upper()}")
        
        # Initialize optimizers
        self.state_optimizer = torch.optim.Adam(self.state_model.parameters(), lr=config.state_lr)
        self.reward_optimizer = torch.optim.Adam(self.future_reward_predictor.parameters(), lr=config.reward_lr)
          # Add reward optimizer to actor-critic
        self.actor_critic.reward_optimizer = self.reward_optimizer
        
        # Training state
        self.phase = 1
        self.episode_count = 0
        self.batch_count = 0
        
        # Episode tracking
        self.episode_lines_cleared = []  # Track lines cleared per episode
        self.current_episode_lines = 0  # Lines cleared in current episode
        
        # NEW: Piece presence reward tracking
        self.total_episodes_completed = 0  # Track total episodes for piece presence decay
        
        # Data storage
        self.exploration_data = []
        self.experience_buffer = ReplayBuffer(config.buffer_size)
        
        # Logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Set TensorBoard writer for ActorCritic logging
        self.actor_critic.set_writer(self.writer)
        
        # NEW: Batch statistics tracking
        self.batch_stats = {
            'exploration': {},
            'state_model': {},
            'reward_predictor': {},
            'exploitation': {},
            'ppo': {},
            'evaluation': {},
            'rnd': {}
        }
        
    def run_training(self):
        """
        Main training loop implementing the 6-phase algorithm
        """
        print(f"🚀 Starting Unified Training: {self.config.num_batches} batches × {self.config.exploration_episodes + self.config.exploitation_episodes} episodes = {self.config.num_batches * (self.config.exploration_episodes + self.config.exploitation_episodes)} total episodes\n")
        
        for batch in range(self.config.num_batches):
            print(f"{'='*80}")
            print(f"🔄 BATCH {batch + 1}/{self.config.num_batches} - TRAINING CYCLE")
            print(f"{'='*80}")
            
            # Phase 1: Exploration data collection
            self.phase_1_exploration(batch)
            
            # Phase 2: State model learning
            self.phase_2_state_learning(batch)
            
            # Phase 3: Future reward prediction
            self.phase_3_reward_prediction(batch)
            
            # Phase 4: Enhanced Multi-Attempt Goal-Focused Policy Exploitation
            self.phase_4_exploitation(batch)
            
            # Phase 5: PPO-style training
            self.phase_5_ppo_training(batch)
            
            # Phase 6: Model evaluation
            self.phase_6_evaluation(batch)
            
            # Print comprehensive batch summary
            self.print_batch_summary(batch)
            
            # Save checkpoints
            if (batch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(batch)
        
        print(f"\n🎉 Training completed successfully! Total episodes: {self.total_episodes_completed}")
        self.writer.close()
        
    def print_batch_summary(self, batch):
        """
        Print a clean, comprehensive batch summary with GOAL-FOCUSED TRAINING emphasis
        """
        print(f"\n{'='*80}")
        print(f"📊 BATCH {batch+1} SUMMARY - GOAL-FOCUSED TRAINING")
        print(f"{'='*80}")
        
        # Training Progress
        progress = (batch + 1) / self.config.num_batches * 100
        print(f"📈 PROGRESS: {progress:.1f}% complete • Episode {self.total_episodes_completed}/{self.config.num_batches * self.config.exploitation_episodes} • ε={self.actor_critic.epsilon:.4f}")
        print(f"🎯 TRAINING MODE: PPO learns from GOAL ACHIEVEMENT, not game rewards")
        
        # Phase summaries in order with goal focus
        phases = [
            ('🔍 EXPLORATION', 'exploration', ['avg_terminal', 'success_rate', 'new_terminals_this_batch']),
            ('🎯 STATE MODEL', 'state_model', ['total_loss', 'loss_improvement']),
            ('🔮 REWARD PRED', 'reward_predictor', ['total_loss']),
            ('🎮 MULTI-ATTEMPT + HER EXPLOIT', 'exploitation', ['avg_reward', 'avg_goal_matches', 'step_goal_success_rate', 'episode_goal_success_rate']),  # Enhanced with HER
            ('🏋️ PPO TRAINING', 'ppo', ['actor_loss', 'critic_loss', 'future_state_loss']),
            ('📊 DUAL EVAL', 'evaluation', ['avg_goal_reward', 'avg_game_reward'])  # Both metrics
        ]
        
        summary_line = []
        for phase_name, phase_key, metrics in phases:
            if phase_key in self.batch_stats and self.batch_stats[phase_key]:
                stats = self.batch_stats[phase_key]
                metric_strs = []
                for metric in metrics:
                    if metric in stats:
                        value = stats[metric]
                        if metric.endswith('_rate') or metric.endswith('factor') or metric.endswith('consistency'):
                            metric_strs.append(f"{value*100:.0f}%")
                        elif metric.endswith('_loss'):
                            metric_strs.append(f"{value:.4f}")
                        elif metric == 'loss_improvement':
                            metric_strs.append(f"+{value*100:.0f}%")
                        elif metric == 'new_terminals_this_batch':
                            metric_strs.append(f"+{int(value)}")
                        elif metric == 'avg_goal_matches':
                            metric_strs.append(f"{value:.1f} goals/ep")
                        elif metric == 'avg_attempts_per_episode':
                            metric_strs.append(f"{value:.1f} attempts/ep")
                        elif metric == 'step_goal_success_rate':
                            metric_strs.append(f"{value*100:.0f}% step")
                        elif metric == 'episode_goal_success_rate':
                            metric_strs.append(f"{value*100:.0f}% episode")
                        else:
                            metric_strs.append(f"{value:.1f}")
                summary_line.append(f"{phase_name}: {' • '.join(metric_strs)}")
        
        for summary in summary_line:
            print(f"{summary}")
        
        # Enhanced goal-game alignment check with HER info
        if 'exploitation' in self.batch_stats and self.batch_stats['exploitation']:
            exploit_stats = self.batch_stats['exploitation']
            goal_aligned = exploit_stats.get('goal_game_aligned', False)
            multi_attempt_enabled = exploit_stats.get('multi_attempt_enabled', False)
            avg_attempts = exploit_stats.get('avg_attempts_per_episode', 0)
            step_success_rate = exploit_stats.get('step_goal_success_rate', 0)
            episode_success_rate = exploit_stats.get('episode_goal_success_rate', 0)
            hindsight_trajectories = exploit_stats.get('hindsight_trajectories', 0)
            
            alignment_status = "✅ ALIGNED" if goal_aligned else "⚠️ MISALIGNED"
            
            if multi_attempt_enabled:
                multi_attempt_status = f"🚀 MULTI-ATTEMPT ({avg_attempts:.1f}/ep)"
                her_status = f"🧠 HER ENABLED ({hindsight_trajectories} trajectories)" if hindsight_trajectories > 0 else "❌ HER NOT WORKING"
                success_status = f"📊 SUCCESS: {step_success_rate*100:.0f}% step, {episode_success_rate*100:.0f}% episode"
            else:
                multi_attempt_status = "❌ SINGLE-ATTEMPT"
                her_status = "❌ HER DISABLED"
                success_status = "📊 SUCCESS: LIMITED"
            
            print(f"🔗 GOAL-GAME ALIGNMENT: {alignment_status}")
            print(f"🎯 ACTOR ENHANCEMENT: {multi_attempt_status}")
            print(f"🧠 HINDSIGHT EXPERIENCE REPLAY: {her_status}")
            print(f"🏆 GOAL ACHIEVEMENT: {success_status}")
        
        if 'evaluation' in self.batch_stats and self.batch_stats['evaluation']:
            eval_stats = self.batch_stats['evaluation']
            goal_aligned = eval_stats.get('goal_game_aligned', False)
            if not goal_aligned:
                print(f"💡 SUGGESTION: Multi-attempt + HER mechanism should improve goal alignment in upcoming batches")
        
        print(f"{'='*80}\n")
    
    def phase_1_exploration(self, batch):
        """
        Phase 1: Collect exploration data using different exploration strategies
        Supports RND, random, and deterministic exploration modes
        ENHANCEMENT: Deterministic mode now uses sequential chain exploration
        """
        print(f"\n🔍 Phase 1: {self.exploration_mode.upper()} Exploration (Batch {batch+1})")
        
        # Collect placement data based on exploration mode
        if self.exploration_mode == 'rnd':
            placement_data = self.exploration_actor.collect_placement_data(
                num_episodes=self.config.exploration_episodes
            )
        elif self.exploration_mode == 'random':
            placement_data = self.exploration_actor.collect_random_placement_data(
                num_episodes=self.config.exploration_episodes
            )
        elif self.exploration_mode == 'deterministic':
            # NEW: Sequential chain exploration generates variable number of states
            sequence_length = 10  # 10 pieces in the sequential chain
            placement_data = self.exploration_actor.generate_all_terminal_states(
                sequence_length=sequence_length
            )
            print(f"   🔗 Sequential exploration generated {len(placement_data)} terminal states")
        else:
            # Fallback to RND
            placement_data = self.exploration_actor.collect_placement_data(
                num_episodes=self.config.exploration_episodes
            )
        
        self.exploration_data.extend(placement_data)
        
        # Log detailed exploration statistics
        if placement_data:
            rewards = [d['terminal_reward'] for d in placement_data]
            
            # Basic statistics
            self.writer.add_scalar('Exploration/AvgTerminalReward', np.mean(rewards), batch)
            self.writer.add_scalar('Exploration/StdTerminalReward', np.std(rewards), batch)
            self.writer.add_scalar('Exploration/MaxTerminalReward', np.max(rewards), batch)
            self.writer.add_scalar('Exploration/MinTerminalReward', np.min(rewards), batch)
            self.writer.add_scalar('Exploration/NumPlacements', len(placement_data), batch)
            
            # Mode-specific statistics
            if self.exploration_mode == 'rnd':
                # RND-specific statistics
                intrinsic_rewards = [d.get('intrinsic_reward', 0) for d in placement_data]
                prediction_errors = [d.get('prediction_error', 0) for d in placement_data]
                
                if intrinsic_rewards and any(r != 0 for r in intrinsic_rewards):
                    self.writer.add_scalar('Exploration/AvgIntrinsicReward', np.mean(intrinsic_rewards), batch)
                    self.writer.add_scalar('Exploration/StdIntrinsicReward', np.std(intrinsic_rewards), batch)
                    self.writer.add_scalar('Exploration/AvgPredictionError', np.mean(prediction_errors), batch)
                    self.writer.add_scalar('Exploration/StdPredictionError', np.std(prediction_errors), batch)
                
                # Get novelty statistics
                novelty_stats = self.exploration_actor._get_novelty_stats() if hasattr(self.exploration_actor, '_get_novelty_stats') else {}
                
                print(f"📊 Phase 1 Results:")
                print(f"   • Terminal rewards: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
                print(f"   • Intrinsic motivation: {np.mean(intrinsic_rewards):.3f} ± {np.std(intrinsic_rewards):.3f}")
                print(f"   • Novel states discovered: {novelty_stats.get('unique_terminals', 0)}")
                print(f"   • Distinct terminal states this batch: {novelty_stats.get('unique_terminals', 0) - novelty_stats.get('prev_unique_terminals', 0)}")
            
            elif self.exploration_mode == 'deterministic':
                # Enhanced deterministic exploration statistics for sequential chains
                chain_depths = [d.get('chain_depth', 0) for d in placement_data]
                is_chain_terminals = [d.get('is_chain_terminal', False) for d in placement_data]
                sequence_positions = [d.get('sequence_position', 0) for d in placement_data]
                placement_histories = [len(d.get('placement_history', [])) for d in placement_data]
                
                # Chain-specific metrics
                max_chain_depth = max(chain_depths) if chain_depths else 0
                num_chain_terminals = sum(is_chain_terminals)
                num_intermediate_states = len(placement_data) - num_chain_terminals
                avg_placement_history = np.mean(placement_histories) if placement_histories else 0
                
                # Log sequential chain metrics
                self.writer.add_scalar('Exploration/MaxChainDepth', max_chain_depth, batch)
                self.writer.add_scalar('Exploration/NumChainTerminals', num_chain_terminals, batch)
                self.writer.add_scalar('Exploration/NumIntermediateStates', num_intermediate_states, batch)
                self.writer.add_scalar('Exploration/AvgPlacementHistory', avg_placement_history, batch)
                
                # Chain depth distribution
                depth_distribution = {}
                for depth in chain_depths:
                    depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
                
                print(f"📊 Phase 1 Results (Sequential Chain Exploration):")
                print(f"   • Terminal rewards: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
                print(f"   • Total terminal states: {len(placement_data)} (Variable batch size)")
                print(f"   • Max chain depth reached: {max_chain_depth}")
                print(f"   • Chain terminals: {num_chain_terminals}")
                print(f"   • Intermediate states: {num_intermediate_states}")
                print(f"   • States by depth: {dict(sorted(depth_distribution.items()))}")
                print(f"   • Value range: {np.min(rewards):.1f} to {np.max(rewards):.1f}")
                
                # Piece type distribution in chains
                piece_types = [d.get('target_piece_type', 0) for d in placement_data]
                piece_distribution = {}
                for piece_type in piece_types:
                    piece_distribution[piece_type] = piece_distribution.get(piece_type, 0) + 1
                print(f"   • Piece distribution: {dict(sorted(piece_distribution.items()))}")
            
            else:  # random exploration
                print(f"📊 Phase 1 Results:")
                print(f"   • Terminal rewards: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
                print(f"   • Random exploration coverage: {len(placement_data)} states")
                print(f"   • Value distribution: uniform random")
            
            # Common success metrics
            high_reward_threshold = -50  # Adjust based on reward scale
            successful_placements = [r for r in rewards if r > high_reward_threshold]
            success_rate = len(successful_placements) / len(rewards)
            
            self.writer.add_scalar('Exploration/SuccessfulPlacementRate', success_rate, batch)
            
            # Store batch statistics for end-of-batch summary
            exploration_stats = {
                'avg_terminal': np.mean(rewards),
                'std_terminal': np.std(rewards),
                'num_placements': len(placement_data),
                'success_rate': success_rate,
                'exploration_mode': self.exploration_mode
            }
            
            # Add mode-specific stats
            if self.exploration_mode == 'deterministic':
                exploration_stats.update({
                    'max_chain_depth': max_chain_depth,
                    'num_chain_terminals': num_chain_terminals,
                    'num_intermediate_states': num_intermediate_states,
                    'is_variable_batch': True
                })
            
            self.update_batch_stats('exploration', exploration_stats)
    
    def phase_2_state_learning(self, batch):
        """
        Phase 2: Train state model to learn optimal placements from terminal rewards
        """
        print(f"\n🎯 Phase 2: State Model Training (Batch {batch+1})")
        
        if not self.exploration_data:
            print("⚠️  No exploration data available for state learning")
            return
            
        # Train state model from exploration data
        loss_info = self.state_model.train_from_placements(
            self.exploration_data[-self.config.state_training_samples:],
            self.state_optimizer,
            num_epochs=self.config.state_epochs
        )
        
        if loss_info:
            # Log detailed state model losses
            self.writer.add_scalar('StateModel/TotalLoss', loss_info['total_loss'], batch)
            self.writer.add_scalar('StateModel/RotationLoss', loss_info['rotation_loss'], batch)
            self.writer.add_scalar('StateModel/XPositionLoss', loss_info['x_position_loss'], batch)
            self.writer.add_scalar('StateModel/YPositionLoss', loss_info['y_position_loss'], batch)
            self.writer.add_scalar('StateModel/ValueLoss', loss_info['value_loss'], batch)
            
            # Log training progression (average over all epochs)
            all_total = loss_info['all_total_losses']
            all_rot = loss_info['all_rotation_losses']
            all_x = loss_info['all_x_position_losses']
            all_y = loss_info['all_y_position_losses']
            all_val = loss_info['all_value_losses']
            
            # IMMEDIATE STATISTICS REPORTING
            improvement = 0.0
            if len(all_total) > 1:
                total_improvement = (all_total[0] - all_total[-1]) / all_total[0] if all_total[0] > 0 else 0
                improvement = total_improvement
                self.writer.add_scalar('StateModel/TotalLossImprovement', total_improvement, batch)
                self.writer.add_scalar('StateModel/FinalEpochLoss', all_total[-1], batch)
            
            print(f"📊 Phase 2 Results:")
            print(f"   • Total loss: {loss_info['total_loss']:.4f} (improvement: {improvement*100:.1f}%)")
            print(f"   • Rotation loss: {loss_info['rotation_loss']:.4f}")
            print(f"   • X position loss: {loss_info['x_position_loss']:.4f}")
            print(f"   • Y position loss: {loss_info['y_position_loss']:.4f}")
            print(f"   • Value loss: {loss_info['value_loss']:.4f}")
            
            # Store batch statistics
            self.update_batch_stats('state_model', {
                'total_loss': loss_info['total_loss'],
                'rotation_loss': loss_info['rotation_loss'],
                'x_position_loss': loss_info['x_position_loss'],
                'y_position_loss': loss_info['y_position_loss'],
                'value_loss': loss_info['value_loss'],
                'loss_improvement': improvement
            })
    
    def phase_3_reward_prediction(self, batch):
        """
        Phase 3: Train future reward predictor on terminal placement data
        """
        print(f"\n🔮 Phase 3: Future Reward Prediction (Batch {batch+1})")
        
        if not self.exploration_data:
            print("⚠️  No exploration data available for reward prediction training")
            return
            
        # Train future reward predictor specifically on terminal placements
        loss_info = self.future_reward_predictor.train_on_terminal_placements(
            self.exploration_data[-self.config.state_training_samples:],
            self.reward_optimizer,
            num_epochs=self.config.state_epochs
        )
        
        if loss_info:
            # Log detailed reward predictor losses
            self.writer.add_scalar('RewardPredictor/TotalLoss', loss_info['total_loss'], batch)
            self.writer.add_scalar('RewardPredictor/RewardPredictionLoss', loss_info['reward_loss'], batch)
            self.writer.add_scalar('RewardPredictor/ValuePredictionLoss', loss_info['value_loss'], batch)
            
            # IMMEDIATE STATISTICS REPORTING
            print(f"📊 Phase 3 Results:")
            print(f"   • Total loss: {loss_info['total_loss']:.4f}")
            print(f"   • Reward prediction: {loss_info['reward_loss']:.4f}")
            print(f"   • Value prediction: {loss_info['value_loss']:.4f}")
            
            # Store batch statistics
            self.update_batch_stats('reward_predictor', {
                'total_loss': loss_info['total_loss'],
                'reward_loss': loss_info['reward_loss'],
                'value_loss': loss_info['value_loss']
            })
    
    def phase_4_exploitation(self, batch):
        """
        Phase 4: Enhanced Multi-Attempt Goal-Focused Policy Exploitation
        ENHANCEMENT: Multiple attempts per placement with hindsight trajectory relabeling
        Actor learns from highest reward terminal states with trajectory replay
        """
        print(f"\n🎮 Phase 4: Multi-Attempt Goal-Focused Policy Exploitation (Batch {batch+1})")
        
        total_reward = 0
        total_goal_reward = 0
        total_steps = 0
        episode_rewards = []
        episode_goal_rewards = []
        episode_steps = []
        batch_lines_cleared = []
        goal_achievement_metrics = []
        
        # NEW: Multi-attempt tracking
        total_attempts = 0
        successful_goal_matches = 0
        hindsight_trajectories = []  # Store trajectories for hindsight relabeling
        total_hindsight_experiences = 0  # Track individual hindsight experiences
        
        for episode in range(self.config.exploitation_episodes):
            # Enable visualization only for the last episode of each batch if visualize flag is set
            visualize_this_episode = (self.config.visualize and 
                                    episode == self.config.exploitation_episodes - 1)
            
            # Recreate environment with appropriate headless setting for this episode
            if visualize_this_episode:
                self.env.close()
                self.env = TetrisEnv(single_player=True, headless=False)
                logging.info(f"Enabling visualization for last exploitation episode of batch {batch}")
            elif episode == 0 and not visualize_this_episode:
                # Ensure environment is headless for non-visualized episodes
                if hasattr(self.env, 'headless') and not self.env.headless:
                    self.env.close()
                    self.env = TetrisEnv(single_player=True, headless=True)
            
            obs = self.env.reset()
            episode_reward = 0
            episode_goal_reward = 0
            steps = 0
            done = False
            episode_lines = 0
            episode_goal_matches = 0
            
            # NEW: Episode trajectory for hindsight relabeling
            episode_trajectory = []
            episode_attempts = []
            all_attempt_experiences = []  # Store ALL individual attempt experiences
            
            while not done and steps < self.config.max_episode_steps:
                # Convert observation to state vector
                state = self._obs_to_state_vector(obs)
                
                # ENHANCED MULTI-ATTEMPT MECHANISM: Sample multiple actions and choose best
                attempts_per_placement = 3  # Generate 3 different action candidates
                attempt_actions = []
                attempt_rewards = []
                
                for attempt in range(attempts_per_placement):
                    total_attempts += 1
                    
                    # Generate action with varying exploration levels
                    if attempt == 0:
                        # First attempt: use current policy (exploitation)
                        action_candidate = self.actor_critic.select_action(state)
                    elif attempt == 1:
                        # Second attempt: moderate exploration
                        old_epsilon = self.actor_critic.epsilon
                        self.actor_critic.epsilon = min(0.2, old_epsilon * 1.5)
                        action_candidate = self.actor_critic.select_action(state)
                        self.actor_critic.epsilon = old_epsilon
                    else:
                        # Third attempt: high exploration for novelty
                        old_epsilon = self.actor_critic.epsilon
                        self.actor_critic.epsilon = min(0.4, old_epsilon * 3.0)
                        action_candidate = self.actor_critic.select_action(state)
                        self.actor_critic.epsilon = old_epsilon
                    
                    # Estimate goal achievement potential for this action
                    # Use state model to predict value of this action choice
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        goal_vector = self.state_model.get_placement_goal_vector(state_tensor)
                        
                        if goal_vector is not None:
                            # Extract goal components
                            goal_rotation = torch.argmax(goal_vector[0, :4]).item()
                            goal_x_pos = torch.argmax(goal_vector[0, 4:14]).item()
                            goal_y_pos = torch.argmax(goal_vector[0, 14:34]).item()
                            goal_confidence = goal_vector[0, 35].item()
                            
                            # Convert action to placement (approximate)
                            action_idx = np.argmax(action_candidate)
                            # Map action to placement parameters (simplified)
                            if action_idx < 4:  # Rotation actions
                                action_rotation = action_idx
                                action_x = 5  # Default center
                            elif action_idx < 7:  # Movement actions  
                                action_rotation = 0  # Default rotation
                                action_x = (action_idx - 4) * 3 + 2  # Spread across board
                            else:
                                action_rotation = 0
                                action_x = 5
                            
                            # Calculate predicted goal achievement
                            rotation_similarity = 1.0 - abs(action_rotation - goal_rotation) / 4.0
                            x_similarity = 1.0 - abs(action_x - goal_x_pos) / 10.0
                            
                            predicted_goal_reward = (rotation_similarity + x_similarity) * goal_confidence * 15.0
                        else:
                            predicted_goal_reward = 1.0  # Default if no goal available
                    
                    attempt_actions.append(action_candidate.copy())
                    attempt_rewards.append(predicted_goal_reward)
                    
                    # STORE EACH ATTEMPT FOR INDIVIDUAL HER: Create experience for every attempt
                    attempt_experience = {
                        'step_index': steps,
                        'attempt_index': attempt,
                        'state': state.copy(),
                        'action': action_candidate.copy(),
                        'predicted_reward': predicted_goal_reward,
                        'obs': obs.copy(),
                        'was_selected': False  # Will be updated for the selected attempt
                    }
                    all_attempt_experiences.append(attempt_experience)
                
                # SELECT BEST ACTION: Choose action with highest predicted goal achievement
                best_attempt_idx = np.argmax(attempt_rewards)
                action = attempt_actions[best_attempt_idx]
                predicted_best_reward = attempt_rewards[best_attempt_idx]
                
                # Mark the selected attempt
                selected_attempt_global_idx = len(all_attempt_experiences) - attempts_per_placement + best_attempt_idx
                all_attempt_experiences[selected_attempt_global_idx]['was_selected'] = True
                
                # Store all attempts for hindsight learning
                episode_attempts.extend([{
                    'action': act,
                    'predicted_reward': pred_rew,
                    'was_selected': i == best_attempt_idx
                } for i, (act, pred_rew) in enumerate(zip(attempt_actions, attempt_rewards))])
                
                # Execute the selected best action
                next_obs, game_reward, done, info = self.env.step(action)
                next_state = self._obs_to_state_vector(next_obs)
                
                # Calculate actual goal achievement reward
                goal_achievement_reward = self.calculate_goal_achievement_reward(state, action, next_state, info)
                
                # Track goal achievement metrics with adjusted threshold
                if goal_achievement_reward > 10.0:  # Step-level goal achievement threshold
                    episode_goal_matches += 1
                    successful_goal_matches += 1
                
                # Update the selected attempt with actual results
                all_attempt_experiences[selected_attempt_global_idx].update({
                    'next_state': next_state.copy(),
                    'next_obs': next_obs.copy(),
                    'actual_goal_reward': goal_achievement_reward,
                    'game_reward': game_reward,
                    'done': done,
                    'info': info.copy() if info else {}
                })
                
                # ENHANCED TRAJECTORY COLLECTION: Store rich information for hindsight learning
                trajectory_step = {
                    'state': state.copy(),
                    'action': action.copy(),
                    'goal_reward': goal_achievement_reward,
                    'game_reward': game_reward,
                    'next_state': next_state.copy(),
                    'done': done,
                    'info': info.copy() if info else {},
                    'obs': obs.copy(),  # Store original observation dict for replay buffer
                    'next_obs': next_obs.copy(),  # Store next observation dict for replay buffer
                    'all_attempts': [{'action': act, 'predicted_reward': pred} for act, pred in zip(attempt_actions, attempt_rewards)],
                    'predicted_best_reward': predicted_best_reward,
                    'actual_reward': goal_achievement_reward,
                    'prediction_accuracy': abs(predicted_best_reward - goal_achievement_reward)
                }
                episode_trajectory.append(trajectory_step)
                
                # Store experience with ACTUAL goal achievement reward
                self.experience_buffer.push(
                    obs, action, goal_achievement_reward, next_obs, done, info
                )
                
                # Keep track of lines cleared for analysis
                if info and 'lines_cleared' in info:
                    episode_lines += info['lines_cleared']
                
                obs = next_obs
                episode_reward += game_reward
                episode_goal_reward += goal_achievement_reward
                steps += 1
                
                if visualize_this_episode:
                    self.env.render()
                    import time
                    time.sleep(0.1)
            
            # ENHANCED ALL-ATTEMPT HINDSIGHT RELABELING: Create hindsight for ALL attempts
            if len(all_attempt_experiences) > 0:
                # Create hindsight experiences for ALL attempts, not just selected ones
                attempt_hindsight_experiences = self._create_all_attempt_hindsight_experiences(
                    all_attempt_experiences, episode_trajectory
                )
                total_hindsight_experiences += len(attempt_hindsight_experiences)
                
                # Add all hindsight experiences to buffer
                for hindsight_exp in attempt_hindsight_experiences:
                    self.experience_buffer.push(
                        hindsight_exp['obs'], 
                        hindsight_exp['action'], 
                        hindsight_exp['hindsight_reward'],
                        hindsight_exp['next_obs'], 
                        hindsight_exp['done'], 
                        hindsight_exp['info']
                    )
            
            # ENHANCED HINDSIGHT TRAJECTORY RELABELING: Learn from highest reward trajectories with randomized future goals
            if len(episode_trajectory) > 0:
                # Create hindsight experiences using randomized future state goals
                hindsight_trajectory = self._create_hindsight_trajectory_with_future_goals(episode_trajectory)
                if hindsight_trajectory:
                    hindsight_trajectories.append(hindsight_trajectory)
                    total_hindsight_experiences += len(hindsight_trajectory)
                    
                    # Add hindsight experiences to buffer
                    for hindsight_step in hindsight_trajectory:
                        self.experience_buffer.push(
                            hindsight_step['obs'], 
                            hindsight_step['action'], 
                            hindsight_step['hindsight_reward'],  # Enhanced reward
                            hindsight_step['next_obs'], 
                            hindsight_step['done'], 
                            hindsight_step['info']
                        )
            
            # After visualization episode, revert to headless
            if visualize_this_episode:
                self.env.close()
                self.env = TetrisEnv(single_player=True, headless=True)
                logging.info(f"Visualization complete, reverting to headless mode")
            
            # Record episode statistics
            episode_rewards.append(episode_reward)
            episode_goal_rewards.append(episode_goal_reward)
            episode_steps.append(steps)
            batch_lines_cleared.append(episode_lines)
            self.episode_lines_cleared.append(episode_lines)
            goal_achievement_metrics.append(episode_goal_matches)
            
            total_reward += episode_reward
            total_goal_reward += episode_goal_reward
            total_steps += steps
            self.episode_count += 1
            
            # Update total episodes completed
            self.total_episodes_completed += 1
            
            # Log individual episode statistics with MULTI-ATTEMPT FOCUS
            self.writer.add_scalar('Exploitation/EpisodeGameReward', episode_reward, self.episode_count)
            self.writer.add_scalar('Exploitation/EpisodeGoalReward', episode_goal_reward, self.episode_count)
            self.writer.add_scalar('Exploitation/EpisodeSteps', steps, self.episode_count)
            self.writer.add_scalar('Exploitation/EpisodeLinesCleared', episode_lines, self.episode_count)
            self.writer.add_scalar('Exploitation/EpisodeGoalMatches', episode_goal_matches, self.episode_count)
            self.writer.add_scalar('Exploitation/EpisodeAttempts', len(episode_attempts), self.episode_count)
            self.writer.add_scalar('Exploitation/GoalRewardRatio', episode_goal_reward / max(1, steps), self.episode_count)
        
        # Calculate enhanced statistics
        avg_game_reward = total_reward / self.config.exploitation_episodes
        avg_goal_reward = total_goal_reward / self.config.exploitation_episodes
        avg_steps = total_steps / self.config.exploitation_episodes
        avg_lines_cleared = np.mean(batch_lines_cleared) if batch_lines_cleared else 0
        avg_goal_matches = np.mean(goal_achievement_metrics) if goal_achievement_metrics else 0
        
        # NEW: Multi-attempt statistics with clearer labeling
        avg_attempts_per_episode = total_attempts / self.config.exploitation_episodes
        step_goal_success_rate = successful_goal_matches / max(1, total_attempts)  # Step-level success
        episode_goal_success_rate = sum(1 for matches in goal_achievement_metrics if matches > 0) / self.config.exploitation_episodes  # Episode-level success
        hindsight_trajectories_created = len(hindsight_trajectories)
        
        # Log enhanced metrics with clear labels
        self.writer.add_scalar('Exploitation/BatchAvgGameReward', avg_game_reward, batch)
        self.writer.add_scalar('Exploitation/BatchAvgSteps', avg_steps, batch)
        self.writer.add_scalar('Exploitation/BatchAvgLinesCleared', avg_lines_cleared, batch)
        self.writer.add_scalar('Exploitation/BatchAvgGoalReward', avg_goal_reward, batch)
        self.writer.add_scalar('Exploitation/BatchStdGoalReward', np.std(episode_goal_rewards), batch)
        self.writer.add_scalar('Exploitation/BatchMaxGoalReward', np.max(episode_goal_rewards), batch)
        self.writer.add_scalar('Exploitation/BatchMinGoalReward', np.min(episode_goal_rewards), batch)
        self.writer.add_scalar('Exploitation/BatchAvgGoalMatches', avg_goal_matches, batch)
        
        # Enhanced multi-attempt metrics with clear distinctions
        self.writer.add_scalar('Exploitation/AvgAttemptsPerEpisode', avg_attempts_per_episode, batch)
        self.writer.add_scalar('Exploitation/StepGoalSuccessRate', step_goal_success_rate, batch)  # Step-level
        self.writer.add_scalar('Exploitation/EpisodeGoalSuccessRate', episode_goal_success_rate, batch)  # Episode-level
        self.writer.add_scalar('Exploitation/HindsightTrajectoriesCreated', hindsight_trajectories_created, batch)
        self.writer.add_scalar('Exploitation/TotalHindsightExperiences', total_hindsight_experiences, batch)  # NEW: Track all hindsight experiences
        
        # Goal achievement consistency (episode-level)
        episode_goal_consistency = 1.0 - (np.std(episode_goal_rewards) / max(1e-6, np.mean(episode_goal_rewards)))
        self.writer.add_scalar('Exploitation/EpisodeGoalConsistency', episode_goal_consistency, batch)
        
        # ENHANCED STATISTICS REPORTING with clearer labels
        print(f"📊 Phase 4 Results (Multi-Attempt Enhancement):")
        print(f"   🎯 Episode goal rewards: {avg_goal_reward:.2f} ± {np.std(episode_goal_rewards):.2f} (TRAINING SIGNAL)")
        print(f"   🎮 Game performance rewards: {avg_game_reward:.2f} ± {np.std(episode_rewards):.2f} (analysis only)")
        print(f"   📏 Episode steps: {avg_steps:.1f} ± {np.std(episode_steps):.1f}")
        print(f"   📐 Lines cleared: {avg_lines_cleared:.1f} ± {np.std(batch_lines_cleared):.1f} (analysis only)")
        print(f"   🎯 Goal matches per episode: {avg_goal_matches:.1f}")
        print(f"   🔄 Attempts per episode: {avg_attempts_per_episode:.1f}")
        print(f"   ✅ Step-level goal success: {step_goal_success_rate:.3f} ({step_goal_success_rate*100:.1f}%)")
        print(f"   🏆 Episode-level goal success: {episode_goal_success_rate:.3f} ({episode_goal_success_rate*100:.1f}%)")
        print(f"   🧠 Hindsight trajectories: {hindsight_trajectories_created}")
        print(f"   🎯 Total hindsight experiences: {total_hindsight_experiences}")
        print(f"   📈 Episode consistency: {episode_goal_consistency:.3f}")
        print(f"   🚀 Multi-attempt actor with randomized future goal hindsight: ENABLED")
        
        # Store enhanced batch statistics with clearer metrics
        self.update_batch_stats('exploitation', {
            'avg_reward': avg_goal_reward,
            'avg_game_reward': avg_game_reward,
            'std_reward': np.std(episode_goal_rewards),
            'avg_steps': avg_steps,
            'std_steps': np.std(episode_steps),
            'avg_lines': avg_lines_cleared,
            'std_lines': np.std(batch_lines_cleared),
            'avg_goal_matches': avg_goal_matches,
            'goal_consistency': episode_goal_consistency,
            'goal_focused_training': True,
            'multi_attempt_enabled': True,
            'avg_attempts_per_episode': avg_attempts_per_episode,
            'step_goal_success_rate': step_goal_success_rate,  # Clear distinction
            'episode_goal_success_rate': episode_goal_success_rate,  # Clear distinction
            'hindsight_trajectories': hindsight_trajectories_created,
            'total_hindsight_experiences': total_hindsight_experiences  # NEW: Track total experiences
        })
    
    def _create_hindsight_trajectory_with_future_goals(self, episode_trajectory):
        """
        Create hindsight trajectory using randomized future states as goals
        ENHANCEMENT: Use HER (Hindsight Experience Replay) with future state goal relabeling
        
        Args:
            episode_trajectory: Complete episode trajectory
        Returns:
            List of hindsight experience steps with future goal relabeling
        """
        hindsight_trajectory = []
        
        if len(episode_trajectory) < 2:
            return hindsight_trajectory
        
        # Select random future states as goals for hindsight relabeling
        for i, current_step in enumerate(episode_trajectory[:-1]):  # Exclude last step
            # Randomly select a future state as the "achieved goal"
            future_indices = list(range(i + 1, len(episode_trajectory)))
            if not future_indices:
                continue
                
            # Use strategy: select from top 50% of remaining future states by reward
            future_steps = [episode_trajectory[idx] for idx in future_indices]
            future_rewards = [step['goal_reward'] for step in future_steps]
            
            # Select from top half of future rewards for better hindsight learning
            sorted_future_indices = sorted(future_indices, 
                                         key=lambda idx: episode_trajectory[idx]['goal_reward'], 
                                         reverse=True)
            top_half_count = max(1, len(sorted_future_indices) // 2)
            top_future_indices = sorted_future_indices[:top_half_count]
            
            # Randomly select from top performers
            goal_step_idx = np.random.choice(top_future_indices)
            goal_step = episode_trajectory[goal_step_idx]
            goal_state = goal_step['next_state']
            
            # Calculate hindsight reward: how well current action led toward this future goal
            current_state = current_step['state']
            current_action = current_step['action']
            achieved_state = current_step['next_state']
            
            # Calculate goal achievement using the future state as target
            hindsight_goal_reward = self._calculate_hindsight_goal_achievement(
                current_state, current_action, achieved_state, goal_state
            )
            
            # Enhanced hindsight reward combines multiple signals
            temporal_discount = 0.95 ** (goal_step_idx - i)  # Discount factor for temporal distance
            future_quality = (goal_step['goal_reward'] + 50) / 100.0  # Normalize future reward quality
            
            # Final hindsight reward
            final_hindsight_reward = (
                hindsight_goal_reward * 0.7 +          # Primary: goal achievement toward future state
                current_step['goal_reward'] * 0.2 +    # Secondary: original reward
                temporal_discount * future_quality * 5.0  # Bonus: temporal and quality weighting
            )
            
            # Create hindsight experience
            hindsight_step = {
                'obs': current_step['obs'],  # Use stored observation dict, not state vector
                'action': current_action,
                'hindsight_reward': min(60.0, max(-10.0, final_hindsight_reward)),  # Clamp to reasonable range
                'next_obs': current_step['next_obs'],  # Use stored next observation dict, not state vector
                'done': current_step['done'],
                'info': current_step['info'],
                'original_reward': current_step['goal_reward'],
                'goal_state': goal_state,  # Future state used as goal
                'temporal_distance': goal_step_idx - i,
                'future_reward_quality': goal_step['goal_reward'],
                'hindsight_type': 'randomized_future_goal'
            }
            
            hindsight_trajectory.append(hindsight_step)
        
        return hindsight_trajectory
    
    def _calculate_hindsight_goal_achievement(self, current_state, action, achieved_state, goal_state):
        """
        Calculate how well an action achieved progress toward a future goal state
        
        Args:
            current_state: Starting state
            action: Action taken
            achieved_state: State reached after action
            goal_state: Future state used as goal
        Returns:
            hindsight_goal_reward: Reward for progress toward goal
        """
        try:
            # Calculate state similarity between achieved state and goal state
            achieved_tensor = torch.FloatTensor(achieved_state).unsqueeze(0).to(self.device)
            goal_tensor = torch.FloatTensor(goal_state).unsqueeze(0).to(self.device)
            
            # Cosine similarity for overall state alignment
            overall_similarity = F.cosine_similarity(achieved_tensor, goal_tensor, dim=1).item()
            overall_similarity = max(0.0, overall_similarity)
            
            # Specific component similarities for better granularity
            # Current piece grid similarity (indices 0-199)
            piece_similarity = F.cosine_similarity(
                achieved_tensor[:, :200], goal_tensor[:, :200], dim=1
            ).item() if achieved_tensor.shape[1] >= 200 else 0.0
            
            # Empty grid similarity (indices 200-399)  
            empty_similarity = F.cosine_similarity(
                achieved_tensor[:, 200:400], goal_tensor[:, 200:400], dim=1
            ).item() if achieved_tensor.shape[1] >= 400 else 0.0
            
            # Metadata similarity (indices 407-409)
            if achieved_tensor.shape[1] >= 410:
                metadata_achieved = achieved_tensor[:, 407:410]
                metadata_goal = goal_tensor[:, 407:410]
                
                # L2 distance for metadata (rotation, x_pos, y_pos)
                metadata_distance = F.mse_loss(metadata_achieved, metadata_goal, reduction='mean').item()
                metadata_similarity = max(0.0, 1.0 - metadata_distance)  # Convert distance to similarity
            else:
                metadata_similarity = 0.0
            
            # Combine similarities with weighting
            hindsight_goal_reward = (
                overall_similarity * 10.0 +      # Overall state alignment
                piece_similarity * 5.0 +         # Piece configuration similarity
                empty_similarity * 3.0 +         # Board state similarity
                metadata_similarity * 15.0       # Position/rotation similarity
            )
            
            return hindsight_goal_reward
            
        except Exception as e:
            print(f"Error in hindsight goal calculation: {e}")
            return 1.0  # Fallback reward

    def phase_5_ppo_training(self, batch):
        """
        Phase 5: Enhanced PPO training with Hindsight Experience Replay
        """
        print(f"\n🏋️ Phase 5: Enhanced PPO with Hindsight Relabelling (Batch {batch+1})")
        
        if len(self.experience_buffer) < self.config.min_buffer_size:
            print("⚠️  Insufficient experience for PPO training")
            return
            
        # Multiple PPO training iterations with hindsight relabelling
        total_actor_loss = 0
        total_critic_loss = 0
        total_reward_loss = 0
        total_future_state_loss = 0
        total_aux_loss = 0
        successful_iterations = 0
        
        for iteration in range(self.config.ppo_iterations):
            # Use enhanced PPO with hindsight relabelling
            losses = self.actor_critic.train_ppo_with_hindsight(
                batch_size=self.config.ppo_batch_size,
                ppo_epochs=self.config.ppo_epochs
            )
            
            if losses:
                actor_loss, critic_loss, reward_loss, aux_loss, future_state_loss = losses
                    
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_reward_loss += reward_loss
                total_future_state_loss += future_state_loss
                total_aux_loss += aux_loss
                successful_iterations += 1
                
                # Log per-iteration losses
                global_step = batch * self.config.ppo_iterations + iteration
                self.writer.add_scalar('PPO/ActorLoss_Iteration', actor_loss, global_step)
                self.writer.add_scalar('PPO/CriticLoss_Iteration', critic_loss, global_step)
                self.writer.add_scalar('PPO/RewardLoss_Iteration', reward_loss, global_step)
                self.writer.add_scalar('PPO/AuxiliaryLoss_Iteration', aux_loss, global_step)
                self.writer.add_scalar('PPO/FutureStateLoss_Iteration', future_state_loss, global_step)
                self.writer.add_scalar('PPO/HindsightSuccess', 1.0, global_step)  # Track hindsight success
        
        # Log average losses for this batch
        if successful_iterations > 0:
            avg_actor_loss = total_actor_loss / successful_iterations
            avg_critic_loss = total_critic_loss / successful_iterations
            avg_reward_loss = total_reward_loss / successful_iterations
            avg_future_state_loss = total_future_state_loss / successful_iterations
            avg_aux_loss = total_aux_loss / successful_iterations
            
            self.writer.add_scalar('PPO/ActorLoss', avg_actor_loss, batch)
            self.writer.add_scalar('PPO/CriticLoss', avg_critic_loss, batch)
            self.writer.add_scalar('PPO/RewardLoss', avg_reward_loss, batch)
            self.writer.add_scalar('PPO/FutureStateLoss', avg_future_state_loss, batch)
            self.writer.add_scalar('PPO/AuxiliaryLoss', avg_aux_loss, batch)
            
            # Log performance ratios
            total_loss = avg_actor_loss + avg_critic_loss + avg_reward_loss + avg_future_state_loss
            if total_loss > 0:
                self.writer.add_scalar('PPO/ActorLossRatio', avg_actor_loss / total_loss, batch)
                self.writer.add_scalar('PPO/CriticLossRatio', avg_critic_loss / total_loss, batch)
                self.writer.add_scalar('PPO/RewardLossRatio', avg_reward_loss / total_loss, batch)
                self.writer.add_scalar('PPO/FutureStateLossRatio', avg_future_state_loss / total_loss, batch)
            
            logging.info(f"Enhanced PPO training completed - Actor: {avg_actor_loss:.4f}, "
                        f"Critic: {avg_critic_loss:.4f}, "
                        f"Reward: {avg_reward_loss:.4f}, "
                        f"Future State: {avg_future_state_loss:.4f}, "
                        f"Auxiliary: {avg_aux_loss:.4f}")
                        
            # IMMEDIATE STATISTICS REPORTING
            print(f"📊 Phase 5 Results:")
            print(f"   • Actor loss: {avg_actor_loss:.6f}")
            print(f"   • Critic loss: {avg_critic_loss:.6f}")
            print(f"   • Future state loss: {avg_future_state_loss:.6f}")
            print(f"   • Hindsight success rate: {successful_iterations / self.config.ppo_iterations * 100:.1f}%")
            print(f"   • Goal-conditioned training: ✅ ENABLED")
            
            # Store batch statistics
            self.update_batch_stats('ppo', {
                'actor_loss': avg_actor_loss,
                'critic_loss': avg_critic_loss,
                'reward_loss': avg_reward_loss,
                'future_state_loss': avg_future_state_loss,
                'auxiliary_loss': avg_aux_loss,
                'success_rate': successful_iterations / self.config.ppo_iterations,
                'hindsight_enabled': True
            })
        else:
            print("⚠️  No successful PPO training iterations with hindsight relabelling")
        
        # Log training efficiency
        self.writer.add_scalar('PPO/SuccessfulIterations', successful_iterations, batch)
        self.writer.add_scalar('PPO/IterationSuccessRate', successful_iterations / self.config.ppo_iterations, batch)
        self.writer.add_scalar('PPO/HindsightRelabellingEnabled', 1.0, batch)
    
    def phase_6_evaluation(self, batch):
        """
        Phase 6: Evaluate current policy performance on both GOAL ACHIEVEMENT and GAME PERFORMANCE
        ENHANCEMENT: Track both goal fulfillment and actual Tetris performance
        """
        print(f"\n📊 Phase 6: Goal Achievement & Game Performance Evaluation (Batch {batch+1})")
        
        total_game_reward = 0
        total_goal_reward = 0
        total_steps = 0
        total_lines_cleared = 0
        goal_matches = 0
        
        # Run evaluation episodes without exploration
        old_epsilon = self.actor_critic.epsilon
        self.actor_critic.epsilon = 0  # Pure exploitation
        
        for episode in range(self.config.eval_episodes):
            obs = self.env.reset()
            episode_game_reward = 0
            episode_goal_reward = 0
            episode_lines = 0
            episode_goal_matches = 0
            steps = 0
            done = False
            
            while not done and steps < self.config.max_episode_steps:
                state = self._obs_to_state_vector(obs)
                action = self.actor_critic.select_action(state)
                next_obs, game_reward, done, info = self.env.step(action)
                next_state = self._obs_to_state_vector(next_obs)
                
                # Calculate both game reward and goal achievement
                goal_achievement_reward = self.calculate_goal_achievement_reward(state, action, next_state, info)
                
                # Track lines cleared
                if info and 'lines_cleared' in info:
                    episode_lines += info['lines_cleared']
                
                # Track high goal achievements
                if goal_achievement_reward > 20.0:
                    episode_goal_matches += 1
                
                episode_game_reward += game_reward
                episode_goal_reward += goal_achievement_reward
                steps += 1
                
                obs = next_obs
            
            total_game_reward += episode_game_reward
            total_goal_reward += episode_goal_reward
            total_lines_cleared += episode_lines
            total_steps += steps
            goal_matches += episode_goal_matches
        
        # Restore epsilon
        self.actor_critic.epsilon = old_epsilon
        
        # Calculate averages
        avg_game_reward = total_game_reward / self.config.eval_episodes
        avg_goal_reward = total_goal_reward / self.config.eval_episodes
        avg_steps = total_steps / self.config.eval_episodes
        avg_lines_cleared = total_lines_cleared / self.config.eval_episodes
        avg_goal_matches = goal_matches / self.config.eval_episodes
        
        # Log both types of performance
        self.writer.add_scalar('Evaluation/AvgGameReward', avg_game_reward, batch)
        self.writer.add_scalar('Evaluation/AvgGoalReward', avg_goal_reward, batch)
        self.writer.add_scalar('Evaluation/AvgSteps', avg_steps, batch)
        self.writer.add_scalar('Evaluation/AvgLinesCleared', avg_lines_cleared, batch)
        self.writer.add_scalar('Evaluation/AvgGoalMatches', avg_goal_matches, batch)
        
        # Goal-to-game performance correlation
        if avg_game_reward != 0:
            goal_game_correlation = avg_goal_reward / abs(avg_game_reward)
            self.writer.add_scalar('Evaluation/GoalGameCorrelation', goal_game_correlation, batch)
        
        # IMMEDIATE STATISTICS REPORTING with both metrics
        print(f"📊 Phase 6 Results:")
        print(f"   🎯 Pure policy goal achievement: {avg_goal_reward:.2f}")
        print(f"   🎮 Pure policy game performance: {avg_game_reward:.2f}")
        print(f"   📏 Pure policy steps: {avg_steps:.1f}")
        print(f"   📐 Pure policy lines cleared: {avg_lines_cleared:.1f}")
        print(f"   🏆 Goal matches per episode: {avg_goal_matches:.1f}")
        print(f"   🔗 Goal-game alignment: {'✅ GOOD' if avg_goal_reward > 0 and avg_game_reward > 0 else '⚠️ NEEDS ATTENTION'}")
        
        # Store batch statistics
        self.update_batch_stats('evaluation', {
            'avg_game_reward': avg_game_reward,
            'avg_goal_reward': avg_goal_reward,
            'avg_steps': avg_steps,
            'avg_lines_cleared': avg_lines_cleared,
            'avg_goal_matches': avg_goal_matches,
            'goal_game_aligned': avg_goal_reward > 0 and avg_game_reward > 0
        })
    
    def _obs_to_state_vector(self, obs):
        """
        Convert simplified observation dict to flattened state vector (410-dimensional)
        ENHANCED: Validates complete active block representation throughout training
        """
        # COMPLETE BLOCK REPRESENTATION: Flatten grids containing ALL active block coordinates
        current_piece_flat = obs['current_piece_grid'].flatten()  # 20*10 = 200 (ALL active block coordinates)
        empty_grid_flat = obs['empty_grid'].flatten()  # 20*10 = 200 (complete occupancy info)
        
        # Get one-hot encoding and metadata
        next_piece = obs['next_piece']  # 7 values (removed hold piece)
        metadata = np.array([
            obs['current_rotation'],
            obs['current_x'], 
            obs['current_y']
        ])  # 3 values
        
        # ENHANCED VALIDATION: Ensure complete active block representation
        active_blocks_count = np.sum(current_piece_flat > 0)
        if active_blocks_count > 0:
            # There should be at least 1 and at most 4 active block cells (typical Tetris piece size)
            if active_blocks_count < 1 or active_blocks_count > 4:
                print(f"     🔍 Training validation: Unusual active block count: {active_blocks_count}")
            
            # Spatial validation: Check that active blocks form a reasonable Tetris piece shape
            current_piece_grid = current_piece_flat.reshape(20, 10)
            active_y, active_x = np.where(current_piece_grid > 0)
            
            if len(active_y) > 0:
                # Calculate bounding box of active blocks
                min_y, max_y = np.min(active_y), np.max(active_y)
                min_x, max_x = np.min(active_x), np.max(active_x)
                span_x, span_y = max_x - min_x, max_y - min_y
                
                # Tetris pieces should span at most 3 units in any direction
                if span_x > 3 or span_y > 3:
                    print(f"     🔍 Training validation: Large piece span detected: x_span={span_x}, y_span={span_y}")
                
                # Connectivity check: all active blocks should be reasonably close
                active_positions = list(zip(active_x, active_y))
                max_distance = 0
                for i, (x1, y1) in enumerate(active_positions):
                    for x2, y2 in active_positions[i+1:]:
                        distance = abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance
                        max_distance = max(max_distance, distance)
                
                # Tetris pieces should have max Manhattan distance of ~5 between any two blocks
                if max_distance > 6:
                    print(f"     🔍 Training validation: Disconnected piece detected: max_distance={max_distance}")
        
        # SPATIAL CONSISTENCY: Validate empty grid consistency
        empty_blocks_count = np.sum(empty_grid_flat == 1)  # Count truly empty cells
        occupied_from_empty = 200 - empty_blocks_count  # Occupied cells from empty grid
        
        # Log spatial information for debugging
        if hasattr(self, 'spatial_validation_count'):
            self.spatial_validation_count += 1
        else:
            self.spatial_validation_count = 1
            
        # Log every 100th validation for monitoring
        if self.spatial_validation_count % 100 == 0:
            print(f"     📊 Spatial validation #{self.spatial_validation_count}: active={active_blocks_count}, occupied_from_empty={occupied_from_empty}")
        
        # Concatenate all components: 200 + 200 + 7 + 3 = 410 (complete active block representation)
        state_vector = np.concatenate([
            current_piece_flat, 
            empty_grid_flat,
            next_piece,
            metadata
        ])
        
        # COMPREHENSIVE VALIDATION: Ensure complete state vector
        if len(state_vector) != 410:
            raise ValueError(f"State vector should be 410 dimensions, got {len(state_vector)}. "
                           f"Breakdown: current_piece={len(current_piece_flat)}, "
                           f"empty_grid={len(empty_grid_flat)}, "
                           f"next_piece={len(next_piece)}, "
                           f"metadata={len(metadata)}")
        
        return state_vector

    def _validate_complete_block_representation(self, obs):
        """
        Validate that the observation contains complete representation of all active blocks
        ENHANCED: Comprehensive validation for training pipeline integration
        """
        current_piece_grid = obs['current_piece_grid']
        empty_grid = obs['empty_grid']
        
        # Count active cells in the current piece grid
        active_cells = np.where(current_piece_grid > 0)
        active_count = len(active_cells[0])
        
        # Spatial analysis of active blocks
        if active_count > 0:
            # Verify that active cells form a connected component (valid Tetris piece)
            active_positions = list(zip(active_cells[1], active_cells[0]))  # (x, y) format
            
            # Enhanced connectivity check: all blocks should be within reasonable distance
            if len(active_positions) > 1:
                x_coords = [pos[0] for pos in active_positions]
                y_coords = [pos[1] for pos in active_positions]
                
                x_span = max(x_coords) - min(x_coords)
                y_span = max(y_coords) - min(y_coords)
                
                # Tetris pieces have max span of 3 in any direction
                if x_span > 3 or y_span > 3:
                    print(f"     ⚠️  Training warning: Active block span too large: x_span={x_span}, y_span={y_span}")
                    return False, active_count, active_positions, "span_too_large"
                
                # Check block connectivity (each block should be adjacent to at least one other)
                if len(active_positions) > 1:
                    connected_blocks = set()
                    
                    for i, (x1, y1) in enumerate(active_positions):
                        for j, (x2, y2) in enumerate(active_positions):
                            if i != j:
                                # Check if blocks are adjacent (distance = 1)
                                if abs(x1 - x2) + abs(y1 - y2) == 1:
                                    connected_blocks.add(i)
                                    connected_blocks.add(j)
                    
                    # At least some blocks should be connected
                    if len(connected_blocks) < len(active_positions) - 1:
                        print(f"     🔍 Training info: Some disconnected blocks detected (may be valid for certain pieces)")
            
            # Grid consistency check
            occupied_from_empty = np.sum(empty_grid == 0)  # Count occupied cells from empty grid
            
            return True, active_count, active_positions, "valid"
        
        return False, 0, [], "no_active_blocks"

    def save_checkpoint(self, batch):
        """Save comprehensive training checkpoint with all model components"""
        checkpoint = {
            'batch': batch,
            'phase': self.phase,
            'episode_count': self.episode_count,
            'total_episodes_completed': self.total_episodes_completed,
            'state_model': self.state_model.state_dict(),
            'future_reward_predictor': self.future_reward_predictor.state_dict(),
            'actor_critic_network': self.actor_critic.network.state_dict(),
            'actor_optimizer': self.actor_critic.actor_optimizer.state_dict(),
            'critic_optimizer': self.actor_critic.critic_optimizer.state_dict(),
            'future_state_optimizer': self.actor_critic.future_state_optimizer.state_dict(),
            'state_optimizer': self.state_optimizer.state_dict(),
            'reward_optimizer': self.reward_optimizer.state_dict(),
            'epsilon': self.actor_critic.epsilon,
            'exploration_data': self.exploration_data[-1000:],  # Keep recent data
            'episode_lines_cleared': self.episode_lines_cleared[-100:],  # Keep recent performance
            'rnd_exploration_state': {
                'visited_terminal_states': list(self.exploration_actor.visited_terminal_states) if hasattr(self.exploration_actor, 'visited_terminal_states') else [],
                'terminal_value_history': self.exploration_actor.terminal_value_history[-500:] if hasattr(self.exploration_actor, 'terminal_value_history') else [],
                'rnd_stats': self.exploration_actor.rnd_exploration.get_exploration_stats() if hasattr(self.exploration_actor, 'rnd_exploration') else {}
            },
            'config': {
                'state_dim': self.tetris_config.STATE_DIM,
                'action_dim': self.tetris_config.ACTION_DIM,
                'goal_dim': self.tetris_config.GOAL_DIM
            }
        }
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_batch_{batch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save the latest checkpoint
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        print(f"💾 Checkpoint saved: batch_{batch}.pt")
        
    def load_checkpoint(self, checkpoint_path):
        """Load comprehensive training checkpoint with all model components"""
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load training state
        self.batch_count = checkpoint.get('batch', 0)
        self.phase = checkpoint.get('phase', 1)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.total_episodes_completed = checkpoint.get('total_episodes_completed', 0)
        
        # Load model states
        self.state_model.load_state_dict(checkpoint['state_model'])
        self.future_reward_predictor.load_state_dict(checkpoint['future_reward_predictor'])
        self.actor_critic.network.load_state_dict(checkpoint['actor_critic_network'])
        
        # Load optimizers
        self.actor_critic.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.actor_critic.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        # Load future state optimizer if available (backward compatibility)
        if 'future_state_optimizer' in checkpoint:
            self.actor_critic.future_state_optimizer.load_state_dict(checkpoint['future_state_optimizer'])
        
        self.state_optimizer.load_state_dict(checkpoint['state_optimizer'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer'])
        
        # Load training parameters
        self.actor_critic.epsilon = checkpoint['epsilon']
        
        # Load data
        self.exploration_data = checkpoint.get('exploration_data', [])
        self.episode_lines_cleared = checkpoint.get('episode_lines_cleared', [])
        
        # Load RND exploration state if available
        if 'rnd_exploration_state' in checkpoint:
            rnd_state = checkpoint['rnd_exploration_state']
            if hasattr(self.exploration_actor, 'visited_terminal_states'):
                self.exploration_actor.visited_terminal_states = set(rnd_state.get('visited_terminal_states', []))
            if hasattr(self.exploration_actor, 'terminal_value_history'):
                self.exploration_actor.terminal_value_history = rnd_state.get('terminal_value_history', [])
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   • Batch: {self.batch_count}, Episodes: {self.total_episodes_completed}")
        print(f"   • Exploration data: {len(self.exploration_data)} placements")
        print(f"   • Epsilon: {self.actor_critic.epsilon:.4f}")
        
        return checkpoint

    def calculate_piece_presence_reward(self, obs):
        """
        Calculate piece presence reward that decreases over the first half of training
        Provides +1 reward per step when pieces are present on board (not +num_pieces per step)
        Args:
            obs: Current observation dict
        Returns:
            piece_presence_reward: Float reward (+1 per step with decay, or 0)
        """
        config = self.tetris_config.RewardConfig
        
        # Calculate decay factor based on total episodes completed
        max_episodes = config.PIECE_PRESENCE_DECAY_STEPS  # 500 episodes (first half)
        if self.total_episodes_completed >= max_episodes:
            return config.PIECE_PRESENCE_MIN  # 0.0 after first half
        
        # Linear decay from 1.0 to 0.0 over first 500 episodes
        decay_factor = 1.0 - (self.total_episodes_completed / max_episodes)
        
        # Check if there are pieces on the board
        # Use current_piece_grid and empty_grid to determine if pieces are present
        current_piece_cells = np.sum(obs['current_piece_grid'] > 0)
        
        # Estimate placed pieces from empty grid (inverse of empty spaces)
        total_grid_cells = obs['empty_grid'].size
        empty_cells = np.sum(obs['empty_grid'] == 0)  # Empty cells
        placed_piece_cells = total_grid_cells - empty_cells
        
        total_piece_cells = current_piece_cells + placed_piece_cells
        
        # Reward is +1 per step if pieces are present, scaled by decay factor
        # Not +num_pieces per step, just +1 for having pieces present
        if total_piece_cells > 0:
            piece_presence_reward = config.PIECE_PRESENCE_REWARD * decay_factor  # +1 * decay_factor
        else:
            piece_presence_reward = 0.0  # No reward if no pieces present
        
        return piece_presence_reward

    def calculate_goal_achievement_reward(self, state, action, next_state, info):
        """
        Calculate reward based purely on goal achievement, not game performance
        ENHANCEMENT: PPO actor is rewarded only for fulfilling state model goals
        
        Args:
            state: Current state vector
            action: Action taken (one-hot)
            next_state: Resulting state vector
            info: Episode info dict
        Returns:
            goal_achievement_reward: Float reward based on goal fulfillment
        """
        if self.state_model is None:
            # Fallback to small neutral reward if no state model
            return 0.1
        
        try:
            # Convert state to tensor for state model
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Get goal vector from state model
            with torch.no_grad():
                goal_vector = self.state_model.get_placement_goal_vector(state_tensor)
                
                if goal_vector is None:
                    return 0.1  # Neutral reward if no goal available
                
                # Extract goal components (36D goal vector structure)
                # [rotation_one_hot(4) + x_position_one_hot(10) + y_position_one_hot(20) + value(1) + confidence(1)]
                goal_rotation = torch.argmax(goal_vector[0, :4]).item()  # 0-3
                goal_x_pos = torch.argmax(goal_vector[0, 4:14]).item()   # 0-9
                goal_y_pos = torch.argmax(goal_vector[0, 14:34]).item()  # 0-19
                goal_value = goal_vector[0, 34].item()                   # Expected value
                goal_confidence = goal_vector[0, 35].item()              # Confidence (0-1)
            
            # Extract actual placement from next_state (metadata indices 407-409)
            actual_rotation = int(next_state[407] * 4)  # Denormalize from [0,1] to [0,3]
            actual_x_pos = int(next_state[408] * 10)    # Denormalize from [0,1] to [0,9]
            actual_y_pos = int(next_state[409] * 20)    # Denormalize from [0,1] to [0,19]
            
            # Calculate direct goal fulfillment (the more direct the mapping, the better)
            rotation_match = 1.0 if abs(actual_rotation - goal_rotation) == 0 else max(0.0, 1.0 - abs(actual_rotation - goal_rotation) / 4.0)
            x_pos_match = 1.0 if abs(actual_x_pos - goal_x_pos) == 0 else max(0.0, 1.0 - abs(actual_x_pos - goal_x_pos) / 10.0)
            y_pos_match = 1.0 if abs(actual_y_pos - goal_y_pos) == 0 else max(0.0, 1.0 - abs(actual_y_pos - goal_y_pos) / 20.0)
            
            # Calculate state similarity reward (how close achieved state is to goal expectation)
            state_similarity = F.cosine_similarity(state_tensor, next_state_tensor, dim=1).item()
            state_similarity = max(0.0, state_similarity)  # Ensure non-negative
            
            # DIRECT GOAL MAPPING REWARD (this is the primary signal)
            direct_mapping_reward = (
                rotation_match * 10.0 +     # Rotation accuracy (max +10)
                x_pos_match * 10.0 +        # X position accuracy (max +10)  
                y_pos_match * 10.0 +        # Y position accuracy (max +10)
                state_similarity * 5.0      # State coherence (max +5)
            )
            
            # Goal quality weighting (higher confidence goals get higher rewards)
            confidence_weight = max(0.1, goal_confidence)  # Minimum 0.1 to avoid zero rewards
            value_weight = max(0.1, (goal_value + 50) / 100.0)  # Normalize value to [0.1, 1.5] range
            
            # Final goal achievement reward
            goal_achievement_reward = direct_mapping_reward * confidence_weight * value_weight
            
            # Log detailed goal tracking for analysis
            if hasattr(self, 'writer') and self.writer is not None:
                step = getattr(self, 'goal_tracking_step', 0)
                self.writer.add_scalar('GoalAchievement/RotationMatch', rotation_match, step)
                self.writer.add_scalar('GoalAchievement/XPositionMatch', x_pos_match, step)
                self.writer.add_scalar('GoalAchievement/YPositionMatch', y_pos_match, step)
                self.writer.add_scalar('GoalAchievement/StateSimilarity', state_similarity, step)
                self.writer.add_scalar('GoalAchievement/DirectMappingReward', direct_mapping_reward, step)
                self.writer.add_scalar('GoalAchievement/ConfidenceWeight', confidence_weight, step)
                self.writer.add_scalar('GoalAchievement/ValueWeight', value_weight, step)
                self.writer.add_scalar('GoalAchievement/FinalReward', goal_achievement_reward, step)
                self.goal_tracking_step = step + 1
            
            # Clamp to reasonable range
            goal_achievement_reward = max(-10.0, min(50.0, goal_achievement_reward))
            
            return goal_achievement_reward
            
        except Exception as e:
            print(f"Error calculating goal achievement reward: {e}")
            return 0.1  # Neutral fallback reward

    def update_batch_stats(self, phase, stats_dict):
        """Update batch statistics for a specific phase"""
        self.batch_stats[phase].update(stats_dict)

    def _create_all_attempt_hindsight_experiences(self, all_attempt_experiences, episode_trajectory):
        """
        Create hindsight experiences for ALL individual attempts, not just selected ones
        ENHANCEMENT: Ensures every attempt gets hindsight relabeling for maximum learning
        
        Args:
            all_attempt_experiences: List of all attempt experiences (selected + unselected)
            episode_trajectory: Complete episode trajectory for future goal selection
        Returns:
            List of hindsight experiences for all attempts
        """
        all_hindsight_experiences = []
        
        if len(all_attempt_experiences) == 0 or len(episode_trajectory) == 0:
            return all_hindsight_experiences
        
        # Process each attempt individually
        for attempt_exp in all_attempt_experiences:
            # Skip attempts that weren't executed (no next_state available)
            if 'next_state' not in attempt_exp:
                # For unselected attempts, create synthetic hindsight based on the selected attempt's outcome
                step_index = attempt_exp['step_index']
                
                # Find the corresponding executed step in episode trajectory
                if step_index < len(episode_trajectory):
                    executed_step = episode_trajectory[step_index]
                    
                    # Create synthetic hindsight: "What if this attempt was selected?"
                    synthetic_goal_reward = self._calculate_synthetic_attempt_reward(
                        attempt_exp['state'], 
                        attempt_exp['action'],
                        executed_step['next_state']  # Use actual achieved state
                    )
                    
                    # Create hindsight experience for unselected attempt
                    hindsight_exp = {
                        'obs': attempt_exp['obs'],
                        'action': attempt_exp['action'],
                        'hindsight_reward': min(40.0, max(-5.0, synthetic_goal_reward)),  # Clamp reward
                        'next_obs': executed_step['next_obs'],  # Use actual next observation
                        'done': executed_step['done'],
                        'info': executed_step['info'],
                        'hindsight_type': 'unselected_attempt',
                        'original_predicted_reward': attempt_exp['predicted_reward']
                    }
                    all_hindsight_experiences.append(hindsight_exp)
            else:
                # For selected attempts, create enhanced hindsight using future goals
                step_index = attempt_exp['step_index']
                
                # Find future high-reward states as goals
                future_steps = [step for i, step in enumerate(episode_trajectory) if i > step_index]
                if future_steps:
                    # Select from top 50% of future rewards
                    future_rewards = [step['goal_reward'] for step in future_steps]
                    sorted_indices = sorted(range(len(future_steps)), 
                                          key=lambda i: future_rewards[i], 
                                          reverse=True)
                    top_half_count = max(1, len(sorted_indices) // 2)
                    top_future_indices = sorted_indices[:top_half_count]
                    
                    # Randomly select a high-reward future state as goal
                    goal_step = future_steps[np.random.choice(top_future_indices)]
                    
                    # Calculate hindsight reward toward this future goal
                    hindsight_goal_reward = self._calculate_hindsight_goal_achievement(
                        attempt_exp['state'],
                        attempt_exp['action'], 
                        attempt_exp['next_state'],
                        goal_step['next_state']
                    )
                    
                    # Enhanced reward combining multiple signals
                    temporal_distance = len([s for s in episode_trajectory[step_index:] if s['goal_reward'] == goal_step['goal_reward']])
                    temporal_discount = 0.95 ** temporal_distance
                    future_quality = (goal_step['goal_reward'] + 50) / 100.0
                    
                    final_hindsight_reward = (
                        hindsight_goal_reward * 0.6 +              # Primary: goal achievement
                        attempt_exp['actual_goal_reward'] * 0.2 +  # Secondary: original reward
                        temporal_discount * future_quality * 10.0  # Bonus: temporal and quality
                    )
                    
                    # Create hindsight experience for selected attempt
                    hindsight_exp = {
                        'obs': attempt_exp['obs'],
                        'action': attempt_exp['action'],
                        'hindsight_reward': min(60.0, max(-10.0, final_hindsight_reward)),
                        'next_obs': attempt_exp['next_obs'],
                        'done': attempt_exp['done'],
                        'info': attempt_exp['info'],
                        'hindsight_type': 'selected_attempt_enhanced',
                        'original_goal_reward': attempt_exp['actual_goal_reward'],
                        'future_goal_reward': goal_step['goal_reward'],
                        'temporal_distance': temporal_distance
                    }
                    all_hindsight_experiences.append(hindsight_exp)
        
        return all_hindsight_experiences
    
    def _calculate_synthetic_attempt_reward(self, attempt_state, attempt_action, achieved_state):
        """
        Calculate synthetic reward for unselected attempts
        Estimates what reward this attempt would have achieved
        
        Args:
            attempt_state: State when attempt was considered
            attempt_action: Action that was considered but not selected
            achieved_state: Actual state that was achieved by selected action
        Returns:
            synthetic_reward: Estimated reward for this attempt
        """
        try:
            # Calculate how well the attempt action aligns with the achieved outcome
            attempt_tensor = torch.FloatTensor(attempt_state).unsqueeze(0).to(self.device)
            achieved_tensor = torch.FloatTensor(achieved_state).unsqueeze(0).to(self.device)
            
            # Use state model to evaluate the attempt action
            with torch.no_grad():
                goal_vector = self.state_model.get_placement_goal_vector(attempt_tensor)
                
                if goal_vector is not None:
                    # Extract goal preferences
                    goal_rotation = torch.argmax(goal_vector[0, :4]).item()
                    goal_x_pos = torch.argmax(goal_vector[0, 4:14]).item()
                    goal_confidence = goal_vector[0, 35].item()
                    
                    # Convert attempt action to placement
                    action_idx = np.argmax(attempt_action)
                    if action_idx < 4:
                        action_rotation = action_idx
                        action_x = 5
                    elif action_idx < 7:
                        action_rotation = 0
                        action_x = (action_idx - 4) * 3 + 2
                    else:
                        action_rotation = 0
                        action_x = 5
                    
                    # Calculate alignment with state model goals
                    rotation_alignment = 1.0 - abs(action_rotation - goal_rotation) / 4.0
                    x_alignment = 1.0 - abs(action_x - goal_x_pos) / 10.0
                    
                    # Calculate state progression similarity
                    state_similarity = F.cosine_similarity(attempt_tensor, achieved_tensor, dim=1).item()
                    state_similarity = max(0.0, state_similarity)
                    
                    synthetic_reward = (
                        rotation_alignment * 8.0 +      # Goal alignment
                        x_alignment * 8.0 +             # Position alignment  
                        state_similarity * 6.0 +        # State progression
                        goal_confidence * 5.0           # Confidence weighting
                    )
                    
                    return synthetic_reward
                else:
                    # Fallback: basic action quality estimation
                    return 2.0
                    
        except Exception as e:
            print(f"Error in synthetic attempt reward: {e}")
            return 1.0

class TrainingConfig:
    """Configuration for unified training"""
    def __init__(self):
        # Get centralized config
        tetris_config = TetrisConfig()
        
        # Training parameters - use centralized config for 1000 total episodes
        self.num_batches = tetris_config.TrainingConfig.NUM_BATCHES  # 50
        self.batch_size = tetris_config.TrainingConfig.BATCH_SIZE  # 32
        self.exploration_episodes = tetris_config.TrainingConfig.EXPLORATION_EPISODES  # 20 (50 * 20 = 1000 total)
        self.exploitation_episodes = tetris_config.TrainingConfig.EXPLOITATION_EPISODES  # 20 (50 * 20 = 1000 total)
        self.eval_episodes = tetris_config.TrainingConfig.EVAL_EPISODES  # 10
        self.max_episode_steps = tetris_config.TrainingConfig.MAX_EPISODE_STEPS  # 2000
        
        # Device detection
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"GPU detected: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            print("Apple Silicon GPU (MPS) detected")
        else:
            self.device = 'cpu'
            print("Using CPU - no GPU detected")
        
        # Model parameters - use centralized config
        self.state_lr = tetris_config.TrainingConfig.STATE_LEARNING_RATE  # 1e-3
        self.reward_lr = tetris_config.TrainingConfig.REWARD_LEARNING_RATE  # 1e-3
        self.clip_ratio = tetris_config.TrainingConfig.PPO_CLIP_RATIO  # 0.2
        
        # Training phases - use centralized config
        self.state_training_samples = tetris_config.TrainingConfig.STATE_TRAINING_SAMPLES  # 1000
        self.state_epochs = tetris_config.TrainingConfig.STATE_EPOCHS  # 5
        self.ppo_iterations = tetris_config.TrainingConfig.PPO_ITERATIONS  # 3
        self.ppo_batch_size = tetris_config.TrainingConfig.PPO_BATCH_SIZE  # 64
        self.ppo_epochs = tetris_config.TrainingConfig.PPO_EPOCHS  # 4
        self.reward_batch_size = tetris_config.TrainingConfig.REWARD_BATCH_SIZE  # 64
        
        # Buffer parameters - use centralized config
        self.buffer_size = tetris_config.TrainingConfig.BUFFER_SIZE  # 100000
        self.min_buffer_size = tetris_config.TrainingConfig.MIN_BUFFER_SIZE  # 1000
        
        # Logging and saving - use centralized config
        self.log_dir = tetris_config.LoggingConfig.LOG_DIR  # 'logs/unified_training'
        self.checkpoint_dir = tetris_config.LoggingConfig.CHECKPOINT_DIR  # 'checkpoints/unified'
        self.save_interval = tetris_config.LoggingConfig.SAVE_INTERVAL  # 10
        self.visualize = False  # Default to no visualization
        
        # NEW: Exploration mode configuration
        self.exploration_mode = 'rnd'  # Options: 'rnd', 'random', 'deterministic'

def main():
    parser = argparse.ArgumentParser(description="Unified Tetris RL Training")
    parser.add_argument('--num_batches', type=int, default=50, help='Number of training batches (50 * 20 episodes = 1000 total)')
    parser.add_argument('--visualize', action='store_true', help='Visualize training')
    parser.add_argument('--log_dir', type=str, default='logs/unified_training', help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/unified', help='Checkpoint directory')
    
    # NEW: Exploration mode options
    parser.add_argument('--exploration_mode', type=str, default='rnd', 
                       choices=['rnd', 'random', 'deterministic'],
                       help='Exploration strategy: rnd (Random Network Distillation), random (true random), deterministic (systematic coverage)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('unified_training.log')
        ]
    )
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create config
    config = TrainingConfig()
    config.num_batches = args.num_batches
    config.visualize = args.visualize
    config.log_dir = args.log_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.exploration_mode = args.exploration_mode  # NEW: Set exploration mode
    
    # Print configuration summary
    print(f"\n🎮 Tetris RL Training Configuration:")
    print(f"   📦 Batches: {config.num_batches}")
    print(f"   🎯 Total Episodes: {config.num_batches * (config.exploration_episodes + config.exploitation_episodes)}")
    print(f"   🔍 Exploration Mode: {config.exploration_mode.upper()}")
    print(f"   👁️  Visualization: {'Enabled' if config.visualize else 'Disabled'}")
    print(f"   📊 Logging: {config.log_dir}")
    print(f"   💾 Checkpoints: {config.checkpoint_dir}")
    
    if config.exploration_mode == 'rnd':
        print(f"   🧠 Using Random Network Distillation with curiosity-driven exploration")
    elif config.exploration_mode == 'random':
        print(f"   🎲 Using true random exploration for unbiased coverage")
    elif config.exploration_mode == 'deterministic':
        print(f"   🎯 Using deterministic systematic exploration for comprehensive coverage")
    
    print(f"   🤖 Enhanced PPO with Hindsight Experience Replay: ENABLED")
    print()
    
    # Initialize and run trainer
    trainer = UnifiedTrainer(config)
    trainer.run_training()

if __name__ == '__main__':
    main()
