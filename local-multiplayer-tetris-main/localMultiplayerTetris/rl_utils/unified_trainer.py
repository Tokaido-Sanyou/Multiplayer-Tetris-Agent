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
            try:
                from .rnd_exploration import RNDExplorationActor
            except ImportError:
                # Direct execution fallback
                from rnd_exploration import RNDExplorationActor
            self.exploration_actor = RNDExplorationActor(self.env)
        elif self.exploration_mode == 'random':
            try:
                from .rnd_exploration import TrueRandomExplorer
            except ImportError:
                # Direct execution fallback
                from rnd_exploration import TrueRandomExplorer
            self.exploration_actor = TrueRandomExplorer(self.env)
        elif self.exploration_mode == 'deterministic':
            try:
                from .rnd_exploration import DeterministicTerminalExplorer
            except ImportError:
                # Direct execution fallback
                from rnd_exploration import DeterministicTerminalExplorer
            self.exploration_actor = DeterministicTerminalExplorer(self.env)
        else:
            # Default to RND if unknown mode
            try:
                from .rnd_exploration import RNDExplorationActor
            except ImportError:
                # Direct execution fallback
                from rnd_exploration import RNDExplorationActor
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
        self.experience_buffer = ReplayBuffer(config.buffer_size, device=self.device)  # CUDA optimization
        
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
    
    def phase_2_state_learning(self, batch, previous_batch_min_loss=float('inf'), extra_training=False):
        """
        Phase 2: State Model Learning
        Trains the state model on data from exploration_data.
        """
        print(f"🧠 Phase 2: State Model Learning (Batch {batch+1})")
        if not self.exploration_data:
            print("   ⚠️ No exploration data for state model training.")
            self.update_batch_stats('state_model', {'total_loss': 0, 'loss_improvement': 0, 'epochs_trained': 0})
            return {'total_loss': 0, 'rotation_loss': 0, 'x_pos_loss': 0, 'y_pos_loss': 0, 'value_loss': 0, 'current_min_loss': float('inf'), 'epochs_trained_this_call': 0}

        # Use normal fixed epochs per call (adaptive logic now handled by StagedUnifiedTrainer)
        epochs_per_call = self.tetris_config.TrainingConfig.STATE_EPOCHS  # 3 epochs
        
        # Transform exploration data format to expected format before validation
        print(f"   🔄 Transforming exploration data format...")
        transformed_exploration_data = []
        transform_errors = 0
        
        for i, data_item in enumerate(self.exploration_data):
            if isinstance(data_item, dict):
                try:
                    # Transform to expected format
                    transformed_item = {}
                    
                    # Map 'state' to 'state_vector'
                    if 'state' in data_item:
                        transformed_item['state_vector'] = data_item['state']
                    elif 'state_vector' in data_item:
                        transformed_item['state_vector'] = data_item['state_vector']
                    
                    # Extract rotation, x, y from 'placement' tuple
                    if 'placement' in data_item:
                        placement = data_item['placement']
                        if isinstance(placement, (tuple, list)) and len(placement) >= 3:
                            transformed_item['true_rotation'] = placement[0]
                            transformed_item['true_x'] = placement[1] 
                            transformed_item['true_y'] = placement[2]
                        else:
                            raise ValueError(f"Invalid placement format: {placement}")
                    elif all(key in data_item for key in ['true_rotation', 'true_x', 'true_y']):
                        # Already in expected format
                        transformed_item['true_rotation'] = data_item['true_rotation']
                        transformed_item['true_x'] = data_item['true_x']
                        transformed_item['true_y'] = data_item['true_y']
                    
                    # Copy terminal_reward
                    if 'terminal_reward' in data_item:
                        transformed_item['terminal_reward'] = data_item['terminal_reward']
                    
                    transformed_exploration_data.append(transformed_item)
                    
                except Exception as e:
                    transform_errors += 1
                    if transform_errors <= 3:  # Only log first 3 errors to avoid spam
                        print(f"   🚨 Transform error at index {i}: {e}. Available keys: {list(data_item.keys())}")
            else:
                transform_errors += 1
                if transform_errors <= 3:
                    print(f"   🚨 Non-dict entry at index {i}: {type(data_item)}")
        
        if transform_errors > 0:
            print(f"   🚨 {transform_errors} transformation errors encountered. Successfully transformed {len(transformed_exploration_data)} entries.")
        else:
            print(f"   ✅ Successfully transformed {len(transformed_exploration_data)} exploration data entries.")

        # NEW: Filter transformed data to ensure all required keys are present
        valid_exploration_data = []
        malformed_count = 0
        required_keys = ['state_vector', 'true_rotation', 'true_x', 'true_y', 'terminal_reward']
        
        for i, data_item in enumerate(transformed_exploration_data):
            if isinstance(data_item, dict):
                missing_keys = [key for key in required_keys if key not in data_item]
                if not missing_keys:
                    valid_exploration_data.append(data_item)
                else:
                    malformed_count += 1
                    # Concise logging for malformed dicts
                    print(f"   🚨 Malformed dict entry in self.exploration_data at index {i}. Missing keys: {missing_keys}. Existing keys: {list(data_item.keys())}")
            else:
                malformed_count += 1
                # Concise logging for non-dict entries
                entry_type = type(data_item)
                entry_len = len(data_item) if hasattr(data_item, '__len__') else 'N/A'
                print(f"   🚨 Malformed non-dict entry in self.exploration_data at index {i}. Type: {entry_type}, Len: {entry_len}")

        if malformed_count > 0:
            print(f"   🚨 Found {malformed_count} malformed entries in self.exploration_data. Using {len(valid_exploration_data)} valid entries for this training call.")

        if not valid_exploration_data:
            print("   ⚠️ No VALID exploration data available for state model training after filtering.")
            self.update_batch_stats('state_model', {'total_loss': 0, 'loss_improvement': 0, 'epochs_trained': 0, 'valid_samples_used': 0})
            return {'total_loss': 0, 'rotation_loss': 0, 'x_pos_loss': 0, 'y_pos_loss': 0, 'value_loss': 0, 'current_min_loss': float('inf'), 'epochs_trained_this_call': 0}

        # Use valid_exploration_data for subsequent processing
        states = torch.FloatTensor(np.array([d['state_vector'] for d in valid_exploration_data])).to(self.device)
        true_rotations = torch.LongTensor(np.array([d['true_rotation'] for d in valid_exploration_data])).to(self.device)
        true_x_positions = torch.LongTensor(np.array([d['true_x'] for d in valid_exploration_data])).to(self.device)
        true_y_positions = torch.LongTensor(np.array([d['true_y'] for d in valid_exploration_data])).to(self.device)
        terminal_rewards = torch.FloatTensor(np.array([d['terminal_reward'] for d in valid_exploration_data])).unsqueeze(1).to(self.device)

        # Calculate reward weights
        reward_weights = torch.clamp(terminal_rewards / self.tetris_config.AlgorithmConfig.REWARD_WEIGHT_NORMALIZATION, min=0.1, max=1.0).squeeze()
        
        dataset = torch.utils.data.TensorDataset(states, true_rotations, true_x_positions, true_y_positions, terminal_rewards, reward_weights)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.tetris_config.TrainingConfig.BATCH_SIZE, shuffle=True)

        initial_loss = float('inf')
        final_loss = float('inf')
        min_loss_this_call = float('inf')
        epochs_trained_this_call = 0
        
        all_epoch_total_losses = []
        all_epoch_rot_losses = []
        all_epoch_x_losses = []
        all_epoch_y_losses = []
        all_epoch_val_losses = []

        print(f"   🔄 Training for {epochs_per_call} epochs (normal call)")

        for epoch in range(epochs_per_call):
            epochs_trained_this_call = epoch + 1
            epoch_total_loss = 0
            epoch_rotation_loss = 0
            epoch_x_pos_loss = 0
            epoch_y_pos_loss = 0
            epoch_value_loss = 0
            
            for s, rot, x, y, rew, weights in dataloader:
                self.state_optimizer.zero_grad()
                pred_rot, pred_x, pred_y, pred_value = self.state_model(s)
                
                loss_rot = F.cross_entropy(pred_rot, rot, reduction='none')
                loss_x = F.cross_entropy(pred_x, x, reduction='none')
                loss_y = F.cross_entropy(pred_y, y, reduction='none')
                loss_value = F.mse_loss(pred_value, rew, reduction='none').squeeze()
                
                weighted_loss = (torch.mean(loss_rot * weights) +
                                 torch.mean(loss_x * weights) +
                                 torch.mean(loss_y * weights) +
                                 torch.mean(loss_value * weights))
                
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.state_model.parameters(), self.tetris_config.TrainingConfig.GRADIENT_CLIP_NORM)
                self.state_optimizer.step()
                
                epoch_total_loss += weighted_loss.item()
                epoch_rotation_loss += torch.mean(loss_rot * weights).item()
                epoch_x_pos_loss += torch.mean(loss_x * weights).item()
                epoch_y_pos_loss += torch.mean(loss_y * weights).item()
                epoch_value_loss += torch.mean(loss_value * weights).item()

            avg_epoch_total_loss = epoch_total_loss / len(dataloader)
            avg_epoch_rotation_loss = epoch_rotation_loss / len(dataloader)
            avg_epoch_x_pos_loss = epoch_x_pos_loss / len(dataloader)
            avg_epoch_y_pos_loss = epoch_y_pos_loss / len(dataloader)
            avg_epoch_value_loss = epoch_value_loss / len(dataloader)

            all_epoch_total_losses.append(avg_epoch_total_loss)
            all_epoch_rot_losses.append(avg_epoch_rotation_loss)
            all_epoch_x_losses.append(avg_epoch_x_pos_loss)
            all_epoch_y_losses.append(avg_epoch_y_pos_loss)
            all_epoch_val_losses.append(avg_epoch_value_loss)
            
            if epoch == 0:
                initial_loss = avg_epoch_total_loss
            final_loss = avg_epoch_total_loss
            min_loss_this_call = min(min_loss_this_call, avg_epoch_total_loss)

            print(f"     Epoch {epochs_trained_this_call}/{epochs_per_call} (State Model): Loss={avg_epoch_total_loss:.4f} (MinThisCall: {min_loss_this_call:.4f})")

        loss_improvement = (initial_loss - final_loss) / initial_loss if initial_loss > 0 and initial_loss != float('inf') else 0
        
        stats = {
            'total_loss': final_loss if final_loss != float('inf') else 0, 
            'rotation_loss': all_epoch_rot_losses[-1] if all_epoch_rot_losses else 0,
            'x_pos_loss': all_epoch_x_losses[-1] if all_epoch_x_losses else 0,
            'y_pos_loss': all_epoch_y_losses[-1] if all_epoch_y_losses else 0,
            'value_loss': all_epoch_val_losses[-1] if all_epoch_val_losses else 0,
            'loss_improvement': loss_improvement,
            'initial_loss': initial_loss if initial_loss != float('inf') else 0,
            'current_min_loss': min_loss_this_call if min_loss_this_call != float('inf') else 0, # Key for StagedUnifiedTrainer
            'epochs_trained_this_call': epochs_trained_this_call,
            'all_total_losses': all_epoch_total_losses, # For more detailed logging if needed
            'valid_samples_used': len(valid_exploration_data)
        }
        self.update_batch_stats('state_model', stats)
        
        # Log to TensorBoard (final epoch stats for this call)
        if epochs_trained_this_call > 0: # Only log if training actually happened
            self.writer.add_scalar('StateModel/TotalLoss_Call', stats['total_loss'], batch * 10 + epochs_trained_this_call) 
            self.writer.add_scalar('StateModel/RotationLoss_Call', stats['rotation_loss'], batch * 10 + epochs_trained_this_call)
            self.writer.add_scalar('StateModel/XPositionLoss_Call', stats['x_pos_loss'], batch * 10 + epochs_trained_this_call)
            self.writer.add_scalar('StateModel/YPositionLoss_Call', stats['y_pos_loss'], batch * 10 + epochs_trained_this_call)
            self.writer.add_scalar('StateModel/ValueLoss_Call', stats['value_loss'], batch * 10 + epochs_trained_this_call)
            self.writer.add_scalar('StateModel/MinLossThisCall', stats['current_min_loss'], batch * 10 + epochs_trained_this_call)
        
        self.writer.add_scalar('StateModel/EpochsTrainedThisCall', epochs_trained_this_call, batch)
        self.writer.add_scalar('StateModel/ValidSamplesUsed', len(valid_exploration_data), batch)
        
        # Return the stats dictionary for StagedUnifiedTrainer
        return stats
    
    def phase_3_reward_prediction(self, batch):
        # Implementation of phase 3
        pass

    def phase_4_exploitation(self, batch):
        # Implementation of phase 4
        pass

    def phase_5_ppo_training(self, batch):
        # Implementation of phase 5
        pass

    def phase_6_evaluation(self, batch):
        # Implementation of phase 6
        pass

    def save_checkpoint(self, batch):
        # Implementation of saving checkpoint
        pass

    def update_batch_stats(self, phase, stats):
        # Implementation of updating batch statistics
        pass
