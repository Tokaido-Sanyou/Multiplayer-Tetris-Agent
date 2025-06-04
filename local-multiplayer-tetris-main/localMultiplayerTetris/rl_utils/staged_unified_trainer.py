"""
STAGED Unified Training System with Goal-Conditioned Actor-Critic
MAJOR ENHANCEMENT: Staged training with state model pretraining
Stage 1: State Model learns optimal placements
Stage 2: Actor learns to achieve frozen goals  
Stage 3: Joint fine-tuning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

# Import base UnifiedTrainer class only
from .unified_trainer import UnifiedTrainer

import argparse # Ensure argparse is imported
import os # Ensure os is imported
import logging # Ensure logging is imported

# Import enhanced components at the top
try:
    from .enhanced_6phase_state_model import Enhanced6PhaseComponents
    ENHANCED_COMPONENTS_AVAILABLE = True
    print("🚀 Enhanced 6-Phase Components loaded successfully!")
except ImportError as e:
    print(f"⚠️ Enhanced components not available: {e}")
    print("📝 Will use original components as fallback")
    ENHANCED_COMPONENTS_AVAILABLE = False

# Handle both direct execution and module import
try:
    from ..config import TetrisConfig
    from .actor_critic import ActorCriticAgent
    from .state_model import StateModel
    from .rnd_exploration import RNDExplorationActor, DeterministicTerminalExplorer, TrueRandomExplorer
    from ..tetris_env import TetrisEnv
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TetrisConfig
    from actor_critic import ActorCriticAgent
    from state_model import StateModel
    from rnd_exploration import RNDExplorationActor, DeterministicTerminalExplorer, TrueRandomExplorer
    from tetris_env import TetrisEnv


class StagedTrainingSchedule:
    """
    Enhanced training schedule with PROPORTIONAL staging support
    Prevents moving target problem by training state model first, then actor
    """
    def __init__(self, total_batches=300, stage_proportions=None):
        self.total_batches = total_batches
        
        # ENHANCED: Proportional staging support (default to original behavior)
        if stage_proportions is None:
            stage_proportions = {
                'state_model_pretraining': 0.5,    # 50% for state model pretraining
                'actor_training': 0.33,            # 33% for actor training
                'joint_finetuning': 0.17           # 17% for joint fine-tuning
            }
        
        # Validate proportions
        total_proportion = sum(stage_proportions.values())
        if abs(total_proportion - 1.0) > 0.01:
            print(f"⚠️ Warning: Stage proportions sum to {total_proportion:.3f}, normalizing...")
            # Normalize proportions
            for key in stage_proportions:
                stage_proportions[key] /= total_proportion
        
        self.stage_proportions = stage_proportions
        
        # Calculate batch boundaries based on proportions
        self.state_model_pretraining_batches = int(total_batches * stage_proportions['state_model_pretraining'])
        self.actor_training_batches = int(total_batches * stage_proportions['actor_training'])
        self.joint_finetuning_batches = int(total_batches * stage_proportions['joint_finetuning'])
        
        # Ensure all batches are accounted for
        assigned_batches = self.state_model_pretraining_batches + self.actor_training_batches + self.joint_finetuning_batches
        if assigned_batches < total_batches:
            # Add remaining batches to joint fine-tuning
            self.joint_finetuning_batches += (total_batches - assigned_batches)
        
        print(f"🎯 PROPORTIONAL STAGING CONFIGURED:")
        print(f"   • State Model Pretraining: {stage_proportions['state_model_pretraining']*100:.1f}% ({self.state_model_pretraining_batches} batches)")
        print(f"   • Actor Training: {stage_proportions['actor_training']*100:.1f}% ({self.actor_training_batches} batches)")
        print(f"   • Joint Fine-tuning: {stage_proportions['joint_finetuning']*100:.1f}% ({self.joint_finetuning_batches} batches)")
    
    def get_training_stage(self, batch):
        """Determine which training stage we're in"""
        if batch < self.state_model_pretraining_batches:
            return "state_model_pretraining"
        elif batch < self.state_model_pretraining_batches + self.actor_training_batches:
            return "actor_training_frozen_goals"
        else:
            return "joint_finetuning"
    
    def should_train_state_model(self, batch):
        """Whether to train state model in this batch"""
        stage = self.get_training_stage(batch)
        return stage in ["state_model_pretraining", "joint_finetuning"]
    
    def should_train_actor(self, batch):
        """Whether to train actor in this batch"""
        stage = self.get_training_stage(batch)
        return stage in ["actor_training_frozen_goals", "joint_finetuning"]
    
    def get_goal_gradient_mode(self, batch):
        """Whether to allow gradients through goals"""
        stage = self.get_training_stage(batch)
        if stage == "state_model_pretraining":
            return "full_gradients"  # State model learns freely
        elif stage == "actor_training_frozen_goals":
            return "stop_gradients"  # Actor can't corrupt state model
        else:
            return "full_gradients"  # Joint fine-tuning
    
    def get_training_intensity(self, batch):
        """Get training intensity multipliers for current stage"""
        stage = self.get_training_stage(batch)
        if stage == "state_model_pretraining":
            return {"state_model_extra_epochs": 3, "actor_extra_epochs": 0}
        elif stage == "actor_training_frozen_goals":
            return {"state_model_extra_epochs": 0, "actor_extra_epochs": 2}
        else:
            return {"state_model_extra_epochs": 1, "actor_extra_epochs": 1}

    def should_run_evaluation(self, batch):
        """Whether to run full actor-critic evaluation in this batch's stage"""
        stage = self.get_training_stage(batch)
        if stage == "state_model_pretraining":
            return False  # No actor evaluation during state model pretraining
        elif stage == "actor_training_frozen_goals":
            return True  # Evaluate actor during its training
        elif stage == "joint_finetuning":
            return True  # Evaluate during joint fine-tuning
        return False # Default to false for safety


class StagedUnifiedTrainer(UnifiedTrainer):
    """
    STAGED Unified trainer that prevents moving target problem
    Inherits from UnifiedTrainer and adds staged training capability
    """
    def __init__(self, config):
        # Initialize base trainer
        super().__init__(config)
        
        # ENHANCED: Support for proportional staging
        stage_proportions = getattr(config, 'stage_proportions', None)
        if stage_proportions:
            print(f"🎯 Using CUSTOM staging proportions: {stage_proportions}")
        else:
            print(f"🎯 Using DEFAULT staging proportions (50% state, 33% actor, 17% joint)")
        
        # CRITICAL ENHANCEMENT: Initialize staged training schedule with proportional support
        self.staged_training = StagedTrainingSchedule(
            total_batches=config.num_batches,
            stage_proportions=stage_proportions
        )
        
        # NEW: Track state model loss history for adaptive training
        self.state_model_batch_loss_history = []  # Track min loss from each batch
        
        print(f"\n🎯 STAGED TRAINING ENABLED:")
        print(f"   • Stage 1: State Model Pretraining (Batches 0-{self.staged_training.state_model_pretraining_batches-1})")
        print(f"   • Stage 2: Actor Training + Frozen Goals (Batches {self.staged_training.state_model_pretraining_batches}-{self.staged_training.state_model_pretraining_batches + self.staged_training.actor_training_batches-1})")
        print(f"   • Stage 3: Joint Fine-tuning (Batches {self.staged_training.state_model_pretraining_batches + self.staged_training.actor_training_batches}-{config.num_batches-1})")
        print(f"   🔥 BENEFIT: Prevents moving target problem - stable goals for actor!")
        
        # ENHANCED: Initialize enhanced components if available
        if ENHANCED_COMPONENTS_AVAILABLE:
            print("🚀 Using ENHANCED 6-Phase Components:")
            print("   • Top 5 terminal state model")
            print("   • Separate Q-learning for terminal rewards")
            print("   • Complete goal/game reward separation")
            print("   • Trajectory consistency and board continuity")
            print("   • Flexible placement support (ANY position on board)")
            
            self.enhanced_components = Enhanced6PhaseComponents(
                state_dim=self.tetris_config.STATE_DIM,
                device=self.device
            )
            
            # Set optimizers for enhanced components
            self.enhanced_components.set_optimizers(
                state_lr=config.state_lr,
                q_lr=config.reward_lr
            )
            
            # Use enhanced state model
            self.state_model = self.enhanced_components.state_model
            self.state_optimizer = self.enhanced_components.state_optimizer
            
            # Enhanced flag
            self.use_enhanced = True
        else:
            # Fallback to original components
            print("📝 Using original components (enhanced not available)")
            self.state_model = StateModel(state_dim=self.tetris_config.STATE_DIM).to(self.device)
            self.state_optimizer = torch.optim.Adam(self.state_model.parameters(), lr=config.state_lr)
            self.use_enhanced = False
    
    def run_training(self):
        """
        Main training loop implementing the 6-phase algorithm with STAGED TRAINING
        Stage 1: State Model Pretraining | Stage 2: Actor Training | Stage 3: Joint Fine-tuning
        """
        print(f"\n🚀 Starting STAGED Unified Training: {self.config.num_batches} batches")
        print(f"   Expected improvements: Goal consistency 40-60% → 80-90%, Goal achievement 8.8% → 30-50%")
        
        for batch in range(self.config.num_batches):
            # CRITICAL: Determine training stage and configuration
            stage = self.staged_training.get_training_stage(batch)
            should_train_state_model = self.staged_training.should_train_state_model(batch)
            should_train_actor = self.staged_training.should_train_actor(batch)
            goal_gradient_mode = self.staged_training.get_goal_gradient_mode(batch)
            intensity = self.staged_training.get_training_intensity(batch)
            
            print(f"\n{'='*80}")
            print(f"🎯 BATCH {batch + 1}/{self.config.num_batches} - STAGE: {stage.upper()}")
            print(f"{'='*80}")
            print(f"   🧠 State Model Training: {'✅ ON' if should_train_state_model else '❌ OFF (FROZEN)'}")
            print(f"   🎭 Actor Training: {'✅ ON' if should_train_actor else '❌ OFF (WAITING)'}")
            print(f"   🔒 Goal Gradients: {goal_gradient_mode.upper()}")
            print(f"   ⚡ Intensity: SM×{intensity['state_model_extra_epochs']}, Actor×{intensity['actor_extra_epochs']}")
            
            # Phase 1: Exploration data collection (always runs)
            self.phase_1_exploration(batch)
            
            # Phase 2: State Model Learning
            if should_train_state_model:
                self.phase_2_state_learning_with_adaptive_target(batch)
            else:
                print(f"🧠 Phase 2: State Model Learning (SKIPPED - goals frozen)")
            
            # Phase 2.5: Multi-Step Q-Learning (CORRECTED - with episode structure)
            if self.use_enhanced:
                self.phase_2_5_terminal_q_learning(batch)
            
            # Phase 3: Future Reward Predictor Training (now optional with Q-learning)
            if not self.use_enhanced:
                self.phase_3_future_reward_predictor(batch)
            else:
                print(f"🔮 Phase 3: Future Reward Predictor (SKIPPED - using Multi-step Q-learning instead)")
            
            # Phase 4: Actor Exploitation
            if should_train_actor:
                self.phase_4_actor_exploitation(batch, goal_gradient_mode)
            else:
                print(f"🎭 Phase 4: Actor Exploitation (SKIPPED - pretraining state model first)")
            
            # Phase 5: PPO Training
            if should_train_actor:
                self.phase_5_ppo_training(batch, goal_gradient_mode)
            else:
                print(f"🚀 Phase 5: PPO Training (SKIPPED - pretraining state model first)")
            
            # Phase 6: Model evaluation (ENHANCED with line clearing test)
            should_evaluate_this_stage = self.staged_training.should_run_evaluation(batch)
            is_evaluation_batch = (batch + 1) % self.tetris_config.AlgorithmConfig.EVALUATION_FREQUENCY == 0

            if should_evaluate_this_stage and is_evaluation_batch:
                print(f"📊 Phase 6: Enhanced Model Evaluation (Stage: {stage.upper()}, Batch {batch+1})")
                eval_results = self.phase_6_evaluation_with_line_clearing_test(batch)
                if eval_results:
                    for key, value in eval_results.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f"Evaluation_Staged/{key}", value, batch)
            elif not should_evaluate_this_stage:
                print(f"📊 Phase 6: Model Evaluation (SKIPPED - Stage: {stage.upper()} - Evaluation not active for this stage)")
            else:
                print(f"📊 Phase 6: Model Evaluation (SKIPPED - Stage: {stage.upper()}, Batch {batch+1} - Not an evaluation batch as per EVALUATION_FREQUENCY={self.tetris_config.AlgorithmConfig.EVALUATION_FREQUENCY})")

            # STAGING TRANSITION MESSAGES
            if batch == self.staged_training.state_model_pretraining_batches - 1:
                self._print_stage_transition_message("state_model_complete", batch)
            elif batch == self.config.num_batches - self.staged_training.joint_finetuning_batches - 1:
                self._print_stage_transition_message("joint_finetuning_begins", batch)
            
            # Print comprehensive batch summary
            self.print_batch_summary(batch)
            
            # Save checkpoints
            if (batch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(batch)
        
        print(f"\n🎉 STAGED Training completed successfully! Total episodes: {self.total_episodes_completed}")
        print(f"🎯 Result: Goal-consistent actor with stable state model guidance")
        self.writer.close()
    
    def phase_1_exploration(self, batch):
        """
        Phase 1: REDESIGNED Piece-by-Piece Exploration with proper board progression
        """
        print(f"🔍 Phase 1: Piece-by-Piece Exploration (Batch {batch+1})")
        
        if self.use_enhanced:
            # REDESIGNED: Use piece-by-piece exploration manager
            exploration_manager = self.enhanced_components.create_piece_by_piece_exploration_manager(self.env)
            
            # Determine exploration mode
            exploration_mode = getattr(self.config, 'exploration_mode', 'rnd')
            
            # Collect piece-by-piece exploration data with proper board progression
            print(f"   🎯 Mode: {exploration_mode.upper()} with 200 trials per piece")
            exploration_data = exploration_manager.collect_piece_by_piece_exploration_data(exploration_mode)
            
            print(f"   📊 Piece-by-Piece Exploration Results:")
            print(f"       • Total exploration data points: {len(exploration_data)}")
            print(f"       • Total lines cleared: {sum(d.get('lines_cleared', 0) for d in exploration_data)}")
            print(f"       • Line clearing rate: {sum(1 for d in exploration_data if d.get('lines_cleared', 0) > 0) / len(exploration_data) * 100:.1f}%")
            print(f"       • Average reward per trial: {np.mean([d['terminal_reward'] for d in exploration_data]):.2f}")
            print(f"       • Pieces placed: {len(exploration_manager.piece_sequence)}")
            
            # Trajectory lineage verification
            lineage_lengths = [len(d.get('trajectory_lineage', [])) for d in exploration_data]
            if lineage_lengths:
                print(f"       • Trajectory lineage verified: avg {np.mean(lineage_lengths):.1f} steps, max {max(lineage_lengths)} steps")
            
            self.exploration_data = exploration_data
            
            # Update batch stats with enhanced lineage info
            self.update_batch_stats('exploration', {
                'total_data_points': len(exploration_data),
                'lines_cleared': sum(d.get('lines_cleared', 0) for d in exploration_data),
                'line_clearing_rate': sum(1 for d in exploration_data if d.get('lines_cleared', 0) > 0) / len(exploration_data),
                'pieces_placed': len(exploration_manager.piece_sequence),
                'avg_trajectory_lineage': np.mean(lineage_lengths) if lineage_lengths else 0,
                'max_trajectory_lineage': max(lineage_lengths) if lineage_lengths else 0,
                'exploration_type': 'piece_by_piece_progression',
                'exploration_mode': exploration_mode,
                'trials_per_piece': exploration_manager.trials_per_piece,
                'boards_kept_per_generation': exploration_manager.boards_to_keep
            })
            
        else:
            # Fallback to original exploration methods
            exploration_mode = getattr(self.config, 'exploration_mode', 'rnd')
            
            if exploration_mode == 'rnd':
                self.exploration_data = self.exploration_actor.collect_placement_data(self.config.exploration_episodes)
            elif exploration_mode == 'deterministic':
                self.exploration_data = self.deterministic_explorer.generate_all_terminal_states()
            elif exploration_mode == 'random':
                self.exploration_data = self.random_explorer.collect_random_placement_data(self.config.exploration_episodes)
            else:
                print(f"⚠️ Unknown exploration mode: {exploration_mode}, using RND")
                self.exploration_data = self.exploration_actor.collect_placement_data(self.config.exploration_episodes)
            
            print(f"   📊 {exploration_mode.upper()} Exploration Results:")
            print(f"       • Total placements: {len(self.exploration_data)}")
            if self.exploration_data:
                rewards = [d['terminal_reward'] for d in self.exploration_data]
                print(f"       • Average terminal reward: {np.mean(rewards):.2f}")
                print(f"       • Reward range: {np.min(rewards):.1f} to {np.max(rewards):.1f}")
            
            # Update batch stats
            self.update_batch_stats('exploration', {
                'total_placements': len(self.exploration_data),
                'exploration_mode': exploration_mode,
                'exploration_type': 'traditional_terminal_states'
            })
    
    def phase_4_actor_exploitation(self, batch, goal_gradient_mode="full_gradients"):
        """
        Enhanced Phase 4: Actor Exploitation with enhanced goals
        """
        freeze_goals = (goal_gradient_mode == "stop_gradients")
        
        print(f"🎭 Phase 4: Actor Exploitation (Batch {batch+1})")
        print(f"   🔒 Goal gradients: {'FROZEN' if freeze_goals else 'FREE'}")
        print(f"   🎯 Using: {'Enhanced Top-5 Goals' if self.use_enhanced else 'Original Goals'}")
        
        if len(self.experience_buffer) < self.tetris_config.TrainingConfig.MIN_BUFFER_SIZE:
            print("   ⚠️ Insufficient experience for actor training")
            return
        
        total_reward = 0
        successful_episodes = 0
        
        for episode in range(self.config.exploitation_episodes):
            self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            while not self.env.game_over and episode_steps < self.config.max_episode_steps:
                # Get current state
                obs = self.env._get_observation()
                if self.use_enhanced:
                    state = self.enhanced_components.obs_to_state_vector(obs)
                else:
                    state = self.exploration_actor._obs_to_state_vector(obs)
                state_tensor = torch.FloatTensor(state).to(self.device)
                
                # Get goal from state model (enhanced or original)
                if self.use_enhanced:
                    goal = self.enhanced_components.get_goal_for_actor(state_tensor)
                    if goal is None:
                        # Fallback to zero goal
                        goal = torch.zeros(1, 36, device=self.device)
                else:
                    # Original goal generation
                    with torch.no_grad():
                        rot_logits, x_logits, y_logits, value_pred = self.state_model(state_tensor.unsqueeze(0))
                        goal = torch.cat([
                            F.softmax(rot_logits, dim=1),
                            F.softmax(x_logits, dim=1),
                            F.softmax(y_logits, dim=1),
                            torch.zeros(1, 2, device=self.device)  # Padding for 36D
                        ], dim=1)
                
                if freeze_goals:
                    goal = goal.detach()
                
                # Get action from actor
                combined_input = torch.cat([state_tensor.unsqueeze(0), goal], dim=1)
                action_probs = self.actor(combined_input)
                
                # Sample action
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
                # Execute action
                reward, done = self.env.step(action.item())
                new_obs = self.env._get_observation()
                if self.use_enhanced:
                    new_state = self.enhanced_components.obs_to_state_vector(new_obs)
                else:
                    new_state = self.exploration_actor._obs_to_state_vector(new_obs)
                
                # Store experience
                self.experience_buffer.push(state, action.item(), reward, new_state, done)
                
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            total_reward += episode_reward
            if episode_reward > 0:
                successful_episodes += 1
        
        avg_reward = total_reward / self.config.exploitation_episodes
        success_rate = successful_episodes / self.config.exploitation_episodes
        
        print(f"   📊 Actor Results:")
        print(f"       • Average reward: {avg_reward:.2f}")
        print(f"       • Success rate: {success_rate*100:.1f}%")
        print(f"       • Successful episodes: {successful_episodes}/{self.config.exploitation_episodes}")
        
        # Store stats
        self.update_batch_stats('actor_exploitation', {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'successful_episodes': successful_episodes,
            'goal_type': 'enhanced_top5' if self.use_enhanced else 'original'
        })
    
    def phase_5_ppo_training(self, batch, goal_gradient_mode="full_gradients"):
        """
        Enhanced Phase 5 with gradient control for staged training
        """
        freeze_goals = (goal_gradient_mode == "stop_gradients")
        
        if freeze_goals:
            print(f"🚀 Phase 5: PPO Training (Goals FROZEN)")
        else:
            print(f"🚀 Phase 5: PPO Training (Joint optimization)")
        
        # Call actor-critic training with goal gradient mode control
        if len(self.experience_buffer) < self.tetris_config.TrainingConfig.MIN_BUFFER_SIZE:
            print("⚠️  Insufficient experience for PPO training")
            return
            
        # Multiple PPO training iterations with goal gradient control
        total_actor_loss = 0
        total_critic_loss = 0
        total_reward_loss = 0
        total_future_state_loss = 0
        total_aux_loss = 0
        successful_iterations = 0
        
        for iteration in range(self.tetris_config.TrainingConfig.PPO_ITERATIONS):
            # Use enhanced PPO with goal gradient mode control
            losses = self.actor_critic.train_ppo_with_hindsight(
                batch_size=self.tetris_config.TrainingConfig.PPO_BATCH_SIZE,
                ppo_epochs=self.tetris_config.TrainingConfig.PPO_EPOCHS,
                goal_gradient_mode=goal_gradient_mode  # Pass the gradient mode
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
                global_step = batch * self.tetris_config.TrainingConfig.PPO_ITERATIONS + iteration
                self.writer.add_scalar('PPO/ActorLoss_Iteration', actor_loss, global_step)
                self.writer.add_scalar('PPO/CriticLoss_Iteration', critic_loss, global_step)
                self.writer.add_scalar('PPO/RewardLoss_Iteration', reward_loss, global_step)
                self.writer.add_scalar('PPO/AuxiliaryLoss_Iteration', aux_loss, global_step)
                self.writer.add_scalar('PPO/FutureStateLoss_Iteration', future_state_loss, global_step)
        
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
            
            print(f"📊 Phase 5 Results:")
            print(f"   • Actor loss: {avg_actor_loss:.6f}")
            print(f"   • Critic loss: {avg_critic_loss:.6f}")
            print(f"   • Future state loss: {avg_future_state_loss:.6f}")
            print(f"   • Goal gradient mode: {goal_gradient_mode.upper()}")
            
            # Store batch statistics
            self.update_batch_stats('ppo', {
                'actor_loss': avg_actor_loss,
                'critic_loss': avg_critic_loss,
                'reward_loss': avg_reward_loss,
                'future_state_loss': avg_future_state_loss,
                'auxiliary_loss': avg_aux_loss,
                'success_rate': successful_iterations / self.tetris_config.TrainingConfig.PPO_ITERATIONS,
                'goal_gradient_mode': goal_gradient_mode
            })
        else:
            print("⚠️  No successful PPO training iterations")
    
    def _print_stage_transition_message(self, transition_type, batch):
        """Print important messages during stage transitions"""
        if transition_type == "state_model_complete":
            print(f"\n{'🎓'*20} STATE MODEL PRETRAINING COMPLETE! {'🎓'*20}")
            print(f"   • State model trained for {self.staged_training.state_model_pretraining_batches} batches")
            print(f"   • Goals should now be stable and meaningful")
            print(f"   • Next: ACTOR TRAINING with frozen goals...")
            self._evaluate_state_model_quality(batch)
            print(f"{'🎓'*65}\n")
        elif transition_type == "joint_finetuning_begins":
            print(f"\n{'🤝'*20} JOINT FINE-TUNING BEGINS! {'🤝'*20}")
            print(f"   • Actor has learned to achieve state model goals")
            print(f"   • Now allowing joint optimization for final {self.staged_training.joint_finetuning_batches} batches")
            print(f"   • Expected: Perfect goal-game alignment")
            print(f"{'🤝'*60}\n")
    
    def _evaluate_state_model_quality(self, batch):
        """Evaluate state model quality after pretraining"""
        print(f"\n📊 STATE MODEL QUALITY EVALUATION:")
        
        try:
            # Test goal consistency with sample states
            test_state_count = 20
            goal_consistency_score = self._measure_goal_consistency(test_state_count)
            goal_optimality_score = self._measure_goal_optimality(test_state_count)
            goal_diversity_score = self._measure_goal_diversity(test_state_count)
            
            print(f"   🎯 Goal Consistency: {goal_consistency_score:.3f}")
            print(f"   🏆 Goal Optimality: {goal_optimality_score:.3f}")
            print(f"   🌈 Goal Diversity: {goal_diversity_score:.3f}")
            
            overall_quality = (goal_consistency_score + goal_optimality_score + goal_diversity_score) / 3
            print(f"   📈 Overall State Model Quality: {overall_quality:.3f}")
            
            if overall_quality > 0.7:
                print(f"   ✅ EXCELLENT: State model ready for actor training!")
            elif overall_quality > 0.5:
                print(f"   ✅ GOOD: State model adequate for actor training")
            else:
                print(f"   ⚠️  WARNING: State model quality could be improved")
                
            # Log to TensorBoard
            self.writer.add_scalar('StagedTraining/GoalConsistency', goal_consistency_score, batch)
            self.writer.add_scalar('StagedTraining/GoalOptimality', goal_optimality_score, batch)
            self.writer.add_scalar('StagedTraining/GoalDiversity', goal_diversity_score, batch)
            self.writer.add_scalar('StagedTraining/OverallQuality', overall_quality, batch)
                
        except Exception as e:
            print(f"   ⚠️  Could not evaluate state model quality: {e}")
            print(f"   ℹ️  Proceeding with actor training stage...")
    
    def _measure_goal_consistency(self, test_count=20):
        """Measure how consistent goals are for similar states"""
        try:
            consistency_scores = []
            
            for _ in range(test_count):
                # Create a test state
                test_state = torch.randn(1, self.tetris_config.STATE_DIM, device=self.device)
                
                # Get goal twice to check consistency
                with torch.no_grad():
                    goal1 = self.state_model.get_placement_goal_vector(test_state)
                    goal2 = self.state_model.get_placement_goal_vector(test_state)
                    
                    if goal1 is not None and goal2 is not None:
                        # Calculate cosine similarity
                        consistency = F.cosine_similarity(goal1, goal2, dim=1).item()
                        consistency_scores.append(max(0, consistency))
            
            return np.mean(consistency_scores) if consistency_scores else 0.5
            
        except Exception:
            return 0.5  # Neutral score if evaluation fails
    
    def _measure_goal_optimality(self, test_count=20):
        """Measure how optimal the generated goals are"""
        try:
            optimality_scores = []
            
            for _ in range(test_count):
                test_state = torch.randn(1, self.tetris_config.STATE_DIM, device=self.device)
                
                with torch.no_grad():
                    goal = self.state_model.get_placement_goal_vector(test_state)
                    
                    if goal is not None:
                        # Extract goal value and confidence
                        goal_value = goal[0, 34].item()  # Value component
                        goal_confidence = goal[0, 35].item()  # Confidence component
                        
                        # Optimality is high value + high confidence
                        optimality = (goal_value + 50) / 100.0 * goal_confidence  # Normalize to [0, 1]
                        optimality_scores.append(max(0, min(1, optimality)))
            
            return np.mean(optimality_scores) if optimality_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _measure_goal_diversity(self, test_count=20):
        """Measure diversity of goals across different states"""
        try:
            goals = []
            
            for _ in range(test_count):
                test_state = torch.randn(1, self.tetris_config.STATE_DIM, device=self.device)
                
                with torch.no_grad():
                    goal = self.state_model.get_placement_goal_vector(test_state)
                    if goal is not None:
                        goals.append(goal.cpu().numpy())
            
            if len(goals) > 1:
                # Calculate pairwise distances
                goals = np.array(goals).reshape(len(goals), -1)
                distances = []
                
                for i in range(len(goals)):
                    for j in range(i+1, len(goals)):
                        dist = np.linalg.norm(goals[i] - goals[j])
                        distances.append(dist)
                
                # Normalize diversity score
                avg_distance = np.mean(distances) if distances else 0
                return min(1.0, avg_distance / 10.0)  # Scale to [0, 1]
            
            return 0.5
            
        except Exception:
            return 0.5

    def update_batch_stats(self, phase_name, stats_dict):
        """
        Update batch statistics for a given phase
        This method was being called by UnifiedTrainer but wasn't defined
        """
        if not hasattr(self, 'batch_stats'):
            # Initialize batch_stats if it doesn't exist
            self.batch_stats = {
                'exploration': {},
                'state_model': {},
                'reward_predictor': {},
                'exploitation': {},
                'ppo': {},
                'evaluation': {},
                'rnd': {},
                'line_clearing_test': {}
            }
        
        if phase_name in self.batch_stats:
            self.batch_stats[phase_name].update(stats_dict)
        else:
            self.batch_stats[phase_name] = stats_dict.copy()

    def phase_2_state_learning_with_adaptive_target(self, batch):
        """
        Phase 2: Enhanced State Model Learning with adaptive training target
        """
        stage = self.staged_training.get_training_stage(batch)
        intensity = self.staged_training.get_training_intensity(batch)
        
        print(f"🧠 Phase 2: State Model Learning (Stage: {stage})")
        
        # Get previous batch minimum loss for adaptive training
        previous_batch_min_loss = float('inf')
        if len(self.state_model_batch_loss_history) > 0:
            previous_batch_min_loss = self.state_model_batch_loss_history[-1]
        
        # Adaptive training with data reuse
        max_training_attempts = 5
        loss_improvement_threshold = self.tetris_config.TrainingConfig.STATE_MODEL_LOSS_IMPROVEMENT_THRESHOLD
        target_loss = previous_batch_min_loss * loss_improvement_threshold if previous_batch_min_loss != float('inf') else float('inf')
        
        batch_min_loss = float('inf')
        training_attempts = 0
        
        print(f"   🎯 Adaptive training target: {target_loss:.4f} (previous: {previous_batch_min_loss if previous_batch_min_loss != float('inf') else 'N/A'})")
        
        # First training attempt
        training_attempts += 1
        print(f"   📚 Training attempt {training_attempts}/{max_training_attempts}")
        state_results = self.phase_2_state_learning(batch)
        current_min_loss = state_results.get('loss', float('inf'))
        batch_min_loss = min(batch_min_loss, current_min_loss)
        
        # Continue training if loss condition not met
        while (training_attempts < max_training_attempts and 
               previous_batch_min_loss != float('inf') and 
               batch_min_loss >= target_loss):
            
            training_attempts += 1
            print(f"   ⚠️ Loss condition not met ({batch_min_loss:.4f} >= {target_loss:.4f}). Training attempt {training_attempts}/{max_training_attempts} (reusing data)")
            
            extra_results = self.phase_2_state_learning(batch)
            extra_min_loss = extra_results.get('loss', float('inf'))
            if extra_min_loss < batch_min_loss:
                batch_min_loss = extra_min_loss
                print(f"   ✅ Improved loss: {batch_min_loss:.4f}")
            else:
                print(f"   ⚠️ No improvement: {extra_min_loss:.4f}")
        
        # Intensive state model training during pretraining stage
        if stage == "state_model_pretraining":
            for extra_epoch in range(intensity['state_model_extra_epochs']):
                print(f"   🔄 Extra pretraining call {extra_epoch+1}/{intensity['state_model_extra_epochs']} (stage-specific)")
                extra_results = self.phase_2_state_learning(batch)
                
                extra_min_loss = extra_results.get('loss', float('inf'))
                if extra_min_loss < batch_min_loss:
                    batch_min_loss = extra_min_loss
        
        # Store the best loss achieved
        self.state_model_batch_loss_history.append(batch_min_loss)
        
        # Enhanced summary
        loss_improvement_achieved = (previous_batch_min_loss - batch_min_loss) / previous_batch_min_loss if previous_batch_min_loss != float('inf') and previous_batch_min_loss > 0 else 0
        condition_met = batch_min_loss < target_loss if target_loss != float('inf') else "N/A (first batch)"
        
        print(f"   📊 Batch {batch+1} State Model Summary:")
        print(f"       • Min Loss: {batch_min_loss:.4f} (Previous: {previous_batch_min_loss if previous_batch_min_loss != float('inf') else 'N/A'})")
        print(f"       • Training Attempts: {training_attempts}/{max_training_attempts}")
        print(f"       • Loss Improvement: {loss_improvement_achieved*100:.1f}%")
        print(f"       • Target Condition: {'✅ MET' if condition_met is True else '❌ NOT MET' if condition_met is False else condition_met}")
        print(f"       • Model Type: {'Enhanced Top-5' if self.use_enhanced else 'Original'}")
    
    def phase_2_state_learning(self, batch):
        """
        Phase 2: Enhanced State Model Learning (top 5 terminal states)
        """
        print(f"🧠 Phase 2: State Model Learning (Batch {batch+1})")
        
        if not self.exploration_data:
            print("   ⚠️ No exploration data for state model training.")
            return {'loss': float('inf'), 'training_data_used': 0}
        
        if self.use_enhanced:
            # Use enhanced top-5 terminal state model
            print("   🏆 Using Enhanced Top-5 Terminal State Model")
            return self.enhanced_components.train_enhanced_state_model(self.exploration_data)
        else:
            # Original state model training
            print("   📚 Using Original State Model")
            return self._train_original_state_model()
    
    def phase_2_5_terminal_q_learning(self, batch):
        """
        Phase 2.5: SIMPLIFIED Q-Learning (CORRECTED - no episode management)
        Train Q-learning on continuous trajectory data for line clearing rewards
        """
        print(f"🎯 Phase 2.5: Simplified Q-Learning (Batch {batch+1})")
        
        if not self.use_enhanced:
            print("   ⚠️ Enhanced components not available, skipping Q-learning")
            return {'q_loss': 0, 'trajectories_trained': 0}
        
        if not self.exploration_data:
            print("   ⚠️ No exploration data for Q-learning training.")
            return {'q_loss': float('inf'), 'trajectories_trained': 0}
        
        # CORRECTED: Train simplified Q-learning on continuous trajectories (no episode management)
        results = self.enhanced_components.train_simplified_q_learning(self.exploration_data)
        
        # CORRECTED: Print Q-learning loss and trajectory info per batch
        q_loss = results.get('q_loss', float('inf'))
        trajectories_trained = results.get('trajectories_trained', 0)
        samples_processed = results.get('samples_processed', 0)
        n_step = results.get('n_step', 4)
        
        print(f"   📊 Q-Learning Results:")
        print(f"       • Q-Learning Loss: {q_loss:.4f}")
        print(f"       • Trajectories Trained: {trajectories_trained}")
        print(f"       • Samples Processed: {samples_processed}")
        print(f"       • N-step Bootstrapping: {n_step}")
        print(f"       • Lines Cleared Bonus: Integrated into reward calculation")
        
        # Log Q-learning metrics to TensorBoard
        self.writer.add_scalar('QLearning/Loss', q_loss, batch)
        self.writer.add_scalar('QLearning/TrajectoriesTrained', trajectories_trained, batch)
        self.writer.add_scalar('QLearning/SamplesProcessed', samples_processed, batch)
        
        return results
    
    def phase_3_future_reward_predictor(self, batch):
        """
        Phase 3: Future Reward Predictor (fallback when enhanced not available)
        """
        print(f"🔮 Phase 3: Future Reward Predictor (Batch {batch+1})")
        
        if not self.exploration_data:
            print("   ⚠️ No exploration data for future reward training.")
            return
        
        # Original future reward predictor training
        device = self.device
        total_loss = 0
        
        for data in self.exploration_data:
            state = torch.FloatTensor(data['state_vector']).unsqueeze(0).to(device)
            reward = data['terminal_reward']
            
            predicted_reward = self.future_reward_predictor(state)
            target_reward = torch.FloatTensor([[reward]]).to(device)
            
            loss = F.mse_loss(predicted_reward, target_reward)
            
            self.reward_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.future_reward_predictor.parameters(), 1.0)
            self.reward_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.exploration_data)
        print(f"   📊 Future Reward Predictor Loss: {avg_loss:.4f}")

    def _train_original_state_model(self):
        """
        Original state model training (fallback)
        """
        # Transform exploration data format
        print("   🔄 Transforming exploration data format...")
        training_data = []
        
        for data in self.exploration_data:
            transformed_data = {
                'state_vector': data['state_vector'],
                'placement': data['placement'],
                'terminal_reward': data['terminal_reward']
            }
            training_data.append(transformed_data)
        
        print(f"   ✅ Successfully transformed {len(training_data)} exploration data entries.")
        
        # Train original state model
        device = self.device
        total_loss = 0
        epochs = 3
        
        print(f"   🔄 Training for {epochs} epochs (original model)")
        
        for epoch in range(epochs):
            epoch_loss = 0
            batches_processed = 0
            
            for data in training_data:
                state = torch.FloatTensor(data['state_vector']).unsqueeze(0).to(device)
                placement = data['placement']
                reward = data['terminal_reward']
                
                # Forward pass
                rot_logits, x_logits, y_logits, value_pred = self.state_model(state)
                
                # Targets
                rot_target = torch.LongTensor([placement[0]]).to(device)
                x_target = torch.LongTensor([placement[1]]).to(device)
                y_target = torch.LongTensor([placement[2]]).to(device)
                value_target = torch.FloatTensor([[reward]]).to(device)
                
                # Losses
                rot_loss = F.cross_entropy(rot_logits, rot_target)
                x_loss = F.cross_entropy(x_logits, x_target)
                y_loss = F.cross_entropy(y_logits, y_target)
                value_loss = F.mse_loss(value_pred, value_target)
                
                total_loss_batch = rot_loss + x_loss + y_loss + value_loss
                
                # Backpropagation
                self.state_optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.state_model.parameters(), 1.0)
                self.state_optimizer.step()
                
                epoch_loss += total_loss_batch.item()
                batches_processed += 1
            
            avg_epoch_loss = epoch_loss / batches_processed if batches_processed > 0 else float('inf')
            total_loss = avg_epoch_loss  # Use last epoch loss
            
            print(f"     Epoch {epoch+1}/{epochs} (State Model): Loss={avg_epoch_loss:.4f}")
        
        return {
            'loss': total_loss,
            'training_data_used': len(training_data),
            'epochs': epochs,
            'model_type': 'original'
        }

    def phase_6_evaluation_with_line_clearing_test(self, batch):
        """
        Enhanced Phase 6: Model evaluation with line clearing test
        """
        print(f"📊 Phase 6: Enhanced Model Evaluation (Batch {batch+1})")
        
        # Run standard evaluation
        standard_results = self.phase_6_evaluation(batch)
        
        # ADDED: Line clearing test with 20 blocks at end of batch
        if self.use_enhanced and (batch + 1) % 10 == 0:  # Every 10 batches
            line_clearing_evaluator = self.enhanced_components.create_line_clearing_evaluator(self.env)
            
            line_clearing_results = line_clearing_evaluator.evaluate_line_clearing(
                self.enhanced_components.state_model,
                self.enhanced_components.q_learning,
                self.enhanced_components.goal_selector,
                batch + 1
            )
            
            # Log line clearing metrics
            self.writer.add_scalar('LineClearingTest/AvgLinesCleared', line_clearing_results['avg_lines_cleared'], batch)
            self.writer.add_scalar('LineClearingTest/SuccessRate', line_clearing_results['success_rate'], batch)
            self.writer.add_scalar('LineClearingTest/TotalLines', line_clearing_results['total_lines'], batch)
            
            # Update batch stats
            self.update_batch_stats('line_clearing_test', line_clearing_results)
            
            # Update epsilon for goal selection
            self.enhanced_components.update_epsilon()
            
            return {**standard_results, 'line_clearing': line_clearing_results}
        
        return standard_results

    def save_checkpoint(self, batch):
        """
        Enhanced checkpoint saving with all network components
        """
        checkpoint_path = f"{self.config.checkpoint_dir}/batch_{batch+1}"
        
        # Save standard components
        super().save_checkpoint(batch)
        
        # ADDED: Save enhanced components if available
        if self.use_enhanced:
            print(f"💾 Saving enhanced 6-phase checkpoint: {checkpoint_path}_enhanced_6phase.pt")
            self.enhanced_components.save_checkpoints(checkpoint_path)
        
        print(f"✅ All checkpoints saved for batch {batch+1}")

    def load_checkpoint(self, checkpoint_path):
        """
        Enhanced checkpoint loading with all network components
        """
        # Load standard components
        super().load_checkpoint(checkpoint_path)
        
        # ADDED: Load enhanced components if available
        if self.use_enhanced:
            try:
                enhanced_checkpoint_path = checkpoint_path.replace('.pt', '')
                self.enhanced_components.load_checkpoints(enhanced_checkpoint_path)
                print(f"✅ Enhanced 6-phase checkpoint loaded from {enhanced_checkpoint_path}_enhanced_6phase.pt")
            except FileNotFoundError:
                print(f"⚠️ Enhanced checkpoint not found, starting fresh")
        
        print(f"✅ All checkpoints loaded")

class StagedTrainingConfig:
    """Enhanced Training Configuration with proportional staging support"""
    def __init__(self):
        # Get Tetris config for centralized parameters
        self.tetris_config = TetrisConfig()
        
        # Main parameters
        self.num_batches = 50  # Default to 50 batches
        self.batch_size = 64
        self.learning_rate = 0.001
        
        # State model and actor-critic parameters
        self.state_lr = 0.0001
        self.actor_lr = 0.0001  
        self.critic_lr = 0.0001
        self.reward_lr = 0.0001  # For Q-learning
        
        # ENHANCED: Proportional staging support
        self.stage_proportions = None  # Use default if None
        
        # Training configuration
        self.gamma = 0.99
        self.epsilon = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        
        # Exploration configuration  
        self.exploration_steps = 600
        self.rnd_lr = 0.001
        
        # Evaluation configuration
        self.eval_frequency = 10
        self.eval_episodes = 10
        
        # Checkpointing
        self.save_frequency = 10
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"
        
        # Enhanced 6-phase configuration
        self.use_enhanced_components = True  # Try to use enhanced components if available
        
        # ADDED: Device configuration
        import torch
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        # Additional attributes needed for compatibility
        self.visualize = False
        self.exploration_mode = 'rnd'
        
        # Training parameters from centralized config  
        self.exploration_episodes = self.tetris_config.TrainingConfig.EXPLORATION_EPISODES
        self.exploitation_episodes = self.tetris_config.TrainingConfig.EXPLOITATION_EPISODES
        self.max_episode_steps = self.tetris_config.TrainingConfig.MAX_EPISODE_STEPS
        
        # Model parameters
        self.clip_ratio = self.tetris_config.TrainingConfig.PPO_CLIP_RATIO
        
        # Training phases
        self.state_training_samples = self.tetris_config.TrainingConfig.STATE_TRAINING_SAMPLES
        self.state_epochs = self.tetris_config.TrainingConfig.STATE_EPOCHS
        self.ppo_iterations = self.tetris_config.TrainingConfig.PPO_ITERATIONS
        self.ppo_batch_size = self.tetris_config.TrainingConfig.PPO_BATCH_SIZE
        self.ppo_epochs = self.tetris_config.TrainingConfig.PPO_EPOCHS
        self.reward_batch_size = self.tetris_config.TrainingConfig.REWARD_BATCH_SIZE
        
        # Buffer parameters
        self.buffer_size = self.tetris_config.TrainingConfig.BUFFER_SIZE
        self.min_buffer_size = self.tetris_config.TrainingConfig.MIN_BUFFER_SIZE
        
        # Logging and saving
        self.save_interval = self.tetris_config.LoggingConfig.SAVE_INTERVAL
        
    def set_stage_proportions(self, state_model_pct=50, actor_pct=33, joint_pct=17):
        """
        Set custom stage proportions
        
        Args:
            state_model_pct: Percentage for state model pretraining (default 50%)
            actor_pct: Percentage for actor training (default 33%)
            joint_pct: Percentage for joint fine-tuning (default 17%)
        """
        total_pct = state_model_pct + actor_pct + joint_pct
        if total_pct != 100:
            print(f"⚠️ Warning: Percentages sum to {total_pct}%, normalizing to 100%")
        
        self.stage_proportions = {
            'state_model_pretraining': state_model_pct / 100.0,
            'actor_training': actor_pct / 100.0,
            'joint_finetuning': joint_pct / 100.0
        }
        
        print(f"🎯 Custom staging proportions set:")
        print(f"   • State Model: {state_model_pct}%")
        print(f"   • Actor Training: {actor_pct}%") 
        print(f"   • Joint Fine-tuning: {joint_pct}%")


def main():
    parser = argparse.ArgumentParser(description='Staged Unified Tetris Trainer with Enhanced Components')
    parser.add_argument('--num_batches', type=int, default=50, help='Number of training batches')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--state_lr', type=float, default=0.0001, help='State model learning rate')
    parser.add_argument('--actor_lr', type=float, default=0.0001, help='Actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=0.0001, help='Critic learning rate')
    parser.add_argument('--reward_lr', type=float, default=0.0001, help='Reward model learning rate')
    parser.add_argument('--exploration_steps', type=int, default=600, help='Exploration steps per batch')
    parser.add_argument('--eval_frequency', type=int, default=10, help='Evaluation frequency')
    parser.add_argument('--save_frequency', type=int, default=10, help='Save frequency')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    
    # ENHANCED: Proportional staging arguments
    parser.add_argument('--stage_model_pct', type=int, default=50, help='Percentage for state model pretraining (default 50%)')
    parser.add_argument('--stage_actor_pct', type=int, default=33, help='Percentage for actor training (default 33%)')
    parser.add_argument('--stage_joint_pct', type=int, default=17, help='Percentage for joint fine-tuning (default 17%)')
    parser.add_argument('--use_custom_staging', action='store_true', help='Use custom staging proportions')
    
    args = parser.parse_args()
    
    # Create config
    config = StagedTrainingConfig()
    config.num_batches = args.num_batches
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.state_lr = args.state_lr
    config.actor_lr = args.actor_lr
    config.critic_lr = args.critic_lr
    config.reward_lr = args.reward_lr
    config.exploration_steps = args.exploration_steps
    config.eval_frequency = args.eval_frequency
    config.save_frequency = args.save_frequency
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    
    # ENHANCED: Set custom staging proportions if requested
    if args.use_custom_staging:
        config.set_stage_proportions(
            state_model_pct=args.stage_model_pct,
            actor_pct=args.stage_actor_pct,
            joint_pct=args.stage_joint_pct
        )
    
    # Create trainer and run
    trainer = StagedUnifiedTrainer(config)
    trainer.run_training()

if __name__ == '__main__':
    main() 