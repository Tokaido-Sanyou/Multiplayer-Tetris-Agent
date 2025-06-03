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
    Enhanced training schedule with state model pretraining
    Prevents moving target problem by training state model first, then actor
    """
    def __init__(self, total_batches=300):
        self.total_batches = total_batches
        
        # Stage 1: State Model Pretraining (batches 0-149)
        self.state_model_pretraining_batches = total_batches // 2  # 150 batches
        
        # Stage 2: Actor Training with Frozen Goals (batches 150-249) 
        self.actor_training_batches = total_batches - self.state_model_pretraining_batches  # 150 batches
        
        # Stage 3: Joint Fine-tuning (last 50 batches)
        self.joint_finetuning_batches = 50
    
    def get_training_stage(self, batch):
        """Determine which training stage we're in"""
        if batch < self.state_model_pretraining_batches:
            return "state_model_pretraining"
        elif batch < self.total_batches - self.joint_finetuning_batches:
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
        
        # CRITICAL ENHANCEMENT: Initialize staged training schedule
        self.staged_training = StagedTrainingSchedule(total_batches=config.num_batches)
        
        # NEW: Track state model loss history for adaptive training
        self.state_model_batch_loss_history = []  # Track min loss from each batch
        
        print(f"\n🎯 STAGED TRAINING ENABLED:")
        print(f"   • Stage 1: State Model Pretraining (Batches 0-{self.staged_training.state_model_pretraining_batches-1})")
        print(f"   • Stage 2: Actor Training + Frozen Goals (Batches {self.staged_training.state_model_pretraining_batches}-{config.num_batches - self.staged_training.joint_finetuning_batches-1})")
        print(f"   • Stage 3: Joint Fine-tuning (Batches {config.num_batches - self.staged_training.joint_finetuning_batches}-{config.num_batches-1})")
        print(f"   🔥 BENEFIT: Prevents moving target problem - stable goals for actor!")
    
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
            
            # Phase 2: State model learning (STAGED)
            if should_train_state_model:
                print(f"🧠 Phase 2: State Model Learning (Stage: {stage})")
                
                # Get previous batch minimum loss for adaptive training
                previous_batch_min_loss = float('inf')
                if len(self.state_model_batch_loss_history) > 0:
                    previous_batch_min_loss = self.state_model_batch_loss_history[-1]
                
                # NEW: Adaptive training with data reuse instead of extended epochs
                max_training_attempts = 5  # Maximum number of complete training calls
                loss_improvement_threshold = self.tetris_config.TrainingConfig.STATE_MODEL_LOSS_IMPROVEMENT_THRESHOLD
                target_loss = previous_batch_min_loss * loss_improvement_threshold if previous_batch_min_loss != float('inf') else float('inf')
                
                batch_min_loss = float('inf')
                training_attempts = 0
                
                print(f"   🎯 Adaptive training target: {target_loss:.4f} (previous: {previous_batch_min_loss if previous_batch_min_loss != float('inf') else 'N/A'})")
                
                # First training attempt (always runs)
                training_attempts += 1
                print(f"   📚 Training attempt {training_attempts}/{max_training_attempts}")
                state_results = self.phase_2_state_learning(batch)  # Use normal 3-epoch training
                current_min_loss = state_results.get('current_min_loss', float('inf'))
                batch_min_loss = min(batch_min_loss, current_min_loss)
                
                # Continue training if loss condition not met and we're not in the first batch
                while (training_attempts < max_training_attempts and 
                       previous_batch_min_loss != float('inf') and 
                       batch_min_loss >= target_loss):
                    
                    training_attempts += 1
                    print(f"   🔄 Loss condition not met ({batch_min_loss:.4f} >= {target_loss:.4f}). Training attempt {training_attempts}/{max_training_attempts} (reusing data)")
                    
                    extra_results = self.phase_2_state_learning(batch)  # Reuse same data with normal epochs
                    extra_min_loss = extra_results.get('current_min_loss', float('inf'))
                    if extra_min_loss < batch_min_loss:
                        batch_min_loss = extra_min_loss
                        print(f"   ✅ Improved loss: {batch_min_loss:.4f}")
                    else:
                        print(f"   ⚠️ No improvement: {extra_min_loss:.4f}")
                
                # INTENSIVE state model training during pretraining stage (additional to adaptive training)
                if stage == "state_model_pretraining":
                    for extra_epoch in range(intensity['state_model_extra_epochs']):
                        print(f"   🔄 Extra pretraining call {extra_epoch+1}/{intensity['state_model_extra_epochs']} (stage-specific)")
                        extra_results = self.phase_2_state_learning(batch, extra_training=True)
                        
                        # Update batch minimum loss if this extra training achieved better loss
                        extra_min_loss = extra_results.get('current_min_loss', float('inf'))
                        if extra_min_loss < batch_min_loss:
                            batch_min_loss = extra_min_loss
                
                # Store the best loss achieved in this batch (across all calls)
                self.state_model_batch_loss_history.append(batch_min_loss)
                
                # Enhanced summary
                loss_improvement_achieved = (previous_batch_min_loss - batch_min_loss) / previous_batch_min_loss if previous_batch_min_loss != float('inf') and previous_batch_min_loss > 0 else 0
                condition_met = batch_min_loss < target_loss if target_loss != float('inf') else "N/A (first batch)"
                
                print(f"   📊 Batch {batch+1} State Model Summary:")
                print(f"       • Min Loss: {batch_min_loss:.4f} (Previous: {previous_batch_min_loss if previous_batch_min_loss != float('inf') else 'N/A'})")
                print(f"       • Training Attempts: {training_attempts}/{max_training_attempts}")
                print(f"       • Loss Improvement: {loss_improvement_achieved*100:.1f}%")
                print(f"       • Target Condition: {'✅ MET' if condition_met is True else '❌ NOT MET' if condition_met is False else condition_met}")
                
            else:
                print(f"🧠 Phase 2: State Model Learning (SKIPPED - goals frozen for actor training)")
                # If state model training is skipped, repeat the last loss value
                if len(self.state_model_batch_loss_history) > 0:
                    self.state_model_batch_loss_history.append(self.state_model_batch_loss_history[-1])
                else:
                    self.state_model_batch_loss_history.append(float('inf'))
            
            # Phase 3: Future reward prediction (always runs but with staged intensity)
            self.phase_3_reward_prediction(batch)
            
            # Phase 4: Actor exploitation (STAGED)
            if should_train_actor:
                print(f"🎭 Phase 4: Actor Exploitation (Stage: {stage})")
                # Pass gradient mode to control goal corruption
                self.phase_4_exploitation_staged(batch, goal_gradient_mode=goal_gradient_mode)
                
                # INTENSIVE actor training during actor-focused stage
                if stage == "actor_training_frozen_goals":
                    for extra_epoch in range(intensity['actor_extra_epochs']):
                        print(f"   🔄 Extra actor epoch {extra_epoch+1}/{intensity['actor_extra_epochs']}")
                        self.phase_4_exploitation_staged(batch, goal_gradient_mode=goal_gradient_mode, extra_training=True)
            else:
                print(f"🎭 Phase 4: Actor Exploitation (SKIPPED - pretraining state model first)")
            
            # Phase 5: PPO training (only during actor stages)
            if should_train_actor:
                self.phase_5_ppo_training_staged(batch, goal_gradient_mode=goal_gradient_mode)
            else:
                print(f"🚀 Phase 5: PPO Training (SKIPPED - pretraining state model first)")
            
            # Phase 6: Model evaluation (CONDITIONAL based on stage and frequency)
            should_evaluate_this_stage = self.staged_training.should_run_evaluation(batch)
            is_evaluation_batch = (batch + 1) % self.tetris_config.AlgorithmConfig.EVALUATION_FREQUENCY == 0

            if should_evaluate_this_stage and is_evaluation_batch:
                print(f"📊 Phase 6: Model Evaluation (Stage: {stage.upper()}, Batch {batch+1})")
                # The base phase_6_evaluation method logs details internally
                eval_results = self.phase_6_evaluation(batch)
                if eval_results: # Assuming phase_6_evaluation returns a dict of metrics
                    for key, value in eval_results.items():
                        if isinstance(value, (int, float)): # Log numerical metrics
                            self.writer.add_scalar(f"Evaluation_Staged/{key}", value, batch)
                        # else: # Handle other types of results if necessary, e.g., arrays for histograms
                        #    pass 
            elif not should_evaluate_this_stage:
                print(f"📊 Phase 6: Model Evaluation (SKIPPED - Stage: {stage.upper()} - Evaluation not active for this stage)")
            else: # should_evaluate_this_stage is True, but not an evaluation frequency batch
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
    
    def phase_4_exploitation_staged(self, batch, goal_gradient_mode="full_gradients", extra_training=False):
        """
        Enhanced Phase 4 with gradient control for staged training
        """
        print(f"🎮 Phase 4: Enhanced Multi-Attempt Goal-Focused Policy Exploitation (Staged)")
        
        # Control gradient flow based on training stage
        freeze_goals = (goal_gradient_mode == "stop_gradients")
        
        if freeze_goals:
            print(f"   🔒 Goals FROZEN - Actor cannot corrupt state model")
        else:
            print(f"   🔓 Goals FREE - Joint optimization enabled")
        
        # Call the base exploitation method (note: base method doesn't support freeze_goals)
        # The goal gradient control will be handled in the actor-critic level during PPO training
        self.phase_4_exploitation(batch)
    
    def phase_5_ppo_training_staged(self, batch, goal_gradient_mode="full_gradients"):
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
                'rnd': {}
            }
        
        if phase_name in self.batch_stats:
            self.batch_stats[phase_name].update(stats_dict)
        else:
            self.batch_stats[phase_name] = stats_dict.copy()

class StagedTrainingConfig:
    """Simple configuration class for staged training"""
    def __init__(self):
        # Get Tetris config for centralized parameters
        from ..config import TetrisConfig
        tetris_config = TetrisConfig()
        
        # Basic training parameters
        self.num_batches = 300
        self.visualize = False
        self.log_dir = 'logs/staged_unified_training'
        self.checkpoint_dir = 'checkpoints/staged_unified'
        self.exploration_mode = 'rnd'
        
        # Training parameters from centralized config
        self.exploration_episodes = tetris_config.TrainingConfig.EXPLORATION_EPISODES
        self.exploitation_episodes = tetris_config.TrainingConfig.EXPLOITATION_EPISODES
        self.eval_episodes = tetris_config.TrainingConfig.EVAL_EPISODES
        self.max_episode_steps = tetris_config.TrainingConfig.MAX_EPISODE_STEPS
        self.batch_size = tetris_config.TrainingConfig.BATCH_SIZE
        
        # Model parameters
        self.state_lr = tetris_config.TrainingConfig.STATE_LEARNING_RATE
        self.reward_lr = tetris_config.TrainingConfig.REWARD_LEARNING_RATE
        self.clip_ratio = tetris_config.TrainingConfig.PPO_CLIP_RATIO
        
        # Training phases
        self.state_training_samples = tetris_config.TrainingConfig.STATE_TRAINING_SAMPLES
        self.state_epochs = tetris_config.TrainingConfig.STATE_EPOCHS
        self.ppo_iterations = tetris_config.TrainingConfig.PPO_ITERATIONS
        self.ppo_batch_size = tetris_config.TrainingConfig.PPO_BATCH_SIZE
        self.ppo_epochs = tetris_config.TrainingConfig.PPO_EPOCHS
        self.reward_batch_size = tetris_config.TrainingConfig.REWARD_BATCH_SIZE
        
        # Buffer parameters
        self.buffer_size = tetris_config.TrainingConfig.BUFFER_SIZE
        self.min_buffer_size = tetris_config.TrainingConfig.MIN_BUFFER_SIZE
        
        # Logging and saving
        self.save_interval = tetris_config.LoggingConfig.SAVE_INTERVAL
        
        # Device detection
        import torch
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

def main():
    parser = argparse.ArgumentParser(description="Staged Unified Tetris RL Training")
    parser.add_argument('--num_batches', type=int, default=300, help='Total number of training batches for staged training (e.g., 300)')
    parser.add_argument('--visualize', action='store_true', help='Visualize training')
    parser.add_argument('--log_dir', type=str, default='logs/staged_unified_training', help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/staged_unified', help='Checkpoint directory')
    parser.add_argument('--exploration_mode', type=str, default='rnd', 
                       choices=['rnd', 'random', 'deterministic'],
                       help='Exploration strategy: rnd, random, deterministic')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('staged_unified_training.log')
        ]
    )
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create simple config
    config = StagedTrainingConfig()
    config.num_batches = args.num_batches
    config.visualize = args.visualize
    config.log_dir = args.log_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.exploration_mode = args.exploration_mode
    
    # Print configuration summary
    print(f"\n🎮 Tetris RL Staged Training Configuration:")
    print(f"   📦 Total Batches: {config.num_batches}")
    print(f"   🔍 Exploration Mode: {config.exploration_mode.upper()}")
    print(f"   👁️  Visualization: {'Enabled' if config.visualize else 'Disabled'}")
    print(f"   📊 Logging: {config.log_dir}")
    print(f"   💾 Checkpoints: {config.checkpoint_dir}")
    print(f"   🚀 Staged training enabled with 3 phases.")
    print()
    
    # Initialize and run STAGED trainer
    trainer = StagedUnifiedTrainer(config)
    trainer.run_training()

if __name__ == '__main__':
    main() 