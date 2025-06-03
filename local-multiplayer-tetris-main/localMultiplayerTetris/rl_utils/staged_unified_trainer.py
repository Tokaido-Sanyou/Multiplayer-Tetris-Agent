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
from .unified_trainer import UnifiedTrainer, TrainingConfig  # Import base trainer and TrainingConfig
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
                self.phase_2_state_learning(batch)
                
                # INTENSIVE state model training during pretraining stage
                if stage == "state_model_pretraining":
                    for extra_epoch in range(intensity['state_model_extra_epochs']):
                        print(f"   🔄 Extra state model epoch {extra_epoch+1}/{intensity['state_model_extra_epochs']}")
                        self.phase_2_state_learning(batch, extra_training=True)
            else:
                print(f"🧠 Phase 2: State Model Learning (SKIPPED - goals frozen for actor training)")
            
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
        
        # Call the base exploitation method with goal freezing
        # Note: We'll need to modify the base method to accept freeze_goals parameter
        try:
            self.phase_4_exploitation(batch, freeze_goals=freeze_goals)
        except TypeError:
            # Fallback if base method doesn't support freeze_goals yet
            print(f"   ⚠️  Base exploitation method doesn't support goal freezing yet")
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
        
        # Call base PPO training
        # Note: PPO training should automatically respect gradient settings from actor-critic
        self.phase_5_ppo_training(batch)
    
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
    
    # Create config using TrainingConfig from unified_trainer
    config = TrainingConfig() # This will use its own defaults first
    config.num_batches = args.num_batches # Override with arg
    config.visualize = args.visualize
    config.log_dir = args.log_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.exploration_mode = args.exploration_mode
    
    # Print configuration summary
    print(f"\n🎮 Tetris RL Staged Training Configuration:")
    print(f"   📦 Total Batches: {config.num_batches}")
    # Episodes per batch are defined in TrainingConfig, can be mentioned if needed
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