#!/usr/bin/env python3
"""
STAGED TRAINING PROPOSAL: State Model → Actor
Train state model first to learn good goals, then train actor to achieve them
"""

class StagedTrainingSchedule:
    """
    Enhanced training schedule with state model pretraining
    """
    def __init__(self, total_batches=300):
        self.total_batches = total_batches
        
        # Stage 1: State Model Pretraining (batches 0-149)
        self.state_model_pretraining_batches = total_batches // 2  # 150 batches
        
        # Stage 2: Actor Training with Frozen Goals (batches 150-299) 
        self.actor_training_batches = total_batches - self.state_model_pretraining_batches  # 150 batches
        
        # Optional Stage 3: Joint Fine-tuning (last 50 batches)
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


def enhanced_unified_trainer_with_staging(self, batch):
    """
    ENHANCED 6-PHASE TRAINING WITH STAGING
    """
    schedule = StagedTrainingSchedule()
    stage = schedule.get_training_stage(batch)
    
    print(f"🎯 BATCH {batch+1}/300 - STAGE: {stage.upper()}")
    print(f"   State Model Training: {'✅ ON' if schedule.should_train_state_model(batch) else '❌ OFF'}")
    print(f"   Actor Training: {'✅ ON' if schedule.should_train_actor(batch) else '❌ OFF'}")
    print(f"   Goal Gradients: {schedule.get_goal_gradient_mode(batch).upper()}")
    
    # Phase 1: Exploration (always runs)
    placement_data = self.phase_1_exploration(batch)
    
    # Phase 2: State Model Learning (staged)
    if schedule.should_train_state_model(batch):
        print(f"🧠 Phase 2: State Model Learning (Stage: {stage})")
        state_model_loss = self.phase_2_state_model_learning(batch, placement_data)
        
        if stage == "state_model_pretraining":
            # INTENSIVE state model training
            for extra_epoch in range(3):  # 3x more training
                extra_loss = self.phase_2_state_model_learning(batch, placement_data)
                print(f"   🔄 Extra epoch {extra_epoch+1}: loss {extra_loss:.4f}")
    else:
        print(f"🧠 Phase 2: State Model Learning (SKIPPED - using frozen goals)")
        state_model_loss = 0.0
    
    # Phase 3: Goal Generation (always runs, but with different gradient modes)
    goal_gradient_mode = schedule.get_goal_gradient_mode(batch)
    generated_goals = self.phase_3_goal_generation(batch, goal_gradient_mode)
    
    # Phase 4: Actor Training (staged)
    if schedule.should_train_actor(batch):
        print(f"🎭 Phase 4: Actor Exploitation (Stage: {stage})")
        if stage == "actor_training_frozen_goals":
            # INTENSIVE actor training with frozen goals
            for extra_epoch in range(2):  # 2x more training
                extra_loss = self.phase_4_exploitation(batch, freeze_goals=True)
                print(f"   🔄 Extra actor epoch {extra_epoch+1}: loss {extra_loss:.4f}")
        else:
            self.phase_4_exploitation(batch, freeze_goals=False)
    else:
        print(f"🎭 Phase 4: Actor Exploitation (SKIPPED - pretraining state model)")
    
    # Phase 5: PPO Training (only during actor stages)
    if schedule.should_train_actor(batch):
        self.phase_5_ppo_training(batch)
    else:
        print(f"🚀 Phase 5: PPO Training (SKIPPED - pretraining state model)")
    
    # Phase 6: Evaluation (always runs)
    self.phase_6_evaluation(batch)
    
    # STAGING BENEFITS ANALYSIS
    if batch == schedule.state_model_pretraining_batches:
        print(f"\n🎓 STATE MODEL PRETRAINING COMPLETE!")
        print(f"   • State model trained for {schedule.state_model_pretraining_batches} batches")
        print(f"   • Goals should now be stable and meaningful")
        print(f"   • Switching to ACTOR TRAINING with frozen goals...")
        
        # Evaluate state model quality
        self._evaluate_state_model_quality(batch)
    
    if batch == schedule.total_batches - schedule.joint_finetuning_batches:
        print(f"\n🤝 JOINT FINE-TUNING BEGINS!")
        print(f"   • Actor has learned to achieve state model goals")
        print(f"   • Now allowing joint optimization for final {schedule.joint_finetuning_batches} batches")


def _evaluate_state_model_quality(self, batch):
    """Evaluate state model quality after pretraining"""
    print(f"\n📊 STATE MODEL QUALITY EVALUATION:")
    
    # Test goal consistency
    test_states = self._generate_test_states(100)
    goal_consistency_score = self._measure_goal_consistency(test_states)
    
    # Test goal optimality  
    goal_optimality_score = self._measure_goal_optimality(test_states)
    
    # Test goal diversity
    goal_diversity_score = self._measure_goal_diversity(test_states)
    
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
        print(f"   ⚠️  WARNING: State model quality low - consider more pretraining")


def modified_goal_encoder_with_gradient_control(self, state, goal, freeze_goals=False):
    """
    Modified goal encoder with gradient control
    """
    if freeze_goals:
        # Stop gradients from actor corrupting state model goals
        goal_feat = self.goal_encoder(goal.detach())
    else:
        # Allow full gradient flow
        goal_feat = self.goal_encoder(goal)
    
    return goal_feat


"""
EXPECTED BENEFITS OF STAGED TRAINING:

🎯 STAGE 1 BENEFITS (State Model Pretraining):
• State model learns optimal placements without actor interference
• Goals become stable and meaningful before actor sees them
• Line clearing detection can be properly learned
• 3x intensive training on state model alone

🎭 STAGE 2 BENEFITS (Actor Training with Frozen Goals):  
• Actor learns to achieve consistent, high-quality goals
• No goal corruption from policy gradients
• Actor focuses purely on goal achievement
• 2x intensive training on actor alone

🤝 STAGE 3 BENEFITS (Joint Fine-tuning):
• Final alignment between state model and actor
• Handles any remaining goal-achievement gaps
• Fine-tunes the full system end-to-end

📈 EXPECTED IMPROVEMENTS:
• Goal consistency: 40-60% → 80-90%
• Goal achievement: 8.8% → 30-50%
• Training stability: Much more stable losses
• System alignment: Perfect goal-game alignment
""" 