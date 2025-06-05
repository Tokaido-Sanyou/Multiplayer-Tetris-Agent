#!/usr/bin/env python3
"""
Long Training Configuration for 10,000 Episodes
Demonstrates the complete AIRL training process with detailed explanations
"""

import os
from pytorch_airl_complete import create_training_config, PyTorchAIRLTrainer, MultiplayerAIRLTrainer

def create_10k_episode_config(training_type: str = "single") -> dict:
    """
    Create configuration for 10,000 episode training.
    
    For SINGLE-PLAYER:
    - Uses iterations where each iteration = episodes_per_iteration episodes
    - 10k episodes = max_iterations × episodes_per_iteration
    
    For MULTIPLAYER:
    - Uses direct episode count (max_episodes = 10000)
    """
    
    if training_type == "single":
        # Single-player uses iteration-based training
        episodes_per_iteration = 10  # Episodes collected per iteration
        updates_per_iteration = 20   # Network updates per iteration
        max_iterations = 1000        # 1000 iterations × 10 episodes = 10k episodes
        
        config = {
            # Environment
            'expert_trajectory_dir': 'expert_trajectories_dqn_adapter',
            'max_episode_steps': 500,
            'headless': True,  # No visualization for long runs
            
            # Training schedule (SINGLE-PLAYER)
            'max_iterations': max_iterations,
            'episodes_per_iteration': episodes_per_iteration,
            'updates_per_iteration': updates_per_iteration,
            
            # Network parameters
            'discriminator_lr': 3e-4,
            'policy_lr': 1e-4,
            'gamma': 0.99,
            'batch_size': 64,      # Larger batch for stability
            'buffer_size': 50000,  # Larger buffer for more diverse data
            
            # Logging
            'use_tensorboard': True,
            'use_cuda': True,
        }
        
        print(f"📊 SINGLE-PLAYER 10K CONFIGURATION:")
        print(f"   Total Episodes: {max_iterations} × {episodes_per_iteration} = {max_iterations * episodes_per_iteration}")
        print(f"   Network Updates: {max_iterations} × {updates_per_iteration} = {max_iterations * updates_per_iteration}")
        print(f"   Expected Duration: ~8-12 hours (depending on hardware)")
        
    else:  # multiplayer
        # Multiplayer uses direct episode-based training
        max_episodes = 10000
        max_episode_steps = 500
        
        config = {
            # Environment
            'expert_trajectory_dir': 'expert_trajectories_dqn_adapter',
            'max_episode_steps': max_episode_steps,
            'headless': True,
            
            # Training schedule (MULTIPLAYER)
            'max_episodes': max_episodes,
            
            # Network parameters
            'discriminator_lr': 3e-4,
            'policy_lr': 1e-4,
            'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 50000,
            
            # Logging
            'use_tensorboard': True,
            'use_cuda': True,
        }
        
        print(f"📊 MULTIPLAYER 10K CONFIGURATION:")
        print(f"   Total Episodes: {max_episodes}")
        print(f"   Max Steps per Episode: {max_episode_steps}")
        print(f"   Expected Duration: ~6-10 hours (depending on hardware)")
    
    return config

def explain_training_process():
    """Explain what happens during training."""
    print("\n🔄 TRAINING PROCESS BREAKDOWN")
    print("=" * 60)
    
    print("\n1️⃣ SINGLE-PLAYER AIRL PROCESS:")
    print("   Per Iteration:")
    print("   • Collect learner episodes using current policy")
    print("   • Sample expert batch from trajectory buffer")
    print("   • Sample learner batch from collected episodes")
    print("   • Update discriminator (expert vs learner classification)")
    print("   • Update policy using AIRL rewards from discriminator")
    print("   • Log metrics to TensorBoard")
    print()
    print("   Expert Imitation Learning:")
    print("   ✅ YES - Happens every batch update!")
    print("   • Expert trajectories sampled randomly each batch")
    print("   • Discriminator learns to distinguish expert vs learner")
    print("   • Policy receives rewards to mimic expert behavior")
    
    print("\n2️⃣ MULTIPLAYER COMPETITIVE PROCESS:")
    print("   Per Episode:")
    print("   • Reset both player environments")
    print("   • Run competitive episode (P1 vs P2)")
    print("   • Apply competitive rewards (win/loss bonuses)")
    print("   • Log episode results and winner")
    print("   • TensorBoard logging")
    print()
    print("   Expert Learning in Multiplayer:")
    print("   ⚠️  Currently basic - could be enhanced with:")
    print("   • Periodic AIRL updates for each player")
    print("   • Expert demonstration replay")
    print("   • Self-play + expert guidance")
    
    print("\n3️⃣ BATCH-LEVEL DETAILS:")
    print("   Every Discriminator Update:")
    print("   • Sample 64 expert transitions")
    print("   • Sample 64 learner transitions") 
    print("   • Train discriminator: label expert=1, learner=0")
    print("   • Discriminator learns expert behavioral patterns")
    print()
    print("   Every Policy Update:")
    print("   • Use discriminator to score learner actions")
    print("   • AIRL reward = log(D/(1-D)) where D = discriminator output")
    print("   • Policy gradient using AIRL rewards")
    print("   • Agent learns to maximize expert-like behavior")

def show_tensorboard_commands():
    """Show how to monitor training with TensorBoard."""
    print("\n📊 TENSORBOARD MONITORING")
    print("=" * 60)
    
    print("1️⃣ START TENSORBOARD:")
    print("   Command: tensorboard --logdir=logs")
    print("   URL: http://localhost:6006")
    print("   Alternative port: tensorboard --logdir=logs --port=6007")
    
    print("\n2️⃣ KEY METRICS TO WATCH:")
    print("   Single-Player AIRL:")
    print("   • discriminator_loss: Should stabilize around 0.7-1.0")
    print("   • overall_accuracy: Should reach 0.7-0.9 (discriminator learning)")
    print("   • policy_loss: Should decrease over time")
    print("   • mean_airl_reward: Should become less negative")
    print("   • learner/buffer_size: Should grow then stabilize")
    
    print("\n   Multiplayer Competitive:")
    print("   • Episode/P1_Reward & Episode/P2_Reward: Should increase")
    print("   • Episode/Winner: Should approach 50/50 balance as agents improve")
    print("   • Episode/Steps: Should increase as agents get better")
    print("   • Training/Total_Steps: Cumulative step counter")
    
    print("\n3️⃣ LOG DIRECTORY STRUCTURE:")
    print("   logs/")
    print("   ├── airl_YYYYMMDD_HHMMSS/          # Single-player logs")
    print("   └── multiplayer_airl_YYYYMMDD_HHMMSS/  # Multiplayer logs")
    
    print("\n4️⃣ EXAMPLE TENSORBOARD COMMANDS:")
    print("   # View latest single-player run")
    print("   tensorboard --logdir=logs/airl_20250604_030413")
    print()
    print("   # View latest multiplayer run") 
    print("   tensorboard --logdir=logs/multiplayer_airl_20250604_030430")
    print()
    print("   # Compare multiple runs")
    print("   tensorboard --logdir=logs")

def create_training_script():
    """Create training scripts for easy execution."""
    
    # Single-player 10k script
    single_script = """#!/usr/bin/env python3
# 10K Single-Player AIRL Training
import sys
sys.path.append('.')
from long_training_config import create_10k_episode_config
from pytorch_airl_complete import PyTorchAIRLTrainer

def main():
    print("🚀 STARTING 10,000 EPISODE SINGLE-PLAYER AIRL TRAINING")
    config = create_10k_episode_config("single")
    trainer = PyTorchAIRLTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
"""
    
    # Multiplayer 10k script
    multi_script = """#!/usr/bin/env python3
# 10K Multiplayer Competitive Training  
import sys
sys.path.append('.')
from long_training_config import create_10k_episode_config
from pytorch_airl_complete import MultiplayerAIRLTrainer

def main():
    print("🚀 STARTING 10,000 EPISODE MULTIPLAYER COMPETITIVE TRAINING")
    config = create_10k_episode_config("multiplayer")
    trainer = MultiplayerAIRLTrainer(config)
    trainer.train_competitive()

if __name__ == "__main__":
    main()
"""
    
    # Write scripts
    with open("train_10k_single.py", "w") as f:
        f.write(single_script)
        
    with open("train_10k_multiplayer.py", "w") as f:
        f.write(multi_script)
    
    print("\n📝 TRAINING SCRIPTS CREATED:")
    print("   train_10k_single.py - Single-player 10K episodes")
    print("   train_10k_multiplayer.py - Multiplayer 10K episodes")
    print()
    print("   Usage:")
    print("   python train_10k_single.py")
    print("   python train_10k_multiplayer.py")

def main():
    """Demonstrate 10K episode training configuration."""
    print("🎯 10,000 EPISODE AIRL TRAINING GUIDE")
    print("=" * 60)
    
    # Show configurations
    print("\n📋 CONFIGURATIONS:")
    single_config = create_10k_episode_config("single")
    print()
    multi_config = create_10k_episode_config("multiplayer") 
    
    # Explain process
    explain_training_process()
    
    # TensorBoard guide
    show_tensorboard_commands()
    
    # Create training scripts
    create_training_script()
    
    print("\n✅ READY FOR 10K EPISODE TRAINING!")
    print("   1. Choose single-player or multiplayer")
    print("   2. Run training script")
    print("   3. Monitor with TensorBoard")
    print("   4. Wait 6-12 hours for completion")

if __name__ == "__main__":
    main() 