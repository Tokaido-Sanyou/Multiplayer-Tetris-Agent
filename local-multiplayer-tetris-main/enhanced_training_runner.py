#!/usr/bin/env python3
"""
Enhanced 6-Phase Training Runner
Standalone training script for the Enhanced 6-Phase system with goal-conditioned actor.
NO PPO dependencies - uses simplified supervised learning throughout.
"""

import sys
import os
sys.path.append('localMultiplayerTetris')

import torch
import numpy as np
import argparse
from datetime import datetime

from localMultiplayerTetris.rl_utils.enhanced_6phase_state_model import Enhanced6PhaseComponents
from localMultiplayerTetris.rl_utils.goal_conditioned_actor import GoalConditionedActorIntegration
from localMultiplayerTetris.tetris_env import TetrisEnv


class Enhanced6PhaseTrainer:
    """
    Standalone trainer for Enhanced 6-Phase system
    Features:
    - 5D goal vectors (rotation + x + y + validity)
    - Goal-conditioned actor (NO PPO)
    - Iterative exploration mode
    - 3-stage training pipeline
    """
    
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu'
        
        print(f"🚀 Enhanced 6-Phase Trainer Starting...")
        print(f"   • Device: {self.device}")
        print(f"   • Exploration mode: {config.exploration_mode}")
        print(f"   • Training batches: {config.num_batches}")
        print(f"   • Goal-conditioned actor: NO PPO")
        print()
        
        # Initialize environment
        self.env = TetrisEnv(single_player=True, headless=True)
        
        # Initialize Enhanced 6-Phase system
        self.enhanced_system = Enhanced6PhaseComponents(
            state_dim=210, 
            goal_dim=5,  # rotation(2) + x + y + validity
            device=self.device
        )
        
        # Set optimizers
        self.enhanced_system.set_optimizers(
            state_lr=config.state_lr,
            q_lr=config.q_lr
        )
        
        # Initialize goal-conditioned actor integration
        self.actor_integration = GoalConditionedActorIntegration(
            enhanced_system=self.enhanced_system,
            state_dim=210,
            goal_dim=5,
            action_dim=8,
            device=self.device
        )
        
        # Create exploration manager
        self.exploration_manager = self.enhanced_system.create_piece_by_piece_exploration_manager(self.env)
        self.exploration_manager.max_pieces = config.max_pieces
        self.exploration_manager.boards_to_keep = config.boards_to_keep
        
        # Training metrics
        self.training_metrics = {
            'state_losses': [],
            'q_losses': [],
            'actor_losses': [],
            'lines_cleared_total': [],
            'exploration_efficiency': []
        }
        
        print(f"✅ Enhanced 6-Phase Trainer initialized!")
        print()
    
    def run_training(self):
        """Run the complete 3-stage training pipeline"""
        print(f"🎯 STARTING ENHANCED 6-PHASE TRAINING")
        print(f"=" * 80)
        
        for batch in range(self.config.num_batches):
            print(f"\n📦 BATCH {batch+1}/{self.config.num_batches}")
            print("-" * 40)
            
            # Determine training stage
            stage = self._get_training_stage(batch)
            print(f"🎯 Stage: {stage}")
            
            # Collect exploration data
            exploration_data = self._collect_exploration_data()
            
            if not exploration_data:
                print("⚠️ No exploration data collected, skipping batch")
                continue
            
            # Stage-specific training
            if stage == 'state_model_only':
                self._train_stage_1_state_model(exploration_data, batch)
            elif stage == 'actor_only':
                self._train_stage_2_actor(exploration_data, batch)
            elif stage == 'joint_training':
                self._train_stage_3_joint(exploration_data, batch)
            
            # Update exploration parameters
            self.enhanced_system.update_epsilon()
            
            # Log metrics
            self._log_batch_metrics(batch)
            
            # Save checkpoint
            if (batch + 1) % 50 == 0:
                self._save_checkpoint(batch)
        
        print(f"\n🎉 TRAINING COMPLETED!")
        self._save_final_results()
    
    def _get_training_stage(self, batch):
        """Determine training stage based on batch number"""
        total_batches = self.config.num_batches
        
        if batch < total_batches * 0.4:  # First 40%
            return 'state_model_only'
        elif batch < total_batches * 0.7:  # Next 30%
            return 'actor_only'
        else:  # Last 30%
            return 'joint_training'
    
    def _collect_exploration_data(self):
        """Collect exploration data using specified mode"""
        if self.config.exploration_mode == 'iterative':
            return self.exploration_manager.collect_piece_by_piece_exploration_data('iterative')
        elif self.config.exploration_mode == 'deterministic':
            # Map deterministic to iterative (most similar)
            return self.exploration_manager.collect_piece_by_piece_exploration_data('iterative')
        else:
            return self.exploration_manager.collect_piece_by_piece_exploration_data('rnd')
    
    def _train_stage_1_state_model(self, exploration_data, batch):
        """Stage 1: State model training only"""
        print("🎯 Stage 1: State Model Training (Actor FROZEN)")
        
        # Train state model
        state_results = self.enhanced_system.train_enhanced_state_model(exploration_data)
        
        # Train Q-learning for value estimation
        q_results = self.enhanced_system.train_simplified_q_learning(exploration_data)
        
        self.training_metrics['state_losses'].append(state_results['loss'])
        self.training_metrics['q_losses'].append(q_results['q_loss'])
        self.training_metrics['actor_losses'].append(0)  # Actor not training
        
        print(f"   • State loss: {state_results['loss']:.4f}")
        print(f"   • Q loss: {q_results['q_loss']:.4f}")
        print(f"   • Top performers: {state_results['top_performers_used']}")
    
    def _train_stage_2_actor(self, exploration_data, batch):
        """Stage 2: Actor training only (state model frozen)"""
        print("🎯 Stage 2: Goal-Conditioned Actor Training (State Model FROZEN)")
        
        # Train actor only
        actor_results = self.actor_integration.train_actor_on_exploration_data(exploration_data)
        
        self.training_metrics['state_losses'].append(0)  # State model not training
        self.training_metrics['q_losses'].append(0)     # Q-learning not training
        self.training_metrics['actor_losses'].append(actor_results['actor_loss'])
        
        print(f"   • Actor loss: {actor_results['actor_loss']:.4f}")
        print(f"   • Goal→action pairs: {actor_results['goal_action_pairs']}")
        print(f"   • Training approach: {actor_results['training_approach']}")
    
    def _train_stage_3_joint(self, exploration_data, batch):
        """Stage 3: Joint training of all components"""
        print("🎯 Stage 3: Joint Training (State Model + Actor + Q-Learning)")
        
        # Train all components
        state_results = self.enhanced_system.train_enhanced_state_model(exploration_data)
        q_results = self.enhanced_system.train_simplified_q_learning(exploration_data)
        actor_results = self.actor_integration.train_actor_on_exploration_data(exploration_data)
        
        self.training_metrics['state_losses'].append(state_results['loss'])
        self.training_metrics['q_losses'].append(q_results['q_loss'])
        self.training_metrics['actor_losses'].append(actor_results['actor_loss'])
        
        print(f"   • State loss: {state_results['loss']:.4f}")
        print(f"   • Q loss: {q_results['q_loss']:.4f}")
        print(f"   • Actor loss: {actor_results['actor_loss']:.4f}")
        print(f"   • Coordination: All components improving together")
    
    def _log_batch_metrics(self, batch):
        """Log metrics for current batch"""
        lines_cleared = sum(d.get('lines_cleared', 0) for d in 
                          self.exploration_manager.exploration_data if d.get('lines_cleared'))
        self.training_metrics['lines_cleared_total'].append(lines_cleared)
        
        # Calculate exploration efficiency
        total_samples = len(self.exploration_manager.exploration_data)
        valid_samples = sum(1 for d in self.exploration_manager.exploration_data 
                          if d.get('terminal_reward', 0) > -100)
        efficiency = valid_samples / max(1, total_samples) * 100
        self.training_metrics['exploration_efficiency'].append(efficiency)
        
        print(f"   📊 Batch {batch+1} Metrics:")
        print(f"      • Lines cleared: {lines_cleared}")
        print(f"      • Exploration samples: {total_samples}")
        print(f"      • Exploration efficiency: {efficiency:.1f}%")
        print(f"      • Epsilon: {self.enhanced_system.goal_selector.epsilon:.4f}")
    
    def _save_checkpoint(self, batch):
        """Save training checkpoint"""
        checkpoint_path = f"checkpoints/enhanced_6phase_batch_{batch+1}"
        os.makedirs("checkpoints", exist_ok=True)
        
        self.enhanced_system.save_checkpoints(checkpoint_path)
        self.actor_integration.trainer.save_checkpoint(f"{checkpoint_path}_actor.pt")
        
        print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_final_results(self):
        """Save final training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"enhanced_6phase_results_{timestamp}.txt"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"Enhanced 6-Phase Training Results - {timestamp}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SYSTEM CONFIGURATION:\n")
            f.write(f"- Training batches: {self.config.num_batches}\n")
            f.write(f"- Exploration mode: {self.config.exploration_mode}\n")
            f.write(f"- Max pieces per episode: {self.config.max_pieces}\n")
            f.write(f"- Device: {self.device}\n")
            f.write(f"- Goal-conditioned actor: NO PPO\n")
            f.write(f"- Goal vector dimension: 5D\n\n")
            
            f.write("TRAINING METRICS:\n")
            if self.training_metrics['state_losses']:
                avg_state_loss = np.mean([l for l in self.training_metrics['state_losses'] if l > 0])
                f.write(f"- Average state loss: {avg_state_loss:.4f}\n")
            
            if self.training_metrics['actor_losses']:
                avg_actor_loss = np.mean([l for l in self.training_metrics['actor_losses'] if l > 0])
                f.write(f"- Average actor loss: {avg_actor_loss:.4f}\n")
            
            total_lines = sum(self.training_metrics['lines_cleared_total'])
            f.write(f"- Total lines cleared: {total_lines}\n")
            
            avg_efficiency = np.mean(self.training_metrics['exploration_efficiency'])
            f.write(f"- Average exploration efficiency: {avg_efficiency:.1f}%\n")
        
        print(f"📄 Final results saved: {results_file}")


class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        self.num_batches = 100
        self.exploration_mode = 'iterative'
        self.max_pieces = 3
        self.boards_to_keep = 4
        self.state_lr = 0.001
        self.q_lr = 0.001
        self.use_cuda = True


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Enhanced 6-Phase Training Runner')
    parser.add_argument('--num_batches', type=int, default=100, help='Number of training batches')
    parser.add_argument('--exploration_mode', type=str, default='iterative', 
                       choices=['iterative', 'deterministic', 'rnd'], help='Exploration mode')
    parser.add_argument('--max_pieces', type=int, default=3, help='Max pieces per exploration episode')
    parser.add_argument('--boards_to_keep', type=int, default=4, help='Top boards to keep')
    parser.add_argument('--state_lr', type=float, default=0.001, help='State model learning rate')
    parser.add_argument('--q_lr', type=float, default=0.001, help='Q-learning learning rate')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.num_batches = args.num_batches
    config.exploration_mode = args.exploration_mode
    config.max_pieces = args.max_pieces
    config.boards_to_keep = args.boards_to_keep
    config.state_lr = args.state_lr
    config.q_lr = args.q_lr
    config.use_cuda = args.use_cuda
    
    print(f"🚀 Enhanced 6-Phase Training Runner")
    print(f"Configuration:")
    print(f"  • Batches: {config.num_batches}")
    print(f"  • Exploration: {config.exploration_mode}")
    print(f"  • Max pieces: {config.max_pieces}")
    print(f"  • Learning rates: state={config.state_lr}, q={config.q_lr}")
    print()
    
    # Create and run trainer
    trainer = Enhanced6PhaseTrainer(config)
    trainer.run_training()


if __name__ == "__main__":
    main() 