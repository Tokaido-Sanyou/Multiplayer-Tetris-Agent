#!/usr/bin/env python3

import sys
import torch
from rl_utils.unified_trainer import UnifiedTrainer, TrainingConfig
from rl_utils.actor_critic import ActorCriticAgent
from rl_utils.state_model import StateModel

def test_state_model_outputs():
    """Test that state model returns 4 outputs"""
    print("Testing state model output count...")
    state_model = StateModel(state_dim=206)
    state = torch.randn(1, 206)
    outputs = state_model(state)
    print(f"State model outputs: {len(outputs)} values")
    print(f"Output shapes: {[x.shape for x in outputs]}")
    return len(outputs)

def test_actor_critic_state_model():
    """Test ActorCritic state model integration"""
    print("\nTesting ActorCritic state model integration...")
    
    # Create agent and state model
    agent = ActorCriticAgent(state_dim=206, action_dim=8)
    state_model = StateModel(state_dim=206)
    
    # Test setting state model
    agent.set_state_model(state_model, {})
    print("✓ State model set successfully")
    
    # Test if state model training would work
    states = torch.randn(32, 206)
    try:
        # This should fail due to wrong number of outputs
        rot_logits, x_logits, y_logits = agent.state_model(states)
        print("✗ Expected error did not occur!")
        return False
    except ValueError as e:
        print(f"✓ Expected error occurred: {e}")
        return True

def test_unified_trainer_setup():
    """Test unified trainer initialization"""
    print("\nTesting UnifiedTrainer setup...")
    
    config = TrainingConfig()
    config.visualize = False
    config.device = 'cpu'
    config.num_batches = 1
    
    trainer = UnifiedTrainer(config)
    print("✓ UnifiedTrainer created successfully")
    
    # Check if writer is passed to actor_critic
    has_writer = hasattr(trainer.actor_critic, 'writer') and trainer.actor_critic.writer is not None
    print(f"ActorCritic has writer: {has_writer}")
    
    return trainer

if __name__ == "__main__":
    print("=== Testing Current Logging Implementation ===")
    
    # Test 1: State model outputs
    num_outputs = test_state_model_outputs()
    
    # Test 2: ActorCritic integration
    test_actor_critic_state_model()
    
    # Test 3: Unified trainer
    trainer = test_unified_trainer_setup()
    
    print(f"\n=== Summary ===")
    print(f"State model returns {num_outputs} outputs (expected: 4)")
    print(f"ActorCritic expects 3 outputs - THIS IS THE BUG!")
    print(f"UnifiedTrainer needs to set writer on ActorCritic")
