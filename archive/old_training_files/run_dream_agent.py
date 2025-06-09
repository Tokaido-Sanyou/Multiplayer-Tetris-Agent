#!/usr/bin/env python3
"""
Dream Agent Runner with Dimension Compatibility

This script patches the dream agent to handle 206→212 dimension conversion
and runs the training.
"""

import torch
import numpy as np
import sys
import os

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def patch_dream_agent():
    """Patch the dream agent classes to handle dimension conversion"""
    
    # Import the dream classes
    from dream_tetris_clean import ImprovedDREAMTrainer, ImprovedTetrisWorldModel, ImprovedTetrisActorCritic, RNDNetwork
    
    # Store original methods
    original_collect_trajectory = ImprovedDREAMTrainer.collect_trajectory
    
    def patched_collect_trajectory(self):
        """Patched collect_trajectory with dimension padding"""
        obs = self.env.reset()
        
        # FIXED: Pad observation dimensions (206→212) 
        if obs.shape[0] == 206:
            obs = np.concatenate([obs, np.zeros(6)], axis=0)
        
        trajectory = {
            'observations': [obs],
            'actions': [],
            'rewards': [],  # Store environment rewards directly
            'dones': []
        }
        
        # No step limit - natural episode termination only  
        step = 0
        
        while True:  # Continue until natural episode termination
            # Get action from current policy with RND exploration
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action, log_prob, value = self.actor_critic.get_action_and_value(
                obs_tensor, epsilon=self.epsilon, temperature=self.temperature, rnd_network=self.rnd_network
            )
            action = action.item()
            
            # Take environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # FIXED: Pad next observation dimensions (206→212)
            if next_obs.shape[0] == 206:
                next_obs = np.concatenate([next_obs, np.zeros(6)], axis=0)
            
            # Store transition
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)  # Environment reward used directly
            trajectory['dones'].append(done)
            trajectory['observations'].append(next_obs)
            
            if done:
                break
            
            obs = next_obs
            step += 1
        
        return trajectory
    
    # Apply the patch
    ImprovedDREAMTrainer.collect_trajectory = patched_collect_trajectory
    
    print("✅ Dream agent patched for 206→212 dimension compatibility")
    
    return ImprovedDREAMTrainer

def main():
    """Main function to run dream agent"""
    print("=== DREAM AGENT WITH DIMENSION COMPATIBILITY ===")
    
    # Patch the dream agent
    DREAMTrainer = patch_dream_agent()
    
    # Initialize and run
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ Using device: {device}")
    
    try:
        trainer = DREAMTrainer(device=device, enable_visualization=True)
        print("✅ DREAM trainer initialized successfully")
        
        # Run training
        print("\n=== STARTING DREAM TRAINING ===")
        results = trainer.train(num_episodes=50)  # Start with fewer episodes
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 