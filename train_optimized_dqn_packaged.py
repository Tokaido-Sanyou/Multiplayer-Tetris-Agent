#!/usr/bin/env python3
"""
Optimized DQN Packaged Training Script - FIXED VERSION
"""

import sys
import os
import argparse
import time
import random
import torch
import numpy as np
from typing import Dict

# Add project path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def setup_training_environment():
    """Setup training environment"""
    print("=" * 80)
    print("OPTIMIZED DQN PACKAGED TRAINING")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("GPU: Not available, using CPU")
    
    try:
        from envs.tetris_env import TetrisEnv
        from agents.dqn_locked_agent_optimized import OptimizedLockedStateDQNAgent
        print("Imports: All modules loaded successfully")
    except ImportError as e:
        print(f"Import Error: {e}")
        sys.exit(1)
    
    return device

def shape_reward(raw_reward: float, step_count: int, done: bool) -> float:
    """
    IMPROVED: Better reward shaping for longer episodes and learning
    """
    shaped_reward = raw_reward
    
    # 1. STRONGER survival bonus - encourage longer episodes
    if not done:
        shaped_reward += 3.0  # Stronger positive for each step survived
    
    # 2. DRASTICALLY reduce terminal penalty
    if done and raw_reward <= -50:  # Terminal penalty
        shaped_reward = -5.0  # Much smaller terminal penalty
    
    # 3. Progressive step bonuses for longer episodes
    if step_count > 5:
        shaped_reward += 1.0   # Bonus for getting past very early termination
    if step_count > 10:
        shaped_reward += 1.0   # Additional bonus for longer episodes
    if step_count > 15:
        shaped_reward += 2.0   # Strong bonus for much longer episodes
    
    # 4. Scale down large negative rewards more aggressively
    if raw_reward < -20:
        shaped_reward = -3.0 + (raw_reward + 20) * 0.05  # More aggressive scaling
    
    # 5. Small positive reward baseline
    shaped_reward += 0.5  # Baseline positive to counteract negative bias
    
    return shaped_reward

def create_optimized_agent(device: str, use_valid_action_selection: bool = False):
    """Create and configure optimized agent with IMPROVED parameters"""
    from agents.dqn_locked_agent_optimized import OptimizedLockedStateDQNAgent
    
    agent = OptimizedLockedStateDQNAgent(
        device=device,
        use_valid_action_selection=use_valid_action_selection,
        # IMPROVED FIXES:
        learning_rate=0.005,      # Higher for faster learning
        epsilon_start=0.9,        # Start with less random
        epsilon_end=0.01,         # Lower final epsilon
        epsilon_decay_steps=300,  # Even faster decay
        batch_size=8,             # Smaller for more frequent updates
        memory_size=5000,         # Smaller for faster turnover
        target_update_freq=50,    # More frequent target updates
        gamma=0.95                # Lower discount for immediate rewards
    )
    
    info = agent.get_info()
    print(f"\nIMPROVED AGENT CONFIGURATION:")
    print(f"   Type: {info['type']}")
    print(f"   Parameters: {info['parameters']:,}")
    print(f"   Architecture: {info['architecture']}")
    print(f"   Learning Rate: 0.005 (increased)")
    print(f"   Epsilon Decay: 300 steps (faster)")
    print(f"   Batch Size: 8 (smaller)")
    print(f"   Gamma: 0.95 (lower)")
    print(f"   Device: {info['device']}")
    
    return agent

def run_training_episode(agent, env, episode: int, training: bool = True) -> Dict:
    """Run a single training episode with improvements"""
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    
    episode_reward = 0
    shaped_episode_reward = 0
    episode_length = 0
    valid_actions_count = 0
    total_actions_taken = 0
    training_losses = []
    
    done = False
    max_steps = 1000
    
    while not done and episode_length < max_steps:
        try:
            # IMPROVED: More diverse action selection
            if hasattr(agent, 'use_valid_action_selection') and agent.use_valid_action_selection:
                state = agent.encode_state_with_selection(observation)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                
                with torch.no_grad():
                    output = agent.q_network(state_tensor).cpu().numpy()[0]
                
                if training and random.random() < agent.epsilon:
                    # IMPROVED: More diverse random exploration
                    if episode_length < 3:  # Early exploration
                        action = random.randint(0, 99)   # Top half first
                    else:  # Later exploration
                        action = random.randint(50, 199)  # Full range
                else:
                    # IMPROVED: Use full Q-network output with better mapping
                    action_idx = np.argmax(output)  # Use full network output
                    action = min(199, max(0, action_idx % 200))  # Map to full range
            else:
                action = agent.select_action(observation, training=training, env=env)
        except Exception as e:
            raise RuntimeError(f"Action selection failed at step {episode_length}: {e}")
        
        # Track valid actions
        if isinstance(action, int) and action >= 0:
            total_actions_taken += 1
            valid_actions_count += 1  # Assume valid for speed
        
        # Environment step
        try:
            if action >= 0:
                step_result = env.step([action])
                
                if len(step_result) == 3:
                    reward, done, info = step_result
                    next_observation = env.get_observation(0)
                    episode_reward += reward
                    reward_value = reward
                    done_value = done
                elif len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                    if isinstance(next_obs, list):
                        next_observation = next_obs[0]
                    else:
                        next_observation = next_obs
                    
                    if isinstance(reward, list):
                        episode_reward += reward[0]
                        reward_value = reward[0]
                    else:
                        episode_reward += reward
                        reward_value = reward
                    
                    if isinstance(done, list):
                        done_value = done[0]
                    else:
                        done_value = done
                else:
                    raise ValueError(f"Unexpected step result: {len(step_result)} elements")
                
                # Apply reward shaping for better learning
                raw_reward = reward_value
                shaped_reward = shape_reward(raw_reward, episode_length, done_value)
                shaped_episode_reward += shaped_reward
                
                # Store experience and train with SHAPED reward
                if training and hasattr(agent, 'memory'):
                    state = agent.encode_state_with_selection(observation)
                    next_state = agent.encode_state_with_selection(next_observation)
                    agent.store_experience(state, action, shaped_reward, next_state, done_value)
                    
                    if len(agent.memory) >= agent.batch_size:
                        try:
                            loss_info = agent.train_batch()
                            if loss_info.get('loss', 0) > 0:
                                training_losses.append(loss_info['loss'])
                        except Exception as e:
                            raise RuntimeError(f"Training failed at step {episode_length}: {e}")
                
                observation = next_observation
                done = done_value
                
        except Exception as e:
            raise RuntimeError(f"Environment step failed at step {episode_length}: {e}")
        
        episode_length += 1
    
    valid_action_rate = valid_actions_count / max(total_actions_taken, 1)
    avg_loss = np.mean(training_losses) if training_losses else 0.0
    
    return {
        'reward': episode_reward,  # Raw reward for display
        'shaped_reward': shaped_episode_reward,  # Shaped reward for learning analysis
        'length': episode_length,
        'valid_action_rate': valid_action_rate,
        'avg_loss': avg_loss,
        'num_training_steps': len(training_losses)
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Optimized DQN Packaged Training - FIXED')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--use-valid-selection', action='store_true', help='Use valid action selection mode')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    device = setup_training_environment() if args.device == 'auto' else args.device
    
    if not args.quiet:
        print(f"\nIMPROVED TRAINING CONFIGURATION:")
        print(f"   Episodes: {args.episodes}")
        print(f"   Enhanced Reward Shaping: ENABLED")
        print(f"   Improved Action Selection: ENABLED")
        print(f"   Device: {device}")
    
    try:
        agent = create_optimized_agent(
            device=device,
            use_valid_action_selection=args.use_valid_selection
        )
        
        from envs.tetris_env import TetrisEnv
        env = TetrisEnv(num_agents=1, headless=True, action_mode='locked_position')
        
        episode_rewards = []
        shaped_rewards = []
        episode_lengths = []
        valid_action_rates = []
        
        print(f"\nSTARTING IMPROVED TRAINING...")
        start_time = time.time()
        
        for episode in range(args.episodes):
            try:
                result = run_training_episode(agent, env, episode, training=True)
                
                episode_rewards.append(result['reward'])
                shaped_rewards.append(result['shaped_reward'])
                episode_lengths.append(result['length'])
                valid_action_rates.append(result['valid_action_rate'])
                
                # Enhanced logging with both raw and shaped rewards
                if episode < 10 or (episode + 1) % 10 == 0:
                    episode_time = (time.time() - start_time) / (episode + 1)
                    
                    if episode >= 9:  # For batches of 10
                        recent_raw = episode_rewards[-10:]
                        recent_shaped = shaped_rewards[-10:]
                        recent_lengths = episode_lengths[-10:]
                        
                        raw_improvement = "↗" if np.mean(recent_raw) > np.mean(episode_rewards[:10]) else "→"
                        shaped_improvement = "↗" if np.mean(recent_shaped) > np.mean(shaped_rewards[:10]) else "→"
                        
                        print(f"Episodes {episode-8}-{episode+1}: "
                              f"Raw={np.mean(recent_raw):.1f}{raw_improvement}, "
                              f"Shaped={np.mean(recent_shaped):.1f}{shaped_improvement}, "
                              f"Length={np.mean(recent_lengths):.0f}, "
                              f"Epsilon={agent.epsilon:.3f}, "
                              f"Memory={len(agent.memory)}, "
                              f"Time={episode_time:.1f}s/ep")
                    else:  # Individual episodes
                        print(f"Episode {episode+1:2d}: "
                              f"Raw={result['reward']:6.1f}, "
                              f"Shaped={result['shaped_reward']:6.1f}, "
                              f"Length={result['length']:3d}, "
                              f"Loss={result['avg_loss']:.3f}, "
                              f"Epsilon={agent.epsilon:.3f}")
            
            except Exception as e:
                print(f"Episode {episode+1} failed: {e}")
                break
        
        env.close()
        total_time = time.time() - start_time
        
        # Final analysis with both raw and shaped rewards
        print(f"\nIMPROVED TRAINING COMPLETED!")
        print(f"   Episodes: {len(episode_rewards)}")
        print(f"   Time: {total_time:.1f}s")
        
        if len(episode_rewards) >= 20:
            early_raw = np.mean(episode_rewards[:10])
            late_raw = np.mean(episode_rewards[-10:])
            early_shaped = np.mean(shaped_rewards[:10])
            late_shaped = np.mean(shaped_rewards[-10:])
            
            print(f"   Raw Reward: {early_raw:.1f} → {late_raw:.1f} ({late_raw-early_raw:+.1f})")
            print(f"   Shaped Reward: {early_shaped:.1f} → {late_shaped:.1f} ({late_shaped-early_shaped:+.1f})")
            print(f"   Avg Length: {np.mean(episode_lengths):.1f}")
            
            if late_shaped > early_shaped:
                print("   🎉 LEARNING DETECTED IN SHAPED REWARDS!")
            if late_raw > early_raw:
                print("   🎉 LEARNING DETECTED IN RAW REWARDS!")
        
        return True
        
    except Exception as e:
        print(f"\nTRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 