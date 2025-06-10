#!/usr/bin/env python3
"""
Quick test of Dreamer with lines cleared tracking - reduced episodes for faster feedback
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque
import random
import os
from datetime import datetime
import numpy as np

from tetris_env import TetrisEnv
from train_dreamer_standard import DreamerAgent, ReplayBuffer

def quick_dreamer_test(episodes=100, world_model_pretrain=20):
    """
    Quick Dreamer test to see lines cleared statistics
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = TetrisEnv(reward_mode='lines_only')
    agent = DreamerAgent(device=device)
    buffer = ReplayBuffer(capacity=10000)
    
    # Performance tracking
    episode_rewards = []
    episode_lines_cleared = []
    episode_steps_history = []
    
    print(f"🚀 Quick Dreamer Test on {device}")
    print(f"World Model Parameters: {sum(p.numel() for p in agent.world_model.parameters()):,}")
    print(f"Policy Parameters: {sum(p.numel() for p in agent.actor.parameters()) + sum(p.numel() for p in agent.critic.parameters()):,}")
    
    total_steps = 0
    world_model_updates = 0
    policy_updates = 0
    
    # Phase 1: Random exploration and world model pretraining
    print(f"\n📚 Phase 1: Random exploration ({world_model_pretrain} episodes)...")
    
    for episode in range(world_model_pretrain):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        lines_cleared = 0
        
        while episode_steps < 200:  # Shorter episodes for quick test
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            if 'lines_cleared' in info:
                lines_cleared += info['lines_cleared']
            
            buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lines_cleared.append(lines_cleared)
        episode_steps_history.append(episode_steps)
        
        # Train world model
        if len(buffer) >= agent.batch_size and episode % 3 == 0:
            batch = buffer.sample(agent.batch_size)
            world_losses = agent.train_world_model(batch)
            world_model_updates += 1
        
        if episode % 5 == 0 or episode == world_model_pretrain - 1:
            recent_count = min(10, len(episode_lines_cleared))
            recent_lines = episode_lines_cleared[-recent_count:]
            recent_rewards = episode_rewards[-recent_count:]
            
            avg_lines = np.mean(recent_lines)
            max_lines = max(recent_lines)
            avg_reward = np.mean(recent_rewards)
            
            print(f"  Episode {episode:2d}: Lines={lines_cleared}, Reward={episode_reward:6.2f} | "
                  f"Last {recent_count}: Avg={avg_lines:.1f}, Max={max_lines}")
    
    # Phase 2: Policy learning with imagination
    print(f"\n🧠 Phase 2: Policy learning with imagination ({episodes} episodes)...")
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        lines_cleared = 0
        
        while episode_steps < 200:  # Shorter episodes
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            
            if 'lines_cleared' in info:
                lines_cleared += info['lines_cleared']
            
            buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lines_cleared.append(lines_cleared)
        episode_steps_history.append(episode_steps)
        
        # Training updates
        if len(buffer) >= agent.batch_size:
            batch = buffer.sample(agent.batch_size)
            
            # World model update (less frequent)
            if episode % 3 == 0:
                world_losses = agent.train_world_model(batch)
                world_model_updates += 1
            
            # Policy update using imagination
            policy_losses = agent.train_policy(batch)
            policy_updates += 1
        
        # Performance reporting every 10 episodes
        if episode % 10 == 0 or episode == episodes - 1:
            # Last 50 episodes stats (or all if less than 50)
            recent_count = min(50, len(episode_lines_cleared))
            recent_lines = episode_lines_cleared[-recent_count:]
            recent_rewards = episode_rewards[-recent_count:]
            recent_steps = episode_steps_history[-recent_count:]
            
            # Statistics
            avg_lines = np.mean(recent_lines)
            max_lines = max(recent_lines)
            avg_reward = np.mean(recent_rewards)
            max_reward = max(recent_rewards)
            avg_steps = np.mean(recent_steps)
            
            # Top performances
            sorted_indices = np.argsort(recent_lines)[::-1]
            top_3_lines = [recent_lines[i] for i in sorted_indices[:3]]
            
            print(f"\n📊 Episode {episode:3d} - Performance Report (Last {recent_count} episodes):")
            print(f"  🏆 Current Episode: {lines_cleared} lines, {episode_reward:.2f} reward")
            print(f"  📈 Average Lines/Episode: {avg_lines:.2f}")
            print(f"  🎯 Highest Lines Cleared: {max_lines}")
            print(f"  💰 Average Reward: {avg_reward:.2f}")
            print(f"  🔥 Top 3 Line Performances: {top_3_lines}")
            print(f"  ⏱️  Average Steps: {avg_steps:.1f}")
            print(f"  🧠 Updates: WM={world_model_updates}, Policy={policy_updates}")
    
    # Final summary
    print(f"\n🎉 Quick Test Completed!")
    all_time_best = max(episode_lines_cleared)
    final_50 = episode_lines_cleared[-50:] if len(episode_lines_cleared) >= 50 else episode_lines_cleared
    final_avg = np.mean(final_50)
    final_max = max(final_50)
    
    # Show progression
    if len(episode_lines_cleared) >= 20:
        first_10_avg = np.mean(episode_lines_cleared[:10])
        last_10_avg = np.mean(episode_lines_cleared[-10:])
        improvement = last_10_avg - first_10_avg
        
        print(f"  📊 Performance Evolution:")
        print(f"    First 10 episodes average: {first_10_avg:.2f} lines")
        print(f"    Last 10 episodes average: {last_10_avg:.2f} lines")
        print(f"    Improvement: {improvement:+.2f} lines ({improvement/first_10_avg*100:+.1f}%)")
    
    print(f"  🏆 All-Time Best: {all_time_best} lines cleared")
    print(f"  📈 Final Period Average: {final_avg:.2f} lines")
    print(f"  🔥 Final Period Best: {final_max} lines")
    print(f"  🎯 Total Episodes: {len(episode_lines_cleared)}")
    print(f"  ⚡ Total Steps: {total_steps}")
    
    return {
        'episode_lines_cleared': episode_lines_cleared,
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps_history,
        'all_time_best': all_time_best,
        'final_avg': final_avg,
        'agent': agent
    }

if __name__ == "__main__":
    print("🎮 Starting Quick Dreamer Test for Lines Cleared Statistics\n")
    results = quick_dreamer_test(episodes=80, world_model_pretrain=15)
    
    # Show detailed final stats
    lines = results['episode_lines_cleared']
    print(f"\n📋 Detailed Final Statistics:")
    print(f"  Episodes with 0 lines: {len([x for x in lines if x == 0])}")
    print(f"  Episodes with 1+ lines: {len([x for x in lines if x >= 1])}")
    print(f"  Episodes with 2+ lines: {len([x for x in lines if x >= 2])}")
    print(f"  Episodes with 3+ lines: {len([x for x in lines if x >= 3])}")
    print(f"  Episodes with 4+ lines: {len([x for x in lines if x >= 4])}")
    
    if any(x > 0 for x in lines):
        non_zero_lines = [x for x in lines if x > 0]
        print(f"  Average lines (when clearing): {np.mean(non_zero_lines):.2f}")
        print(f"  Success rate (clearing lines): {len(non_zero_lines)/len(lines)*100:.1f}%") 