#!/usr/bin/env python3
"""
Performance Benchmarking Suite for AIRL Implementations
"""

import sys
import os
import time
import logging

# Add paths
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')

def benchmark_expert_trajectories(trajectory_dir):
    """Benchmark expert trajectory loading."""
    print(f"🔍 Benchmarking: {trajectory_dir}")
    
    if not os.path.exists(trajectory_dir):
        print(f"❌ Directory not found: {trajectory_dir}")
        return
    
    files = [f for f in os.listdir(trajectory_dir) if f.endswith('.pkl')]
    print(f"   📁 Found {len(files)} trajectory files")
    
    # Calculate total size
    total_size = sum(os.path.getsize(os.path.join(trajectory_dir, f)) for f in files)
    print(f"   📊 Total size: {total_size / 1024 / 1024:.2f} MB")

def benchmark_multiplayer_airl():
    """Benchmark multiplayer AIRL performance."""
    print("⚔️ Benchmarking Multiplayer AIRL")
    
    try:
        start_time = time.time()
        
        # Import and test
        from rl_utils.multiplayer_airl import MultiplayerAIRLTrainer
        
        config = {'device': 'cpu'}
        trainer = MultiplayerAIRLTrainer(config)
        
        # Run a few test episodes
        for i in range(3):
            episode_start = time.time()
            episode_data = trainer.run_competitive_episode(max_steps=50)
            episode_time = time.time() - episode_start
            print(f"   Game {i+1}: {episode_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"   ✅ Total time: {total_time:.2f}s")
        print(f"   🎮 Games played: {trainer.metrics['total_games']}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Main benchmarking function."""
    print("🚀 AIRL Performance Benchmark Suite")
    print("=" * 50)
    
    # Benchmark expert trajectories
    benchmark_expert_trajectories('expert_trajectories')
    benchmark_expert_trajectories('expert_trajectories_new')
    
    # Benchmark multiplayer AIRL
    benchmark_multiplayer_airl()
    
    print("🏁 Benchmark Complete!")

if __name__ == "__main__":
    main()
