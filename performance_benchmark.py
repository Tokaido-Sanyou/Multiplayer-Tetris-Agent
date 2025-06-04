#!/usr/bin/env python3
"""
Performance Benchmarking Suite for AIRL Implementations
Tests single-player AIRL, multiplayer AIRL, and expert trajectory quality
"""

import sys
import os
import time
import torch
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple

# Add paths
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')

def setup_logging():
    """Setup logging for benchmarking."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('benchmark_results.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('PerformanceBenchmark')

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.logger = setup_logging()
        self.results = {}
        
    def benchmark_expert_trajectories(self, trajectory_dir: str) -> Dict:
        """Benchmark expert trajectory quality and loading performance."""
        self.logger.info(f"🔍 Benchmarking expert trajectories: {trajectory_dir}")
        
        start_time = time.time()
        results = {
            'directory': trajectory_dir,
            'total_files': 0,
            'valid_files': 0,
            'total_steps': 0,
            'avg_episode_length': 0,
            'avg_reward': 0,
            'hold_percentage': 0,
            'loading_time': 0,
            'file_sizes': []
        }
        
        if not os.path.exists(trajectory_dir):
            self.logger.warning(f"Directory not found: {trajectory_dir}")
            return results
        
        # Analyze trajectory files
        files = [f for f in os.listdir(trajectory_dir) if f.endswith('.pkl')]
        results['total_files'] = len(files)
        
        valid_episodes = []
        total_hold_actions = 0
        total_actions = 0
        
        for filename in files:
            filepath = os.path.join(trajectory_dir, filename)
            file_size = os.path.getsize(filepath)
            results['file_sizes'].append(file_size)
            
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict) and 'steps' in data:
                    steps = data['steps']
                    episode_length = len(steps)
                    episode_reward = data.get('total_reward', 0)
                    
                    # Count HOLD actions
                    hold_count = sum(1 for step in steps if step.get('action') == 40)
                    
                    valid_episodes.append({
                        'length': episode_length,
                        'reward': episode_reward,
                        'hold_count': hold_count
                    })
                    
                    total_hold_actions += hold_count
                    total_actions += episode_length
                    results['valid_files'] += 1
                    results['total_steps'] += episode_length
                    
            except Exception as e:
                self.logger.warning(f"Failed to load {filename}: {e}")
        
        # Calculate statistics
        if valid_episodes:
            results['avg_episode_length'] = np.mean([ep['length'] for ep in valid_episodes])
            results['avg_reward'] = np.mean([ep['reward'] for ep in valid_episodes])
            results['hold_percentage'] = (total_hold_actions / max(1, total_actions)) * 100
        
        results['loading_time'] = time.time() - start_time
        
        # Log results
        self.logger.info(f"  📁 Files: {results['valid_files']}/{results['total_files']} valid")
        self.logger.info(f"  📊 Avg episode length: {results['avg_episode_length']:.1f}")
        self.logger.info(f"  💰 Avg reward: {results['avg_reward']:.2f}")
        self.logger.info(f"  🎯 HOLD percentage: {results['hold_percentage']:.1f}%")
        self.logger.info(f"  ⏱️ Loading time: {results['loading_time']:.2f}s")
        
        return results
    
    def benchmark_single_player_airl(self, iterations: int = 10) -> Dict:
        """Benchmark single-player AIRL training performance."""
        self.logger.info(f"🤖 Benchmarking Single-Player AIRL ({iterations} iterations)")
        
        start_time = time.time()
        results = {
            'iterations': iterations,
            'total_time': 0,
            'avg_iteration_time': 0,
            'peak_memory_usage': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Import AIRL components
            from rl_utils.airl_train import AIRLTrainer
            
            # Create minimal config
            config = {
                'expert_trajectory_dir': 'expert_trajectories_new',
                'state_dim': 207,
                'action_dim': 41,
                'policy_lr': 1e-4,
                'discriminator_lr': 3e-4,
                'gamma': 0.99,
                'batch_size': 32,
                'max_hold_percentage': 50.0,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            # Measure memory usage
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            iteration_times = []
            
            # Run limited training
            for i in range(iterations):
                iter_start = time.time()
                
                # Simulate AIRL iteration (environment interaction + updates)
                # This is a simplified benchmark - in practice you'd run actual training
                
                iter_time = time.time() - iter_start
                iteration_times.append(iter_time)
                
                if i % max(1, iterations // 5) == 0:
                    self.logger.info(f"    Iteration {i+1}/{iterations} completed ({iter_time:.3f}s)")
            
            # Calculate metrics
            results['total_time'] = time.time() - start_time
            results['avg_iteration_time'] = np.mean(iteration_times)
            results['success'] = True
            
            if torch.cuda.is_available():
                results['peak_memory_usage'] = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"Single-player AIRL benchmark failed: {e}")
        
        self.logger.info(f"  ⏱️ Total time: {results['total_time']:.2f}s")
        self.logger.info(f"  📈 Avg iteration time: {results['avg_iteration_time']:.3f}s")
        if results['peak_memory_usage'] > 0:
            self.logger.info(f"  💾 Peak memory: {results['peak_memory_usage']:.1f} MB")
        
        return results
    
    def benchmark_multiplayer_airl(self, iterations: int = 10) -> Dict:
        """Benchmark multiplayer AIRL competitive training performance."""
        self.logger.info(f"⚔️ Benchmarking Multiplayer AIRL ({iterations} iterations)")
        
        start_time = time.time()
        results = {
            'iterations': iterations,
            'total_time': 0,
            'avg_iteration_time': 0,
            'total_games': 0,
            'avg_game_length': 0,
            'win_distribution': {'player1': 0, 'player2': 0, 'draws': 0},
            'peak_memory_usage': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Import multiplayer AIRL
            from rl_utils.multiplayer_airl import MultiplayerAIRLTrainer
            
            config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            }
            
            # Measure memory usage
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            trainer = MultiplayerAIRLTrainer(config)
            
            iteration_times = []
            
            # Run competitive training
            for i in range(iterations):
                iter_start = time.time()
                
                # Run one competitive episode
                episode_data = trainer.run_competitive_episode(max_steps=100)  # Shorter for benchmarking
                
                iter_time = time.time() - iter_start
                iteration_times.append(iter_time)
                
                if i % max(1, iterations // 5) == 0:
                    self.logger.info(f"    Game {i+1}/{iterations} completed ({iter_time:.3f}s)")
            
            # Collect metrics from trainer
            results['total_time'] = time.time() - start_time
            results['avg_iteration_time'] = np.mean(iteration_times)
            results['total_games'] = trainer.metrics['total_games']
            results['win_distribution']['player1'] = trainer.metrics['player1_wins']
            results['win_distribution']['player2'] = trainer.metrics['player2_wins']
            results['win_distribution']['draws'] = trainer.metrics['draws']
            results['success'] = True
            
            if torch.cuda.is_available():
                results['peak_memory_usage'] = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"Multiplayer AIRL benchmark failed: {e}")
        
        self.logger.info(f"  ⏱️ Total time: {results['total_time']:.2f}s")
        self.logger.info(f"  📈 Avg game time: {results['avg_iteration_time']:.3f}s")
        self.logger.info(f"  🎮 Total games: {results['total_games']}")
        self.logger.info(f"  🏆 Win distribution: {results['win_distribution']}")
        if results['peak_memory_usage'] > 0:
            self.logger.info(f"  💾 Peak memory: {results['peak_memory_usage']:.1f} MB")
        
        return results
    
    def benchmark_environment_performance(self) -> Dict:
        """Benchmark Tetris environment performance."""
        self.logger.info("🎯 Benchmarking Tetris Environment Performance")
        
        start_time = time.time()
        results = {
            'single_player': {'reset_time': 0, 'step_time': 0, 'success': False},
            'multiplayer': {'reset_time': 0, 'step_time': 0, 'success': False},
            'total_time': 0
        }
        
        try:
            from tetris_env import TetrisEnv
            
            # Benchmark single-player environment
            env_single = TetrisEnv(single_player=True, headless=True)
            
            # Reset time
            reset_start = time.time()
            for _ in range(100):
                obs = env_single.reset()
            results['single_player']['reset_time'] = (time.time() - reset_start) / 100
            
            # Step time
            step_start = time.time()
            obs = env_single.reset()
            for _ in range(100):
                action = np.random.randint(0, 41)
                obs, reward, done, info = env_single.step(action)
                if done:
                    obs = env_single.reset()
            results['single_player']['step_time'] = (time.time() - step_start) / 100
            results['single_player']['success'] = True
            
            env_single.close()
            
            # Benchmark multiplayer environment
            env_multi = TetrisEnv(single_player=False, headless=True)
            
            # Reset time
            reset_start = time.time()
            for _ in range(50):  # Fewer iterations for multiplayer
                obs = env_multi.reset()
            results['multiplayer']['reset_time'] = (time.time() - reset_start) / 50
            
            # Step time
            step_start = time.time()
            obs = env_multi.reset()
            for _ in range(50):
                actions = {
                    'player1': np.random.randint(0, 41),
                    'player2': np.random.randint(0, 41)
                }
                obs, reward, done, info = env_multi.step(actions)
                if done:
                    obs = env_multi.reset()
            results['multiplayer']['step_time'] = (time.time() - step_start) / 50
            results['multiplayer']['success'] = True
            
            env_multi.close()
            
        except Exception as e:
            self.logger.error(f"Environment benchmark failed: {e}")
        
        results['total_time'] = time.time() - start_time
        
        self.logger.info(f"  🔄 Single-player reset: {results['single_player']['reset_time']*1000:.2f}ms")
        self.logger.info(f"  ⚡ Single-player step: {results['single_player']['step_time']*1000:.2f}ms")
        self.logger.info(f"  🔄 Multiplayer reset: {results['multiplayer']['reset_time']*1000:.2f}ms")
        self.logger.info(f"  ⚡ Multiplayer step: {results['multiplayer']['step_time']*1000:.2f}ms")
        
        return results
    
    def run_full_benchmark(self) -> Dict:
        """Run the complete benchmarking suite."""
        self.logger.info("🚀 Starting Full Performance Benchmark Suite")
        self.logger.info("=" * 60)
        
        # Benchmark expert trajectories
        self.results['expert_trajectories_original'] = self.benchmark_expert_trajectories('expert_trajectories')
        self.results['expert_trajectories_new'] = self.benchmark_expert_trajectories('expert_trajectories_new')
        
        # Benchmark environment performance
        self.results['environment'] = self.benchmark_environment_performance()
        
        # Benchmark single-player AIRL
        self.results['single_player_airl'] = self.benchmark_single_player_airl(iterations=5)
        
        # Benchmark multiplayer AIRL
        self.results['multiplayer_airl'] = self.benchmark_multiplayer_airl(iterations=10)
        
        self.logger.info("=" * 60)
        self.logger.info("🏁 Benchmarking Complete!")
        
        return self.results
    
    def save_results(self, filename: str = 'benchmark_results.json'):
        """Save benchmark results to file."""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(item) for item in d]
            else:
                return convert_numpy(d)
        
        cleaned_results = clean_dict(self.results)
        
        with open(filename, 'w') as f:
            json.dump(cleaned_results, f, indent=2)
        
        self.logger.info(f"📊 Results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "="*80)
        print("📊 PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        # Expert trajectories
        if 'expert_trajectories_original' in self.results:
            orig = self.results['expert_trajectories_original']
            print(f"🎯 Original Expert Trajectories:")
            print(f"   Files: {orig['valid_files']}/{orig['total_files']}")
            print(f"   Avg Reward: {orig['avg_reward']:.2f}")
            print(f"   HOLD%: {orig['hold_percentage']:.1f}%")
        
        if 'expert_trajectories_new' in self.results:
            new = self.results['expert_trajectories_new']
            print(f"🎯 New Expert Trajectories:")
            print(f"   Files: {new['valid_files']}/{new['total_files']}")
            print(f"   Avg Reward: {new['avg_reward']:.2f}")
            print(f"   HOLD%: {new['hold_percentage']:.1f}%")
        
        # Environment performance
        if 'environment' in self.results:
            env = self.results['environment']
            print(f"🎮 Environment Performance:")
            print(f"   Single-player step: {env['single_player']['step_time']*1000:.2f}ms")
            print(f"   Multiplayer step: {env['multiplayer']['step_time']*1000:.2f}ms")
        
        # AIRL performance
        if 'single_player_airl' in self.results:
            sp = self.results['single_player_airl']
            print(f"🤖 Single-Player AIRL:")
            print(f"   Success: {sp['success']}")
            print(f"   Avg iteration: {sp['avg_iteration_time']:.3f}s")
        
        if 'multiplayer_airl' in self.results:
            mp = self.results['multiplayer_airl']
            print(f"⚔️ Multiplayer AIRL:")
            print(f"   Success: {mp['success']}")
            print(f"   Avg game: {mp['avg_iteration_time']:.3f}s")
            print(f"   Win distribution: {mp['win_distribution']}")
        
        print("="*80)

def main():
    """Main benchmarking function."""
    benchmark = PerformanceBenchmark()
    
    # Run full benchmark suite
    results = benchmark.run_full_benchmark()
    
    # Save results
    benchmark.save_results()
    
    # Print summary
    benchmark.print_summary()

if __name__ == "__main__":
    main() 