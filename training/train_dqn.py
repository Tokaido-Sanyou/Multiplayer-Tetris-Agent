#!/usr/bin/env python3
"""
DQN Training Script for Tetris
Main training script using configuration system and proper directory structure
"""

import os
import sys
import argparse
import time
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from configs.dqn_config import DQNConfig, get_config
from algorithms.dqn_trainer import DQNTrainer
from envs.tetris_env import TetrisEnv
from utils.logger import setup_training_logger

def create_trainer_from_config(config: DQNConfig) -> DQNTrainer:
    """Create DQN trainer from configuration"""
    
    # Create environment
    env = TetrisEnv(**config.environment)
    
    # Create trainer
    trainer = DQNTrainer(
        env=env,
        model_config=config.model,
        training_config=config.training,
        experiment_name=config.config_name
    )
    
    return trainer

def train_with_config(config_name: str, custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Train DQN with specified configuration"""
    
    # Load configuration
    if custom_config:
        config = DQNConfig.from_dict(custom_config, f"custom_{config_name}")
    else:
        config = get_config(config_name)
    
    print(f"Starting training with config: {config.config_name}")
    print(f"Environment: {config.environment}")
    print(f"Training parameters: {config.training}")
    
    # Save configuration
    config.save_config()
    
    # Create trainer
    trainer = create_trainer_from_config(config)
    
    # Record start time
    start_time = time.time()
    
    # Train
    try:
        trainer.train()
        
        # Training completed successfully
        training_time = time.time() - start_time
        
        # Final evaluation
        print("Running final evaluation...")
        final_metrics = trainer.evaluate(num_episodes=20)
        
        # Create results summary
        results = {
            'config_name': config.config_name,
            'training_time': training_time,
            'total_episodes': trainer.episode,
            'total_steps': trainer.total_steps,
            'final_metrics': final_metrics,
            'config': config.get_full_config()
        }
        
        # Save results
        results_file = os.path.join(config.paths['results_dir'], 'training_results.json')
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training completed successfully!")
        print(f"Total time: {training_time:.2f} seconds")
        print(f"Final performance: {final_metrics}")
        print(f"Results saved to: {results_file}")
        
        return results
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        training_time = time.time() - start_time
        
        # Save partial results
        results = {
            'config_name': config.config_name,
            'training_time': training_time,
            'total_episodes': trainer.episode,
            'total_steps': trainer.total_steps,
            'status': 'interrupted',
            'config': config.get_full_config()
        }
        
        results_file = os.path.join(config.paths['results_dir'], 'training_results_interrupted.json')
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Partial results saved to: {results_file}")
        return results
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error results
        results = {
            'config_name': config.config_name,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'config': config.get_full_config()
        }
        
        results_file = os.path.join(config.paths['results_dir'], 'training_results_failed.json')
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        raise

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Train DQN agent for Tetris')
    
    parser.add_argument('--config', '-c', type=str, default='default',
                       help='Configuration name (default, fast, research, locked_position, debug)')
    
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to custom configuration file')
    
    parser.add_argument('--episodes', type=int, default=None,
                       help='Override max episodes')
    
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Override learning rate')
    
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    
    parser.add_argument('--action-mode', type=str, default=None,
                       choices=['direct', 'locked_position'],
                       help='Override action mode')
    
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='Override device')
    
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configurations')
    
    args = parser.parse_args()
    
    # List configurations
    if args.list_configs:
        print("Available configurations:")
        from configs.dqn_config import CONFIGS
        for name in CONFIGS.keys():
            config = get_config(name)
            print(f"  {name}:")
            print(f"    Episodes: {config.training['max_episodes']}")
            print(f"    Memory: {config.training['memory_size']}")
            print(f"    Action mode: {config.environment['action_mode']}")
            print(f"    Device: {config.training['device']}")
            print()
        return
    
    # Create custom config with overrides
    custom_config = None
    if any([args.episodes, args.learning_rate, args.batch_size, args.action_mode, args.device]):
        custom_config = {}
        
        if args.episodes or args.learning_rate or args.batch_size or args.device:
            custom_config['training'] = {}
            if args.episodes:
                custom_config['training']['max_episodes'] = args.episodes
            if args.learning_rate:
                custom_config['training']['learning_rate'] = args.learning_rate
            if args.batch_size:
                custom_config['training']['batch_size'] = args.batch_size
            if args.device:
                custom_config['training']['device'] = args.device
        
        if args.action_mode:
            custom_config['environment'] = {'action_mode': args.action_mode}
    
    # Load from file if specified
    if args.config_file:
        config = DQNConfig("file_config")
        config.load_config(args.config_file)
        
        # Apply overrides
        if custom_config:
            if 'training' in custom_config:
                config.training.update(custom_config['training'])
            if 'environment' in custom_config:
                config.environment.update(custom_config['environment'])
        
        # Train with loaded config
        results = train_with_config("file_config", config.get_full_config())
    else:
        # Train with named config
        results = train_with_config(args.config, custom_config)
    
    return results

if __name__ == "__main__":
    main() 