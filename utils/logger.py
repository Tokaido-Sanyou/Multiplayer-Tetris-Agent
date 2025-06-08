"""
Logging utilities for Tetris ML Training
Provides structured logging for training, debugging, and analysis
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import numpy as np

class TetrisLogger:
    """Enhanced logger for Tetris training with multiple output formats"""
    
    def __init__(self, 
                 name: str = "tetris",
                 log_dir: str = "logs",
                 level: int = logging.INFO,
                 console_output: bool = True,
                 file_output: bool = True,
                 json_output: bool = True):
        
        self.name = name
        self.log_dir = log_dir
        self.level = level
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        
        # JSON structured log handler
        if json_output:
            json_file = os.path.join(log_dir, f"{name}_{timestamp}.jsonl")
            self.json_file = json_file
            self.json_handler = open(json_file, 'w')
        else:
            self.json_handler = None
        
        # Training metrics
        self.training_start_time = time.time()
        self.episode_start_time = None
        self.step_times = []
        
    def log_structured(self, level: str, event_type: str, data: Dict[str, Any]):
        """Log structured data in JSON format"""
        if self.json_handler:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'event_type': event_type,
                'logger': self.name,
                **data
            }
            
            # Convert numpy types to native Python types
            log_entry = self._convert_numpy_types(log_entry)
            
            self.json_handler.write(json.dumps(log_entry) + '\n')
            self.json_handler.flush()
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _clean_message_for_console(self, message: str) -> str:
        """Remove emojis for Windows console compatibility"""
        import re
        import sys
        if sys.platform == 'win32':
            # More comprehensive emoji removal for Windows console
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002500-\U00002BEF"  # chinese char
                "\U00002702-\U000027B0"
                "\U00002800-\U000028FF"
                "\U000024C2-\U0001F251"
                "\U0001F900-\U0001F9FF"  # supplemental symbols
                "\U0001FA00-\U0001FA6F"  # extended symbols
                "]+", flags=re.UNICODE)
            clean_msg = emoji_pattern.sub('', message)
            # Additional fallback: replace remaining high unicode chars
            clean_msg = ''.join(char if ord(char) < 0x10000 else '' for char in clean_msg)
            return clean_msg
        return message
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        clean_message = self._clean_message_for_console(message)
        self.logger.info(clean_message)
        if kwargs:
            self.log_structured('INFO', 'message', {'message': message, **kwargs})
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        clean_message = self._clean_message_for_console(message)
        self.logger.debug(clean_message)
        if kwargs:
            self.log_structured('DEBUG', 'message', {'message': message, **kwargs})
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        clean_message = self._clean_message_for_console(message)
        self.logger.warning(clean_message)
        if kwargs:
            self.log_structured('WARNING', 'message', {'message': message, **kwargs})
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        clean_message = self._clean_message_for_console(message)
        self.logger.error(clean_message)
        if kwargs:
            self.log_structured('ERROR', 'message', {'message': message, **kwargs})
    
    def log_training_start(self, config: Dict[str, Any]):
        """Log training session start"""
        self.training_start_time = time.time()
        self.info("🚀 Starting training session")
        self.log_structured('INFO', 'training_start', {
            'config': config,
            'timestamp': time.time()
        })
    
    def log_episode_start(self, episode: int, **kwargs):
        """Log episode start"""
        self.episode_start_time = time.time()
        self.debug(f"📋 Episode {episode} started")
        self.log_structured('INFO', 'episode_start', {
            'episode': episode,
            'timestamp': time.time(),
            **kwargs
        })
    
    def log_episode_end(self, episode: int, reward: float, steps: int, **kwargs):
        """Log episode completion"""
        episode_duration = time.time() - self.episode_start_time if self.episode_start_time else 0
        
        self.info(f"📊 Episode {episode}: reward={reward:.3f}, steps={steps}, duration={episode_duration:.2f}s")
        self.log_structured('INFO', 'episode_end', {
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'duration': episode_duration,
            'timestamp': time.time(),
            **kwargs
        })
    
    def log_step(self, episode: int, step: int, reward: float, action, **kwargs):
        """Log training step"""
        step_time = time.time()
        self.step_times.append(step_time)
        
        # Keep only last 100 step times for performance calculation
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]
        
        if step % 100 == 0:  # Log every 100 steps
            avg_step_time = np.mean(np.diff(self.step_times)) if len(self.step_times) > 1 else 0
            self.debug(f"⚡ Episode {episode}, Step {step}: reward={reward:.3f}, avg_step_time={avg_step_time:.4f}s")
        
        self.log_structured('DEBUG', 'training_step', {
            'episode': episode,
            'step': step,
            'reward': reward,
            'action': action,
            'timestamp': step_time,
            **kwargs
        })
    
    def log_model_checkpoint(self, episode: int, model_path: str, metrics: Dict[str, float]):
        """Log model checkpoint save"""
        self.info(f"💾 Model checkpoint saved: {model_path}")
        self.log_structured('INFO', 'model_checkpoint', {
            'episode': episode,
            'model_path': model_path,
            'metrics': metrics,
            'timestamp': time.time()
        })
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics"""
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.info(f"📈 Performance: {metrics_str}")
        self.log_structured('INFO', 'performance_metrics', {
            'metrics': metrics,
            'timestamp': time.time()
        })
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters"""
        self.info("🔧 Hyperparameters configured")
        self.log_structured('INFO', 'hyperparameters', {
            'hyperparameters': hyperparams,
            'timestamp': time.time()
        })
    
    def log_gpu_info(self, gpu_available: bool, gpu_name: str = None, memory_info: Dict = None):
        """Log GPU information"""
        if gpu_available:
            self.info(f"🎮 GPU available: {gpu_name}")
        else:
            self.info("💻 Running on CPU")
        
        self.log_structured('INFO', 'gpu_info', {
            'gpu_available': gpu_available,
            'gpu_name': gpu_name,
            'memory_info': memory_info,
            'timestamp': time.time()
        })
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with contextual information"""
        import traceback
        
        error_msg = f"❌ Error: {str(error)}"
        self.error(error_msg)
        
        self.log_structured('ERROR', 'exception', {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'timestamp': time.time()
        })
    
    def close(self):
        """Close logger and cleanup"""
        if self.json_handler:
            self.json_handler.close()
        
        # Log training session summary
        total_duration = time.time() - self.training_start_time
        self.info(f"🏁 Training session completed. Total duration: {total_duration:.2f}s")
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

class MetricsTracker:
    """Track and aggregate training metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        
    def add_metric(self, name: str, value: float):
        """Add a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # Keep only window_size values
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name] = self.metrics[name][-self.window_size:]
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics"""
        aggregated = {}
        
        for name, values in self.metrics.items():
            if values:
                aggregated[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'latest': float(values[-1]),
                    'count': len(values)
                }
        
        return aggregated
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()

def get_tetris_logger(name: str = "tetris_training", **kwargs) -> TetrisLogger:
    """Get a configured Tetris logger instance"""
    return TetrisLogger(name=name, **kwargs)

def setup_training_logger(experiment_name: str, config: Dict[str, Any]) -> TetrisLogger:
    """Setup logger for training experiment"""
    logger = get_tetris_logger(
        name=experiment_name,
        log_dir=f"logs/{experiment_name}",
        level=logging.INFO,
        console_output=True,
        file_output=True,
        json_output=True
    )
    
    logger.log_training_start(config)
    return logger 