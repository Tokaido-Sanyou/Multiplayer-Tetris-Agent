# Utilities

This directory contains utility modules for logging, video recording, and other support functions for the Tetris ML environment.

## Overview

The utilities provide essential support functionality including Windows-compatible logging, video recording for training analysis, and various helper functions for the ML training pipeline.

## Files

### `logger.py`
Enhanced logging system with Windows console compatibility and structured JSON logging.

### `video_logger.py`
Video recording system for capturing and analyzing gameplay episodes during training.

## Logging System (`logger.py`)

### Features
- **Windows Console Compatibility**: Automatic emoji filtering for Windows console output
- **Structured JSON Logging**: Machine-readable logs for analysis
- **GPU Monitoring**: Real-time GPU memory and utilization tracking
- **Multi-format Output**: Console, file, and JSON logging simultaneously
- **Configurable Levels**: DEBUG, INFO, WARNING, ERROR levels

### Classes

#### `TetrisLogger`
Main logging class with Windows compatibility and structured output.

```python
from utils.logger import TetrisLogger

# Create logger
logger = TetrisLogger("experiment_name", log_level="INFO")

# Log messages
logger.info("Training started")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")

# Log structured data
logger.log_episode_metrics(episode=100, reward=-150.5, steps=234)
logger.log_gpu_info(available=True, device_name="RTX 3080", memory_info={})
```

#### `MetricsTracker`
Performance metrics tracking with rolling windows.

```python
from utils.logger import MetricsTracker

# Create tracker
tracker = MetricsTracker(window_size=100)

# Add metrics
tracker.add_metric('reward', -150.5)
tracker.add_metric('steps', 234)
tracker.add_metric('duration', 0.15)

# Get statistics
stats = tracker.get_metrics()
# Returns: {'reward_mean': -145.2, 'reward_std': 12.4, ...}
```

### Usage Examples

#### Basic Logging
```python
from utils.logger import setup_training_logger

# Setup logger for training
logger = setup_training_logger("my_experiment", config_dict)

# Log training progress
logger.info(f"Episode {episode}: reward={reward:.3f}, steps={steps}")
logger.log_episode_metrics(episode, reward, steps, duration)
```

#### Windows Compatibility
The logger automatically handles Windows console emoji issues:
```python
# This works on Windows without encoding errors
logger.info("🎮 Training started with GPU acceleration")
logger.info("✅ Episode completed successfully")
```

#### Structured JSON Logging
```python
# Logs are automatically saved in JSON format for analysis
logger.log_hyperparameters({
    'learning_rate': 0.0001,
    'batch_size': 32,
    'memory_size': 100000
})

logger.log_performance_metrics({
    'mean_reward': -145.2,
    'std_reward': 12.4,
    'max_reward': -120.5
})
```

## Video Logging System (`video_logger.py`)

### Features
- **Pygame Surface Capture**: Direct capture from game rendering
- **Animated GIF Generation**: Efficient GIF creation with PIL
- **Episode Metadata**: Action sequences, rewards, and timing data
- **Training Integration**: Seamless integration with training loops
- **Configurable Recording**: Frequency and quality settings

### Classes

#### `VideoLogger`
Basic video recording functionality.

```python
from utils.video_logger import VideoLogger

# Create video logger
video_logger = VideoLogger(
    output_dir="videos",
    fps=10,
    max_frames=1000,
    gif_duration_ms=100
)

# Record episode
video_logger.start_recording("episode_001")
for frame in game_frames:
    video_logger.capture_frame(surface)
gif_path = video_logger.stop_recording()
```

#### `EpisodeRecorder`
Complete episode recording with metadata.

```python
from utils.video_logger import EpisodeRecorder

# Create recorder
recorder = EpisodeRecorder(output_dir="videos")

# Record episode with metadata
recorder.start_episode("training_ep_100", metadata={'config': 'default'})

for step in episode:
    recorder.record_step(
        surface=game_surface,
        action=action_taken,
        reward=step_reward,
        state_info={'score': current_score}
    )

gif_path, episode_data = recorder.end_episode(final_reward=total_reward)
```

#### `TrainingVideoLogger`
Specialized for training session recording.

```python
from utils.video_logger import TrainingVideoLogger

# Create training video logger
training_logger = TrainingVideoLogger(
    output_dir="videos/experiment",
    record_frequency=100,  # Record every 100 episodes
    record_best=True,      # Always record best episodes
    record_evaluations=True  # Record evaluation episodes
)

# Check if should record
if training_logger.should_record_episode(episode_num, reward, is_evaluation=True):
    gif_path, data = training_logger.record_episode(
        episode_id=f"eval_ep_{episode_num}",
        env=environment,
        agent=dqn_agent,
        max_steps=500,
        metadata={'episode': episode_num, 'reward': reward}
    )
```

### Usage Examples

#### Training Integration
```python
# In training loop
if episode % 100 == 0:  # Record every 100 episodes
    video_path, episode_data = video_logger.record_episode(
        episode_id=f"training_ep_{episode}",
        env=env,
        agent=agent,
        max_steps=1000,
        metadata={
            'episode': episode,
            'epsilon': current_epsilon,
            'learning_rate': learning_rate
        }
    )
    
    if video_path:
        logger.info(f"Video recorded: {video_path}")
        logger.info(f"Episode stats: {episode_data}")
```

#### Best Episode Recording
```python
# Track best episodes
best_reward = float('-inf')

for episode in training_episodes:
    reward = run_episode()
    
    if reward > best_reward:
        best_reward = reward
        # Record new best episode
        video_logger.record_episode(
            episode_id=f"best_ep_{episode}",
            env=env,
            agent=agent,
            metadata={'reward': reward, 'episode': episode}
        )
```

#### Evaluation Recording
```python
# Record evaluation episodes
def evaluate_agent(agent, num_episodes=10):
    for i in range(num_episodes):
        video_path, data = video_logger.record_episode(
            episode_id=f"eval_{i}",
            env=eval_env,
            agent=agent,
            max_steps=2000,
            metadata={'evaluation': True, 'episode': i}
        )
        
        # Analyze recorded data
        print(f"Episode {i}: {data['total_reward']:.2f} reward, {data['total_steps']} steps")
```

## Directory Structure

The utilities create the following directory structure:

```
├── logs/[experiment_name]/
│   ├── training.log            # Human-readable logs
│   ├── training.jsonl          # Structured JSON logs
│   ├── gpu_metrics.json        # GPU monitoring data
│   └── metrics_history.json    # Training metrics history
├── videos/[experiment_name]/
│   ├── training_ep_100.gif     # Training episodes
│   ├── eval_ep_200.gif         # Evaluation episodes
│   ├── best_episode.gif        # Best performance
│   └── metadata/               # Episode metadata
│       ├── training_ep_100.json
│       └── eval_ep_200.json
```

## Configuration

### Logger Configuration
```python
logger_config = {
    'log_level': 'INFO',           # Logging level
    'console_output': True,        # Enable console output
    'file_output': True,           # Enable file output
    'json_output': True,           # Enable JSON structured logs
    'log_dir': 'logs/experiment',  # Log directory
    'max_file_size': '10MB',       # Log rotation size
    'backup_count': 5              # Number of backup files
}
```

### Video Logger Configuration
```python
video_config = {
    'output_dir': 'videos',        # Output directory
    'fps': 10,                     # Frames per second
    'max_frames': 1000,            # Maximum frames per episode
    'gif_duration_ms': 100,        # Frame duration in GIF
    'record_frequency': 100,       # Recording frequency
    'record_best': True,           # Record best episodes
    'record_evaluations': True,    # Record evaluation episodes
    'compression_level': 6         # GIF compression level
}
```

## Performance Considerations

### Logging Performance
- **Async Logging**: Non-blocking log writes
- **Batch Processing**: Efficient batch log processing
- **Memory Management**: Automatic log rotation and cleanup
- **Filtering**: Configurable log level filtering

### Video Recording Performance
- **Frame Buffering**: Efficient frame capture and buffering
- **Compression**: Optimized GIF compression
- **Memory Usage**: Automatic frame cleanup
- **Background Processing**: Non-blocking video generation

## Error Handling

### Logging Errors
```python
try:
    logger.info("Training progress")
except Exception as e:
    # Fallback to print if logging fails
    print(f"Logging error: {e}")
```

### Video Recording Errors
```python
try:
    video_logger.record_episode(...)
except Exception as e:
    logger.error(f"Video recording failed: {e}")
    # Continue training without video
```

## Integration with Training

### Complete Training Integration
```python
from utils.logger import setup_training_logger
from utils.video_logger import TrainingVideoLogger

# Setup logging and video recording
logger = setup_training_logger("experiment", config)
video_logger = TrainingVideoLogger("videos/experiment")

# Training loop with full integration
for episode in range(max_episodes):
    # Run episode
    reward, steps = run_training_episode()
    
    # Log progress
    logger.log_episode_metrics(episode, reward, steps, duration)
    
    # Record video if needed
    if video_logger.should_record_episode(episode, reward):
        video_logger.record_episode(f"ep_{episode}", env, agent)
    
    # Evaluation with video
    if episode % eval_freq == 0:
        eval_reward = evaluate_agent()
        logger.log_performance_metrics({'eval_reward': eval_reward})
        
        if video_logger.should_record_episode(episode, eval_reward, True):
            video_logger.record_episode(f"eval_{episode}", env, agent)
```

## Best Practices

1. **Use Appropriate Log Levels**: DEBUG for development, INFO for production
2. **Structure Metadata**: Include relevant context in video metadata
3. **Monitor Disk Usage**: Video files can be large, monitor storage
4. **Regular Cleanup**: Implement log and video cleanup policies
5. **Error Resilience**: Don't let logging/video errors stop training
6. **Performance Monitoring**: Track logging and video recording overhead

## Troubleshooting

### Common Issues

1. **Windows Encoding Errors**
   - Automatic emoji filtering handles most cases
   - Use ASCII-only messages if issues persist

2. **Large Video Files**
   - Reduce fps or max_frames
   - Increase compression_level
   - Implement cleanup policies

3. **Slow Logging**
   - Reduce log level
   - Disable JSON logging if not needed
   - Use async logging for high-frequency logs

4. **Memory Issues**
   - Reduce video buffer sizes
   - Implement frame cleanup
   - Monitor memory usage during recording

This utility system provides robust, production-ready logging and video recording capabilities optimized for ML training workflows. 