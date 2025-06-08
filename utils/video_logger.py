"""
Video/GIF logging utilities for Tetris training
Provides functionality to record gameplay episodes and generate GIFs for analysis
"""

import os
import numpy as np
import pygame
from typing import List, Optional, Tuple
import time
from PIL import Image
import io

class VideoLogger:
    """Video logger for recording Tetris gameplay"""
    
    def __init__(self, 
                 output_dir: str = "videos",
                 fps: int = 10,
                 max_frames: int = 1000,
                 gif_duration_ms: int = 100):
        """
        Initialize video logger
        
        Args:
            output_dir: Directory to save videos/GIFs
            fps: Frames per second for recording
            max_frames: Maximum frames per episode
            gif_duration_ms: Duration per frame in GIF (milliseconds)
        """
        self.output_dir = output_dir
        self.fps = fps
        self.max_frames = max_frames
        self.gif_duration_ms = gif_duration_ms
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Recording state
        self.frames = []
        self.recording = False
        self.episode_id = None
        
    def start_recording(self, episode_id: str):
        """Start recording an episode"""
        self.episode_id = episode_id
        self.frames = []
        self.recording = True
        
    def stop_recording(self) -> Optional[str]:
        """Stop recording and save the episode"""
        if not self.recording or not self.frames:
            return None
        
        self.recording = False
        
        # Save as GIF
        gif_path = self.save_as_gif()
        
        # Clear frames to free memory
        self.frames = []
        
        return gif_path
    
    def capture_frame(self, surface: pygame.Surface):
        """Capture a single frame from pygame surface"""
        if not self.recording or len(self.frames) >= self.max_frames:
            return
        
        # Convert pygame surface to numpy array
        frame_array = pygame.surfarray.array3d(surface)
        # Transpose because pygame uses (width, height, channels) but PIL expects (height, width, channels)
        frame_array = np.transpose(frame_array, (1, 0, 2))
        
        # Convert to PIL Image
        frame_image = Image.fromarray(frame_array)
        self.frames.append(frame_image)
    
    def save_as_gif(self) -> str:
        """Save recorded frames as animated GIF"""
        if not self.frames:
            return None
        
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"tetris_{self.episode_id}_{timestamp}.gif"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save as animated GIF
        self.frames[0].save(
            filepath,
            save_all=True,
            append_images=self.frames[1:],
            duration=self.gif_duration_ms,
            loop=0  # Loop forever
        )
        
        return filepath
    
    def create_comparison_gif(self, 
                            surfaces: List[pygame.Surface], 
                            labels: List[str],
                            filename: str = None) -> str:
        """Create side-by-side comparison GIF from multiple surfaces"""
        
        if not surfaces or len(surfaces) != len(labels):
            raise ValueError("Surfaces and labels must have same length")
        
        # Convert surfaces to images
        images = []
        for surface in surfaces:
            frame_array = pygame.surfarray.array3d(surface)
            frame_array = np.transpose(frame_array, (1, 0, 2))
            images.append(Image.fromarray(frame_array))
        
        # Create side-by-side layout
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        
        combined_image = Image.new('RGB', (total_width, max_height + 30), color='white')
        
        # Paste images side by side
        x_offset = 0
        for i, (img, label) in enumerate(zip(images, labels)):
            combined_image.paste(img, (x_offset, 30))
            x_offset += img.width
        
        # Add labels (simplified - would need PIL text rendering for production)
        # For now, just save the combined image
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{timestamp}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        combined_image.save(filepath)
        
        return filepath

class EpisodeRecorder:
    """Records complete episodes with metadata"""
    
    def __init__(self, video_logger: VideoLogger):
        self.video_logger = video_logger
        self.episode_data = {}
        
    def start_episode(self, episode_id: str, metadata: dict = None):
        """Start recording episode with metadata"""
        self.episode_data = {
            'episode_id': episode_id,
            'start_time': time.time(),
            'frames': 0,
            'actions': [],
            'rewards': [],
            'metadata': metadata or {}
        }
        
        self.video_logger.start_recording(episode_id)
    
    def record_step(self, 
                   surface: pygame.Surface,
                   action: int,
                   reward: float,
                   state_info: dict = None):
        """Record a single step"""
        
        # Capture frame
        self.video_logger.capture_frame(surface)
        
        # Record step data
        self.episode_data['frames'] += 1
        self.episode_data['actions'].append(action)
        self.episode_data['rewards'].append(reward)
        
        if state_info:
            if 'step_info' not in self.episode_data:
                self.episode_data['step_info'] = []
            self.episode_data['step_info'].append(state_info)
    
    def end_episode(self, final_reward: float = None) -> Tuple[str, dict]:
        """End episode recording and save data"""
        
        # Stop video recording
        video_path = self.video_logger.stop_recording()
        
        # Finalize episode data
        self.episode_data['end_time'] = time.time()
        self.episode_data['duration'] = self.episode_data['end_time'] - self.episode_data['start_time']
        self.episode_data['total_reward'] = sum(self.episode_data['rewards'])
        self.episode_data['video_path'] = video_path
        
        if final_reward is not None:
            self.episode_data['final_reward'] = final_reward
        
        # Save episode metadata
        metadata_path = self._save_episode_metadata()
        self.episode_data['metadata_path'] = metadata_path
        
        return video_path, self.episode_data
    
    def _save_episode_metadata(self) -> str:
        """Save episode metadata as JSON"""
        import json
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{self.episode_data['episode_id']}_{timestamp}.json"
        filepath = os.path.join(self.video_logger.output_dir, filename)
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Create serializable copy
        serializable_data = {}
        for key, value in self.episode_data.items():
            if key == 'video_path':
                serializable_data[key] = value
            elif isinstance(value, list):
                serializable_data[key] = [convert_types(item) for item in value]
            else:
                serializable_data[key] = convert_types(value)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        return filepath

class TrainingVideoLogger:
    """Specialized video logger for training sessions"""
    
    def __init__(self, 
                 output_dir: str = "videos/training",
                 record_frequency: int = 100,  # Record every N episodes
                 record_best: bool = True,
                 record_evaluations: bool = True):
        """
        Initialize training video logger
        
        Args:
            output_dir: Directory for training videos
            record_frequency: How often to record episodes
            record_best: Whether to record new best episodes
            record_evaluations: Whether to record evaluation episodes
        """
        self.video_logger = VideoLogger(output_dir)
        self.episode_recorder = EpisodeRecorder(self.video_logger)
        
        self.record_frequency = record_frequency
        self.record_best = record_best
        self.record_evaluations = record_evaluations
        
        self.best_reward = float('-inf')
        self.recorded_episodes = []
        
    def should_record_episode(self, episode_num: int, reward: float = None, is_evaluation: bool = False) -> bool:
        """Determine if episode should be recorded"""
        
        # Always record evaluation episodes if enabled
        if is_evaluation and self.record_evaluations:
            return True
        
        # Record if it's a new best
        if self.record_best and reward is not None and reward > self.best_reward:
            self.best_reward = reward
            return True
        
        # Record based on frequency
        if episode_num % self.record_frequency == 0:
            return True
        
        return False
    
    def record_episode(self, 
                      episode_id: str,
                      env,
                      agent,
                      max_steps: int = 1000,
                      metadata: dict = None) -> Tuple[str, dict]:
        """Record a complete episode of agent playing"""
        
        # Start recording
        self.episode_recorder.start_episode(episode_id, metadata)
        
        # Reset environment
        obs = env.reset()
        total_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Get action from agent
            action = agent.select_action(obs, training=False)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Record step (if environment has rendering)
            if hasattr(env, 'surface') and env.surface is not None:
                self.episode_recorder.record_step(
                    env.surface, 
                    action, 
                    reward,
                    state_info=info
                )
            
            obs = next_obs
            step += 1
        
        # End recording
        video_path, episode_data = self.episode_recorder.end_episode(total_reward)
        
        # Store episode info
        self.recorded_episodes.append({
            'episode_id': episode_id,
            'total_reward': total_reward,
            'steps': step,
            'video_path': video_path,
            'metadata': metadata
        })
        
        return video_path, episode_data
    
    def create_training_summary_video(self, episodes_data: List[dict]) -> str:
        """Create summary video showing training progress"""
        # This would create a compilation of key moments
        # Implementation would depend on specific requirements
        pass

def create_demo_gif(env, agent, output_path: str = "demo.gif", max_steps: int = 500):
    """Create a demo GIF of agent playing"""
    
    video_logger = VideoLogger(output_dir=os.path.dirname(output_path) or ".")
    video_logger.start_recording("demo")
    
    obs = env.reset()
    step = 0
    
    while step < max_steps:
        # Get action
        action = agent.select_action(obs, training=False)
        
        # Take step  
        obs, reward, done, info = env.step(action)
        
        # Capture frame
        if hasattr(env, 'surface') and env.surface is not None:
            env.render()  # Ensure surface is updated
            video_logger.capture_frame(env.surface)
        
        step += 1
        
        if done:
            obs = env.reset()
    
    # Save GIF
    gif_path = video_logger.stop_recording()
    
    # Move to desired location if different
    if gif_path and output_path != gif_path:
        import shutil
        shutil.move(gif_path, output_path)
        return output_path
    
    return gif_path

# Example usage functions

def record_training_highlights(trainer, episodes_to_record: List[int]):
    """Record specific episodes during training"""
    video_logger = TrainingVideoLogger(
        output_dir=f"videos/training_{trainer.experiment_name}",
        record_frequency=float('inf'),  # Don't use frequency
        record_best=False,
        record_evaluations=False
    )
    
    for episode in episodes_to_record:
        if episode <= trainer.episode:
            # Record this episode
            video_path, data = video_logger.record_episode(
                f"episode_{episode}",
                trainer.env,
                trainer,
                metadata={'training_episode': episode}
            )
            print(f"Recorded episode {episode}: {video_path}")

def compare_agent_performance(agents: List, env, output_dir: str = "videos/comparison"):
    """Create comparison videos of different agents"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, agent in enumerate(agents):
        video_logger = VideoLogger(output_dir)
        recorder = EpisodeRecorder(video_logger)
        
        # Record each agent
        video_path, data = recorder.record_episode(
            f"agent_{i}",
            env,
            agent,
            metadata={'agent_index': i}
        )
        
        print(f"Agent {i} video: {video_path}")
        print(f"Performance: {data['total_reward']} reward in {data['frames']} frames") 