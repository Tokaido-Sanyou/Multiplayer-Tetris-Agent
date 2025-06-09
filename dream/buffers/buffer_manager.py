#!/usr/bin/env python3
"""
🧠 ADVANCED BUFFER MANAGEMENT SYSTEM

Provides intelligent buffer size regulation, memory optimization, and experience prioritization.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time
import gc

class AdvancedBufferManager:
    """Advanced buffer management with intelligent memory regulation"""
    
    def __init__(self, 
                 max_buffer_size: int = 50000,
                 memory_threshold_mb: float = 4000.0,  # 4GB memory threshold
                 cleanup_ratio: float = 0.3,  # Remove 30% when cleaning
                 priority_mode: str = 'recent_reward'):  # 'recent_reward', 'loss_based', 'random'
        
        self.max_buffer_size = max_buffer_size
        self.memory_threshold_mb = memory_threshold_mb
        self.cleanup_ratio = cleanup_ratio
        self.priority_mode = priority_mode
        
        # Buffer statistics
        self.cleanup_count = 0
        self.total_experiences_added = 0
        self.memory_pressure_events = 0
        
        # Performance tracking
        self.cleanup_times = []
        self.buffer_size_history = []
        
        print(f"🧠 Advanced Buffer Manager initialized:")
        print(f"   Max buffer size: {max_buffer_size:,}")
        print(f"   Memory threshold: {memory_threshold_mb:.1f} MB")
        print(f"   Cleanup ratio: {cleanup_ratio*100:.1f}%")
        print(f"   Priority mode: {priority_mode}")
    
    def get_memory_usage_mb(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get comprehensive memory information"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'max_allocated_mb': max_allocated,
                'utilization': allocated / reserved if reserved > 0 else 0.0
            }
        return {'allocated_mb': 0, 'reserved_mb': 0, 'max_allocated_mb': 0, 'utilization': 0}
    
    def should_cleanup_buffer(self, current_buffer_size: int) -> bool:
        """Determine if buffer cleanup is needed"""
        # Size-based cleanup
        if current_buffer_size >= self.max_buffer_size:
            return True
        
        # Memory-based cleanup
        memory_usage = self.get_memory_usage_mb()
        if memory_usage >= self.memory_threshold_mb:
            self.memory_pressure_events += 1
            return True
        
        return False
    
    def calculate_experience_priorities(self, 
                                      experiences: List[Dict], 
                                      recent_losses: List[float] = None) -> np.ndarray:
        """Calculate priority scores for experiences"""
        
        if not experiences:
            return np.array([])
        
        n_experiences = len(experiences)
        priorities = np.ones(n_experiences)
        
        if self.priority_mode == 'recent_reward':
            # Prioritize based on recent rewards (higher absolute rewards = higher priority)
            for i, exp in enumerate(experiences):
                if 'rewards' in exp and exp['rewards']:
                    avg_reward = np.mean(np.abs(exp['rewards']))
                    priorities[i] = avg_reward + 1.0  # Add 1 to avoid zero priorities
        
        elif self.priority_mode == 'loss_based' and recent_losses:
            # Prioritize based on training loss (higher loss = higher priority)
            if len(recent_losses) >= n_experiences:
                priorities = np.array(recent_losses[-n_experiences:]) + 1.0
            else:
                # Pad with average loss
                avg_loss = np.mean(recent_losses) if recent_losses else 1.0
                priorities = np.full(n_experiences, avg_loss + 1.0)
        
        elif self.priority_mode == 'random':
            # Random prioritization
            priorities = np.random.rand(n_experiences) + 0.1
        
        # Add recency bias (recent experiences get slight priority boost)
        recency_boost = np.linspace(0.5, 1.5, n_experiences)
        priorities *= recency_boost
        
        return priorities
    
    def cleanup_buffer_intelligent(self, 
                                 buffer, 
                                 recent_losses: List[float] = None) -> Dict[str, int]:
        """Intelligently clean up buffer based on priorities"""
        
        start_time = time.time()
        original_size = len(buffer)
        
        if original_size == 0:
            return {'original_size': 0, 'removed': 0, 'final_size': 0}
        
        # Calculate how many to remove
        target_remove = int(original_size * self.cleanup_ratio)
        target_size = original_size - target_remove
        
        print(f"🔧 Intelligent buffer cleanup initiated:")
        print(f"   Original size: {original_size:,}")
        print(f"   Target removal: {target_remove:,} ({self.cleanup_ratio*100:.1f}%)")
        print(f"   Target size: {target_size:,}")
        
        try:
            # Get buffer data for priority calculation
            if hasattr(buffer, 'trajectories'):
                experiences = buffer.trajectories
            elif hasattr(buffer, 'memory'):
                experiences = buffer.memory
            else:
                # Fallback: remove from the beginning (oldest first)
                for _ in range(target_remove):
                    if len(buffer) > 0:
                        buffer.popleft() if hasattr(buffer, 'popleft') else buffer.pop(0)
                final_size = len(buffer)
                cleanup_time = time.time() - start_time
                self.cleanup_times.append(cleanup_time)
                self.cleanup_count += 1
                
                print(f"   Fallback cleanup completed in {cleanup_time:.2f}s")
                return {'original_size': original_size, 'removed': original_size - final_size, 'final_size': final_size}
            
            # Calculate priorities
            priorities = self.calculate_experience_priorities(experiences, recent_losses)
            
            if len(priorities) > 0:
                # Get indices of experiences to remove (lowest priority)
                remove_indices = np.argsort(priorities)[:target_remove]
                
                # Remove experiences (in reverse order to maintain indices)
                for idx in sorted(remove_indices, reverse=True):
                    if idx < len(experiences):
                        experiences.pop(idx)
            
            final_size = len(buffer)
            cleanup_time = time.time() - start_time
            self.cleanup_times.append(cleanup_time)
            self.cleanup_count += 1
            
            print(f"   Intelligent cleanup completed in {cleanup_time:.2f}s")
            print(f"   Final size: {final_size:,}")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'original_size': original_size,
                'removed': original_size - final_size,
                'final_size': final_size
            }
            
        except Exception as e:
            print(f"   ⚠️  Intelligent cleanup failed: {e}")
            print(f"   Falling back to simple cleanup...")
            
            # Fallback: simple removal from beginning
            for _ in range(min(target_remove, len(buffer))):
                if hasattr(buffer, 'popleft'):
                    buffer.popleft()
                elif hasattr(buffer, 'pop'):
                    buffer.pop(0)
            
            final_size = len(buffer)
            cleanup_time = time.time() - start_time
            self.cleanup_times.append(cleanup_time)
            self.cleanup_count += 1
            
            return {'original_size': original_size, 'removed': original_size - final_size, 'final_size': final_size}
    
    def monitor_buffer_health(self, buffer) -> Dict[str, any]:
        """Monitor buffer health and performance"""
        current_size = len(buffer)
        self.buffer_size_history.append(current_size)
        
        # Keep only recent history
        if len(self.buffer_size_history) > 1000:
            self.buffer_size_history = self.buffer_size_history[-1000:]
        
        memory_info = self.get_memory_info()
        
        health_metrics = {
            'buffer_size': current_size,
            'buffer_utilization': current_size / self.max_buffer_size,
            'memory_info': memory_info,
            'cleanup_stats': {
                'total_cleanups': self.cleanup_count,
                'total_experiences_added': self.total_experiences_added,
                'memory_pressure_events': self.memory_pressure_events,
                'avg_cleanup_time': np.mean(self.cleanup_times) if self.cleanup_times else 0.0
            },
            'health_status': self._assess_buffer_health(current_size, memory_info)
        }
        
        return health_metrics
    
    def _assess_buffer_health(self, buffer_size: int, memory_info: Dict) -> str:
        """Assess overall buffer health"""
        utilization = buffer_size / self.max_buffer_size
        memory_pressure = memory_info['allocated_mb'] / self.memory_threshold_mb
        
        if utilization > 0.9 or memory_pressure > 0.9:
            return "CRITICAL"
        elif utilization > 0.7 or memory_pressure > 0.7:
            return "WARNING"
        elif utilization > 0.5:
            return "HEALTHY"
        else:
            return "EXCELLENT"
    
    def get_performance_report(self) -> Dict[str, any]:
        """Get comprehensive performance report"""
        return {
            'buffer_management': {
                'max_buffer_size': self.max_buffer_size,
                'total_cleanups': self.cleanup_count,
                'total_experiences': self.total_experiences_added,
                'memory_pressure_events': self.memory_pressure_events,
                'avg_cleanup_time': np.mean(self.cleanup_times) if self.cleanup_times else 0.0,
                'cleanup_efficiency': len(self.cleanup_times) / max(self.total_experiences_added, 1)
            },
            'memory_management': self.get_memory_info(),
            'buffer_size_trend': {
                'recent_sizes': self.buffer_size_history[-10:] if self.buffer_size_history else [],
                'size_variance': np.var(self.buffer_size_history) if len(self.buffer_size_history) > 1 else 0,
                'growth_rate': self._calculate_growth_rate()
            }
        }
    
    def _calculate_growth_rate(self) -> float:
        """Calculate buffer growth rate"""
        if len(self.buffer_size_history) < 2:
            return 0.0
        
        recent_sizes = self.buffer_size_history[-10:]
        if len(recent_sizes) < 2:
            return 0.0
        
        # Simple linear regression for growth rate
        x = np.arange(len(recent_sizes))
        y = np.array(recent_sizes)
        
        if len(x) > 1:
            slope, _ = np.polyfit(x, y, 1)
            return float(slope)
        
        return 0.0
    
    def optimize_memory(self):
        """Optimize memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("🔧 Memory optimization completed")
        print(f"   Current memory: {self.get_memory_usage_mb():.1f} MB")

# Example usage and integration
class BufferIntegratedDREAM:
    """Example integration of buffer manager with DREAM training"""
    
    def __init__(self, max_buffer_size=50000):
        self.buffer_manager = AdvancedBufferManager(
            max_buffer_size=max_buffer_size,
            memory_threshold_mb=3500.0,  # Conservative threshold
            cleanup_ratio=0.25,  # Remove 25% when cleaning
            priority_mode='recent_reward'
        )
        
        # Placeholder for actual buffer
        self.replay_buffer = deque(maxlen=max_buffer_size)
        self.recent_world_losses = []
        self.recent_actor_losses = []
    
    def add_experience(self, experience):
        """Add experience with buffer management"""
        self.replay_buffer.append(experience)
        self.buffer_manager.total_experiences_added += 1
        
        # Check if cleanup is needed
        if self.buffer_manager.should_cleanup_buffer(len(self.replay_buffer)):
            recent_losses = self.recent_world_losses + self.recent_actor_losses
            cleanup_stats = self.buffer_manager.cleanup_buffer_intelligent(
                self.replay_buffer, recent_losses
            )
            return cleanup_stats
        
        return None
    
    def update_loss_history(self, world_loss: float, actor_loss: float):
        """Update loss history for priority calculation"""
        self.recent_world_losses.append(world_loss)
        self.recent_actor_losses.append(actor_loss)
        
        # Keep only recent history
        if len(self.recent_world_losses) > 200:
            self.recent_world_losses = self.recent_world_losses[-200:]
        if len(self.recent_actor_losses) > 200:
            self.recent_actor_losses = self.recent_actor_losses[-200:]
    
    def get_buffer_health_report(self):
        """Get comprehensive buffer health report"""
        return self.buffer_manager.monitor_buffer_health(self.replay_buffer) 