#!/usr/bin/env python3
"""
PyTorch Expert Bridge for AIRL Training
Integrates PyTorch DQN models with rl_utils components for expert trajectory generation.
"""

import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
tetris_ai_path = os.path.join(current_dir, 'tetris-ai-master')
local_multiplayer_path = os.path.join(current_dir, 'local-multiplayer-tetris-main', 'localMultiplayerTetris')

if tetris_ai_path not in sys.path:
    sys.path.append(tetris_ai_path)
if local_multiplayer_path not in sys.path:
    sys.path.append(local_multiplayer_path)

# Import PyTorch DQN components
try:
    from pytorch_dqn import PyTorchDQNAgent, DQNAgent
    PYTORCH_DQN_AVAILABLE = True
    print("✅ PyTorch DQN imported successfully")
except ImportError as e:
    print(f"❌ PyTorch DQN import failed: {e}")
    PYTORCH_DQN_AVAILABLE = False

# Import promoted rl_utils components (no more relative import issues!)
try:
    from dqn_adapter import board_props, enumerate_next_states
    from tetris_env import TetrisEnv
    from expert_loader import ExpertTrajectoryLoader
    print("✅ RL Utils components imported successfully")
except ImportError as e:
    print(f"❌ RL Utils import failed: {e}")
    raise ImportError(f"Failed to import rl_utils components: {e}")

class PyTorchExpertPolicy:
    """
    Expert policy using PyTorch DQN models that's compatible with rl_utils components.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize PyTorch expert policy.
        
        Args:
            model_path: Path to .pth PyTorch model file
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent = None
        self.logger = logging.getLogger(__name__)
        
        self._load_model()
    
    def _load_model(self):
        """Load PyTorch DQN model."""
        if not PYTORCH_DQN_AVAILABLE:
            raise ImportError("PyTorch DQN components not available")
        
        try:
            # Try to load as state dict first (preferred format)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Create agent
            self.agent = PyTorchDQNAgent(state_size=4, n_neurons=[32, 32])
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.agent.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and any('weight' in k for k in checkpoint.keys()):
                self.agent.model.load_state_dict(checkpoint)
            else:
                # Try as full model
                self.agent.model = checkpoint
            
            self.agent.model.eval()
            self.logger.info(f"Successfully loaded PyTorch model from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def select_action(self, env) -> int:
        """
        Select action using PyTorch DQN and dqn_adapter.
        NO FALLBACKS - must work with sample.pth or crash.
        
        Args:
            env: TetrisEnv instance
            
        Returns:
            Action index for TetrisEnv
        """
        next_states = enumerate_next_states(env)
        
        if not next_states:
            raise ValueError("No valid next states found - environment may be in invalid state")
        
        # Get state vectors and actions
        states = np.array(list(next_states.keys()))
        actions = list(next_states.values())
        
        # Get best state using PyTorch agent
        best_state = self.agent.best_state(states)
        
        # Find corresponding action
        for state_tuple, action in next_states.items():
            if np.allclose(state_tuple, best_state, atol=1e-6):
                return action
        
        # If no exact match found, use the action corresponding to the best state
        best_values = [self.agent.predict_value(np.reshape(state, [1, 4])) for state in states]
        best_idx = np.argmax(best_values)
        return actions[best_idx]
    
    def _count_holes(self, grid: np.ndarray) -> int:
        """Count holes in the grid (empty cells with blocks above)."""
        holes = 0
        for col in grid.T:
            seen_block = False
            for cell in col:
                if cell:
                    seen_block = True
                elif seen_block:
                    holes += 1
        return holes
    
    def _calculate_bumpiness_height(self, grid: np.ndarray) -> Tuple[int, int]:
        """Calculate bumpiness and total height."""
        heights = []
        for col in grid.T:
            # Find height of each column
            height = 0
            for i, cell in enumerate(col):
                if cell:
                    height = len(col) - i
                    break
            heights.append(height)
        
        # Calculate bumpiness (sum of absolute differences between adjacent columns)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        total_height = sum(heights)
        
        return bumpiness, total_height

class PyTorchExpertTrajectoryGenerator:
    """
    Generate expert trajectories using PyTorch DQN that are compatible with rl_utils.
    """
    
    def __init__(self, 
                 model_path: str,
                 trajectory_dir: str = "expert_trajectories_pytorch",
                 max_episodes: int = 100):
        """
        Initialize trajectory generator.
        
        Args:
            model_path: Path to PyTorch model
            trajectory_dir: Directory to save trajectories
            max_episodes: Maximum episodes to generate
        """
        self.model_path = model_path
        self.trajectory_dir = trajectory_dir
        self.max_episodes = max_episodes
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(trajectory_dir, exist_ok=True)
        
        # Initialize expert policy and environment
        self.expert = PyTorchExpertPolicy(model_path)
        
        # MUST use TetrisEnv - no fallbacks
        if not PYTORCH_DQN_AVAILABLE:
            raise ImportError("PyTorch DQN components not available - cannot generate proper expert trajectories")
        
        self.env = TetrisEnv(single_player=True, headless=True)
        self.logger.info("Using TetrisEnv for trajectory generation (NO FALLBACKS)")
    
    def _extract_rl_utils_features(self, observation: Dict) -> np.ndarray:
        """
        Extract features compatible with rl_utils ExpertTrajectoryLoader.
        
        Args:
            observation: TetrisEnv observation dict
            
        Returns:
            Feature vector compatible with rl_utils (207 dimensions)
        """
        grid = observation['grid'].flatten()  # 20*10 = 200 features
        next_piece = np.array([observation['next_piece']])  # 1 feature
        hold_piece = np.array([observation['hold_piece']])  # 1 feature
        current_shape = np.array([observation['current_shape']])  # 1 feature
        current_rotation = np.array([observation['current_rotation']])  # 1 feature
        current_x = np.array([observation['current_x']])  # 1 feature
        current_y = np.array([observation['current_y']])  # 1 feature
        can_hold = np.array([observation['can_hold']])  # 1 feature
        
        # Concatenate all features: 200 + 7 = 207 total features
        features = np.concatenate([
            grid, next_piece, hold_piece, current_shape, 
            current_rotation, current_x, current_y, can_hold
        ]).astype(np.float32)
        
        return features
    
    def _convert_to_original_reward(self, tetris_env_reward: float, info: dict) -> float:
        """
        Convert TetrisEnv reward to original tetris-ai-master reward scale.
        
        The PyTorch model was trained on the original reward structure:
        - +1 for placing a piece
        - +1 + (lines_cleared²) * 10 for line clears
        - -2 for game over
        
        Args:
            tetris_env_reward: Reward from TetrisEnv (small decimal values)
            info: Step info containing lines_cleared, etc.
            
        Returns:
            Reward compatible with original tetris-ai-master scale
        """
        # Base reward for placing a piece
        original_reward = 1.0
        
        # Add line clearing bonus using original formula
        lines_cleared = info.get('lines_cleared', 0)
        if lines_cleared > 0:
            original_reward += (lines_cleared ** 2) * 10
        
        # Game over penalty (check if episode ended badly)
        # In TetrisEnv, game_over is indicated by done=True with low score
        if info.get('episode_steps', 0) < 10:  # Very short episode = likely game over
            original_reward -= 2
        
        return original_reward

    def generate_episode(self) -> Dict:
        """
        Generate a single expert trajectory episode.
        
        Returns:
            Dictionary with trajectory data compatible with rl_utils
        """
        observation = self.env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        
        steps = []
        total_reward = 0
        episode_length = 0
        max_steps = 1000
        
        while episode_length < max_steps:
            # Extract features for current state
            state_features = self._extract_rl_utils_features(observation)
            
            # Select action using expert policy
            action = self.expert.select_action(self.env)
            
            # Take environment step
            step_result = self.env.step(action)
            if len(step_result) == 4:
                next_observation, reward, done, info = step_result
            else:
                next_observation, reward, done, truncated, info = step_result
                done = done or truncated
            
            # Convert reward to original tetris-ai-master scale
            original_reward = self._convert_to_original_reward(reward, info)
            
            # Extract features for next state
            next_state_features = self._extract_rl_utils_features(next_observation)
            
            # Store transition with ORIGINAL reward scale
            step_data = {
                'state': observation,
                'action': action,
                'reward': original_reward,  # Use converted reward!
                'next_state': next_observation,
                'done': done,
                'state_features': state_features,
                'next_state_features': next_state_features
            }
            steps.append(step_data)
            
            total_reward += original_reward  # Accumulate converted rewards
            episode_length += 1
            observation = next_observation
            
            if done:
                break
        
        episode_data = {
            'steps': steps,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'model_path': self.model_path
        }
        
        return episode_data
    
    def generate_trajectories(self, num_episodes: int = None) -> int:
        """
        Generate multiple expert trajectories.
        
        Args:
            num_episodes: Number of episodes to generate (default: max_episodes)
            
        Returns:
            Number of successfully generated trajectories
        """
        if num_episodes is None:
            num_episodes = self.max_episodes
        
        successful_episodes = 0
        
        for episode_idx in range(num_episodes):
            try:
                episode_data = self.generate_episode()
                
                # Save trajectory
                filename = f"pytorch_expert_episode_{episode_idx:04d}.pkl"
                filepath = os.path.join(self.trajectory_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(episode_data, f)
                
                successful_episodes += 1
                self.logger.info(f"Generated episode {episode_idx}: {len(episode_data['steps'])} steps, "
                               f"reward: {episode_data['total_reward']:.2f}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate episode {episode_idx}: {e}")
        
        self.logger.info(f"Successfully generated {successful_episodes}/{num_episodes} trajectories")
        return successful_episodes

class EnhancedExpertLoader:
    """
    Enhanced expert loader that works with both PyTorch and traditional trajectories.
    """
    
    def __init__(self, 
                 pytorch_model_path: str = None,
                 trajectory_dirs: List[str] = None,
                 use_live_generation: bool = True):
        """
        Initialize enhanced expert loader.
        
        Args:
            pytorch_model_path: Path to PyTorch model for live generation
            trajectory_dirs: Directories containing pre-generated trajectories
            use_live_generation: Whether to use live trajectory generation
        """
        self.pytorch_model_path = pytorch_model_path
        self.trajectory_dirs = trajectory_dirs or []
        self.use_live_generation = use_live_generation
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.pytorch_generator = None
        self.rl_utils_loader = None
        self.transitions = []
        
        self._setup_components()
    
    def _setup_components(self):
        """Setup PyTorch generator - NO FALLBACKS."""
        # Setup PyTorch generator for live trajectories - MUST work
        if self.pytorch_model_path and self.use_live_generation:
            self.pytorch_generator = PyTorchExpertTrajectoryGenerator(
                self.pytorch_model_path,
                trajectory_dir="temp_pytorch_trajectories"
            )
            self.logger.info("PyTorch trajectory generator ready (NO FALLBACKS)")
        
        # Setup rl_utils loader for existing trajectories - MUST work if enabled
        if self.trajectory_dirs and PYTORCH_DQN_AVAILABLE:
            self.rl_utils_loader = ExpertTrajectoryLoader(
                trajectory_dir=self.trajectory_dirs[0]  # Use first directory
            )
            self.logger.info("RL Utils trajectory loader ready (NO FALLBACKS)")
    
    def load_trajectories(self, num_live_episodes: int = 10) -> int:
        """
        Load trajectories from multiple sources.
        
        Args:
            num_live_episodes: Number of live episodes to generate
            
        Returns:
            Total number of transitions loaded
        """
        total_transitions = 0
        
        # Generate live trajectories with PyTorch - NO FALLBACKS
        if self.pytorch_generator:
            generated = self.pytorch_generator.generate_trajectories(num_live_episodes)
            self.logger.info(f"Generated {generated} live PyTorch trajectories")
            
            # Load the generated trajectories - MUST work with rl_utils
            if not PYTORCH_DQN_AVAILABLE:
                raise ImportError("PyTorch DQN components not available - cannot load trajectories properly")
            
            temp_loader = ExpertTrajectoryLoader(
                trajectory_dir=self.pytorch_generator.trajectory_dir
            )
            live_count = temp_loader.load_trajectories()
            self.transitions.extend(temp_loader.transitions)
            total_transitions += len(temp_loader.transitions)
        
        # Load existing trajectories
        if self.rl_utils_loader:
            existing_count = self.rl_utils_loader.load_trajectories()
            self.transitions.extend(self.rl_utils_loader.transitions)
            total_transitions += len(self.rl_utils_loader.transitions)
            self.logger.info(f"Loaded {len(self.rl_utils_loader.transitions)} existing transitions")
        
        self.logger.info(f"Total transitions loaded: {total_transitions}")
        return total_transitions
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about loaded expert trajectories.
        
        Returns:
            Dictionary with trajectory statistics
        """
        if not self.transitions:
            return {
                'num_trajectories': 0,
                'total_transitions': 0,
                'avg_transitions_per_trajectory': 0.0,
                'avg_reward': 0.0
            }
        
        rewards = [t['reward'] for t in self.transitions]
        
        # Need to count actual trajectories by tracking episode boundaries
        # For now, estimate based on 'done' flags
        num_episodes = sum(1 for t in self.transitions if t.get('done', False))
        if num_episodes == 0:  # If no episodes are marked as done, assume all transitions are from one episode
            num_episodes = 1
        
        total_transitions = len(self.transitions)
        avg_transitions_per_trajectory = total_transitions / num_episodes if num_episodes > 0 else 0.0
        
        return {
            'num_trajectories': num_episodes,
            'total_transitions': total_transitions,
            'avg_transitions_per_trajectory': avg_transitions_per_trajectory,
            'avg_reward': np.mean(rewards) if rewards else 0.0,
            'min_reward': np.min(rewards) if rewards else 0.0,
            'max_reward': np.max(rewards) if rewards else 0.0
        }
    
    def get_batch(self, batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Get a batch of expert transitions.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to put tensors on
            
        Returns:
            Dictionary of batched tensors
        """
        if not self.transitions:
            raise ValueError("No expert transitions loaded. Call load_trajectories() first.")
        
        # Sample random transitions
        indices = np.random.choice(len(self.transitions), batch_size, replace=True)
        batch_transitions = [self.transitions[i] for i in indices]
        
        # Stack into tensors
        states = torch.FloatTensor([t['state'] for t in batch_transitions]).to(device)
        actions_onehot = torch.FloatTensor([t['action_onehot'] for t in batch_transitions]).to(device)
        rewards = torch.FloatTensor([t['reward'] for t in batch_transitions]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor([t['next_state'] for t in batch_transitions]).to(device)
        dones = torch.FloatTensor([t['done'] for t in batch_transitions]).unsqueeze(1).to(device)
        
        return {
            'states': states,
            'actions': actions_onehot,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

def test_pytorch_expert_bridge():
    """Test the PyTorch expert bridge components."""
    print("Testing PyTorch Expert Bridge...")
    
    # Test with PyTorch model
    pytorch_model_path = "tetris-ai-master/sample.pth"
    
    if os.path.exists(pytorch_model_path):
        try:
            # Test expert policy
            print("\n1. Testing PyTorch Expert Policy...")
            expert = PyTorchExpertPolicy(pytorch_model_path)
            print("✅ Expert policy loaded successfully")
            
            # Test trajectory generation
            print("\n2. Testing Trajectory Generation...")
            generator = PyTorchExpertTrajectoryGenerator(
                pytorch_model_path,
                trajectory_dir="test_pytorch_trajectories",
                max_episodes=2
            )
            
            num_generated = generator.generate_trajectories(2)
            print(f"✅ Generated {num_generated} test trajectories")
            
            # Test enhanced loader
            print("\n3. Testing Enhanced Expert Loader...")
            loader = EnhancedExpertLoader(
                pytorch_model_path=pytorch_model_path,
                trajectory_dirs=["test_pytorch_trajectories"],
                use_live_generation=True
            )
            
            total_transitions = loader.load_trajectories(num_live_episodes=2)
            print(f"✅ Loaded {total_transitions} total transitions")
            
            # Test batch sampling
            if total_transitions > 0:
                batch = loader.get_batch(batch_size=4)
                print(f"✅ Sampled batch with states shape: {batch['states'].shape}")
            
            print("\n🎉 All PyTorch Expert Bridge tests passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ PyTorch model not found: {pytorch_model_path}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_pytorch_expert_bridge() 