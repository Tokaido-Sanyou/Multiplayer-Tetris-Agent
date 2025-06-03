"""
Unified Training Script: Orchestrates exploration and exploitation phases
Implements the 6-phase algorithm with batch processing and episode rollouts
"""
import os
import sys
import argparse
import logging
import torch
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter

# Handle both direct execution and module import
try:
    from ..tetris_env import TetrisEnv
    from ..config import TetrisConfig  # Import centralized config
    from .exploration_actor import ExplorationActor
    from .state_model import StateModel
    from .actor_critic import ActorCriticAgent
    from .future_reward_predictor import FutureRewardPredictor
    from .replay_buffer import ReplayBuffer
except ImportError:
    # Direct execution - add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tetris_env import TetrisEnv
    from config import TetrisConfig  # Import centralized config
    from rl_utils.exploration_actor import ExplorationActor
    from rl_utils.state_model import StateModel
    from rl_utils.actor_critic import ActorCriticAgent
    from rl_utils.future_reward_predictor import FutureRewardPredictor
    from rl_utils.replay_buffer import ReplayBuffer

class UnifiedTrainer:
    """
    Unified trainer that implements the 6-phase exploration-exploitation algorithm
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Get centralized network config
        self.tetris_config = TetrisConfig()
        
        # Initialize environment - headless by default, enable rendering only when needed
        self.env = TetrisEnv(single_player=True, headless=True)
        
        # Initialize models with centralized dimensions
        self.state_model = StateModel(state_dim=self.tetris_config.STATE_DIM).to(self.device)
        self.future_reward_predictor = FutureRewardPredictor(
            state_dim=self.tetris_config.STATE_DIM, 
            action_dim=self.tetris_config.ACTION_DIM
        ).to(self.device)
        self.actor_critic = ActorCriticAgent(
            state_dim=self.tetris_config.STATE_DIM, 
            action_dim=self.tetris_config.ACTION_DIM, 
            clip_ratio=config.clip_ratio,
            state_model=self.state_model
        )
        
        # Initialize exploration actor
        self.exploration_actor = ExplorationActor(self.env)
        
        # Initialize optimizers
        self.state_optimizer = torch.optim.Adam(self.state_model.parameters(), lr=config.state_lr)
        self.reward_optimizer = torch.optim.Adam(self.future_reward_predictor.parameters(), lr=config.reward_lr)
          # Add reward optimizer to actor-critic
        self.actor_critic.reward_optimizer = self.reward_optimizer
        
        # Training state
        self.phase = 1
        self.episode_count = 0
        self.batch_count = 0
        
        # Data storage
        self.exploration_data = []
        self.experience_buffer = ReplayBuffer(config.buffer_size)
        
        # Logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Set TensorBoard writer for ActorCritic logging
        self.actor_critic.set_writer(self.writer)
        
    def run_training(self):
        """
        Main training loop implementing the 6-phase algorithm
        """
        logging.info("Starting unified training with 6-phase algorithm")
        
        for batch in range(self.config.num_batches):
            logging.info(f"Starting batch {batch + 1}/{self.config.num_batches}")
            
            # Phase 1: Exploration data collection
            self.phase_1_exploration(batch)
            
            # Phase 2: State model learning
            self.phase_2_state_learning(batch)
            
            # Phase 3: Future reward prediction
            self.phase_3_reward_prediction(batch)
            
            # Phase 4: Exploitation episodes
            self.phase_4_exploitation(batch)
            
            # Phase 5: PPO-style training
            self.phase_5_ppo_training(batch)
            
            # Phase 6: Model evaluation
            self.phase_6_evaluation(batch)
            
            # Save checkpoints
            if (batch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(batch)
                self.writer.close()
        logging.info("Training completed")
    
    def phase_1_exploration(self, batch):
        """
        Phase 1: Collect exploration data using systematic placement trials
        """
        logging.info(f"Phase 1: Exploration data collection (batch {batch})")
        
        # Collect placement data
        placement_data = self.exploration_actor.collect_placement_data(
            num_episodes=self.config.exploration_episodes
        )
        
        self.exploration_data.extend(placement_data)
        
        # Log detailed exploration statistics
        if placement_data:
            rewards = [d['terminal_reward'] for d in placement_data]
            
            # Basic statistics
            self.writer.add_scalar('Exploration/AvgTerminalReward', np.mean(rewards), batch)
            self.writer.add_scalar('Exploration/StdTerminalReward', np.std(rewards), batch)
            self.writer.add_scalar('Exploration/MaxTerminalReward', np.max(rewards), batch)
            self.writer.add_scalar('Exploration/MinTerminalReward', np.min(rewards), batch)
            self.writer.add_scalar('Exploration/NumPlacements', len(placement_data), batch)
            
            # Reward distribution
            positive_rewards = [r for r in rewards if r > 0]
            negative_rewards = [r for r in rewards if r <= 0]
            self.writer.add_scalar('Exploration/PositivePlacementRate', len(positive_rewards) / len(rewards), batch)
            self.writer.add_scalar('Exploration/AvgPositiveReward', np.mean(positive_rewards) if positive_rewards else 0, batch)
            self.writer.add_scalar('Exploration/AvgNegativeReward', np.mean(negative_rewards) if negative_rewards else 0, batch)
            
            # Placement success metrics
            high_reward_threshold = -50  # Adjust based on reward scale
            successful_placements = [r for r in rewards if r > high_reward_threshold]
            self.writer.add_scalar('Exploration/SuccessfulPlacementRate', len(successful_placements) / len(rewards), batch)
            logging.info(f"Collected {len(placement_data)} placements, avg reward: {np.mean(rewards):.3f}, success rate: {len(successful_placements)/len(rewards):.2%}")
    
    def phase_2_state_learning(self, batch):
        """
        Phase 2: Train state model to learn optimal placements from terminal rewards
        """
        logging.info(f"Phase 2: State model learning (batch {batch})")
        
        if not self.exploration_data:
            logging.warning("No exploration data available for state learning")
            return
            
        # Train state model from exploration data
        loss_info = self.state_model.train_from_placements(
            self.exploration_data[-self.config.state_training_samples:],
            self.state_optimizer,
            num_epochs=self.config.state_epochs
        )
        
        if loss_info:
            # Log detailed state model losses
            self.writer.add_scalar('StateModel/TotalLoss', loss_info['total_loss'], batch)
            self.writer.add_scalar('StateModel/RotationLoss', loss_info['rotation_loss'], batch)
            self.writer.add_scalar('StateModel/XPositionLoss', loss_info['x_position_loss'], batch)
            self.writer.add_scalar('StateModel/ValueLoss', loss_info['value_loss'], batch)
            
            # Log training progression (average over all epochs)
            all_total = loss_info['all_total_losses']
            all_rot = loss_info['all_rotation_losses']
            all_x = loss_info['all_x_position_losses']
            all_val = loss_info['all_value_losses']
            
            if len(all_total) > 1:                # Log improvement metrics
                total_improvement = (all_total[0] - all_total[-1]) / all_total[0] if all_total[0] > 0 else 0
                self.writer.add_scalar('StateModel/TotalLossImprovement', total_improvement, batch)
                self.writer.add_scalar('StateModel/FinalEpochLoss', all_total[-1], batch)
                logging.info(f"State model training - Total: {loss_info['total_loss']:.4f}, "
                        f"Rotation: {loss_info['rotation_loss']:.4f}, "
                        f"X-pos: {loss_info['x_position_loss']:.4f}, "
                        f"Value: {loss_info['value_loss']:.4f}")
    
    def phase_3_reward_prediction(self, batch):
        """
        Phase 3: Train future reward predictor on collected experience
        """
        logging.info(f"Phase 3: Future reward prediction training (batch {batch})")
        
        if len(self.experience_buffer) < self.config.min_buffer_size:
            logging.warning("Insufficient experience for reward prediction training")
            return
            
        # Sample batch and train future reward predictor
        batch_data = self.experience_buffer.sample(self.config.reward_batch_size)
        if batch_data is None:
            return
            
        states, actions, rewards, next_states, dones, _, _, _ = batch_data
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        
        # Convert actions to one-hot
        action_one_hot = torch.nn.functional.one_hot(actions.long(), 8).float()
        
        # Train future reward predictor
        reward_pred, value_pred = self.future_reward_predictor(states, action_one_hot)
        reward_loss = torch.nn.functional.mse_loss(reward_pred.squeeze(), rewards)
        value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), rewards)  # Use rewards as value target
        total_loss = reward_loss + value_loss
        
        self.reward_optimizer.zero_grad()
        total_loss.backward()
        self.reward_optimizer.step()
        
        # Log detailed reward predictor losses
        self.writer.add_scalar('RewardPredictor/TotalLoss', total_loss.item(), batch)
        self.writer.add_scalar('RewardPredictor/RewardPredictionLoss', reward_loss.item(), batch)
        self.writer.add_scalar('RewardPredictor/ValuePredictionLoss', value_loss.item(), batch)
        
        # Log prediction accuracy metrics
        with torch.no_grad():
            reward_mae = torch.nn.functional.l1_loss(reward_pred.squeeze(), rewards)
            value_mae = torch.nn.functional.l1_loss(value_pred.squeeze(), rewards)
            self.writer.add_scalar('RewardPredictor/RewardMAE', reward_mae.item(), batch)
            self.writer.add_scalar('RewardPredictor/ValueMAE', value_mae.item(), batch)
            logging.info(f"Future reward prediction - Total: {total_loss.item():.4f}, "
                     f"Reward: {reward_loss.item():.4f}, "
                     f"Value: {value_loss.item():.4f}")
    
    def phase_4_exploitation(self, batch):
        """
        Phase 4: Run exploitation episodes using current policy
        """
        logging.info(f"Phase 4: Exploitation episodes (batch {batch})")
        
        total_reward = 0
        total_steps = 0
        episode_rewards = []
        episode_steps = []
        
        for episode in range(self.config.exploitation_episodes):
            # Enable visualization only for the last episode of each batch if visualize flag is set
            visualize_this_episode = (self.config.visualize and 
                                    episode == self.config.exploitation_episodes - 1)
            
            # Recreate environment with appropriate headless setting for this episode
            if visualize_this_episode:
                self.env.close()
                self.env = TetrisEnv(single_player=True, headless=False)
                logging.info(f"Enabling visualization for last exploitation episode of batch {batch}")
            elif episode == 0 and not visualize_this_episode:
                # Ensure environment is headless for non-visualized episodes
                if hasattr(self.env, 'headless') and not self.env.headless:
                    self.env.close()
                    self.env = TetrisEnv(single_player=True, headless=True)
            
            obs = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < self.config.max_episode_steps:
                # Convert observation to state vector
                state = self._obs_to_state_vector(obs)
                
                # Select action using actor-critic
                action = self.actor_critic.select_action(state)
                
                # Take step
                next_obs, reward, done, info = self.env.step(action)
                next_state = self._obs_to_state_vector(next_obs)
                
                # Store experience
                self.experience_buffer.push(
                    obs, action, reward, next_obs, done, info
                )
                
                obs = next_obs
                episode_reward += reward
                steps += 1
                
                if visualize_this_episode:
                    self.env.render()
                    # Small delay to make visualization watchable
                    import time
                    time.sleep(0.05)
            
            # After visualization episode, revert to headless
            if visualize_this_episode:
                self.env.close()
                self.env = TetrisEnv(single_player=True, headless=True)
                logging.info(f"Visualization complete, reverting to headless mode")
            
            episode_rewards.append(episode_reward)
            episode_steps.append(steps)
            total_reward += episode_reward
            total_steps += steps
            self.episode_count += 1
            
            # Log individual episode statistics
            self.writer.add_scalar('Exploitation/EpisodeReward', episode_reward, self.episode_count)
            self.writer.add_scalar('Exploitation/EpisodeSteps', steps, self.episode_count)
        
        # Log batch statistics
        avg_reward = total_reward / self.config.exploitation_episodes
        avg_steps = total_steps / self.config.exploitation_episodes
        
        self.writer.add_scalar('Exploitation/BatchAvgReward', avg_reward, batch)
        self.writer.add_scalar('Exploitation/BatchAvgSteps', avg_steps, batch)
        self.writer.add_scalar('Exploitation/BatchStdReward', np.std(episode_rewards), batch)
        self.writer.add_scalar('Exploitation/BatchStdSteps', np.std(episode_steps), batch)
        self.writer.add_scalar('Exploitation/BatchMaxReward', np.max(episode_rewards), batch)
        self.writer.add_scalar('Exploitation/BatchMinReward', np.min(episode_rewards), batch)
          # Performance trend analysis
        if len(episode_rewards) > 1:
            reward_trend = np.polyfit(range(len(episode_rewards)), episode_rewards, 1)[0]
            self.writer.add_scalar('Exploitation/BatchRewardTrend', reward_trend, batch)
        
        logging.info(f"Exploitation: avg reward={avg_reward:.2f}±{np.std(episode_rewards):.2f}, "
                    f"avg steps={avg_steps:.1f}±{np.std(episode_steps):.1f}")
    
    def phase_5_ppo_training(self, batch):
        """
        Phase 5: PPO-style training on collected experience
        """
        logging.info(f"Phase 5: PPO training (batch {batch})")
        
        if len(self.experience_buffer) < self.config.min_buffer_size:
            logging.warning("Insufficient experience for PPO training")
            return
            
        # Multiple PPO training iterations
        total_actor_loss = 0
        total_critic_loss = 0
        total_reward_loss = 0
        successful_iterations = 0
        
        for iteration in range(self.config.ppo_iterations):
            losses = self.actor_critic.train_ppo(
                batch_size=self.config.ppo_batch_size,
                ppo_epochs=self.config.ppo_epochs
            )
            if losses:
                actor_loss, critic_loss, reward_loss, aux_loss = losses
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_reward_loss += reward_loss
                successful_iterations += 1
                
                # Log per-iteration losses
                global_step = batch * self.config.ppo_iterations + iteration
                self.writer.add_scalar('PPO/ActorLoss_Iteration', actor_loss, global_step)
                self.writer.add_scalar('PPO/CriticLoss_Iteration', critic_loss, global_step)
                self.writer.add_scalar('PPO/RewardLoss_Iteration', reward_loss, global_step)
                self.writer.add_scalar('PPO/AuxiliaryLoss_Iteration', aux_loss, global_step)
        
        # Log average losses for this batch
        if successful_iterations > 0:
            avg_actor_loss = total_actor_loss / successful_iterations
            avg_critic_loss = total_critic_loss / successful_iterations
            avg_reward_loss = total_reward_loss / successful_iterations
            
            self.writer.add_scalar('PPO/ActorLoss', avg_actor_loss, batch)
            self.writer.add_scalar('PPO/CriticLoss', avg_critic_loss, batch)
            self.writer.add_scalar('PPO/RewardLoss', avg_reward_loss, batch)
            
            # Log performance ratios
            total_loss = avg_actor_loss + avg_critic_loss + avg_reward_loss
            if total_loss > 0:
                self.writer.add_scalar('PPO/ActorLossRatio', avg_actor_loss / total_loss, batch)
                self.writer.add_scalar('PPO/CriticLossRatio', avg_critic_loss / total_loss, batch)
                self.writer.add_scalar('PPO/RewardLossRatio', avg_reward_loss / total_loss, batch)
            
            logging.info(f"PPO training completed - Actor: {avg_actor_loss:.4f}, "
                        f"Critic: {avg_critic_loss:.4f}, "
                        f"Reward: {avg_reward_loss:.4f}")
        else:
            logging.warning("No successful PPO training iterations")
        
        # Log training efficiency
        self.writer.add_scalar('PPO/SuccessfulIterations', successful_iterations, batch)
        self.writer.add_scalar('PPO/IterationSuccessRate', successful_iterations / self.config.ppo_iterations, batch)
    
    def phase_6_evaluation(self, batch):
        """
        Phase 6: Evaluate current policy performance
        """
        logging.info(f"Phase 6: Policy evaluation (batch {batch})")
        
        total_reward = 0
        total_steps = 0
        
        # Run evaluation episodes without exploration
        old_epsilon = self.actor_critic.epsilon
        self.actor_critic.epsilon = 0  # Pure exploitation
        
        for episode in range(self.config.eval_episodes):
            obs = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < self.config.max_episode_steps:
                state = self._obs_to_state_vector(obs)
                action = self.actor_critic.select_action(state)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                steps += 1
            
            total_reward += episode_reward
            total_steps += steps
        
        # Restore epsilon
        self.actor_critic.epsilon = old_epsilon
        
        avg_reward = total_reward / self.config.eval_episodes
        avg_steps = total_steps / self.config.eval_episodes
        
        self.writer.add_scalar('Evaluation/AvgReward', avg_reward, batch)
        self.writer.add_scalar('Evaluation/AvgSteps', avg_steps, batch)
        
        logging.info(f"Evaluation: avg reward={avg_reward:.2f}, avg steps={avg_steps:.1f}")
    
    def _obs_to_state_vector(self, obs):
        """Convert simplified observation dict to flattened state vector (NEW: 410-dimensional)"""
        # Flatten the simplified grids
        current_piece_flat = obs['current_piece_grid'].flatten()  # 20*10 = 200
        empty_grid_flat = obs['empty_grid'].flatten()  # 20*10 = 200
        
        # Get one-hot encoding and metadata
        next_piece = obs['next_piece']  # 7 values (removed hold piece)
        metadata = np.array([
            obs['current_rotation'],
            obs['current_x'], 
            obs['current_y']
        ])  # 3 values
        
        # Concatenate all components: 200 + 200 + 7 + 3 = 410 (removed piece_grids + hold_piece)
        return np.concatenate([
            current_piece_flat, 
            empty_grid_flat,
            next_piece,
            metadata
        ])
    
    def save_checkpoint(self, batch):
        """Save training checkpoint"""
        checkpoint = {
            'batch': batch,
            'phase': self.phase,
            'episode_count': self.episode_count,
            'state_model': self.state_model.state_dict(),
            'future_reward_predictor': self.future_reward_predictor.state_dict(),
            'actor_critic': self.actor_critic.network.state_dict(),
            'state_optimizer': self.state_optimizer.state_dict(),
            'reward_optimizer': self.reward_optimizer.state_dict(),
            'exploration_data': self.exploration_data[-1000:],  # Keep recent data
        }
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_batch_{batch}.pt')
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")

class TrainingConfig:
    """Configuration for unified training"""
    def __init__(self):
        # Training parameters
        self.num_batches = 100
        self.batch_size = 32  # Add batch_size parameter
        self.exploration_episodes = 50
        self.exploitation_episodes = 20
        self.eval_episodes = 10
        self.max_episode_steps = 2000
        
        # Device detection
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"GPU detected: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            print("Apple Silicon GPU (MPS) detected")
        else:
            self.device = 'cpu'
            print("Using CPU - no GPU detected")
        
        # Model parameters
        self.state_lr = 1e-3
        self.reward_lr = 1e-3
        self.clip_ratio = 0.2
        
        # Training phases
        self.state_training_samples = 1000
        self.state_epochs = 5
        self.ppo_iterations = 3
        self.ppo_batch_size = 64
        self.ppo_epochs = 4
        self.reward_batch_size = 64
        
        # Buffer parameters
        self.buffer_size = 100000
        self.min_buffer_size = 1000
        
        # Logging and saving
        self.log_dir = 'logs/unified_training'
        self.checkpoint_dir = 'checkpoints/unified'
        self.save_interval = 10
        self.visualize = False  # Default to no visualization

def main():
    parser = argparse.ArgumentParser(description="Unified Tetris RL Training")
    parser.add_argument('--num_batches', type=int, default=100, help='Number of training batches')
    parser.add_argument('--visualize', action='store_true', help='Visualize training')
    parser.add_argument('--log_dir', type=str, default='logs/unified_training', help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/unified', help='Checkpoint directory')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('unified_training.log')
        ]
    )
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create config
    config = TrainingConfig()
    config.num_batches = args.num_batches
    config.visualize = args.visualize
    config.log_dir = args.log_dir
    config.checkpoint_dir = args.checkpoint_dir
    
    # Initialize and run trainer
    trainer = UnifiedTrainer(config)
    trainer.run_training()

if __name__ == '__main__':
    main()
