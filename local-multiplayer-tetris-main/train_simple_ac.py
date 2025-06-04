#!/usr/bin/env python3
"""
Simple Actor-Critic training script (CPU-only version)
"""
import os
import numpy as np
import logging
from localMultiplayerTetris.tetris_env import TetrisEnv
from localMultiplayerTetris.rl_utils.actor_critic_cpu import ActorCriticAgentCPU
from localMultiplayerTetris.rl_utils.train import preprocess_state

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def train_simple_ac(num_episodes=10):
    """
    Simple Actor-Critic training function
    """
    logging.info("Starting simple Actor-Critic training...")
    
    try:
        # Create environment
        logging.info("Creating TetrisEnv...")
        env = TetrisEnv(single_player=True, headless=True, total_episodes=num_episodes)
        
        # Create agent
        logging.info("Creating ActorCriticAgentCPU...")
        state_dim = 202  # 20x10 grid + next_piece + hold_piece scalars
        action_dim = 8   # 8 possible actions
        agent = ActorCriticAgentCPU(state_dim, action_dim)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            logging.info(f"Starting episode {episode + 1}/{num_episodes}")
            
            # Set current episode for progressive reward calculation
            env.set_current_episode(episode)
            
            # Initialize episode
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done and episode_length < 1000:  # Limit episode length
                # Preprocess observation
                state_array = preprocess_state(obs)
                
                # Select and perform action
                action = agent.select_action(state_array)
                step_result = env.step(action)
                
                if isinstance(step_result, tuple) and len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                    
                    # Update metrics
                    episode_reward += reward
                    episode_length += 1
                    
                    # Store transition
                    agent.memory.push(obs, action, reward, next_obs, done, info)
                    
                    # Train agent (every 10 steps)
                    if episode_length % 10 == 0:
                        train_result = agent.train()
                        if train_result is not None:
                            actor_loss, critic_loss = train_result
                            logging.info(f"  Step {episode_length}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
                    
                    obs = next_obs
                else:
                    logging.error(f"Invalid step result: {step_result}")
                    break
            
            # Update exploration rate
            agent.update_epsilon()
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            logging.info(f"Episode {episode + 1} completed: Reward: {episode_reward:.2f}, Length: {episode_length}, Epsilon: {agent.epsilon:.3f}")
        
        # Save model
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = 'checkpoints/actor_critic_cpu_simple.pth'
        agent.save(checkpoint_path)
        logging.info(f"Model saved to: {checkpoint_path}")
        
        env.close()
        
        # Print summary
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        logging.info(f"Training completed! Average reward: {avg_reward:.2f}, Average length: {avg_length:.1f}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Simple Actor-Critic Training')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to train')
    args = parser.parse_args()
    
    success = train_simple_ac(num_episodes=args.episodes)
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")
