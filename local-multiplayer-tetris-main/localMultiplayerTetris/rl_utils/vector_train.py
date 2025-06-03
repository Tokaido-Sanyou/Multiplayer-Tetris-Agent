import os
import numpy as np
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
from gym.vector import AsyncVectorEnv, SyncVectorEnv # Or SyncVectorEnv
from ..tetris_env import TetrisEnv
from .actor_critic import ActorCriticAgent
from .replay_buffer import ReplayBuffer # Assuming preprocess_state is here or accessible

# Attempt to log script entry immediately
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vectorized_training.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.debug("vector_train.py: Script execution started.")

def make_env(env_id, seed, headless=True):
    def _init():
        env = TetrisEnv(single_player=True, headless=headless)
        # It's good practice to seed environments in a vectorized setup
        # for reproducibility, though TetrisEnv.seed might need to be implemented
        # if it doesn't exist or doesn't seed all random components.
        env.seed(seed + env_id)
        return env
    return _init

def preprocess_obs_batch(batched_obs_dict, device):
    """
    Preprocesses a dictionary of batched observations from the vectorized environments
    into a batch of state tensors.
    Args:
        batched_obs_dict: A dictionary where keys are observation component names (e.g., 'grid')
                          and values are numpy arrays of batched observations (e.g., shape (num_envs, 20, 10) for grid).
        device: The torch device to send the tensor to.
    Returns:
        A batched tensor of states, shape (num_envs, feature_size).
    """
    # logger.debug(f"Preprocessing batched_obs_dict. Keys: {batched_obs_dict.keys()}")
    # print(f"[DEBUG] preprocess_obs_batch: batched_obs_dict. Keys: {batched_obs_dict.keys()}")
    # for key, value in batched_obs_dict.items():
    #     print(f"[DEBUG] preprocess_obs_batch: Key: {key}, Type: {type(value)}, Shape: {getattr(value, 'shape', 'N/A')}")

    num_envs = batched_obs_dict['grid'].shape[0]
    processed_tensors = []

    for i in range(num_envs):
        grid_flat = batched_obs_dict['grid'][i].flatten() # Shape (200,)
        metadata = np.array([
            batched_obs_dict['next_piece'][i],
            batched_obs_dict['hold_piece'][i],
            batched_obs_dict['current_shape'][i],
            batched_obs_dict['current_rotation'][i],
            batched_obs_dict['current_x'][i],
            batched_obs_dict['current_y'][i],
            batched_obs_dict['can_hold'][i]
        ]).astype(np.float32) # Shape (7,)
        
        combined = np.concatenate([grid_flat, metadata]).astype(np.float32)
        processed_tensors.append(torch.from_numpy(combined))

    return torch.stack(processed_tensors).to(device)


def get_obs_dict_from_batch(batched_obs_dict, index):
    """Extracts a single observation dictionary for a given environment index from a batched observation dictionary."""
    obs_dict = {}
    for key, value_batch in batched_obs_dict.items():
        obs_dict[key] = value_batch[index]
    return obs_dict

def evaluate_agent(agent, num_eval_episodes=10, headless=True):
    """
    Evaluate the agent's performance over a number of episodes.
    Args:
        agent: The agent to evaluate.
        num_eval_episodes: Number of episodes to run for evaluation.
        headless: Whether to run the environment in headless mode.
    Returns:
        A dictionary containing average metrics (reward, length, lines, score).
    """
    logger.info(f"Starting evaluation for {num_eval_episodes} episodes...")
    eval_env = TetrisEnv(single_player=True, headless=headless)
    
    total_rewards = []
    total_lengths = []
    total_lines = []
    total_scores = []

    for episode in range(num_eval_episodes):
        obs_dict, info = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_lines = 0
        episode_score = 0

        while not done:
            # Preprocess observation for the agent (similar to single env preprocessing)
            # This needs to match how individual states are processed if not using a batch preprocessor.
            # Assuming agent.memory._state_to_tensor can handle a single obs_dict.
            state_tensor = agent.memory._state_to_tensor(obs_dict).unsqueeze(0).to(agent.device) 
            
            # Select action in evaluation mode (greedy)
            action = agent.select_actions_batch(state_tensor, eval_mode=True)[0] # Get single action from batch method
            
            next_obs_dict, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            episode_lines += info.get('lines_cleared', 0)
            episode_score += info.get('score', 0)
            
            obs_dict = next_obs_dict
        
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        total_lines.append(episode_lines)
        total_scores.append(episode_score)
        logger.debug(f"Eval Episode {episode+1}/{num_eval_episodes} - Score: {episode_score}, Reward: {episode_reward:.2f}, Length: {episode_length}")

    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(total_lengths)
    avg_lines = np.mean(total_lines)
    avg_score = np.mean(total_scores)

    logger.info(f"Evaluation finished. Avg Score: {avg_score:.2f}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Avg Lines: {avg_lines:.2f}")
    return {
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "avg_lines": avg_lines,
        "avg_score": avg_score
    }

def train_vectorized(num_envs=4, num_episodes=10000, save_interval=100, eval_interval=500, # Increased eval_interval
                     checkpoint=None, no_eval=False, verbose=False, headless_eval=True):
    """
    Train an agent using vectorized environments.
    """
    writer = SummaryWriter(log_dir='logs/vectorized_tensorboard')
    
    # Create vectorized environments
    # Important: Each environment in AsyncVectorEnv runs in a separate process,
    # so they need to be picklable. Ensure TetrisEnv and its components are.
    logger.debug(f"train_vectorized: About to create SyncVectorEnv with {num_envs} environments (switched from AsyncVectorEnv for debugging).")
    # envs = AsyncVectorEnv([make_env(i, seed=i, headless=True) for i in range(num_envs)])
    envs = SyncVectorEnv([make_env(i, seed=i, headless=True) for i in range(num_envs)]) # DEBUG: Using SyncVectorEnv
    logger.debug(f"train_vectorized: SyncVectorEnv created: {envs}")
    # For debugging, SyncVectorEnv can be easier as it runs in the main process:
    # envs = SyncVectorEnv([make_env(i, seed=i, headless=True) for i in range(num_envs)])

    # Assuming observation_space and action_space are consistent across envs
    # Use the first env to get space dimensions if needed, though TetrisEnv has fixed dims
    # For ActorCriticAgent, state_dim is the flattened obs, action_dim is discrete
    # state_dim = 200 (grid) + 7 (metadata) = 207
    state_dim = 207 
    # 4 rotations × 10 columns = 40 flattened placement actions + 1 hold action
    action_dim = 4 * 10 + 1

    agent = ActorCriticAgent(
        state_dim,
        action_dim,
        schedule_episodes=num_episodes # Epsilon and gamma scheduling
    )
    logger.info(f"Using device: {agent.device}")

    if checkpoint:
        try:
            agent.load(checkpoint)
            logger.info(f"Loaded checkpoint from {checkpoint}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")

    os.makedirs('checkpoints_vectorized', exist_ok=True)

    # --- Training Loop ---
    # `obs` will be a list of observations, one for each environment
    logger.debug("train_vectorized: About to call envs.reset().")
    # print("[DEBUG] train_vectorized: About to call envs.reset().") # Direct print
    current_batched_obs, infos_after_reset = envs.reset() # Name changed to current_batched_obs
    logger.debug(f"train_vectorized: envs.reset() returned. Type of current_batched_obs: {type(current_batched_obs)}, Keys: {current_batched_obs.keys() if isinstance(current_batched_obs, dict) else 'N/A'}")
    # print(f"[DEBUG] train_vectorized: envs.reset() returned. Type: {type(current_batched_obs)}, Value: {current_batched_obs}") # Direct print
    logger.debug(f"train_vectorized: envs.reset() returned. Type of infos_after_reset: {type(infos_after_reset)}, Value: {infos_after_reset}")
    
    # Track episode stats for each environment
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)
    episode_lines = np.zeros(num_envs)
    episode_scores = np.zeros(num_envs)
    # ... any other per-env metrics

    # Global episode counter (counts episodes across all envs)
    total_episodes_completed = 0
    
    # Main loop runs for a total number of agent steps or episodes
    # For simplicity, let's use a large number of steps, assuming episodes will complete
    # Or, adapt to run until `total_episodes_completed` reaches `num_episodes`

    # Let's aim for roughly num_episodes * average_episode_length steps
    # This is a simplification; a more robust loop might run for a fixed number of agent steps
    # or until a certain number of total episodes (sum over envs) is met.
    # For now, we'll iterate and break when enough *individual env* episodes are done,
    # which means some envs might run more episodes than others if we just count `num_episodes`.
    # A better approach is to run for a total number of agent steps or a total number of completed episodes.

    # Let's use a step-based loop for a fixed number of interactions
    # Say, num_episodes * 200 steps on average (typical Tetris episode length)
    # total_steps_to_run = num_episodes * 200 # Heuristic
    # For a simpler loop based on completed episodes:
    
    completed_episodes_this_iteration = [0] * num_envs

    for step in range(1, (num_episodes * 1000) // num_envs +1): # Iterate enough times for episodes to complete
        # Preprocess batch of observations
        current_states_batch_tensor = preprocess_obs_batch(current_batched_obs, agent.device)

        # Select actions for the batch
        actions = agent.select_actions_batch(current_states_batch_tensor) # Expects batched tensor

        # decode flattened actions into (rotation, column) pairs for MultiDiscrete
        hold_mask = actions == 40
        rot = (actions % 40) // 10
        col = (actions % 40) % 10
        multi_actions = np.stack([rot, col], axis=1)
        # For hold actions we will send 40 directly (scalar) so env interprets hold
        multi_actions[hold_mask] = 40

        # Step in all environments
        # `next_obs_list` is a list of obs dicts
        # `rewards_array`, `dones_array` are numpy arrays of shape (num_envs,)
        # next_batched_obs, rewards_array, dones_array, infos_list = envs.step(actions) # Renamed to next_batched_obs
        step_result = envs.step(multi_actions)
        # Unpack based on Gym version: 4-tuple or 5-tuple
        if len(step_result) == 4:
            next_batched_obs, rewards_array, dones_array, infos = step_result
            # No separate terminated/truncated in Gym<0.26
            terminated_array = dones_array
            truncated_array = np.zeros_like(dones_array)
        else:
            next_batched_obs, rewards_array, terminated_array, truncated_array, infos = step_result
            dones_array = np.logical_or(terminated_array, truncated_array)

        # Update per-environment episode metrics
        episode_rewards += rewards_array
        episode_lengths += 1

        # Normalize infos into a list of per-env dicts
        per_env_infos = []
        if isinstance(infos, (list, tuple)):
            per_env_infos = list(infos)
        elif isinstance(infos, dict):
            # infos is a dict of batched values
            for idx in range(num_envs):
                info_i = {}
                for key, value in infos.items():
                    if isinstance(value, (np.ndarray, list, tuple)) and len(value) == num_envs:
                        info_i[key] = value[idx]
                    else:
                        info_i[key] = value
                per_env_infos.append(info_i)
        else:
            logger.error(f"Unexpected infos type: {type(infos)}; using empty dicts.")
            per_env_infos = [{} for _ in range(num_envs)]

        # Update episode-specific metrics and store transitions
        for i in range(num_envs):
            info = per_env_infos[i]
            episode_lines[i] += info.get('lines_cleared', 0)
            episode_scores[i] += info.get('score', 0)

            obs_i = get_obs_dict_from_batch(current_batched_obs, i)
            next_obs_i = get_obs_dict_from_batch(next_batched_obs, i)
            agent.memory.push(
                obs_i,
                actions[i],
                rewards_array[i],
                next_obs_i,
                dones_array[i],  # use dones_array instead of undefined dones
                info
            )

        # Train agent (e.g., every step or after N steps)
        if len(agent.memory) > agent.batch_size:
            losses = agent.train() # train() should handle batch sampling from memory
            if losses:
                actor_loss, critic_loss = losses
                # Log losses (e.g., average over training steps or episodes)
                # For simplicity, log instantaneous loss here, or average them
                writer.add_scalar('Loss/Actor_vec', actor_loss, step)
                writer.add_scalar('Loss/Critic_vec', critic_loss, step)
        
        # Handle dones: if an environment is done, log its episode stats and reset
        for i in range(num_envs):
            # if dones_array[i]:
            if dones_array[i]:  # use dones_array instead of undefined dones
                total_episodes_completed += 1
                completed_episodes_this_iteration[i] +=1

                logger.info(
                    f"Env {i}, Episode {completed_episodes_this_iteration[i]} finished. "
                    f"Reward: {episode_rewards[i]:.2f}, Length: {episode_lengths[i]}, "
                    f"Lines: {episode_lines[i]}, Score: {episode_scores[i]}"
                )
                # Log to TensorBoard
                writer.add_scalar(f'Train/Env{i}/Reward', episode_rewards[i], total_episodes_completed)
                writer.add_scalar(f'Train/Env{i}/Length', episode_lengths[i], total_episodes_completed)
                writer.add_scalar(f'Train/Env{i}/Lines', episode_lines[i], total_episodes_completed)
                writer.add_scalar(f'Train/Env{i}/Score', episode_scores[i], total_episodes_completed)
                # Add more scalars as needed

                # Reset the finished environment's state
                reset_obs_i, _ = envs.envs[i].reset()
                for key, value in reset_obs_i.items():
                    current_batched_obs[key][i] = value

                # Reset metrics for this environment
                episode_rewards[i] = 0
                episode_lengths[i] = 0
                episode_lines[i] = 0
                episode_scores[i] = 0
                
                # Epsilon and gamma are updated per *agent* episode, not per env episode.
                # This means they decay based on the total number of completed episodes.
                agent.update_schedules(total_episodes_completed) # Changed from update_epsilon
                # agent.update_gamma()   # if you add this method # Removed as it's part of update_schedules
                writer.add_scalar('Train/Epsilon_vec', agent.epsilon, total_episodes_completed)
                # writer.add_scalar('Train/Gamma_vec', agent.gamma, total_episodes_completed) # Removed

                if total_episodes_completed >= num_episodes:
                    logger.info(f"Target number of episodes ({num_episodes}) reached.")
                    break # Break the outer step loop
            
        if total_episodes_completed >= num_episodes:
            break 

        # Update current observations
        current_batched_obs = next_batched_obs # next_batched_obs is the new batched_obs from envs.step()
        
        # Save model checkpoint periodically (based on total_episodes_completed or steps)
        if total_episodes_completed > 0 and total_episodes_completed % save_interval == 0:
            # Ensure we don't save multiple times for the same episode count if multiple envs finish
            # A simple way is to use a flag or check against a list of saved episode counts.
            # Or, more simply, save based on the 'step' or a primary episode counter.
            # Let's use total_episodes_completed, but be mindful of multiple saves if not careful.
            # A robust way: save if `total_episodes_completed // save_interval` changes.
            current_save_milestone = total_episodes_completed // save_interval
            if not hasattr(train_vectorized, 'last_saved_milestone') or \
               train_vectorized.last_saved_milestone != current_save_milestone:
                try:
                    save_path = f'checkpoints_vectorized/actor_critic_vec_ep_{total_episodes_completed}.pt'
                    agent.save(save_path)
                    logger.info(f"Saved checkpoint: {save_path}")
                    train_vectorized.last_saved_milestone = current_save_milestone
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {str(e)}")
        
        # (Optional) Evaluation step
        # Evaluation can be done similarly, perhaps with a separate set of sync envs or by pausing training.
        # For simplicity, this example omits a detailed eval loop within the vectorized training.
        if not no_eval and total_episodes_completed > 0 and total_episodes_completed % eval_interval == 0:
            # Ensure evaluation doesn't happen multiple times for the same milestone
            current_eval_milestone = total_episodes_completed // eval_interval
            if not hasattr(train_vectorized, 'last_eval_milestone') or \
               train_vectorized.last_eval_milestone != current_eval_milestone:
                logger.info(f"Starting evaluation at {total_episodes_completed} total episodes.")
                eval_metrics = evaluate_agent(agent, num_eval_episodes=10, headless=headless_eval)
                writer.add_scalar('Eval/AvgReward', eval_metrics["avg_reward"], total_episodes_completed)
                writer.add_scalar('Eval/AvgLength', eval_metrics["avg_length"], total_episodes_completed)
                writer.add_scalar('Eval/AvgLines', eval_metrics["avg_lines"], total_episodes_completed)
                writer.add_scalar('Eval/AvgScore', eval_metrics["avg_score"], total_episodes_completed)
                train_vectorized.last_eval_milestone = current_eval_milestone
                # Potentially save a checkpoint after evaluation if it's the best so far
                # (requires tracking best score and comparing)

    # --- End of Training Loop ---

    logger.info("Vectorized training finished.")
    try:
        agent.save('checkpoints_vectorized/actor_critic_vec_final.pt')
        logger.info("Saved final vectorized model.")
    except Exception as e:
        logger.error(f"Error saving final model: {str(e)}")
    
    writer.close()
    envs.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train Actor-Critic agent with vectorized environments.")
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel environments.')
    parser.add_argument('--episodes', type=int, default=1000, help='Total number of episodes to train for.')
    parser.add_argument('--save-interval', type=int, default=500, help='How often to save checkpoints (episodes).')
    parser.add_argument('--eval-interval', type=int, default=1000, help='How often to run evaluation (episodes).')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint to load and continue training.')
    parser.add_argument('--no-eval', action='store_true', help='Disable periodic evaluation during training.')
    parser.add_argument('--headless-eval', action=argparse.BooleanOptionalAction, default=True, help='Run evaluation in headless mode.')

    args = parser.parse_args()

    train_vectorized(
        num_envs=args.num_envs,
        num_episodes=args.episodes,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        checkpoint=args.checkpoint,
        no_eval=args.no_eval,
        headless_eval=args.headless_eval
    )
