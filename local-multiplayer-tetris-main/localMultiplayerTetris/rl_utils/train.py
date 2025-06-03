# train.py
# Main training pipeline: exploration, supervised pretraining, and reinforcement learning
# Enhanced with the new 6-phase exploration-exploitation algorithm

import os
import argparse
import logging
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Known issues that may block execution:
# 1. RewardModel.default action_dim is 8 but Tetris has 7 actions -> must instantiate with action_dim=7.
# 2. ActorCriticAgent.__init__ stores state_model but doesn't create state_model_optimizer; must call set_state_model() to enable auxiliary loss.
# 3. DQNAgent.train in dqn_agent.py is incomplete; this script uses ActorCriticAgent instead.


def main():
    parser = argparse.ArgumentParser(description="Tetris RL training pipeline")
    parser.add_argument('--mode', choices=['explore', 'pretrain', 'train', 'unified'], default='unified',
                        help='Pipeline stage to run - unified mode uses new 6-phase algorithm')
    parser.add_argument('--explore_dir', type=str, default='exploration_data',
                        help='Directory to save exploration transitions')
    parser.add_argument('--pretrain_epochs', type=int, default=5,
                        help='Epochs for state-model pretraining')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of training episodes (legacy mode)')
    parser.add_argument('--num_batches', type=int, default=100,
                        help='Number of training batches (unified mode)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for unified training')
    parser.add_argument('--exploration_episodes', type=int, default=50,
                        help='Number of exploration episodes per batch')
    parser.add_argument('--exploitation_episodes', type=int, default=20,
                        help='Number of exploitation episodes per batch')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cpu/cuda)')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Checkpoint save interval (episodes/batches)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization during training')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])

    # Check if unified mode is selected
    if args.mode == 'unified':
        logging.info("Starting unified 6-phase training algorithm")
        config = TrainingConfig()
        config.num_batches = args.num_batches
        config.batch_size = args.batch_size
        config.exploration_episodes = args.exploration_episodes
        config.exploitation_episodes = args.exploitation_episodes
        config.device = args.device
        config.visualize = args.visualize
        config.log_dir = args.log_dir
        config.checkpoint_dir = args.checkpoint_dir
        config.save_interval = args.save_interval
        
        # Create directories
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize and run unified trainer
        trainer = UnifiedTrainer(config)
        trainer.run_training()
        return

    # Legacy training modes below
    # prepare directories
    os.makedirs(args.explore_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # initialize environment (headless)
    env = TetrisEnv(headless=True)

    # exploration stage: collect one-step transitions
    explorer = StateExplorer(save_dir=args.explore_dir)
    if args.mode == 'explore':
        transitions = explorer.collect_initial_transitions(env)
        explorer.save('initial_transitions.pkl')
        logging.info(f"Collected {len(transitions)} transitions. Summary: {explorer.summary()}")
        return

    # load or collect transitions for pretraining
    data_path = os.path.join(args.explore_dir, 'initial_transitions.pkl')
    if os.path.exists(data_path):
        import pickle

# Handle both direct execution and module import
try:
    from ..tetris_env import TetrisEnv
    from .state_explorer import StateExplorer
    from .state_model import StateModel
    from .reward_model import RewardModel
    from .actor_critic import ActorCriticAgent
    from .exploration_actor import ExplorationActor
    from .unified_trainer import UnifiedTrainer, TrainingConfig
except ImportError:
    # Direct execution - imports without relative paths
    from .tetris_env import TetrisEnv
    from state_explorer import StateExplorer
    from state_model import StateModel
    from reward_model import RewardModel
    from actor_critic import ActorCriticAgent
    from exploration_actor import ExplorationActor
    from unified_trainer import UnifiedTrainer, TrainingConfig
        with open(data_path, 'rb') as f:
            explorer.initial_transitions = pickle.load(f)['initial_transitions']
    else:
        explorer.initial_transitions = explorer.collect_initial_transitions(env)
    summary = explorer.summary()
    # instantiate RewardModel with correct action_dim
    reward_model = RewardModel(state_dim=206, action_dim=7)
    # derive simple validity rules: action -> has any valid examples
    state_rules = {action: counts['valid'] > 0 for action, counts in summary.items()}
    logging.info(f"Action validity rules: {state_rules}")

    # supervised pretraining of state-model
    state_model = StateModel(state_dim=206)
    optimizer = torch.optim.Adam(state_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    if args.mode in ['pretrain', 'train']:
        # prepare dataset: only valid transitions
        examples = [(obs, next_obs) for obs, _, next_obs, valid in explorer.initial_transitions if valid]
        if examples:
            for epoch in range(args.pretrain_epochs):
                total_loss = 0.0
                np.random.shuffle(examples)
                for obs, next_obs in examples:
                    # convert obs to tensor, and extract labels from next_obs
                    state_vec = torch.FloatTensor(np.concatenate([obs['grid'].flatten(),
                                                                 [obs['cur_shape'], obs['cur_rot'], obs['cur_x'], obs['cur_y'], obs['next_piece'], obs['hold_piece']]])).unsqueeze(0)
                    rot_label = torch.LongTensor([next_obs['cur_rot']])
                    x_label = torch.LongTensor([next_obs['cur_x']])
                    y_label = torch.LongTensor([next_obs['cur_y']])
                    # forward
                    rot_logits, x_logits, y_logits, _ = state_model(state_vec)
                    loss = (criterion(rot_logits, rot_label)
                            + criterion(x_logits, x_label)
                            + criterion(y_logits, y_label))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                logging.info(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}, Loss: {total_loss/len(examples):.4f}")
        else:
            logging.warning("No valid transitions found for pretraining; skipping state-model pretraining.")

    # reinforcement learning stage
    if args.mode == 'train':
        writer = SummaryWriter(log_dir='logs/tensorboard')
        # initialize actor-critic agent with pretrained state_model and rules
        agent = ActorCriticAgent(state_dim=206, action_dim=7, state_model=state_model)
        agent.set_state_model(state_model, state_rules)

        for episode in range(args.num_episodes):
            obs = env.reset()
            # flatten state
            state = np.concatenate([obs['grid'].flatten(),
                                    [obs['cur_shape'], obs['cur_rot'], obs['cur_x'], obs['cur_y'], obs['next_piece'], obs['hold_piece']]])
            done = False
            ep_reward = 0.0
            steps = 0
            while not done:
                action = agent.select_action(state)
                next_obs, reward, done, info = env.step(action)
                next_state = np.concatenate([next_obs['grid'].flatten(),
                                             [next_obs['cur_shape'], next_obs['cur_rot'], next_obs['cur_x'], next_obs['cur_y'], next_obs['next_piece'], next_obs['hold_piece']]])
                agent.memory.push({'grid': obs['grid'], 'next_piece': obs['next_piece'], 'hold_piece': obs['hold_piece']},
                                  action, reward,
                                  {'grid': next_obs['grid'], 'next_piece': next_obs['next_piece'], 'hold_piece': next_obs['hold_piece']},
                                  done, info)
                losses = agent.train()
                if losses:
                    actor_loss, critic_loss = losses
                    writer.add_scalar('Loss/Actor', actor_loss, episode)
                    writer.add_scalar('Loss/Critic', critic_loss, episode)
                state = next_state
                obs = next_obs
                ep_reward += reward
                steps += 1

            agent.update_epsilon()
            writer.add_scalar('Episode/Reward', ep_reward, episode)
            writer.add_scalar('Episode/Steps', steps, episode)
            if (episode+1) % args.save_interval == 0:
                ckpt_path = f"checkpoints/actor_critic_ep{episode+1}.pt"
                agent.save(ckpt_path)
                logging.info(f"Saved checkpoint: {ckpt_path}")
        # save final model
        agent.save('checkpoints/actor_critic_final.pt')
        writer.close()

if __name__ == '__main__':
    main()
