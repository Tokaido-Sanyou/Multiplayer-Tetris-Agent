import os
import time
import torch
from .rl_utils.actor_critic import ActorCriticAgent
from .rl_utils.train import preprocess_state
from .tetris_env import TetrisEnv

def main():
    # Create environment in GUI mode
    env = TetrisEnv(single_player=True, headless=False)
    # Initialize agent
    state_dim = 202  # 20x10 grid + next + hold
    action_dim = 8   # 0-7 actions
    agent = ActorCriticAgent(state_dim, action_dim)

    # Auto-load latest checkpoint
    ckpt_dir = 'checkpoints'
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No checkpoints found in '{ckpt_dir}'")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading checkpoint: {latest}")
    agent.load(latest)

    # Switch to eval mode and correct device
    agent.network.eval()
    agent.network.to(agent.device)
    print(f"Running on device: {agent.device}")

    # Main play loop: pick greedy actions, render, no training
    obs = env.reset()
    state = preprocess_state(obs)
    try:
        while True:
            # Forward pass
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action_probs, _ = agent.network(state_tensor)
                action = int(action_probs.argmax(dim=1).item())

            # Step environment
            obs, _, done, _ = env.step(action)
            state = preprocess_state(obs)
            env.render()

            # Restart episode if game over
            if done:
                obs = env.reset()
                state = preprocess_state(obs)

            # Slow down for visibility
    except KeyboardInterrupt:
        print("Play interrupted by user")
    finally:
        env.close()

if __name__ == '__main__':
    main()
