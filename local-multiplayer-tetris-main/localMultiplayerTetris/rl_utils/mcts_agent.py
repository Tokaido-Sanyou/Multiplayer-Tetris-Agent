"""
MCTSAgent: Monte Carlo rollout-based planning agent using learned models
"""
import torch
import torch.nn.functional as F
import numpy as np
import random

class MCTSAgent:
    def __init__(self, state_model, reward_model, action_dim, state_dim, device=None,
                 num_simulations=10, max_depth=5):
        """
        Args:
            state_model: model predicting next-state from state and action
            reward_model: model predicting immediate reward from state and action
            action_dim: number of discrete actions
            state_dim: dimension of flattened state vector
            num_simulations: number of Monte Carlo rollouts per candidate action
            max_depth: rollout depth
        """
        self.state_model = state_model.to(device)
        self.reward_model = reward_model.to(device)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device or torch.device("cpu")
        self.num_sim = num_simulations
        self.max_depth = max_depth

    def select_action(self, state_vec):
        """
        Select action by Monte Carlo rollouts:
        For each action, perform multiple rollouts using learned models,
        average cumulative reward, and pick best action.
        """
        # Convert state to tensor
        s0 = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        avg_rewards = []
        for a in range(self.action_dim):
            total_r = 0.0
            for _ in range(self.num_sim):
                total_r += self._rollout(s0, a)
            avg_rewards.append(total_r / self.num_sim)
        # Select action with highest estimated reward
        best_a = int(np.argmax(avg_rewards))
        return best_a

    def _rollout(self, state, action):
        """
        Simulate a single rollout starting with action, return cumulative reward
        """
        # First action
        a = torch.LongTensor([action]).to(self.device)
        # Predict immediate reward
        r = self.reward_model(state, a).item()
        cum_r = r
        # Predict next state
        with torch.no_grad():
            grid_p, piece_logits = self.state_model(state, a)
        # Build next-state vector
        s = torch.cat([grid_p, piece_logits], dim=1)
        # Further random actions up to max_depth
        for _ in range(self.max_depth - 1):
            a_rand = random.randrange(self.action_dim)
            a_t = torch.LongTensor([a_rand]).to(self.device)
            r = self.reward_model(s, a_t).item()
            cum_r += r
            with torch.no_grad():
                grid_p, piece_logits = self.state_model(s, a_t)
            s = torch.cat([grid_p, piece_logits], dim=1)
        return cum_r
