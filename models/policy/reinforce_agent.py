from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)  


class ReinforceAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float       = 0.99,
        lr: float          = 1e-3,
        hidden_dim: int    = 128,
        device: str        = "cpu",
        grad_clip: float   = 0.5,
    ):
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        self.net = PolicyValueNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.states:  List[np.ndarray] = []
        self.actions: List[int]        = []
        self.rewards: List[float]      = []

        self.last_loss: Optional[float] = None

    def select_action(self, state: np.ndarray) -> int:
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.net(s_t)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        self.states.append(state)
        self.actions.append(action)
        return action

    def step(self, reward: float, done: bool):
        self.rewards.append(reward)
        if done:
            self._update_policy()
            self._reset_episode()

    def _discounted_returns(self) -> torch.Tensor:
        R, out = 0.0, []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            out.insert(0, R)
        return torch.tensor(out, dtype=torch.float32, device=self.device)

    def _update_policy(self):
        states  = torch.as_tensor(np.vstack(self.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions, dtype=torch.int64, device=self.device)
        returns = self._discounted_returns()

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        logits, values = self.net(states)
        log_probs = F.log_softmax(logits, dim=1)
        log_pi_a  = log_probs[torch.arange(len(actions)), actions]

        advantage = returns - values.detach()
        policy_loss = -(log_pi_a * advantage).mean()

        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss
        self.last_loss = loss.item()

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optim.step()

    def _reset_episode(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str, map_location="cpu"):
        self.net.load_state_dict(torch.load(path, map_location=map_location))

    def get_last_loss(self):
        return self.last_loss
