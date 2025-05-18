# es_agent.py
import torch
import torch.nn as nn
import numpy as np
import torch

import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)



class ESAgent:
    def __init__(self, state_dim=5, action_dim=3, hidden_dim=128, device="cpu"):
        self.device = torch.device(device)
        self.net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)

    def select_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            logits = self.net(torch.as_tensor(state, dtype=torch.float32,
                                              device=self.device).unsqueeze(0))
        return int(torch.argmax(logits, dim=1).item())

    def step(self, *_, **__):  
        pass

    def get_flat_params(self) -> torch.Tensor:
        return torch.nn.utils.parameters_to_vector(self.net.parameters()).detach()

    def set_flat_params(self, flat: torch.Tensor):
        """Записывает 1-D тензор параметров обратно в сеть"""
        torch.nn.utils.vector_to_parameters(flat.to(self.device), self.net.parameters())

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str, map_location="cpu"):
        self.net.load_state_dict(torch.load(path, map_location=map_location))

