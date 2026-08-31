from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class NoisePredictor(nn.Module):
    """Residual MLP noise predictor with a full-rank skip applied outside the module."""

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 8196,
        time_embed_dim: int = 128,
        n_hidden_layers: int = 1,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
        )
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers - 1)]
        )
        self.fc2 = nn.Linear(hidden_dim, data_dim)
        self.zero_init_output = zero_init_output
        if zero_init_output:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        self.act = nn.SiLU()

    def residual(self, x: Tensor, t: Tensor) -> Tensor:
        hidden = self.act(self.fc1(x) + self.time_embed(t))
        for layer in self.hidden_layers:
            hidden = self.act(layer(hidden))
        return self.fc2(hidden)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.residual(x, t)
