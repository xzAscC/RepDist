from __future__ import annotations

import torch
from torch import Tensor


class LatentPCA:
    """Train-only PCA map used to diffuse in a shared low-dimensional basis."""

    def __init__(self, mean: Tensor, components: Tensor, scale: Tensor) -> None:
        self.mean = mean.float()
        self.components = components.float()
        self.scale = scale.float().reshape(-1)

    @property
    def rank(self) -> int:
        return int(self.components.shape[1])

    @classmethod
    def fit(cls, hidden: Tensor, rank: int) -> LatentPCA:
        if rank < 1:
            raise ValueError("latent rank must be >= 1")
        mean = hidden.mean(dim=0)
        centered = hidden - mean
        q = min(rank, *centered.shape)
        _, _, v = torch.pca_lowrank(centered, q=q, center=False)
        components = v[:, :rank].contiguous()
        latent = centered @ components
        scale = latent.std(dim=0, unbiased=False).clamp_min(1e-8)
        return cls(mean.cpu(), components.cpu(), scale.cpu())

    def encode(self, hidden: Tensor) -> Tensor:
        mean = self.mean.to(hidden.device)
        components = self.components.to(hidden.device)
        scale = self.scale.to(hidden.device)
        return ((hidden - mean) @ components) / scale

    def decode(self, latent: Tensor) -> Tensor:
        components = self.components.to(latent.device)
        mean = self.mean.to(latent.device)
        scale = self.scale.to(latent.device)
        return (latent * scale) @ components.T + mean

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "mean": self.mean,
            "components": self.components,
            "scale": self.scale,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> LatentPCA:
        return cls(state["mean"], state["components"], state["scale"])
