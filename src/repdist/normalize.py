from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Normalizer:
    """Training-set centering plus a global scalar scale."""

    mean: Tensor
    scale: float

    def encode(self, hidden: Tensor) -> Tensor:
        return (
            hidden - self.mean.to(device=hidden.device, dtype=hidden.dtype)
        ) / self.scale

    def decode(self, x: Tensor) -> Tensor:
        return x * self.scale + self.mean.to(device=x.device, dtype=x.dtype)

    def state_dict(self) -> dict[str, Tensor | float]:
        return {"mean": self.mean.detach().cpu().clone(), "scale": float(self.scale)}

    def fingerprint(self) -> str:
        mean = self.mean.detach().cpu().float().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(mean.shape)).encode())
        digest.update(mean.numpy().tobytes())
        digest.update(repr(float(self.scale)).encode())
        return digest.hexdigest()

    def matches_fingerprint(self, fingerprint: str) -> bool:
        return self.fingerprint() == fingerprint

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor | float]) -> Normalizer:
        mean = state["mean"]
        if not isinstance(mean, Tensor):
            raise TypeError("normalizer mean must be a tensor")
        return cls(mean=mean.float(), scale=float(state["scale"]))

    @classmethod
    def fit(cls, hidden: Tensor) -> Normalizer:
        mean = hidden.mean(dim=0)
        centered = hidden - mean
        scale = float(centered.pow(2).mean().sqrt().clamp_min(1e-8).item())
        return cls(mean=mean.cpu(), scale=scale)
