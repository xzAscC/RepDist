from __future__ import annotations

import math

import torch
from torch import Tensor


class CosineSchedule:
    """Nichol & Dhariwal cosine schedule with beta cap."""

    def __init__(
        self,
        timesteps: int,
        cosine_s: float = 0.008,
        beta_max: float = 0.999,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if timesteps < 1:
            raise ValueError("timesteps must be >= 1")
        self.timesteps = timesteps
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        f = (
            torch.cos(((t / timesteps) + cosine_s) / (1.0 + cosine_s) * math.pi / 2)
            ** 2
        )
        alpha_bar = f / f[0]
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0, beta_max)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]]
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_variance[0] = betas[0]

        def _cast(x: Tensor) -> Tensor:
            return x.to(device=device, dtype=dtype)

        self.betas = _cast(betas)
        self.alphas = _cast(alphas)
        self.alphas_cumprod = _cast(alphas_cumprod)
        self.alphas_cumprod_prev = _cast(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = _cast(alphas_cumprod.sqrt())
        self.sqrt_one_minus_alphas_cumprod = _cast((1.0 - alphas_cumprod).sqrt())
        self.sqrt_recip_alphas = _cast(alphas.rsqrt())
        self.posterior_variance = _cast(posterior_variance)
        self.posterior_log_variance_clipped = _cast(
            posterior_variance.clamp(min=1e-20).log()
        )
        self.sqrt_posterior_variance = _cast(posterior_variance.sqrt())

    def to(
        self, device: torch.device | str, dtype: torch.dtype | None = None
    ) -> CosineSchedule:
        for name, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, name, value.to(device=device, dtype=dtype or value.dtype))
        return self


def extract(schedule_tensor: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
    """Gather schedule values for a batch of timesteps and broadcast to x."""
    out = schedule_tensor.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))
