from __future__ import annotations

import torch
from torch import Tensor, nn

from repdist.schedule import CosineSchedule, extract


def q_sample(x0: Tensor, t: Tensor, noise: Tensor, schedule: CosineSchedule) -> Tensor:
    """Forward process: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps."""
    return (
        extract(schedule.sqrt_alphas_cumprod, t, x0.shape) * x0
        + extract(schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
    )


def diffusion_loss(
    model: nn.Module,
    x0: Tensor,
    schedule: CosineSchedule,
    generator: torch.Generator | None = None,
) -> Tensor:
    batch = x0.shape[0]
    t = torch.randint(
        0,
        schedule.timesteps,
        (batch,),
        device=x0.device,
        generator=generator,
    )
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    xt = q_sample(x0, t, noise, schedule)
    pred = model(xt, t)
    return torch.mean((noise - pred) ** 2)


@torch.no_grad()
def p_sample(
    model: nn.Module,
    xt: Tensor,
    t_index: int,
    schedule: CosineSchedule,
    generator: torch.Generator | None = None,
) -> Tensor:
    batch = xt.shape[0]
    t = torch.full((batch,), t_index, device=xt.device, dtype=torch.long)
    eps = model(xt, t)
    alpha_t = extract(schedule.alphas, t, xt.shape)
    beta_t = extract(schedule.betas, t, xt.shape)
    sqrt_one_minus = extract(schedule.sqrt_one_minus_alphas_cumprod, t, xt.shape)
    mean = extract(schedule.sqrt_recip_alphas, t, xt.shape) * (
        xt - beta_t / sqrt_one_minus * eps
    )
    if t_index == 0:
        return mean
    noise = torch.randn(xt.shape, device=xt.device, dtype=xt.dtype, generator=generator)
    sigma = extract(schedule.sqrt_posterior_variance, t, xt.shape)
    return mean + sigma * noise


@torch.no_grad()
def sample(
    model: nn.Module,
    n: int,
    dim: int,
    schedule: CosineSchedule,
    device: torch.device | str,
    batch_size: int = 256,
    generator: torch.Generator | None = None,
) -> Tensor:
    model.eval()
    chunks: list[Tensor] = []
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        xt = torch.randn(b, dim, device=device, generator=generator)
        for t_index in range(schedule.timesteps - 1, -1, -1):
            xt = p_sample(model, xt, t_index, schedule, generator=generator)
        chunks.append(xt.cpu())
        remaining -= b
    return torch.cat(chunks, dim=0)
