from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from repdist.schedule import CosineSchedule, extract


@dataclass(frozen=True)
class SampleDiagnostics:
    samples: Tensor
    reverse_rms: dict[int, float]
    final_rms: float
    max_rms: float
    finite: bool


def q_sample(x0: Tensor, t: Tensor, noise: Tensor, schedule: CosineSchedule) -> Tensor:
    """Forward process: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps."""
    return (
        extract(schedule.sqrt_alphas_cumprod, t, x0.shape) * x0
        + extract(schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
    )


def predict_epsilon(
    model: nn.Module, xt: Tensor, t: Tensor, schedule: CosineSchedule
) -> Tensor:
    skip = extract(schedule.sqrt_one_minus_alphas_cumprod, t, xt.shape)
    return skip * xt + model(xt, t)


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
    pred = predict_epsilon(model, xt, t, schedule)
    return torch.mean((noise - pred) ** 2)


@torch.no_grad()
def epsilon_diagnostics(
    model: nn.Module,
    x0: Tensor,
    schedule: CosineSchedule,
    timesteps: list[int] | tuple[int, ...],
    seed: int = 0,
) -> dict[str, object]:
    """Measure seeded epsilon MSE at fixed timesteps and against zero prediction."""
    if not timesteps:
        raise ValueError("diagnostic timesteps must not be empty")
    if any(t < 0 or t >= schedule.timesteps for t in timesteps):
        raise ValueError("diagnostic timestep is outside the schedule")
    model.eval()
    generator = torch.Generator(device=x0.device).manual_seed(seed)
    mse: dict[str, float] = {}
    zero_mse: dict[str, float] = {}
    for t_index in timesteps:
        t = torch.full((x0.shape[0],), t_index, device=x0.device, dtype=torch.long)
        noise = torch.randn(
            x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
        )
        xt = q_sample(x0, t, noise, schedule)
        pred = predict_epsilon(model, xt, t, schedule)
        mse[str(t_index)] = float(torch.mean((noise - pred) ** 2).item())
        zero_mse[str(t_index)] = float(torch.mean(noise.square()).item())
    return {
        "timesteps": list(timesteps),
        "mse": mse,
        "zero_predictor_mse": zero_mse,
    }


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
    eps = predict_epsilon(model, xt, t, schedule)
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


@torch.no_grad()
def sample_with_diagnostics(
    model: nn.Module,
    n: int,
    dim: int,
    schedule: CosineSchedule,
    device: torch.device | str,
    batch_size: int = 256,
    generator: torch.Generator | None = None,
    selected_steps: list[int] | tuple[int, ...] = (),
) -> SampleDiagnostics:
    """Sample without clipping and report reverse-trajectory RMS health."""
    if any(t < 0 or t >= schedule.timesteps for t in selected_steps):
        raise ValueError("selected reverse timestep is outside the schedule")
    if n < 1 or batch_size < 1:
        raise ValueError("n and batch_size must be >= 1")
    model.eval()
    selected = set(selected_steps)
    rms_squares: dict[int, float] = {t: 0.0 for t in selected_steps}
    rms_counts: dict[int, int] = {t: 0 for t in selected_steps}
    chunks: list[Tensor] = []
    max_rms = 0.0
    finite = True
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        xt = torch.randn(b, dim, device=device, generator=generator)
        for t_index in range(schedule.timesteps - 1, -1, -1):
            xt = p_sample(model, xt, t_index, schedule, generator=generator)
            is_finite = bool(torch.isfinite(xt).all().item())
            finite = finite and is_finite
            rms = float(torch.sqrt(torch.mean(xt.square())).item())
            max_rms = max(max_rms, rms) if torch.isfinite(xt).all() else float("nan")
            if t_index in selected:
                rms_squares[t_index] += float(xt.square().sum().item())
                rms_counts[t_index] += xt.numel()
        chunks.append(xt.cpu())
        remaining -= b
    reverse_rms = {
        t: (rms_squares[t] / rms_counts[t]) ** 0.5
        for t in selected_steps
        if rms_counts[t]
    }
    final_rms = reverse_rms.get(0)
    if final_rms is None:
        final_rms = float(torch.sqrt(torch.mean(torch.cat(chunks).square())).item())
    samples = torch.cat(chunks, dim=0)
    finite = finite and bool(torch.isfinite(samples).all().item())
    return SampleDiagnostics(samples, reverse_rms, final_rms, max_rms, finite)
