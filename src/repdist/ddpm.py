from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


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
        self.residual_scale = hidden_dim**-0.5
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
        return self.fc2(hidden) * self.residual_scale

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.residual(x, t)


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
    min_snr_gamma: float = 0.0,
) -> Tensor:
    batch = x0.shape[0]
    t = torch.randint(
        0, schedule.timesteps, (batch,), device=x0.device, generator=generator
    )
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    xt = q_sample(x0, t, noise, schedule)
    pred = predict_epsilon(model, xt, t, schedule)
    per_sample = torch.mean((noise - pred) ** 2, dim=1)
    if min_snr_gamma <= 0:
        return per_sample.mean()
    alpha_bar = extract(schedule.alphas_cumprod, t, x0.shape).squeeze(-1)
    snr = alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)
    weight = torch.clamp(snr, max=min_snr_gamma) / snr.clamp(min=1e-8)
    return torch.mean(weight * per_sample)


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
    return {"timesteps": list(timesteps), "mse": mse, "zero_predictor_mse": zero_mse}


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
