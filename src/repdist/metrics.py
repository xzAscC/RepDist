from __future__ import annotations

import torch
from torch import Tensor


def covariance_spectrum(hidden: Tensor) -> Tensor:
    """Eigenvalues of the empirical covariance, descending."""
    x = hidden - hidden.mean(dim=0, keepdim=True)
    n = x.shape[0]
    _, singular, _ = torch.linalg.svd(x, full_matrices=False)
    return (singular**2) / n


def effective_rank(eigenvalues: Tensor) -> Tensor:
    return (eigenvalues.sum() ** 2) / eigenvalues.square().sum().clamp_min(1e-12)


def sliced_wasserstein_2(
    real: Tensor,
    fake: Tensor,
    n_projections: int = 128,
    seed: int = 0,
) -> Tensor:
    n = min(real.shape[0], fake.shape[0])
    x = real[:n]
    y = fake[:n]
    dim = x.shape[1]
    g = torch.Generator(device=x.device).manual_seed(seed)
    directions = torch.randn(n_projections, dim, device=x.device, generator=g)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    a = x @ directions.T
    b = y @ directions.T
    a_sorted, _ = torch.sort(a, dim=0)
    b_sorted, _ = torch.sort(b, dim=0)
    w2_sq = ((a_sorted - b_sorted) ** 2).mean(dim=0)
    return w2_sq.mean().sqrt()


def pca_basis(train: Tensor, rank: int = 2) -> tuple[Tensor, Tensor, Tensor]:
    mean = train.mean(dim=0)
    centered = train - mean
    q = min(rank, *centered.shape)
    u, s, v = torch.pca_lowrank(centered, q=q, center=False)
    return mean, v[:, :rank], s[:rank]


def project_pca(hidden: Tensor, mean: Tensor, components: Tensor) -> Tensor:
    return (hidden - mean) @ components


def sample_gaussian(mean: Tensor, train: Tensor, n: int, seed: int) -> Tensor:
    """Draw from N(mu_train, Sigma_train) using the training SVD."""
    g = torch.Generator(device=train.device).manual_seed(seed)
    centered = train - mean
    n_train = centered.shape[0]
    u, s, v = torch.linalg.svd(centered, full_matrices=False)
    scale = s / (n_train - 1) ** 0.5
    z = torch.randn(n, scale.numel(), device=train.device, generator=g)
    return mean + z * scale @ v
