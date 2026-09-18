from __future__ import annotations

import torch
from torch import Tensor


def covariance_spectrum(hidden: Tensor) -> Tensor:
    """Eigenvalues of the empirical covariance, descending."""
    x = hidden - hidden.mean(dim=0, keepdim=True)
    n = x.shape[0]
    _, singular, _ = torch.linalg.svd(x, full_matrices=False)
    return (singular**2) / n


def covariance_relative_error(reference: Tensor, candidate: Tensor) -> Tensor:
    reference_centered = reference - reference.mean(dim=0, keepdim=True)
    candidate_centered = candidate - candidate.mean(dim=0, keepdim=True)
    reference_cov = reference_centered.T @ reference_centered / reference.shape[0]
    candidate_cov = candidate_centered.T @ candidate_centered / candidate.shape[0]
    return torch.linalg.matrix_norm(
        candidate_cov - reference_cov
    ) / torch.linalg.matrix_norm(reference_cov).clamp_min(1e-12)


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
    _, s, v = torch.pca_lowrank(centered, q=q, center=False)
    return mean, v[:, :rank], s[:rank]


def project_pca(hidden: Tensor, mean: Tensor, components: Tensor) -> Tensor:
    return (hidden - mean) @ components


def sample_standard_normal(
    n: int, dim: int, seed: int, device: str | torch.device
) -> Tensor:
    """Draw an independent baseline from N(0, I)."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(n, dim, device=device, generator=g)


def pca_subspace_metrics(
    train: Tensor,
    real: Tensor,
    diffusion: Tensor,
    random: Tensor,
    rank: int,
    n_projections: int,
    seed: int,
) -> dict[str, float | int]:
    """Compare distributions in a PCA basis fit exclusively on training data."""
    mean, components, _ = pca_basis(train, rank=rank)
    actual_rank = components.shape[1]
    real_pca = project_pca(real, mean, components)
    diffusion_pca = project_pca(diffusion, mean, components)
    random_pca = project_pca(random, mean, components)
    split = real_pca.shape[0] // 2
    real_left = real_pca[:split]
    real_right = real_pca[split:]

    spectra = {
        "real": covariance_spectrum(real_pca),
        "diffusion": covariance_spectrum(diffusion_pca),
        "random": covariance_spectrum(random_pca),
        "real_split": covariance_spectrum(real_right),
    }
    return {
        "pca_rank": int(actual_rank),
        "pca_d_eff_real": float(effective_rank(spectra["real"])),
        "pca_d_eff_diffusion": float(effective_rank(spectra["diffusion"])),
        "pca_d_eff_random": float(effective_rank(spectra["random"])),
        "pca_d_eff_real_split": float(effective_rank(spectra["real_split"])),
        "pca_swd_real_diffusion": float(
            sliced_wasserstein_2(real_pca, diffusion_pca, n_projections, seed)
        ),
        "pca_swd_real_random": float(
            sliced_wasserstein_2(real_pca, random_pca, n_projections, seed)
        ),
        "pca_swd_real_real_split": float(
            sliced_wasserstein_2(real_left, real_right, n_projections, seed)
        ),
        "pca_covariance_relative_error_diffusion": float(
            covariance_relative_error(real_pca, diffusion_pca)
        ),
        "pca_covariance_relative_error_random": float(
            covariance_relative_error(real_pca, random_pca)
        ),
        "pca_covariance_relative_error_real_split": float(
            covariance_relative_error(real_left, real_right)
        ),
    }
