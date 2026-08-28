# pyright: reportMissingImports=false

import inspect

import torch

from repdist.config import EvalConfig
from repdist.metrics import (
    covariance_relative_error,
    covariance_spectrum,
    effective_rank,
    pca_subspace_metrics,
    sample_standard_normal,
    sliced_wasserstein_2,
)


def test_effective_rank_identity_like():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2000, 10, generator=g)
    eig = covariance_spectrum(x)
    deff = float(effective_rank(eig))
    assert 6.0 < deff < 10.5


def test_effective_rank_low_rank():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(1000, 2, generator=g)
    a = torch.randn(2, 16, generator=g)
    x = z @ a
    deff = float(effective_rank(covariance_spectrum(x)))
    assert deff < 3.5


def test_swd_identical_is_small():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(256, 8, generator=g)
    d = float(sliced_wasserstein_2(x, x.clone(), n_projections=32, seed=0))
    assert d < 1e-5


def test_swd_shifted_is_larger():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(256, 8, generator=g)
    y = x + 3.0
    d = float(sliced_wasserstein_2(x, y, n_projections=32, seed=0))
    assert d > 1.0


def test_standard_normal_baseline_has_no_train_data_dependency():
    parameters = inspect.signature(sample_standard_normal).parameters
    assert list(parameters) == ["n", "dim", "seed", "device"]


def test_standard_normal_baseline_is_seeded_n0_i():
    first = sample_standard_normal(n=20_000, dim=4, seed=17, device="cpu")
    second = sample_standard_normal(n=20_000, dim=4, seed=17, device="cpu")

    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.all(first.mean(dim=0).abs() < 0.03)
    assert torch.all((first.var(dim=0) - 1.0).abs() < 0.04)


def test_covariance_relative_error():
    g = torch.Generator().manual_seed(4)
    real = torch.randn(512, 6, generator=g)

    assert float(covariance_relative_error(real, real.clone())) < 1e-6
    assert float(covariance_relative_error(real, 2.0 * real)) > 2.9


def test_pca_subspace_is_fit_on_training_data_only():
    axis = torch.linspace(-3.0, 3.0, 256)
    train = torch.stack((axis, torch.zeros_like(axis)), dim=1)
    g = torch.Generator().manual_seed(5)
    real = torch.randn(128, 2, generator=g)
    diffusion = real.clone()
    diffusion[:, 1] *= 20.0
    random = torch.randn(128, 2, generator=g)

    metrics = pca_subspace_metrics(
        train,
        real,
        diffusion,
        random,
        rank=1,
        n_projections=16,
        seed=6,
    )

    assert metrics["pca_rank"] == 1
    assert metrics["pca_swd_real_diffusion"] < 1e-6
    assert metrics["pca_covariance_relative_error_diffusion"] < 1e-6
    assert "pca_swd_real_random" in metrics
    assert "pca_swd_real_real_split" in metrics
    assert "pca_d_eff_real_split" in metrics


def test_pca_rank_defaults_to_32():
    assert EvalConfig().pca_rank == 32
