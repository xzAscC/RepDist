import torch

from repdist.metrics import covariance_spectrum, effective_rank, sliced_wasserstein_2


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
