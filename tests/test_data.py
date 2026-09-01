import pytest
import torch

from repdist.data import LatentPCA, Normalizer


def test_encode_decode_roundtrip():
    h = 7.0 * torch.randn(64, 12, generator=torch.Generator().manual_seed(0)) + 3.0
    norm = Normalizer.fit(h)
    assert torch.allclose(norm.decode(norm.encode(h)), h, atol=1e-5)
    assert abs(float(norm.encode(h).pow(2).mean().sqrt()) - 1.0) < 1e-5


def test_normalizer_is_frozen_after_fit():
    with pytest.raises((AttributeError, TypeError)):
        Normalizer.fit(torch.tensor([[1.0, 2.0], [3.0, 4.0]])).__setattr__(
            "scale", 99.0
        )


def test_fitted_normalizer_is_reused_without_refitting():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    norm = Normalizer.fit(hidden)
    before = norm.fingerprint()
    norm.encode(torch.cat([hidden, torch.tensor([[100.0, 200.0]])]))
    assert norm.fingerprint() == before


def test_latent_pca_roundtrip_preserves_subspace():
    g = torch.Generator().manual_seed(0)
    basis = torch.randn(32, 6, generator=g)
    hidden = torch.randn(80, 6, generator=g) @ basis.T
    pca = LatentPCA.fit(hidden, rank=6)
    assert torch.mean((hidden - pca.decode(pca.encode(hidden))).square()) < 1e-4
