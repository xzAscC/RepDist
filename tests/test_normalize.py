import torch
import pytest

from repdist.normalize import Normalizer


def test_encode_decode_roundtrip():
    g = torch.Generator().manual_seed(0)
    h = 7.0 * torch.randn(64, 12, generator=g) + 3.0
    norm = Normalizer.fit(h)
    x = norm.encode(h)
    rec = norm.decode(x)
    assert torch.allclose(rec, h, atol=1e-5)
    assert abs(float(x.mean())) < 1e-5
    assert abs(float(x.pow(2).mean().sqrt()) - 1.0) < 1e-5


def test_normalizer_is_frozen_after_fit():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    norm = Normalizer.fit(hidden)

    with pytest.raises((AttributeError, TypeError)):
        norm.scale = 99.0


def test_fitted_normalizer_can_be_reused_without_refitting():
    initial = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    streamed = torch.tensor([[100.0, 200.0]])
    norm = Normalizer.fit(initial)
    before_mean = norm.mean.clone()
    before_scale = norm.scale

    encoded = norm.encode(torch.cat([initial, streamed]))

    assert torch.equal(norm.mean, before_mean)
    assert norm.scale == before_scale
    assert not torch.allclose(encoded[-1], torch.zeros(2))
