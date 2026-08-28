import torch

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
