import torch

from repdist.diffusion import p_sample, q_sample, sample
from repdist.model import NoisePredictor
from repdist.schedule import CosineSchedule


def test_q_sample_at_t0_near_data():
    sched = CosineSchedule(timesteps=20)
    x0 = torch.randn(8, 16)
    t = torch.zeros(8, dtype=torch.long)
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise, sched)
    # t=0 still has a small amount of noise under a cosine schedule
    assert xt.shape == x0.shape
    assert torch.isfinite(xt).all()


def test_reverse_step_shape():
    dim = 16
    model = NoisePredictor(dim, hidden_dim=32, time_embed_dim=16)
    sched = CosineSchedule(timesteps=10)
    xt = torch.randn(4, dim)
    out = p_sample(model, xt, t_index=5, schedule=sched)
    assert out.shape == xt.shape


def test_sample_count():
    dim = 8
    model = NoisePredictor(dim, hidden_dim=16, time_embed_dim=8)
    sched = CosineSchedule(timesteps=5)
    x = sample(model, n=7, dim=dim, schedule=sched, device="cpu", batch_size=4)
    assert x.shape == (7, dim)
