import torch

from repdist.schedule import CosineSchedule, extract


def test_cosine_schedule_shapes_and_bounds():
    sched = CosineSchedule(timesteps=100, cosine_s=0.008, beta_max=0.999)
    assert sched.betas.shape == (100,)
    assert torch.all(sched.betas > 0)
    assert torch.all(sched.betas <= 0.999)
    assert torch.all(sched.alphas_cumprod[1:] <= sched.alphas_cumprod[:-1])
    assert sched.alphas_cumprod[-1] < 0.1


def test_extract_broadcasts():
    sched = CosineSchedule(timesteps=10)
    t = torch.tensor([0, 3, 9])
    x = torch.zeros(3, 4)
    out = extract(sched.betas, t, x.shape)
    assert out.shape == (3, 1)
