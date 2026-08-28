import torch

from repdist.diffusion import (
    diffusion_loss,
    epsilon_diagnostics,
    p_sample,
    predict_epsilon,
    q_sample,
    sample,
    sample_with_diagnostics,
)
from repdist.model import NoisePredictor
from repdist.schedule import CosineSchedule, extract


class ZeroResidual(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


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
    assert torch.isfinite(out).all()


def test_final_reverse_step_is_deterministic_without_noise():
    dim = 8
    model = NoisePredictor(dim, hidden_dim=16, time_embed_dim=8)
    sched = CosineSchedule(timesteps=10)
    xt = torch.randn(4, dim)
    t = torch.zeros(4, dtype=torch.long)
    eps = predict_epsilon(model, xt, t, sched)
    expected = extract(sched.sqrt_recip_alphas, t, xt.shape) * (
        xt
        - extract(sched.betas, t, xt.shape)
        / extract(sched.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        * eps
    )

    assert torch.equal(p_sample(model, xt, t_index=0, schedule=sched), expected)


def test_zero_residual_epsilon_path_is_full_rank_and_timestep_conditioned():
    dim = 4096
    sched = CosineSchedule(timesteps=1000)
    xt = torch.randn(2, dim)
    t = torch.tensor([1, 999])

    eps = predict_epsilon(ZeroResidual(), xt, t, sched)
    coefficients = sched.sqrt_one_minus_alphas_cumprod[t].reshape(2, 1)

    assert torch.allclose(eps, coefficients * xt)
    assert not torch.isclose(coefficients[0], coefficients[1])
    assert torch.linalg.matrix_rank(torch.diag(eps[0, :8] / xt[0, :8])) == 8


def test_high_t_zero_residual_prediction_approximates_xt():
    sched = CosineSchedule(timesteps=1000)
    xt = torch.randn(4, 32)
    t = torch.full((4,), 999, dtype=torch.long)

    eps = predict_epsilon(ZeroResidual(), xt, t, sched)

    assert torch.allclose(eps, xt, atol=2e-3, rtol=0)


def test_diffusion_loss_uses_full_rank_epsilon_path():
    sched = CosineSchedule(timesteps=10)
    x0 = torch.randn(4, 8)
    generator = torch.Generator().manual_seed(7)
    loss = diffusion_loss(ZeroResidual(), x0, sched, generator=generator)

    expected_generator = torch.Generator().manual_seed(7)
    t = torch.randint(0, sched.timesteps, (4,), generator=expected_generator)
    noise = torch.randn(x0.shape, generator=expected_generator)
    xt = q_sample(x0, t, noise, sched)
    expected = torch.mean((noise - predict_epsilon(ZeroResidual(), xt, t, sched)) ** 2)

    assert torch.allclose(loss, expected)


def test_p_sample_generator_is_deterministic_with_full_rank_path():
    sched = CosineSchedule(timesteps=10)
    xt = torch.randn(3, 8)
    first = p_sample(
        ZeroResidual(), xt, 5, sched, generator=torch.Generator().manual_seed(11)
    )
    second = p_sample(
        ZeroResidual(), xt, 5, sched, generator=torch.Generator().manual_seed(11)
    )

    assert torch.equal(first, second)


def test_sample_count():
    dim = 8
    model = NoisePredictor(dim, hidden_dim=16, time_embed_dim=8)
    sched = CosineSchedule(timesteps=5)
    x = sample(model, n=7, dim=dim, schedule=sched, device="cpu", batch_size=4)
    assert x.shape == (7, dim)


def test_epsilon_diagnostics_are_seeded_and_include_zero_reference():
    sched = CosineSchedule(timesteps=1000)
    x0 = torch.randn(6, 4)
    model = ZeroResidual()

    first = epsilon_diagnostics(model, x0, sched, timesteps=[900, 999], seed=17)
    second = epsilon_diagnostics(model, x0, sched, timesteps=[900, 999], seed=17)

    assert first == second
    assert set(first["timesteps"]) == {900, 999}
    assert set(first["mse"]) == {"900", "999"}
    assert set(first["zero_predictor_mse"]) == {"900", "999"}
    assert all(value >= 0.0 for value in first["zero_predictor_mse"].values())


def test_sample_with_diagnostics_reports_raw_trajectory_health():
    sched = CosineSchedule(timesteps=5)
    result = sample_with_diagnostics(
        ZeroResidual(),
        n=3,
        dim=4,
        schedule=sched,
        device="cpu",
        batch_size=3,
        selected_steps=[4, 2, 0],
        generator=torch.Generator().manual_seed(3),
    )

    assert result.samples.shape == (3, 4)
    assert set(result.reverse_rms) == {4, 2, 0}
    assert result.final_rms == result.reverse_rms[0]
    assert result.max_rms >= result.final_rms
    assert result.finite
