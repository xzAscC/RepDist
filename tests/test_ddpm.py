import torch

from repdist.ddpm import (
    CosineSchedule,
    NoisePredictor,
    diffusion_loss,
    epsilon_diagnostics,
    extract,
    p_sample,
    predict_epsilon,
    q_sample,
    sample,
    sample_with_diagnostics,
)


class ZeroResidual(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_cosine_schedule_shapes_and_bounds():
    sched = CosineSchedule(100)
    assert sched.betas.shape == (100,)
    assert torch.all(sched.betas > 0)
    assert torch.all(sched.betas <= 0.999)
    assert sched.alphas_cumprod[-1] < 0.1


def test_extract_broadcasts():
    sched = CosineSchedule(10)
    assert extract(sched.betas, torch.tensor([0, 3, 9]), torch.Size((3, 4))).shape == (
        3,
        1,
    )


def test_sample_centered_removes_batch_mean():
    sched = CosineSchedule(50)
    model = NoisePredictor(4, hidden_dim=8, time_embed_dim=8)
    g = torch.Generator().manual_seed(0)
    out = sample(
        model, 64, 4, sched, device="cpu", batch_size=64, generator=g, center=True
    )
    assert out.mean(dim=0).abs().max().item() < 1e-6


def test_sample_centered_keeps_single_sample():
    sched = CosineSchedule(10)
    model = NoisePredictor(4, hidden_dim=8, time_embed_dim=8)
    g = torch.Generator().manual_seed(0)
    out = sample(
        model, 1, 4, sched, device="cpu", batch_size=1, generator=g, center=True
    )
    assert out.abs().sum() > 0


def test_noise_predictor_residual_contract():
    model = NoisePredictor(8, hidden_dim=25, time_embed_dim=6, zero_init_output=False)
    torch.nn.init.ones_(model.fc2.weight)
    torch.nn.init.zeros_(model.fc2.bias)
    x = torch.zeros(2, 8)
    t = torch.zeros(2, dtype=torch.long)
    hidden = model.act(model.fc1(x) + model.time_embed(t))
    assert torch.allclose(
        model.residual(x, t), hidden @ model.fc2.weight.T * (25**-0.5)
    )


def test_fresh_noise_predictor_has_zero_residual():
    model = NoisePredictor(32, hidden_dim=16, time_embed_dim=8)
    assert torch.equal(
        model.residual(torch.randn(5, 32), torch.tensor([0, 1, 17, 500, 999])),
        torch.zeros(5, 32),
    )


def test_q_sample_and_reverse_step():
    sched = CosineSchedule(20)
    x0 = torch.randn(8, 16)
    assert (
        q_sample(
            x0, torch.zeros(8, dtype=torch.long), torch.randn_like(x0), sched
        ).shape
        == x0.shape
    )
    out = p_sample(
        NoisePredictor(16, hidden_dim=32, time_embed_dim=16),
        torch.randn(4, 16),
        5,
        sched,
    )
    assert torch.isfinite(out).all()


def test_final_reverse_step_is_deterministic_without_noise():
    sched = CosineSchedule(10)
    model = NoisePredictor(8, hidden_dim=16, time_embed_dim=8)
    xt = torch.randn(4, 8)
    t = torch.zeros(4, dtype=torch.long)
    eps = predict_epsilon(model, xt, t, sched)
    expected = extract(sched.sqrt_recip_alphas, t, xt.shape) * (
        xt
        - extract(sched.betas, t, xt.shape)
        / extract(sched.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        * eps
    )
    assert torch.equal(p_sample(model, xt, 0, sched), expected)


def test_zero_residual_epsilon_path_is_full_rank():
    sched = CosineSchedule(1000)
    xt = torch.randn(2, 4096)
    eps = predict_epsilon(ZeroResidual(), xt, torch.tensor([1, 999]), sched)
    coefficients = sched.sqrt_one_minus_alphas_cumprod[torch.tensor([1, 999])].reshape(
        2, 1
    )
    assert torch.allclose(eps, coefficients * xt)
    assert torch.linalg.matrix_rank(torch.diag(eps[0, :8] / xt[0, :8])) == 8


def test_diffusion_loss_and_sampling():
    sched = CosineSchedule(10)
    x0 = torch.randn(4, 8)
    loss = diffusion_loss(
        ZeroResidual(), x0, sched, generator=torch.Generator().manual_seed(7)
    )
    assert torch.isfinite(loss)
    assert sample(
        NoisePredictor(8, hidden_dim=16, time_embed_dim=8),
        7,
        8,
        CosineSchedule(5),
        "cpu",
        4,
    ).shape == (7, 8)


def test_epsilon_diagnostics_and_sample_diagnostics():
    sched = CosineSchedule(1000)
    model = ZeroResidual()
    x0 = torch.randn(6, 4)
    first = epsilon_diagnostics(model, x0, sched, [900, 999], seed=17)
    assert first == epsilon_diagnostics(model, x0, sched, [900, 999], seed=17)
    result = sample_with_diagnostics(
        model,
        3,
        4,
        CosineSchedule(5),
        "cpu",
        3,
        torch.Generator().manual_seed(3),
        [4, 2, 0],
    )
    assert result.samples.shape == (3, 4)
    assert result.finite
