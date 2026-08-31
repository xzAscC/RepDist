import torch

from repdist.model import NoisePredictor


def test_noise_predictor_exposes_8196_wide_residual_path():
    model = NoisePredictor(data_dim=4096, hidden_dim=8196, time_embed_dim=128)
    x = torch.randn(2, 4096)
    t = torch.tensor([0, 999])

    residual = model.residual(x, t)

    assert residual.shape == x.shape
    assert model.fc1.out_features == 8196
    assert model.fc2.in_features == 8196
    assert torch.isfinite(residual).all()


def test_forward_remains_compatible_residual_prediction():
    model = NoisePredictor(data_dim=8, hidden_dim=4, time_embed_dim=6)
    x = torch.randn(3, 8)
    t = torch.tensor([0, 1, 2])

    assert torch.equal(model(x, t), model.residual(x, t))


def test_fresh_noise_predictor_has_zero_residual():
    model = NoisePredictor(data_dim=32, hidden_dim=16, time_embed_dim=8)
    x = torch.randn(5, 32)
    t = torch.tensor([0, 1, 17, 500, 999])

    assert torch.equal(model.residual(x, t), torch.zeros_like(x))
