from pathlib import Path

import pytest
import torch

from repdist.cli import _layer_cfgs, main
from repdist.config import ExperimentConfig
from repdist.evaluate import compare_layer_runs
from repdist.extract import ensure_train, layer_stores
from repdist.layers import apply_layer_paths, probe_layers, resolved_layers
from repdist.model import NoisePredictor


def test_latent_pca_roundtrip_preserves_subspace():
    from repdist.latent import LatentPCA

    g = torch.Generator().manual_seed(0)
    basis = torch.randn(32, 6, generator=g)
    z = torch.randn(80, 6, generator=g)
    hidden = z @ basis.T
    pca = LatentPCA.fit(hidden, rank=6)
    recon = pca.decode(pca.encode(hidden))
    assert recon.shape == hidden.shape
    assert torch.mean((hidden - recon).square()) < 1e-4


def test_olmo7b_probe_layers_are_first_quartiles_current_and_last():
    assert probe_layers(32, current_layer=16) == [1, 8, 16, 24, 32]


def test_resolved_layers_prefers_explicit_list():
    cfg = ExperimentConfig.load("configs/smoke.yaml")
    cfg.extract.layer = 16
    cfg.extract.layers = [1, 8, 16, 24, 32]
    assert resolved_layers(cfg) == [1, 8, 16, 24, 32]


def test_default_and_rankfix_probe_the_main_layers():
    for path in (
        "configs/default.yaml",
        "configs/rankfix.yaml",
        "configs/rankfix-pilot.yaml",
    ):
        cfg = ExperimentConfig.load(path)
        assert resolved_layers(cfg) == [1, 8, 16, 24, 32]


def test_multi_layer_protocol_pre_extracts_aligned_data_without_streaming():
    cfg = ExperimentConfig.load("configs/default.yaml")
    assert cfg.extract.n_train_initial == cfg.extract.max_train_samples == 50000
    assert cfg.diffusion.stream_on_plateau is False
    for path in ("configs/rankfix.yaml", "configs/rankfix-pilot.yaml"):
        fixed = ExperimentConfig.load(path)
        assert fixed.diffusion.stream_on_plateau is False, path


def test_mlp1024_comparison_config_isolates_runs_and_shares_store():
    cfg = ExperimentConfig.load("configs/mlp-1024.yaml")
    base = ExperimentConfig.load("configs/default.yaml")
    assert cfg.diffusion.hidden_dim == 1024
    assert cfg.paths.data == base.paths.data
    assert cfg.paths.checkpoints.startswith("runs/mlp1024/")
    assert cfg.paths.logs.startswith("runs/mlp1024/")
    assert cfg.paths.outputs.startswith("runs/mlp1024/")
    assert cfg.diffusion.lr == base.diffusion.lr
    assert cfg.diffusion.weight_decay == base.diffusion.weight_decay


def test_layer_cfgs_loop_all_main_layers_unless_restricted():
    cfg = ExperimentConfig.load("configs/default.yaml")
    all_cfgs = _layer_cfgs(cfg, None)
    assert [item.extract.layer for item in all_cfgs] == [1, 8, 16, 24, 32]
    assert all(item.extract.layers == [] for item in all_cfgs)
    one = _layer_cfgs(cfg, 16)
    assert len(one) == 1
    assert one[0].extract.layer == 16
    assert one[0].paths.data.endswith("layer_16")


def test_extract_rejects_layer_flag():
    with pytest.raises(SystemExit):
        main(["--config", "configs/smoke.yaml", "--layer", "16", "extract"])


def test_smoke_stays_single_store_without_layer_nesting():
    cfg = ExperimentConfig.load("configs/smoke.yaml")
    cfgs = _layer_cfgs(cfg, None)
    assert len(cfgs) == 1
    assert cfgs[0].paths.data == cfg.paths.data
    assert cfgs[0].paths.checkpoints == cfg.paths.checkpoints


def test_apply_layer_paths_nests_under_config_roots():
    cfg = ExperimentConfig.load("configs/default.yaml")
    data_root = cfg.paths.data
    ckpt_root = cfg.paths.checkpoints
    apply_layer_paths(cfg, 8)
    assert cfg.extract.layer == 8
    assert cfg.extract.layers == []
    assert cfg.paths.data == str(Path(data_root) / "layer_08")
    assert cfg.paths.checkpoints == str(Path(ckpt_root) / "layer_08")
    assert cfg.paths.normalizer.endswith("layer_08/normalizer.pt")
    assert "data/layers" not in cfg.paths.data
    assert "runs/layers" not in cfg.paths.checkpoints


def test_deeper_predictor_still_zero_at_init():
    model = NoisePredictor(
        data_dim=16, hidden_dim=8, time_embed_dim=6, n_hidden_layers=2
    )
    x = torch.randn(4, 16)
    t = torch.tensor([0, 1, 2, 3])
    assert len(model.hidden_layers) == 1
    assert torch.equal(model.residual(x, t), torch.zeros_like(x))


def test_synthetic_multilayer_extract_writes_aligned_stores(tmp_path: Path):
    cfg = ExperimentConfig.load("configs/smoke.yaml")
    cfg.paths.data = str(tmp_path / "layers")
    cfg.extract.layers = [1, 3]
    cfg.extract.n_val = 32
    cfg.extract.n_test = 32
    cfg.extract.n_train_initial = 64
    cfg.extract.max_train_samples = 64
    cfg.device = "cpu"
    ensure_train(cfg, None, "cpu")
    stores = layer_stores(cfg)
    assert set(stores) == {1, 3}
    assert stores[1].n_train() == stores[3].n_train() == 64
    assert stores[1].manifest["cursor"] == stores[3].manifest["cursor"]
    assert stores[1].load_split("val").shape == stores[3].load_split("val").shape
    assert not torch.equal(stores[1].load_split("val"), stores[3].load_split("val"))


def test_compare_layer_runs_marks_swd_win(tmp_path: Path):
    root = tmp_path / "runs"
    for layer, swd_diff, swd_rand in ((1, 0.2, 0.5), (16, 0.9, 0.4)):
        met = root / f"layer_{layer:02d}" / "outputs" / "metrics"
        met.mkdir(parents=True)
        (met / "eval.json").write_text(
            """
{
  "d_eff_real": 12.0,
  "d_eff_diffusion": 14.0,
  "d_eff_random": 2000.0,
  "swd_real_diffusion": %s,
  "swd_real_random": %s,
  "pca_swd_real_diffusion": 1.0,
  "pca_swd_real_random": 2.0,
  "pca_covariance_relative_error_diffusion": 0.2,
  "pca_covariance_relative_error_random": 0.9
}
"""
            % (swd_diff, swd_rand)
        )
    out = tmp_path / "results"
    comparison = compare_layer_runs(root, out)
    assert comparison["beats_random_swd"]["1"] is True
    assert comparison["beats_random_swd"]["16"] is False
    assert (out / "comparison.json").exists()
    assert (out / "swd_vs_layer.png").exists()
