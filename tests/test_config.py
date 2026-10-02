from pathlib import Path

import pytest

from repdist.cli import REPORT_DEFAULT_RUNS, _layer_cfgs, main
from repdist.config import (
    ExperimentConfig,
    apply_layer_paths,
    probe_layers,
    resolved_layers,
)
from repdist.ddpm import NoisePredictor
from repdist.evaluate import compare_layer_runs
from repdist.extract import ensure_train, layer_stores


def test_probe_and_resolved_layers():
    assert probe_layers(32, 16) == [1, 8, 16, 24, 32]
    cfg = ExperimentConfig.load("configs/smoke.yaml")
    cfg.extract.layers = [1, 8, 16, 24, 32]
    assert resolved_layers(cfg) == [1, 8, 16, 24, 32]


def test_configs_and_paths_contract():
    for name in ("default.yaml", "rankfix.yaml", "rankfix-pilot.yaml"):
        assert resolved_layers(ExperimentConfig.load(f"configs/{name}")) == [
            1,
            8,
            16,
            24,
            32,
        ]
    cfg = ExperimentConfig.load("configs/default.yaml")
    data_root, ckpt_root = cfg.paths.data, cfg.paths.checkpoints
    apply_layer_paths(cfg, 8)
    assert cfg.paths.data == str(Path(data_root) / "layer_08")
    assert cfg.paths.checkpoints == str(Path(ckpt_root) / "layer_08")
    assert cfg.paths.figs.endswith("layer_08")


def test_layer_cfgs_and_smoke_store():
    cfg = ExperimentConfig.load("configs/default.yaml")
    assert [item.extract.layer for item in _layer_cfgs(cfg, None)] == [1, 8, 16, 24, 32]
    smoke = ExperimentConfig.load("configs/smoke.yaml")
    assert len(_layer_cfgs(smoke, None)) == 1


def test_extract_rejects_layer_flag():
    with pytest.raises(SystemExit):
        main(["--config", "configs/smoke.yaml", "--layer", "16", "extract"])


def test_deeper_predictor_and_multilayer_extract(tmp_path: Path):
    model = NoisePredictor(16, hidden_dim=8, time_embed_dim=6, n_hidden_layers=2)
    assert len(model.hidden_layers) == 1
    cfg = ExperimentConfig.load("configs/smoke.yaml")
    cfg.paths.data = str(tmp_path / "layers")
    cfg.extract.layers, cfg.extract.n_val, cfg.extract.n_test = [1, 3], 8, 8
    cfg.extract.n_train_initial = cfg.extract.max_train_samples = 16
    ensure_train(cfg, None, "cpu")
    stores = layer_stores(cfg)
    assert stores[1].n_train() == stores[3].n_train() == 16


def test_compare_layer_runs_reads_flat_logs_and_writes_pdfs(tmp_path: Path):
    root = tmp_path / "logs"
    for layer, diff, random in ((1, 0.2, 0.5), (16, 0.9, 0.4)):
        met = root / f"layer_{layer:02d}"
        met.mkdir(parents=True)
        (met / "eval.json").write_text(
            f'{{"d_eff_real": 12, "d_eff_diffusion": 14, "d_eff_random": 2000, "swd_real_diffusion": {diff}, "swd_real_random": {random}, "pca_swd_real_diffusion": 1, "pca_swd_real_random": 2, "pca_covariance_relative_error_diffusion": 0.2, "pca_covariance_relative_error_random": 0.9}}'
        )
    out = tmp_path / "comparison"
    comparison = compare_layer_runs(root, out)
    assert comparison["beats_random_swd"]["1"] is True
    assert (out / "summary.json").exists()
    assert (out / "swd_vs_layer.pdf").exists()


def test_report_default_runs_is_50k_only():
    assert REPORT_DEFAULT_RUNS == ["50k=logs/layers-50k"]
    assert not any(item.startswith(("1024=", "8196=")) for item in REPORT_DEFAULT_RUNS)
