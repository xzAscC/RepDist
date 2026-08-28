from pathlib import Path

from repdist.config import ExperimentConfig
from repdist.evaluate import evaluate
from repdist.train import train


def _smoke_cfg(tmp_path: Path, max_steps: int) -> ExperimentConfig:
    cfg = ExperimentConfig.load("configs/smoke.yaml")
    cfg.paths.data = str(tmp_path / "data")
    cfg.paths.checkpoints = str(tmp_path / "checkpoints")
    cfg.paths.logs = str(tmp_path / "logs")
    cfg.paths.outputs = str(tmp_path / "outputs")
    cfg.device = "cpu"
    cfg.diffusion.max_steps = max_steps
    cfg.diffusion.eval_every = max(max_steps // 2, 1)
    cfg.diffusion.ckpt_every = max(max_steps // 2, 1)
    cfg.diffusion.log_every = 1
    cfg.diffusion.warmup_steps = 1
    return cfg


def test_train_and_resume(tmp_path: Path):
    cfg = _smoke_cfg(tmp_path, max_steps=8)
    first = train(cfg, resume=False)
    assert first["step"] == 8
    assert (Path(cfg.paths.checkpoints) / "latest.pt").exists()
    cfg.diffusion.max_steps = 12
    second = train(cfg, resume=True)
    assert second["step"] == 12


def test_eval_runs(tmp_path: Path):
    cfg = _smoke_cfg(tmp_path, max_steps=6)
    train(cfg, resume=False)
    metrics = evaluate(cfg, ckpt_name="latest")
    assert metrics["n_test"] == 64
    assert metrics["swd_real_diffusion"] >= 0.0
    assert (Path(cfg.paths.outputs) / "metrics" / "eval.json").exists()
