import json
from pathlib import Path

import torch

from repdist.config import ExperimentConfig
from repdist.diffusion import SampleDiagnostics
from repdist.evaluate import evaluate
from repdist.normalize import Normalizer
from repdist.train import train
from repdist import train as train_module


def _smoke_cfg(tmp_path: Path, max_steps: int) -> ExperimentConfig:
    cfg = ExperimentConfig.load("configs/smoke.yaml")
    cfg.paths.data = str(tmp_path / "data")
    cfg.paths.checkpoints = str(tmp_path / "checkpoints")
    cfg.paths.logs = str(tmp_path / "logs")
    cfg.paths.outputs = str(tmp_path / "outputs")
    cfg.paths.normalizer = str(tmp_path / "normalizer.pt")
    cfg.device = "cpu"
    cfg.diffusion.max_steps = max_steps
    cfg.diffusion.eval_every = max(max_steps // 2, 1)
    cfg.diffusion.ckpt_every = max(max_steps // 2, 1)
    cfg.diffusion.log_every = 1
    cfg.diffusion.warmup_steps = 1
    cfg.diffusion.diagnostic_sample_count = 2
    cfg.diffusion.diagnostic_timesteps = [49, 25, 0]
    return cfg


def test_train_and_resume(tmp_path: Path):
    cfg = _smoke_cfg(tmp_path, max_steps=8)
    first = train(cfg, resume=False)
    assert first["step"] == 8
    assert (Path(cfg.paths.checkpoints) / "latest.pt").exists()
    records = [
        json.loads(line)
        for line in (Path(cfg.paths.logs) / "train.jsonl").read_text().splitlines()
    ]
    diagnostics = [record for record in records if record.get("split") == "diagnostic"]
    assert diagnostics
    assert diagnostics[-1]["finite"] is True
    assert set(diagnostics[-1]["epsilon_mse"]) == {"49", "25", "0"}
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


def test_no_resume_fresh_run_does_not_load_optimizer(tmp_path: Path, monkeypatch):
    cfg = _smoke_cfg(tmp_path, max_steps=2)
    original = train_module.load_checkpoint

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("optimizer/checkpoint state loaded on a fresh run")

    monkeypatch.setattr(train_module, "load_checkpoint", fail_if_loaded)
    train(cfg, resume=False)
    monkeypatch.setattr(train_module, "load_checkpoint", original)


def test_runs_share_store_but_own_normalizers(tmp_path: Path):
    first = _smoke_cfg(tmp_path / "first", max_steps=1)
    second = _smoke_cfg(tmp_path / "second", max_steps=1)
    second.paths.data = first.paths.data

    train(first, resume=False)
    shared_store = Path(first.paths.data)
    first_norm = Normalizer.from_state_dict(
        torch.load(first.paths.normalizer, weights_only=False)
    )
    second_norm = Normalizer.fit(torch.ones(64, 32) * 7.0)
    Path(second.paths.normalizer).parent.mkdir(parents=True, exist_ok=True)
    torch.save(second_norm.state_dict(), second.paths.normalizer)

    train(second, resume=False)

    assert first_norm.fingerprint() != second_norm.fingerprint()
    assert "normalizer_fingerprint" not in json.loads(
        (shared_store / "manifest.json").read_text()
    )
    first_payload = torch.load(
        Path(first.paths.checkpoints) / "latest.pt", weights_only=False
    )
    second_payload = torch.load(
        Path(second.paths.checkpoints) / "latest.pt", weights_only=False
    )
    assert (
        first_payload["normalizer_fingerprint"]
        != second_payload["normalizer_fingerprint"]
    )


def test_resume_refreshes_run_normalizer_from_checkpoint(tmp_path: Path):
    cfg = _smoke_cfg(tmp_path, max_steps=2)
    train(cfg, resume=False)
    checkpoint_path = Path(cfg.paths.checkpoints) / "latest.pt"
    payload = torch.load(checkpoint_path, weights_only=False)

    stale = Normalizer.fit(torch.ones(64, 32) * 99.0)
    torch.save(stale.state_dict(), cfg.paths.normalizer)
    cfg.diffusion.max_steps = 3
    train(cfg, resume=True)

    refreshed = Normalizer.from_state_dict(
        torch.load(cfg.paths.normalizer, weights_only=False)
    )
    assert refreshed.fingerprint() == payload["normalizer_fingerprint"]


def test_unhealthy_reverse_trajectory_cannot_become_best(tmp_path: Path, monkeypatch):
    cfg = _smoke_cfg(tmp_path, max_steps=1)
    cfg.diffusion.max_healthy_final_rms = 0.5
    cfg.diffusion.hard_abort_rms = 10.0

    def unhealthy(*args, **kwargs):
        return SampleDiagnostics(
            samples=torch.zeros(2, 32),
            reverse_rms={0: 1.0},
            final_rms=1.0,
            max_rms=1.0,
            finite=True,
        )

    monkeypatch.setattr(train_module, "sample_with_diagnostics", unhealthy)
    train(cfg, resume=False)

    assert not (Path(cfg.paths.checkpoints) / "best.pt").exists()
    assert (Path(cfg.paths.checkpoints) / "latest.pt").exists()


def test_hard_abort_reverse_trajectory_raises(tmp_path: Path, monkeypatch):
    cfg = _smoke_cfg(tmp_path, max_steps=1)
    cfg.diffusion.hard_abort_rms = 2.0

    def exploding(*args, **kwargs):
        return SampleDiagnostics(
            samples=torch.zeros(2, 32),
            reverse_rms={0: 1.0},
            final_rms=1.0,
            max_rms=2.0,
            finite=True,
        )

    monkeypatch.setattr(train_module, "sample_with_diagnostics", exploding)
    try:
        train(cfg, resume=False)
    except RuntimeError as exc:
        assert "hard-abort" in str(exc)
    else:
        raise AssertionError("hard-abort RMS did not stop training")


def test_train_with_lr_retry_halves_lr_and_restarts_fresh(tmp_path: Path, monkeypatch):
    cfg = _smoke_cfg(tmp_path, max_steps=2)
    calls: list[float] = []

    def flaky(train_cfg, resume=True):
        calls.append(train_cfg.diffusion.lr)
        if len(calls) < 3:
            raise RuntimeError(
                "reverse trajectory RMS 25.1 reached hard-abort threshold 25 at step 13000"
            )
        return {"step": 2, "best_val": 0.4, "last_val": 0.4}

    monkeypatch.setattr(train_module, "train", flaky)
    result = train_module.train_with_lr_retry(cfg, resume=False)
    assert result == {"step": 2, "best_val": 0.4, "last_val": 0.4}
    assert calls == [1.0e-3, 5.0e-4, 2.5e-4]
    records = [
        json.loads(line)
        for line in (Path(cfg.paths.logs) / "train.jsonl").read_text().splitlines()
    ]
    retries = [r for r in records if r.get("event") == "lr_retry"]
    assert [r["lr"] for r in retries] == [5.0e-4, 2.5e-4]
    assert all(r["attempt"] in (1, 2) for r in retries)


def test_train_with_lr_retry_gives_up_after_max_retries(tmp_path: Path, monkeypatch):
    cfg = _smoke_cfg(tmp_path, max_steps=1)

    def always_exploding(train_cfg, resume=True):
        raise RuntimeError("nonfinite reverse trajectory at step 5")

    monkeypatch.setattr(train_module, "train", always_exploding)
    try:
        train_module.train_with_lr_retry(cfg, resume=False, max_lr_retries=2)
    except RuntimeError as exc:
        assert "reverse trajectory" in str(exc)
    else:
        raise AssertionError("retry loop must re-raise after exhausting attempts")
