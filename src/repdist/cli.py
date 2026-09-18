from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from repdist.config import ExperimentConfig, apply_layer_paths, resolved_layers
from repdist.data import HiddenStateStore
from repdist.evaluate import compare_layer_runs, evaluate, visualize_from_saved
from repdist.extract import ensure_heldout, ensure_train, layer_stores
from repdist.report import build_geom_metrics, write_loss_summary
from repdist.select import select_best_swd
from repdist.train import train_with_lr_retry


def _load_cfg(path: str) -> ExperimentConfig:
    return ExperimentConfig.load(path)


def _layer_cfgs(cfg: ExperimentConfig, layer: int | None) -> list[ExperimentConfig]:
    if layer is not None:
        return [apply_layer_paths(deepcopy(cfg), layer)]
    if cfg.extract.layers:
        return [apply_layer_paths(deepcopy(cfg), item) for item in resolved_layers(cfg)]
    return [cfg]


def _ensure_dirs(cfg: ExperimentConfig) -> None:
    for rel in (
        cfg.paths.data,
        cfg.paths.checkpoints,
        cfg.paths.logs,
        cfg.paths.figs,
    ):
        Path(rel).mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="repdist")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="restrict train/eval/viz to one extracted layer",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="extract hidden states")
    p_extract.add_argument("--train-only", action="store_true")

    p_train = sub.add_parser("train", help="train DDPM")
    p_train.add_argument("--no-resume", action="store_true")

    p_eval = sub.add_parser("eval", help="evaluate held-out matching")
    p_eval.add_argument(
        "--ckpt",
        default="best",
        help="checkpoint to evaluate: best, latest, or a step number",
    )

    p_select = sub.add_parser(
        "select-swd", help="rank step checkpoints by validation SWD"
    )
    p_select.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=None,
        help="candidate steps (default: every step_*.pt checkpoint)",
    )
    p_select.add_argument(
        "--apply",
        action="store_true",
        help="re-run full test eval and figures on the winning checkpoint",
    )

    p_report = sub.add_parser(
        "report",
        help="regenerate notebook inputs (geom_metrics.json, loss_summary.json)",
    )
    p_report.add_argument(
        "--runs",
        nargs="+",
        default=["1024=logs/legacy/mlp1024", "8196=logs/legacy", "50k=logs/layers-50k"],
        help="family=logroot pairs feeding geom_metrics.json",
    )
    p_report.add_argument(
        "--main-run", default="50k", help="run holding real/N(0,I) tensors"
    )
    p_report.add_argument("--out", default="notebooks/figs/geom_metrics.json")

    sub.add_parser(
        "viz",
        help="write memo figures from saved normalized eval tensors or by re-running eval",
    )
    sub.add_parser("all", help="extract, train, evaluate")
    sub.add_parser(
        "compare-layers",
        help="aggregate per-layer eval.json files into comparison tables and figures",
    )

    args = parser.parse_args(argv)
    if args.cmd == "compare-layers":
        cfg = _load_cfg(args.config)
        comparison = compare_layer_runs(
            cfg.paths.logs, Path(cfg.paths.logs) / "comparison"
        )
        print(comparison["beats_random_swd"])
        return 0

    if args.cmd == "report":
        cfg = _load_cfg(args.config)
        runs: dict[str, Path] = {
            key: Path(root) for key, root in (item.split("=", 1) for item in args.runs)
        }
        out = Path(args.out)
        frozen = None
        if out.exists():
            frozen = json.loads(out.read_text())
        geom = build_geom_metrics(
            runs,
            layers=resolved_layers(cfg),
            main_run=args.main_run,
            seed=cfg.seed,
            n_projections=cfg.eval.n_projections,
            device=cfg.resolve_device(),
            frozen=frozen,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(geom, indent=2) + "\n")
        summary = write_loss_summary(cfg)
        print(
            json.dumps(
                {
                    "geom_metrics": str(out),
                    "loss_summary": str(Path(cfg.paths.logs) / "loss_summary.json"),
                    "selected_steps": {
                        key: value["best_step"]
                        for key, value in summary["layers"].items()
                    },
                },
                indent=2,
            )
        )
        return 0

    cfg = _load_cfg(args.config)
    if args.cmd == "extract" and args.layer is not None:
        parser.error("--layer applies to train/eval/viz, not extract")

    if args.cmd == "extract":
        _ensure_dirs(cfg)
        store = HiddenStateStore(cfg.paths.data)
        device = cfg.resolve_device()
        if not args.train_only:
            ensure_heldout(cfg, store, device)
        ensure_train(cfg, store, device)
        print({k: v.manifest for k, v in layer_stores(cfg, store).items()})
        return 0

    if args.cmd == "train":
        result = None
        for layer_cfg in _layer_cfgs(cfg, args.layer):
            _ensure_dirs(layer_cfg)
            result = train_with_lr_retry(layer_cfg, resume=not args.no_resume)
            print(result)
        return 0

    if args.cmd == "eval":
        metrics = None
        for layer_cfg in _layer_cfgs(cfg, args.layer):
            _ensure_dirs(layer_cfg)
            metrics = evaluate(layer_cfg, ckpt_name=args.ckpt)
            print(metrics)
        return 0

    if args.cmd == "select-swd":
        for layer_cfg in _layer_cfgs(cfg, args.layer):
            _ensure_dirs(layer_cfg)
            result = select_best_swd(layer_cfg, steps=args.steps)
            print(
                {
                    key: result[key]
                    for key in (
                        "layer",
                        "best_step",
                        "best_swd_val",
                        "swd_val_random",
                        "beats_random",
                    )
                }
            )
            if args.apply:
                print(evaluate(layer_cfg, ckpt_name=str(result["best_step"])))
        return 0

    if args.cmd == "viz":
        metrics = None
        for layer_cfg in _layer_cfgs(cfg, args.layer):
            _ensure_dirs(layer_cfg)
            saved = Path(layer_cfg.paths.logs) / "eval_tensors.pt"
            if saved.exists():
                metrics = visualize_from_saved(layer_cfg)
            else:
                metrics = evaluate(layer_cfg, ckpt_name="best")
            print(metrics)
            print(f"figures -> {layer_cfg.paths.figs}")
        return 0

    if args.cmd == "all":
        _ensure_dirs(cfg)
        store = HiddenStateStore(cfg.paths.data)
        device = cfg.resolve_device()
        ensure_train(cfg, store, device)
        for layer_cfg in _layer_cfgs(cfg, args.layer):
            _ensure_dirs(layer_cfg)
            print(train_with_lr_retry(layer_cfg, resume=True))
            print(evaluate(layer_cfg, ckpt_name="best"))
        return 0

    raise AssertionError(args.cmd)
