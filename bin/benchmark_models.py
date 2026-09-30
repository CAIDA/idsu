#!/usr/bin/env python3
"""Compare extraction models: the IDSU pipeline once per model, scored the same way.

    uv run bin/benchmark_models.py --split training
    uv run bin/benchmark_models.py --models gemma qwen3 --runs 3 --no-llm
    uv run bin/benchmark_models.py --out runs/<dir> --resume    # finish an interrupted benchmark

For each model, runs bin/extract.py into <out>/<model>/, then bin/evaluate.py
on it. Only the extraction model changes: prompt, shots, seed and scorer are
the same for every model, and the LLM judge is evaluation.judge_model whichever
model is being benchmarked. Writes <out>/benchmark.json and benchmark.md, one
row per model.

This is not the harness behind the February four-model table
(experiments/model-extract-eval_rding in GitLab history), which used its own
prompt, matcher and metrics, so its numbers are not comparable with these.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idsu import metrics  # noqa: E402
from idsu.config import load_config, repo_path  # noqa: E402

BIN = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode:
        sys.exit(f"[ERROR] exit {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    cfg = load_config(pre.parse_known_args()[0].config)
    ex = cfg["extraction"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config file (default: config.yaml)")
    ap.add_argument("--models", nargs="+", default=cfg["benchmark"]["models"], help="default: benchmark.models")
    ap.add_argument("--split", action="append", choices=sorted(cfg["data"]["splits"]),
                    help="split to extract; repeat for several (default: training)")
    ap.add_argument("--runs", type=int, default=ex["runs"])
    ap.add_argument("--shots", type=int, default=ex["shots"], choices=[1, 2, 3])
    ap.add_argument("--shot-mode", default=ex["shot_mode"], choices=["partial", "whole"])
    ap.add_argument("--seed", type=int, default=ex["seed"])
    ap.add_argument("--workers", type=int, default=ex["workers"])
    ap.add_argument("--out", default=None, help="output directory (default: <runs_dir>/<timestamp>-benchmark-<splits>)")
    ap.add_argument("--resume", action="store_true", help="keep what is already done in --out")
    ap.add_argument("--no-llm", action="store_true", help="score with fuzzy matching only")
    ap.add_argument("--ground-truth", default="reference", help="passed to bin/evaluate.py")
    ap.add_argument("--dry-run", action="store_true", help="write prompts, call no model; implies --no-llm")
    args = ap.parse_args()
    if len(set(args.models)) != len(args.models):
        sys.exit(f"[ERROR] a model is named twice: {args.models}")

    splits = args.split or ["training"]
    if args.out:
        out = repo_path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = repo_path(ex["runs_dir"]) / f"{stamp}-benchmark-{'+'.join(splits)}"
    if out.exists() and any(out.iterdir()) and not args.resume:
        sys.exit(f"[ERROR] {out} is not empty; pass --resume to continue it")
    no_llm = args.no_llm or args.dry_run
    # The directory bin/evaluate.py writes into: eval-<ground truth stem>-<matcher>.
    gt_path = {"reference": cfg["data"]["ground_truth"], "baseline": cfg["data"]["baseline_ground_truth"]}.get(
        args.ground_truth, args.ground_truth)
    eval_label = f"{repo_path(gt_path).stem}-{'fuzzy' if no_llm else 'llm'}"

    common = [] if args.config is None else ["--config", args.config]
    rows = []
    for model in args.models:
        model_dir = out / model
        cmd = [sys.executable, str(BIN / "extract.py"), *common, "--model", model, "--out", str(model_dir),
               "--runs", str(args.runs), "--shots", str(args.shots), "--shot-mode", args.shot_mode,
               "--seed", str(args.seed), "--workers", str(args.workers)]
        for s in splits:
            cmd += ["--split", s]
        if args.resume:
            cmd.append("--resume")
        if args.dry_run:
            cmd.append("--dry-run")
        run(cmd)

        cmd = [sys.executable, str(BIN / "evaluate.py"), *common, str(model_dir), "--ground-truth", args.ground_truth]
        if no_llm:
            cmd.append("--no-llm")
        run(cmd)
        summary_path = model_dir / f"eval-{eval_label}" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({"model": model, "n_runs": summary["n_runs"], "mean": summary["mean"], "sd": summary["sd"],
                     "summary": str(summary_path.relative_to(out))})

    matcher = "fuzzy" if no_llm else f"fuzzy + LLM judge ({cfg['evaluation']['judge_model']})"
    report = {"created": datetime.now().isoformat(timespec="seconds"), "splits": splits, "runs": args.runs,
              "shots": args.shots, "shot_mode": args.shot_mode, "seed": args.seed, "matcher": matcher,
              "ground_truth": args.ground_truth, "dry_run": args.dry_run, "models": rows}
    lines = [f"### Model benchmark: extraction + used, mean (sample SD) over {args.runs} run(s)", "",
             f"Splits {'+'.join(splits)}; {args.shots} shot(s), {args.shot_mode}; seed {args.seed}; "
             f"matcher: {matcher}; ground truth: {args.ground_truth}.", "",
             "| Model | " + " | ".join(k.capitalize() for k in metrics.METRICS) + " |",
             "|---|" + "---|" * len(metrics.METRICS)]
    for r in rows:
        lines.append(f"| {r['model']} | " + " | ".join(f"{r['mean'][k]:.4f} ± {r['sd'][k]:.4f}" for k in metrics.METRICS) + " |")
    table = "\n".join(lines) + "\n"
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "benchmark.md").write_text(table, encoding="utf-8")
    print("\n" + table + f"\nWritten: {out}")


if __name__ == "__main__":
    main()
