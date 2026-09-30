#!/usr/bin/env python3
"""Score extraction runs: the "extraction + used" F1, mean and SD across runs.

    uv run bin/evaluate.py runs/<dir>                  # fuzzy matching, then the LLM judge
    uv run bin/evaluate.py runs/<dir> --no-llm         # fuzzy matching only; no API needed
    uv run bin/evaluate.py runs/<dir> --ground-truth baseline

<dir> is what bin/extract.py wrote (run-1 ... run-N), or a single run.
Results go to <dir>/eval-<ground truth>-<matcher>/: summary.json, summary.md,
and each paper's matches, per run. Nothing else in <dir> is changed.

Exits non-zero, writing no summary, if a paper has no ground truth, a run holds two
batches for one paper, or a paper the run was asked for has no batch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idsu import evaluate, llm  # noqa: E402
from idsu.config import load_config, repo_path  # noqa: E402
from idsu.data import load_ground_truth  # noqa: E402


def make_judge(model: str, temperature, timeout: float):
    client = llm.make_client(timeout=timeout)

    def judge(prompt: str) -> str:
        # The reported scorer's call: this system line, no JSON mode.
        messages = [{"role": "system", "content": "You are a research assistant"}, {"role": "user", "content": prompt}]
        response = llm.chat(client, model, messages, temperature=temperature, json_mode=False)
        if response is None:
            raise RuntimeError(f"LLM judge ({model}) failed after retries")
        return response.choices[0].message.content
    return judge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", help="directory written by bin/extract.py, or one run inside it")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ground-truth", default="reference",
                    help='"reference" (data.ground_truth), "baseline" (data.baseline_ground_truth), or a path')
    ap.add_argument("--no-llm", action="store_true", help="fuzzy matching only")
    ap.add_argument("--judge-model", default=None, help="default: evaluation.judge_model")
    ap.add_argument("--use-saved-matches", action="store_true",
                    help="where a batch has matched-resources.yaml (the original scorer's hand-approved "
                         "matches), score those instead of matching again")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    gt_path = {"reference": cfg["data"]["ground_truth"], "baseline": cfg["data"]["baseline_ground_truth"]}.get(
        args.ground_truth, args.ground_truth)
    gt_path = repo_path(gt_path)
    ground_truth = load_ground_truth(gt_path)

    root = repo_path(args.runs)
    judge = None
    if not args.no_llm:
        judge = make_judge(args.judge_model or ev["judge_model"], ev["judge_temperature"], cfg["extraction"]["timeout"])
    label = f"{gt_path.stem}-{'fuzzy' if args.no_llm else 'llm'}"
    out_dir = root / f"eval-{label}"

    try:
        runs = evaluate.find_runs(root)
        scored = [evaluate.score_run(r, ground_truth, out_dir, judge=judge,
                                     use_saved_matches=args.use_saved_matches) for r in runs]
    except evaluate.EvaluationError as e:
        sys.exit(f"[ERROR] {e}")

    summary, table = evaluate.summarize(scored, label)
    summary["ground_truth"] = str(gt_path)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    print("\n" + table + f"\nWritten: {out_dir}")


if __name__ == "__main__":
    main()
