"""Score extraction runs against a ground truth.

A port of main() in extraction-eval-avg-new.py, without its console prompts.
Where the original skipped something without saying so, this stops instead:
a batch whose paper has no ground truth, two batches for one paper, or a paper
the run was asked for but has no batch. One scoring change: the original left
out every extracted resource that lacked in_target_list, a field the prompt
never asks for. Here such a resource is scored as on the list, and each run
reports how many there were.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from idsu import metrics
from idsu.match import Judge, match_to_truth

RUN_INFO = "run.yaml"


class EvaluationError(Exception):
    pass


def find_runs(path: Path) -> list[Path]:
    """A run directory holds batch-* directories; a runs root holds run directories."""
    if any(p.is_dir() and p.name.startswith("batch-") for p in path.iterdir()):
        return [path]
    runs = sorted(p for p in path.iterdir() if p.is_dir() and any(c.name.startswith("batch-") for c in p.iterdir()))
    if not runs:
        raise EvaluationError(f"No run directories (containing batch-*) under {path}")
    return runs


def truth_items(resources: list[dict]) -> list[dict]:
    # A ground-truth resource is used if any of its quotes is marked used.
    return [{"id": r["id"], "in_target_list": True,
             "used_in_paper": any(q["used_in_paper"] for q in r["quotes"]),
             "references": r.get("references")} for r in resources or []]


def extracted_items(resources: list[dict]) -> tuple[list[dict], int, int]:
    """The resources to score, how many were malformed (skipped), and how many lacked in_target_list."""
    items, dropped, assumed = [], 0, 0
    for r in resources if isinstance(resources, list) else []:
        if not isinstance(r, dict) or "id" not in r:
            dropped += 1
            continue
        if "in_target_list" not in r:
            assumed += 1
        items.append({"id": r["id"], "in_target_list": r.get("in_target_list", True),
                      "used_in_paper": r.get("used_in_paper"), "references": r.get("references")})
    return items, dropped, assumed


def load_batches(run_dir: Path) -> dict[str, Path]:
    batches: dict[str, Path] = {}
    for d in sorted(run_dir.iterdir()):
        if not (d.is_dir() and (d / "info.yaml").is_file()):
            continue
        with open(d / "info.yaml", encoding="utf-8") as f:
            filename = next(doc for doc in yaml.safe_load_all(f) if doc)["filename"]
        if filename in batches:
            raise EvaluationError(f"{run_dir}: two batches for one paper: {batches[filename].name}, {d.name}")
        batches[filename] = d
    return batches


def score_run(run_dir: Path, ground_truth: list[dict], out_dir: Path, *, judge: Judge | None,
              use_saved_matches: bool = False, log=print) -> dict[str, Any]:
    gt_by_file = {p["filename"]: p.get("resources") or [] for p in ground_truth}
    batches = load_batches(run_dir)

    no_gt = sorted(f for f in batches if f not in gt_by_file)
    if no_gt:
        raise EvaluationError(f"{run_dir}: no ground truth for {len(no_gt)} paper(s): {no_gt}")
    info_path = run_dir / RUN_INFO
    if info_path.is_file():
        expected = yaml.safe_load(info_path.read_text(encoding="utf-8")).get("papers", [])
        missing = sorted(set(expected) - set(batches))
        if missing:
            raise EvaluationError(f"{run_dir}: {len(missing)} paper(s) in {RUN_INFO} have no batch: {missing}")
    else:
        log(f"[WARN] {run_dir.name}: no {RUN_INFO}, so a paper missing from this run cannot be detected")

    totals = {"identity": {}, "used": {}, "extraction_used": {}}
    papers, dropped_total, assumed_total = [], 0, 0
    (out_dir / run_dir.name).mkdir(parents=True, exist_ok=True)
    for filename in sorted(batches):
        batch = batches[filename]
        saved = batch / "matched-resources.yaml"
        if use_saved_matches and saved.is_file():
            matched = yaml.safe_load(saved.read_text(encoding="utf-8"))["matches"]
            dropped, assumed, source = 0, 0, "saved"
        else:
            parsed = json.loads((batch / "extraction-parsed.json").read_text(encoding="utf-8"))
            extracted, dropped, assumed = extracted_items(parsed.get("resources", []))
            matched = match_to_truth(extracted, truth_items(gt_by_file[filename]), judge=judge, log=log)
            source = "computed"
        cms = {"identity": metrics.identity_cm(matched), "used": metrics.used_cm(matched),
               "extraction_used": metrics.extraction_used_cm(matched)}
        for k, cm in cms.items():
            metrics.add_cm(totals[k], cm)
        dropped_total += dropped
        assumed_total += assumed
        if dropped:
            log(f"[WARN] {run_dir.name}/{batch.name}: {dropped} extracted resource(s) malformed (no id); not scored")
        papers.append({"filename": filename, "batch": batch.name, "matches": source,
                       "dropped_malformed": dropped, "assumed_in_target_list": assumed, **cms})
        with open(out_dir / run_dir.name / f"{batch.name}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"filename": filename, "matches": matched}, f, sort_keys=False, allow_unicode=True)

    return {"run": run_dir.name, "papers": len(papers), "dropped_malformed": dropped_total,
            "assumed_in_target_list": assumed_total,
            "confusion": totals, "metrics": {k: metrics.scores(cm) for k, cm in totals.items()},
            "per_paper": papers}


def summarize(runs: list[dict[str, Any]], label: str) -> tuple[dict[str, Any], str]:
    primary = [r["metrics"]["extraction_used"] for r in runs]
    mean, sd = metrics.mean_sd(primary)
    summary = {"label": label, "n_runs": len(runs), "primary": "extraction_used", "mean": mean, "sd": sd,
               "runs": [{k: v for k, v in r.items() if k != "per_paper"} for r in runs]}
    lines = [f"### {label}: extraction + used, mean (sample SD) over {len(runs)} run(s)", "",
             "| Metric | Mean | SD |", "|---|---|---|"]
    lines += [f"| {k.capitalize()} | {mean[k]:.4f} | {sd[k]:.4f} |" for k in metrics.METRICS]
    lines += ["", "| Run | Papers | TP | FP | FN | TN | F1 | No in_target_list (scored as on list) | Malformed (not scored) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        cm = r["confusion"]["extraction_used"]
        lines.append(f"| {r['run']} | {r['papers']} | {cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['tn']} "
                     f"| {r['metrics']['extraction_used']['f1']:.4f} | {r['assumed_in_target_list']} "
                     f"| {r['dropped_malformed']} |")
    return summary, "\n".join(lines) + "\n"
