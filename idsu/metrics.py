"""Confusion matrices and summary statistics.

The primary metric is "extraction + used" (finalReport.md, Evaluation
Methodology): the positive class is "used in the paper". A matched pair both
sides call used is TP; an extraction called used with no ground-truth match is
FP; an extraction called unused with no ground-truth match is TN, not FP.
"""

from __future__ import annotations

import statistics
from typing import Any

METRICS = ("accuracy", "precision", "recall", "f1")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def identity_cm(matched: dict[str, list]) -> dict[str, int]:
    """Matched vs unmatched counts, ignoring the used label."""
    return {"tp": len(matched["matches"]), "fp": len(matched["extracted"]), "fn": len(matched["ground_truth"]), "tn": 0}


def used_cm(matched: dict[str, list]) -> dict[str, int]:
    """Agreement on used_in_paper, over matched pairs only."""
    cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for m in matched["matches"]:
        e, t = as_bool(m.get("used_extraction")), as_bool(m.get("used_truth"))
        cm["tp" if e and t else "fp" if e else "fn" if t else "tn"] += 1
    return cm


def extraction_used_cm(matched: dict[str, list]) -> dict[str, int]:
    """The primary matrix: matched pairs by used label, plus both unmatched sides."""
    cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "fp_extracted_used_not_used_GT": 0, "fp_extracted_not_in_GT": 0}
    for m in matched["matches"]:
        e, t = as_bool(m.get("used_extraction")), as_bool(m.get("used_truth"))
        if e and t:
            cm["tp"] += 1
        elif e:
            cm["fp"] += 1
            cm["fp_extracted_not_in_GT" if m.get("ground_truth") in ("None", None) else "fp_extracted_used_not_used_GT"] += 1
        elif t:
            cm["fn"] += 1
        else:
            cm["tn"] += 1
    for r in matched["extracted"]:
        if as_bool(r.get("used_in_paper")):
            cm["fp"] += 1
            cm["fp_extracted_not_in_GT"] += 1
        else:
            cm["tn"] += 1
    for r in matched["ground_truth"]:
        cm["fn" if as_bool(r.get("used_in_paper")) else "tn"] += 1
    return cm


def add_cm(total: dict[str, int], cm: dict[str, int]) -> None:
    for k, v in cm.items():
        total[k] = total.get(k, 0) + v


def scores(cm: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    total = tp + fp + fn + tn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / total if total else 0.0, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def mean_sd(per_run: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and sample standard deviation across runs (SD is 0 for a single run)."""
    mean = {k: statistics.mean(r[k] for r in per_run) for k in METRICS}
    sd = {k: statistics.stdev([r[k] for r in per_run]) if len(per_run) > 1 else 0.0 for k in METRICS}
    return mean, sd
