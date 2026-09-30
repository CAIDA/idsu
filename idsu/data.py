"""Loaders for the corpus files: ground truth, target list, and split lists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from thefuzz import fuzz

# A ground-truth id counts as on the target list at this thefuzz ratio or above
# (full_process.py's rule, used to tag shots and to classify papers).
TARGET_ID_RATIO = 90


def load_ground_truth(path: str | Path) -> list[dict[str, Any]]:
    """One dict per paper, {filename, resources}, from a multi-document YAML file."""
    papers = []
    with open(path, encoding="utf-8") as f:
        for doc in yaml.safe_load_all(f):
            if isinstance(doc, dict):
                papers.append(doc)
            elif isinstance(doc, list):
                papers.extend(d for d in doc if isinstance(d, dict))
    return papers


def load_target_list(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def load_split(path: str | Path) -> list[str]:
    """Filenames listed in a papers_<split>.yaml file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    return [item["filename"].strip() for item in data if isinstance(item, dict) and item.get("filename")]


def mark_in_target_list(papers: list[dict[str, Any]], target_list: list[dict[str, Any]]) -> None:
    """Set in_target_list on every ground-truth resource, in place.

    The tagged resources are what the few-shot examples show the model, so the
    model learns to emit in_target_list itself; the scorer relies on that key.
    """
    target_ids = [t if isinstance(t, str) else t.get("id", "") for t in target_list]
    for paper in papers:
        for res in paper.get("resources") or []:
            rid = res.get("id", "")
            res["in_target_list"] = any(rid == tid or fuzz.ratio(rid, tid) >= TARGET_ID_RATIO for tid in target_ids)


def is_paper_positive(paper: dict[str, Any]) -> bool:
    """True if any of the paper's resources is on the target list (after mark_in_target_list)."""
    return any(res.get("in_target_list") is True for res in paper.get("resources") or [])
