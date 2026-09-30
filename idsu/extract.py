"""Extraction stage: one prompt per paper, one model call, parsed and filtered output.

A port of experiments/mitigating-falsepositives-nman/final_runs/full_process.py,
the single-pass unified-prompt pipeline behind the reported F1. Prompt text,
shot construction, model parameters and output filtering are unchanged. The
output layout is unchanged too, so idsu.evaluate scores runs made by either.

Per run directory:
    batch-<n>-1/
        extraction-prompt.txt     every message sent (contains paper text; never commit)
        extraction-response.json  the raw API response
        extraction-parsed.json    {"resources": [...]}, after filtering
        info.yaml                 paper, shots, counts
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import yaml
from json_repair import repair_json

from idsu import llm
from idsu.data import is_paper_positive

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_TEMPLATE = (PROMPTS_DIR / "extraction-system.prompt").read_text(encoding="utf-8")
TARGET_INSTRUCTION = (PROMPTS_DIR / "extraction-target.prompt").read_text(encoding="utf-8").strip()

DESIRED_ORDER = ["id", "name", "type", "in_target_list", "used_in_paper", "quotes", "references"]


def format_catalog(target_list: list[dict[str, Any]]) -> str:
    # Each entry is rendered as a Python dict repr: that is what the reported runs sent.
    return "\n".join(f"- {res}" for res in target_list)


def system_prompt(target_list: list[dict[str, Any]]) -> str:
    return SYSTEM_TEMPLATE.replace("{target_resources_list}", format_catalog(target_list)).strip()


def select_shots(pool: list[dict[str, Any]], target_filename: str, num_shots: int,
                 rng: random.Random) -> list[dict[str, Any]]:
    """Half positive and half negative papers (positives get the odd one), topped up at random."""
    candidates = [p for p in pool if p.get("filename") != target_filename]
    pos_pool = [p for p in candidates if is_paper_positive(p)]
    neg_pool = [p for p in candidates if not is_paper_positive(p)]
    want_pos, want_neg = (num_shots + 1) // 2, num_shots // 2

    shots = rng.sample(pos_pool, want_pos) if len(pos_pool) >= want_pos else list(pos_pool)
    shots += rng.sample(neg_pool, want_neg) if len(neg_pool) >= want_neg else list(neg_pool)
    if len(shots) < num_shots:
        chosen = {p.get("filename") for p in shots}
        remaining = [p for p in candidates if p.get("filename") not in chosen]
        needed = num_shots - len(shots)
        shots += rng.sample(remaining, needed) if len(remaining) >= needed else remaining
    rng.shuffle(shots)
    return shots


def _quotes_and_refs(paper: dict[str, Any]) -> tuple[list[str], list[str]]:
    quotes, refs = [], []
    for res in paper.get("resources") or []:
        for q in res.get("quotes") or []:
            if isinstance(q, dict):
                q = q.get("quote", "")
            if isinstance(q, str) and q.strip():
                quotes.append(q.strip())
        for r in res.get("references") or []:
            if isinstance(r, str) and r.strip():
                refs.append(r.strip())
    return quotes, refs


def shot_messages(paper: dict[str, Any], index: int, mode: str) -> list[dict[str, str]]:
    """A user/assistant pair: the paper's ground-truth quotes and references, then its resources."""
    quotes, refs = _quotes_and_refs(paper)
    body = "\n".join(quotes + (["REFERENCES"] if refs else []) + refs)
    if mode == "whole":
        header = (f"# Example Paper {index} (whole worked example)\n"
                  "Below is a complete example of the task applied to a prior paper.\n\n")
    else:
        header = f"# Example Paper {index} (partial):\n"
    return [{"role": "user", "content": header + body},
            {"role": "assistant", "content": json.dumps({"resources": paper.get("resources", [])})}]


def build_messages(paper_text: str, shots: list[dict[str, Any]], target_list: list[dict[str, Any]],
                   mode: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt(target_list)}]
    for i, shot in enumerate(shots, 1):
        messages += shot_messages(shot, i, mode)
    messages.append({"role": "user", "content": TARGET_INSTRUCTION + "\n\n# TARGET PAPER:\n" + paper_text})
    return messages


def parse_response(response_obj: dict[str, Any]) -> dict[str, Any]:
    """Replace each choice's content with its list of resources (JSON repaired if needed)."""
    for choice in response_obj.get("choices", []):
        try:
            if "message" in choice and choice["message"]["content"]:
                parsed = json.loads(repair_json(choice["message"]["content"]))
                if isinstance(parsed, dict) and "resources" in parsed:
                    choice["message"]["content"] = parsed.get("resources", [])
                elif isinstance(parsed, list):
                    choice["message"]["content"] = parsed
                else:
                    choice["message"]["content"] = []
        except Exception as e:
            choice["message"]["content"] = [{"error": str(e), "raw_content": choice["message"].get("content")}]
    return response_obj


def filter_resources(raw: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop non-objects, anything the model marked off-list, and anything without a quote."""
    kept = []
    skipped = {"malformed_json": 0, "filtered_unused": 0, "filtered_not_in_target": 0, "filtered_empty_quotes": 0}
    for res in raw if isinstance(raw, list) else []:
        if not isinstance(res, dict):
            skipped["malformed_json"] += 1
            continue
        if res.get("in_target_list") is False:
            skipped["filtered_not_in_target"] += 1
            continue
        quotes = res.get("quotes")
        if not isinstance(quotes, list) or not quotes:
            skipped["filtered_empty_quotes"] += 1
            continue
        if any((isinstance(q, str) and q.strip()) or (isinstance(q, dict) and q.get("quote", "").strip())
               for q in quotes):
            ordered = {k: res[k] for k in DESIRED_ORDER if k in res}
            ordered.update({k: v for k, v in res.items() if k not in DESIRED_ORDER})
            kept.append(ordered)
        else:
            skipped["filtered_empty_quotes"] += 1
    return kept, skipped


def write_prompt_dump(path: Path, messages: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(f"\n{'=' * 40}\nROLE: {msg.get('role', '').upper()}\n{'=' * 40}\n\n")
            content = msg.get("content")
            f.write(content if isinstance(content, str) else json.dumps(content, indent=2))
            f.write("\n\n")


def extract_paper(*, paper: dict[str, Any], batch_dir: Path, paper_text: str, shots: list[dict[str, Any]],
                  target_list: list[dict[str, Any]], cfg: dict[str, Any], client=None, dry_run: bool = False,
                  log=print) -> dict[str, Any]:
    """Run one paper once and write its batch directory. Returns the info.yaml contents."""
    os.makedirs(batch_dir, exist_ok=True)
    mode = cfg["shot_mode"]
    messages = build_messages(paper_text, shots, target_list, mode)
    write_prompt_dump(batch_dir / "extraction-prompt.txt", messages)

    if dry_run:
        response_obj: dict[str, Any] = {"dry_run": True}
    else:
        response = llm.chat(client, cfg["model"], messages, temperature=cfg["temperature"],
                            max_retries=cfg["max_retries"], log=log)
        response_obj = json.loads(response.model_dump_json()) if response else {"error": "API call failed"}
    with open(batch_dir / "extraction-response.json", "w") as f:
        json.dump(response_obj, f, indent=2)

    raw = parse_response(response_obj).get("choices", [{}])[0].get("message", {}).get("content", [])
    resources, skipped = filter_resources(raw)
    with open(batch_dir / "extraction-parsed.json", "w", encoding="utf-8") as f:
        json.dump({"resources": resources}, f, indent=4)

    positive = is_paper_positive(paper)
    info = {
        "filename": paper["filename"],
        "ground_truth_status": "POSITIVE (Has Target Resource)" if positive else "NEGATIVE (No Target Resource)",
        "extraction_model": cfg["model"],
        "shot_mode": mode,
        "shots_used": len(shots),
        "shot_files": [s["filename"] for s in shots],
        "extracted_resources_count": len(resources),
        "raw_resources_count": len(raw) if isinstance(raw, list) else 0,
        "skipped_resources_details": skipped,
        "api_error": "error" in response_obj,
    }
    with open(batch_dir / "info.yaml", "w") as f:
        yaml.safe_dump(info, f)
    return info
