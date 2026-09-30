"""Match extracted resources to a paper's ground truth.

A port of matchExtractionWithTruth() and performLLMMatching() from
experiments/mitigating-falsepositives-nman/scripts/extraction-eval-avg-new.py,
the scorer behind the reported F1: exact id, then fuzzy id (thefuzz ratio),
then fuzzy reference (token_sort_ratio), each one-to-one, then an optional
LLM judge for whatever is still unmatched. The original followed this with an
interactive console review; that step is gone, so every match here is automatic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from thefuzz import fuzz

DEFAULT_ID_THRESHOLD = 80
DEFAULT_REF_THRESHOLD = 80
PERFECT = 100
NO_MATCH = "None"

JUDGE_TEMPLATE = (Path(__file__).resolve().parent / "prompts" / "matching-judge.prompt").read_text(encoding="utf-8")

# Takes a prompt, returns the model's raw text.
Judge = Callable[[str], str]


def best_reference_ratio(ex_refs, gt_refs) -> int:
    best = 0
    for er in ex_refs or []:
        for tr in gt_refs or []:
            if er is None or tr is None:
                continue
            score = fuzz.token_sort_ratio(str(er), str(tr))
            if score > best:
                best = score
                if best == PERFECT:
                    return PERFECT
    return best


def _extract_json_dict(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("No JSON object found in LLM output")


def judge_prompt(extracted: list[dict], truth: list[dict]) -> str:
    ex = [{"id": r["id"], "references": r.get("references")} for r in extracted]
    gt = [{"id": r["id"], "references": r.get("references")} for r in truth]
    return (JUDGE_TEMPLATE.replace("{extracted_resources}", json.dumps(ex, indent=2))
            .replace("{ground_truth_resources}", json.dumps(gt, indent=2)).strip())


def llm_matches(extracted: list[dict], truth: list[dict], judge: Judge, log=print) -> list[dict]:
    """The judge's one-to-one matches, keeping only pairs whose ground-truth id is real."""
    extracted_ids = [r.get("id", "") for r in extracted]
    if not extracted_ids:
        return []
    truth_ids = [r.get("id", "") for r in truth]
    if not truth_ids:
        return [{"extracted": eid, "ground_truth": None} for eid in extracted_ids]
    raw = judge(judge_prompt(extracted, truth))  # an API failure raises: never score a paper without its judge
    try:
        result = _extract_json_dict(raw)
        valid, used, out = set(truth_ids), set(), []
        for item in result.get("matches", []):
            gt = item.get("ground_truth")
            if gt == NO_MATCH or gt not in valid or gt in used:
                gt = None
            if gt is not None:
                used.add(gt)
            out.append({"extracted": item.get("extracted"), "ground_truth": gt})
        return [m for m in out if m.get("ground_truth") is not None]
    except Exception as e:
        log(f"[WARN] LLM judge returned unusable output, leaving {len(extracted_ids)} resources unmatched: {e}")
        return [{"extracted": eid, "ground_truth": None} for eid in extracted_ids]


def match_to_truth(extracted: list[dict], truth: list[dict], judge: Judge | None = None,
                   id_threshold: int = DEFAULT_ID_THRESHOLD, ref_threshold: int = DEFAULT_REF_THRESHOLD,
                   log=print) -> dict[str, list]:
    """Returns {"matches", "extracted", "ground_truth"}: the pairs, then what is left unmatched on each side."""
    result = {"matches": [], "extracted": list(extracted), "ground_truth": list(truth)}
    matched_ex, matched_gt, used_gt_ids = [], [], set()

    for ex in result["extracted"]:
        best_id, best_id_truth, best_ref = -1, None, -1
        candidates = []
        ex_id = ex.get("id", "")
        ex_refs = ex.get("references") or []

        for gt in result["ground_truth"]:
            gt_id = gt.get("id", "")
            if gt_id in used_gt_ids:
                continue
            gt_refs = gt.get("references") or []
            if ex_id == gt_id:
                best_id, best_id_truth = PERFECT, gt
                best_ref = best_reference_ratio(ex_refs, gt_refs)
                candidates.append({"truth": gt, "id_score": best_id, "ref_score": best_ref})
                break
            id_score = fuzz.ratio(str(ex_id), str(gt_id))
            if id_score > best_id:
                best_id, best_id_truth = id_score, gt
            ref_score = best_reference_ratio(ex_refs, gt_refs)
            best_ref = max(best_ref, ref_score)
            candidates.append({"truth": gt, "id_score": id_score, "ref_score": ref_score})

        # Among the signals over their thresholds, take the strongest; a reference
        # tie across several ground-truth entries defers to the id.
        id_above = best_id_truth is not None and best_id >= id_threshold
        ref_above = best_ref >= ref_threshold
        top_ref = [c for c in candidates if c["ref_score"] == best_ref] if ref_above else []
        chosen = reason = score = None
        if id_above and ref_above:
            if len(top_ref) != 1 or best_id >= best_ref:
                chosen, reason, score = best_id_truth, "id", best_id
            else:
                chosen, reason, score = top_ref[0]["truth"], "reference", best_ref
        elif id_above:
            chosen, reason, score = best_id_truth, "id", best_id
        elif ref_above:
            if len(top_ref) == 1:
                chosen, reason, score = top_ref[0]["truth"], "reference", best_ref
            else:
                tie = max(top_ref, key=lambda c: c["id_score"])
                if tie["id_score"] >= id_threshold:
                    chosen, reason, score = tie["truth"], "reference_tie_broken_by_id", best_ref

        if chosen is not None:
            result["matches"].append({
                "extracted": ex_id,
                "ground_truth": chosen.get("id"),
                "match_reason": reason,
                "match_score": score,
                "in_target_list": ex.get("in_target_list") or chosen.get("in_target_list"),
                "used_extraction": ex.get("used_in_paper"),
                "used_truth": chosen.get("used_in_paper"),
                "references_extracted": ex.get("references"),
                "references_truth": chosen.get("references"),
            })
            matched_ex.append(ex)
            matched_gt.append(chosen)
            used_gt_ids.add(chosen.get("id"))

    for ex in matched_ex:
        if ex in result["extracted"]:
            result["extracted"].remove(ex)
    for gt in matched_gt:
        if gt in result["ground_truth"]:
            result["ground_truth"].remove(gt)

    if judge is not None and result["extracted"]:
        judged = llm_matches(result["extracted"], result["ground_truth"], judge, log=log)
        ex_by_id = {r.get("id"): r for r in result["extracted"]}
        gt_by_id = {r.get("id"): r for r in result["ground_truth"]}
        for m in judged:
            ex, gt = ex_by_id.get(m.get("extracted")), gt_by_id.get(m.get("ground_truth"))
            m["match_reason"] = "llm"
            m["references_extracted"] = (ex.get("references") or []) if ex else []
            m["references_truth"] = (gt.get("references") or []) if gt else []
            m["in_target_list"] = ex.get("in_target_list") if ex else None
            m["used_extraction"] = ex.get("used_in_paper") if ex else False
            m["used_truth"] = gt.get("used_in_paper") if gt else False
        result["matches"].extend(judged)
        done_ex = {m.get("extracted") for m in judged if m.get("extracted")}
        done_gt = {m.get("ground_truth") for m in judged if m.get("ground_truth") not in (None, NO_MATCH)}
        result["extracted"] = [r for r in result["extracted"] if r.get("id") not in done_ex]
        result["ground_truth"] = [r for r in result["ground_truth"] if r.get("id") not in done_gt]

    return result
