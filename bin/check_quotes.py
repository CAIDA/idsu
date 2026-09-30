#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pypdf>=4", "fonttools>=4", "pyyaml>=6", "rapidfuzz>=3"]
# ///
"""Check that every ground-truth quote actually appears in its paper.

For each resource instance in a ground-truth file, every `quotes[].quote` is
searched for in the text of the paper's PDF. Matching ignores case, whitespace,
punctuation and line-break hyphenation, and an ellipsis in a quote matches any
gap. A quote that is not found verbatim is
scored with a fuzzy partial match, which catches PDF-extraction differences.

With --baseline, only instances that are new or whose quotes differ from the
baseline file are checked. This is the evidence check for ground-truth edits.

Paths are relative to the repo root, so the script runs from any directory.

    bin/check_quotes.py
    bin/check_quotes.py --baseline data/imc2023/_metadata/ground-truth.yaml
    bin/check_quotes.py --tsv out.tsv

Exits 1 if any checked quote is missing or its PDF is absent.
"""
import argparse
import collections
import csv
import logging
import re
import sys
import unicodedata
from pathlib import Path

import yaml
from pypdf import PdfReader
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("training", "validation", "evaluation")
logging.getLogger("pypdf").setLevel(logging.ERROR)


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^a-z0-9]", "", text)


def find_in_order(fragments, text):
    """True if every fragment occurs in text, each after the previous one."""
    pos = 0
    for frag in fragments:
        pos = text.find(frag, pos)
        if pos < 0:
            return False
        pos += len(frag)
    return True


def load_gt(path):
    """Return {filename: {id: resource}} from a multi-document ground-truth file."""
    with open(ROOT / path) as f:
        return {
            doc["filename"]: {r["id"]: r for r in doc.get("resources") or []}
            for doc in yaml.safe_load_all(f)
            if doc
        }


def load_splits(papers_dir):
    split = {}
    for name in SPLITS:
        with open(ROOT / papers_dir / f"papers_{name}.yaml") as f:
            for p in yaml.safe_load(f) or []:
                split[p["filename"]] = name
    return split


def quotes_of(resource):
    return [(str(q.get("quote", "")), q.get("used_in_paper")) for q in resource.get("quotes") or []]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default="data/imc2023/_metadata/ground-truth-merged.yaml")
    ap.add_argument("--papers-dir", default="data/imc2023")
    ap.add_argument("--baseline", help="check only instances added or changed relative to this file")
    ap.add_argument("--threshold", type=float, default=90.0, help="fuzzy score counted as found (0-100)")
    ap.add_argument("--tsv", help="write one row per checked quote")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    baseline = load_gt(args.baseline) if args.baseline else None
    split = load_splits(args.papers_dir)

    rows = []
    for filename, resources in gt.items():
        pdf = ROOT / args.papers_dir / filename
        text = None
        if pdf.exists():
            text = normalize(" ".join(page.extract_text() or "" for page in PdfReader(pdf).pages))
        for rid, resource in resources.items():
            if baseline is not None:
                old = baseline.get(filename, {}).get(rid)
                if old is not None and quotes_of(old) == quotes_of(resource):
                    continue
            for quote, used in quotes_of(resource):
                # An annotator's "..." elides text: each piece must appear, in order.
                fragments = [f for f in map(normalize, re.split(r"\.\.\.|…", quote)) if f]
                needle = "".join(fragments)
                if text is None:
                    status, score = "no-pdf", 0.0
                elif not needle:
                    status, score = "empty-quote", 0.0
                elif find_in_order(fragments, text):
                    status, score = "exact", 100.0
                else:
                    score = fuzz.partial_ratio(needle, text)
                    status = "fuzzy" if score >= args.threshold else "missing"
                rows.append({
                    "status": status,
                    "score": round(score, 1),
                    "split": split.get(filename, "unsplit"),
                    "paper": filename,
                    "id": rid,
                    "used_in_paper": used,
                    "quote": " ".join(quote.split()),
                })

    if args.tsv:
        with open(args.tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["status"], delimiter="\t")
            w.writeheader()
            w.writerows(rows)

    counts = collections.Counter((r["split"], r["status"]) for r in rows)
    statuses = sorted({s for _, s in counts})
    print(f"{len(rows)} quotes checked in {args.gt}" + (f" (changed vs {args.baseline})" if baseline is not None else ""))
    print("split".ljust(12) + "".join(s.rjust(13) for s in statuses))
    for sp in sorted({sp for sp, _ in counts}):
        print(sp.ljust(12) + "".join(str(counts[sp, s]).rjust(13) for s in statuses))

    bad = [r for r in rows if r["status"] in ("missing", "no-pdf", "empty-quote")]
    for r in bad:
        print(f"\n[{r['status']} {r['score']}] {r['split']} | {r['paper']} | {r['id']}\n    {r['quote'][:200]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
