#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pypdf>=4", "fonttools>=4", "pyyaml>=6"]
# ///
"""Write data/imc2023/_metadata/papers.yaml: one record per benchmark paper.

Each record holds the filename the ground truth uses, its split, the DOI or
arXiv id read from the PDF's first pages, and the PDF's SHA-256. The hash
lets a fetch script confirm that a downloaded copy is the exact file the
quotes were annotated against.

Reads the local PDFs, so run it where data/imc2023/*.pdf exists:

    bin/build_papers_manifest.py            # writes the manifest
    bin/build_papers_manifest.py --check    # exits 1 if the manifest is stale

Papers whose PDF carries neither identifier get `doi: null`; fill those by hand.
"""
import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

import yaml
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("training", "validation", "evaluation")
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+?)(?=[\s,;)\]]|$)")
# The arXiv stamp in a preprint's margin, e.g. "arXiv:2305.19037v1 [cs.CR]", not a citation.
ARXIV = re.compile(r"arXiv:(\d{4}\.\d{4,5})(v\d+)?\s*\[", re.IGNORECASE)
logging.getLogger("pypdf").setLevel(logging.ERROR)


def identifiers(pdf, pages=2):
    text = " ".join(p.extract_text() or "" for p in PdfReader(pdf).pages[:pages])
    # A DOI split across a line break matches truncated; the longest real one wins.
    dois = [d.rstrip(".") for d in DOI.findall(text) if "nnnn" not in d]
    doi = max(dois, key=len, default=None)
    arxiv = ARXIV.search(text)
    return (doi,
            arxiv.group(1) + (arxiv.group(2) or "") if arxiv else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers-dir", default="data/imc2023")
    ap.add_argument("--out", default="data/imc2023/_metadata/papers.yaml")
    ap.add_argument("--check", action="store_true", help="compare against --out instead of writing it")
    args = ap.parse_args()

    papers_dir = ROOT / args.papers_dir
    records = []
    for split in SPLITS:
        with open(papers_dir / f"papers_{split}.yaml") as f:
            for p in yaml.safe_load(f) or []:
                pdf = papers_dir / p["filename"]
                rec = {"filename": p["filename"], "split": split, "category": p.get("category")}
                if pdf.exists():
                    doi, arxiv = identifiers(pdf)
                    rec.update(doi=doi, arxiv=arxiv, sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
                else:
                    rec.update(doi=None, arxiv=None, sha256=None)
                    print(f"no PDF: {p['filename']}", file=sys.stderr)
                records.append(rec)

    out = ROOT / args.out
    text = yaml.safe_dump(records, sort_keys=False, allow_unicode=True, width=1000)
    if args.check:
        stale = not out.exists() or out.read_text() != text
        print(f"{args.out} is {'stale' if stale else 'current'}")
        return 1 if stale else 0
    out.write_text(text)
    missing = [r["filename"] for r in records if not r["doi"] and not r["arxiv"]]
    print(f"wrote {len(records)} papers to {args.out}; {len(missing)} without an identifier")
    for m in missing:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
