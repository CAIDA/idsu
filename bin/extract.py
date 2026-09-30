#!/usr/bin/env python3
"""Run the extraction pipeline over one or more splits, several times.

    uv run bin/extract.py --split training
    uv run bin/extract.py --split validation --split evaluation    # held-out

Writes <out>/run-<k>/batch-<n>-1/ for each run k and paper n, plus
<out>/run-<k>/run.yaml recording the papers, settings and commit. Score the
result with bin/evaluate.py <out>.

Needs OPENAI_API_KEY and OPENAI_BASE_URL (see .env.example) unless --dry-run,
which writes every prompt and an empty extraction without calling a model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idsu import extract, llm  # noqa: E402
from idsu.config import REPO_ROOT, load_config, repo_path  # noqa: E402
from idsu.data import load_ground_truth, load_split, load_target_list, mark_in_target_list  # noqa: E402
from idsu.pdf import pdf_text  # noqa: E402

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "idsu", "bin", "config.yaml"],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(cfg: dict) -> argparse.Namespace:
    ex = cfg["extraction"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config file (default: config.yaml)")
    ap.add_argument("--split", action="append", choices=sorted(cfg["data"]["splits"]),
                    help="split to extract; repeat for several (default: training)")
    ap.add_argument("--runs", type=int, default=ex["runs"])
    ap.add_argument("--model", default=ex["model"])
    ap.add_argument("--shots", type=int, default=ex["shots"], choices=[1, 2, 3])
    ap.add_argument("--shot-mode", default=ex["shot_mode"], choices=["partial", "whole"])
    ap.add_argument("--seed", type=int, default=ex["seed"])
    ap.add_argument("--workers", type=int, default=ex["workers"])
    ap.add_argument("--out", default=None, help="output directory (default: <runs_dir>/<timestamp>-<model>-<splits>)")
    ap.add_argument("--only", action="append", default=[], metavar="FILENAME", help="restrict to this paper; repeatable")
    ap.add_argument("--resume", action="store_true", help="keep batches already done in --out")
    ap.add_argument("--dry-run", action="store_true", help="write prompts, call no model")
    return ap.parse_args()


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    cfg = load_config(pre.parse_known_args()[0].config)
    args = parse_args(cfg)
    splits = args.split or ["training"]
    data, ex = cfg["data"], dict(cfg["extraction"])
    ex.update(model=args.model, shots=args.shots, shot_mode=args.shot_mode, seed=args.seed)

    papers_dir = repo_path(data["papers_dir"])
    target_list = load_target_list(repo_path(data["target_list"]))
    ground_truth = load_ground_truth(repo_path(data["ground_truth"]))
    mark_in_target_list(ground_truth, target_list)
    by_file = {p["filename"]: p for p in ground_truth}

    def split_papers(name: str) -> list[dict]:
        files = load_split(repo_path(data["splits"][name]))
        missing_gt = [f for f in files if f not in by_file]
        if missing_gt:
            sys.exit(f"[ERROR] split {name}: no ground truth for {missing_gt}")
        return [by_file[f] for f in files]

    shot_pool = split_papers(ex["shot_split"])
    targets = [p for s in splits for p in split_papers(s)]
    if args.only:
        unknown = sorted(set(args.only) - {p["filename"] for p in targets})
        if unknown:
            sys.exit(f"[ERROR] --only names papers not in {splits}: {unknown}")
        targets = [p for p in targets if p["filename"] in args.only]
    missing_pdf = [p["filename"] for p in shot_pool + targets if not (papers_dir / p["filename"]).is_file()]
    if missing_pdf:
        sys.exit(f"[ERROR] {len(missing_pdf)} PDF(s) not in {papers_dir}: {sorted(set(missing_pdf))}")

    if args.out:
        out = repo_path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = repo_path(ex["runs_dir"]) / f"{stamp}-{args.model}-{'+'.join(splits)}"
    if out.exists() and any(out.iterdir()) and not args.resume:
        sys.exit(f"[ERROR] {out} is not empty; pass --resume to continue it")

    client = None if args.dry_run else llm.make_client(timeout=ex["timeout"])
    log(f"Extracting {len(targets)} paper(s) from {'+'.join(splits)}, {args.runs} run(s), model {args.model}"
        f"{' (dry run)' if args.dry_run else ''}\nOutput: {out}")

    texts = {p["filename"]: pdf_text(papers_dir / p["filename"]) for p in targets}
    commit = git_commit()
    prompt_hashes = {p.name: sha256(p) for p in sorted(extract.PROMPTS_DIR.glob("extraction-*.prompt"))}
    tasks = []
    for k in range(1, args.runs + 1):
        run_dir = out / f"run-{k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_info = {"run": k, "created": datetime.now().isoformat(timespec="seconds"), "commit": commit,
                    "splits": splits, "papers": [p["filename"] for p in targets], "extraction": ex,
                    "ground_truth": data["ground_truth"], "target_list": data["target_list"],
                    "prompts_sha256": prompt_hashes, "dry_run": args.dry_run}
        with open(run_dir / "run.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(run_info, f, sort_keys=False)
        for n, paper in enumerate(targets, 1):
            batch_dir = run_dir / f"batch-{n}-1"
            if args.resume and (batch_dir / "info.yaml").is_file():
                done = yaml.safe_load((batch_dir / "info.yaml").read_text(encoding="utf-8"))
                if not done.get("api_error"):
                    continue
            # Each (run, paper) draws its own shots from a seeded generator, so a
            # rerun picks the same shots whatever order the workers finish in.
            rng = random.Random(f"{args.seed}:{k}:{paper['filename']}")
            shots = copy.deepcopy(extract.select_shots(shot_pool, paper["filename"], args.shots, rng))
            tasks.append((k, n, paper, batch_dir, shots))

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract.extract_paper, paper=paper, batch_dir=batch_dir, paper_text=texts[paper["filename"]],
                               shots=shots, target_list=target_list, cfg=ex, client=client,
                               dry_run=args.dry_run, log=log): (k, n, paper) for k, n, paper, batch_dir, shots in tasks}
        for fut in as_completed(futures):
            k, n, paper = futures[fut]
            try:
                info = fut.result()
                failures += info["api_error"]
                log(f"run {k} paper {n:2d}: {info['extracted_resources_count']} resource(s)"
                    f"{'  [API ERROR]' if info['api_error'] else ''}  {paper['filename']}")
            except Exception as e:
                failures += 1
                log(f"[ERROR] run {k} paper {n}: {paper['filename']}: {e}")

    log(f"\nDone: {len(tasks)} batch(es), {failures} failed. Output: {out}")
    if failures:
        log("Rerun with --resume to retry the failed batches.")
        sys.exit(1)


if __name__ == "__main__":
    main()
