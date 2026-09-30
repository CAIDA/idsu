#!/usr/bin/env python3
"""Put the benchmark papers in data.papers_dir, each checked against its recorded hash.

    uv run bin/fetch_papers.py                       # check; download what can be downloaded
    uv run bin/fetch_papers.py --import ~/Downloads  # add PDFs you downloaded yourself
    uv run bin/fetch_papers.py --no-download         # check only

Every paper is identified by the SHA-256 in data.papers_manifest, because the
ground-truth quotes were annotated against those exact files; another version of
the same paper (a preprint, a re-typeset copy) is not accepted in its place.

- Papers with an arXiv id are downloaded from arXiv at the recorded version.
- The rest cannot be fetched by a script (the ACM Digital Library refuses
  scripted downloads), so this lists each with its DOI link. Download them in a
  browser, then run with --import <dir>: every PDF in <dir> whose hash matches a
  missing paper is copied in under the name the ground truth uses. Downloaded
  names do not matter.

Exits non-zero while any paper is missing or differs from its recorded hash.
A file already in place with the wrong hash is reported, never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idsu.config import load_config, repo_path  # noqa: E402

USER_AGENT = "idsu-fetch-papers/0.1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expected: str, timeout: float = 120) -> str | None:
    """Download url to dest if its hash is expected. Returns None on success, else why not."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                shutil.copyfileobj(resp, tmp)
        except OSError as e:
            tmp_path.unlink(missing_ok=True)
            return str(e)
    got = sha256(tmp_path)
    if got != expected:
        tmp_path.unlink()
        return f"downloaded file has a different hash ({got[:12]}…), so it is another version"
    tmp_path.replace(dest)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--import", dest="import_dir", metavar="DIR", help="copy in PDFs from DIR that match a missing paper")
    ap.add_argument("--no-download", action="store_true", help="do not download from arXiv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    papers_dir = repo_path(cfg["data"]["papers_dir"])
    manifest = yaml.safe_load(repo_path(cfg["data"]["papers_manifest"]).read_text(encoding="utf-8"))
    papers_dir.mkdir(parents=True, exist_ok=True)

    missing, wrong = [], []
    for p in manifest:
        path = papers_dir / p["filename"]
        if not path.exists():
            missing.append(p)
        elif sha256(path) != p["sha256"]:
            wrong.append(p)
    print(f"{len(manifest) - len(missing) - len(wrong)}/{len(manifest)} papers present and verified in {papers_dir}")

    if args.import_dir and missing:
        by_hash = {p["sha256"]: p for p in missing}
        for f in sorted(Path(args.import_dir).expanduser().glob("*.pdf")):
            p = by_hash.pop(sha256(f), None)
            if p:
                shutil.copy2(f, papers_dir / p["filename"])
                print(f"  imported  {p['filename']}  (from {f.name})")
        missing = list(by_hash.values())

    if not args.no_download:
        still = []
        for p in missing:
            if not p.get("arxiv"):
                still.append(p)
                continue
            why = download(f"https://arxiv.org/pdf/{p['arxiv']}", papers_dir / p["filename"], p["sha256"])
            if why:
                print(f"  [FAIL] arXiv {p['arxiv']}: {why}  {p['filename']}")
                still.append(p)
            else:
                print(f"  downloaded  arXiv {p['arxiv']}  {p['filename']}")
        missing = still

    for p in wrong:
        print(f"  [DIFFERS] {p['filename']}: in place, but not the recorded version; move it aside and re-fetch")
    if missing:
        print(f"\n{len(missing)} paper(s) to download yourself, then rerun with --import <download dir>:")
        for p in missing:
            link = f"https://doi.org/{p['doi']}" if p.get("doi") else "no DOI or arXiv id recorded; find it by title"
            print(f"  {link}\n      {p['filename']}")
    if missing or wrong:
        sys.exit(1)
    print("All papers present and verified.")


if __name__ == "__main__":
    main()
