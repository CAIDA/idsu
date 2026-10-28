#!/usr/bin/env python3
"""Compare training and validation/held-out evaluation summaries.

    uv run bin/compare_summaries.py --training <training summary> --held-out <held-out summary>
    uv run bin/compare_summaries.py --training <training summary> --validation <validation summary> --output comparison.md

Pass summary.md files or evaluation directories containing summary.md.
Prints a Markdown table; --output also saves it to a file. No API is needed.
Paths are relative to the current directory. Exits non-zero for invalid inputs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

METRICS = ("Accuracy", "Precision", "Recall", "F1")


def read_summary(path: Path) -> tuple[Path, dict[str, tuple[float, float]]]:
    """Read the aggregate metric table, ignoring the per-run confusion matrix."""
    if path.is_dir():
        path = path / "summary.md"
    rows = {}
    in_table = False
    found_table = False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if [cell.lower() for cell in cells] == ["metric", "mean", "sd"]:
            if found_table:
                raise ValueError(f"{path}: multiple metric tables; expected one summary")
            in_table = found_table = True
            continue
        if not in_table:
            continue
        if not line.strip().startswith("|"):
            in_table = False
            continue
        name = next((name for name in METRICS if cells[0].lower() == name.lower()), None)
        if name is None:
            continue
        if name in rows:
            raise ValueError(f"{path}: duplicate {name} row")
        if len(cells) != 3:
            raise ValueError(f"{path}: expected mean and SD for {name}")
        try:
            mean, sd = float(cells[1]), float(cells[2])
        except ValueError:
            raise ValueError(f"{path}: invalid mean or SD for {name}") from None
        if not (math.isfinite(mean) and math.isfinite(sd) and 0 <= mean <= 1 and sd >= 0):
            raise ValueError(f"{path}: {name} requires a finite mean in [0, 1] and nonnegative SD")
        rows[name] = (mean, sd)
    missing = [name for name in METRICS if name not in rows]
    if missing:
        raise ValueError(f"{path}: missing metrics: {', '.join(missing)}")
    return path, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--training", type=Path, required=True, help="training summary.md or evaluation directory")
    ap.add_argument("--validation", type=Path, help="validation summary.md or evaluation directory")
    ap.add_argument("--held-out", type=Path, help="held-out summary.md or evaluation directory")
    ap.add_argument("--output", type=Path, help="also save the comparison as Markdown")
    args = ap.parse_args()
    if args.validation is None and args.held_out is None:
        ap.error("provide --validation, --held-out, or both")

    splits = (("Training", args.training), ("Validation", args.validation), ("Held-out", args.held_out))
    try:
        summaries = [(label, *read_summary(path)) for label, path in splits
                     if path is not None]
        lines = ["### Split comparison", "", "Values are mean (sample SD), as reported in each summary.", "",
                 "| Metric | " + " | ".join(label for label, _, _ in summaries) + " |",
                 "|---|" + "---|" * len(summaries)]
        for metric in METRICS:
            values = [f"{rows[metric][0]:.4f} ({rows[metric][1]:.4f})" for _, _, rows in summaries]
            lines.append(f"| {metric} | " + " | ".join(values) + " |")
        lines += ["", "Sources:", ""]
        lines += [f"- {label}: `{path.as_posix()}`" for label, path, _ in summaries]
        table = "\n".join(lines) + "\n"
        if args.output:
            if any(args.output.resolve() == path.resolve()
                   or (args.output.exists() and args.output.samefile(path)) for _, path, _ in summaries):
                raise ValueError("--output must not overwrite an input summary")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(table, encoding="utf-8")
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    print(table, end="")


if __name__ == "__main__":
    main()
