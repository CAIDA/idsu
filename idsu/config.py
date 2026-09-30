"""Load config.yaml and resolve its paths against the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def repo_path(value: str | Path) -> Path:
    """A path from config or the command line: absolute as given, else relative to the repo root."""
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p
