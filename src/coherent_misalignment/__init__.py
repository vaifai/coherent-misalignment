"""Coherent Misalignment — MSM extension to inverted-persona prevention."""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv_if_present(path: Path = _REPO_ROOT / ".env") -> None:
    """Load KEY=value lines from .env into os.environ.

    - No-op if the file doesn't exist (so Colab, which uses notebook secrets,
      is unaffected — .env exists only on the local Mac).
    - Does NOT overwrite existing env vars, so an explicit `export` or
      Colab's secret-injection still wins.
    - Supports lines like `KEY=value`, `KEY="value with spaces"`,
      `KEY='value'`, and comments starting with `#`.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_if_present()
