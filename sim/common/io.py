"""Atomic output helpers shared by all studies."""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl


def atomic_write_parquet(df: pl.DataFrame, out: Path) -> None:
    """Write ``df`` to ``out`` via a temporary file and a rename."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    df.write_parquet(tmp)
    os.replace(tmp, out)


def atomic_write_json(obj: object, out: Path) -> None:
    """Write ``obj`` as JSON to ``out`` via a temporary file and a rename."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, out)


def unit_seed(study: str, unit: str) -> int:
    """Deterministic 63-bit seed derived from the study and unit ids."""
    import hashlib

    digest = hashlib.sha256(f"{study}:{unit}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1
