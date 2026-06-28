"""Filesystem layout for ReadmitRisk.

A single source of truth for where raw data, the derived cohort, model artifacts,
reports, and the committed sample live. Everything is rooted at the project root
(the directory containing ``pyproject.toml``) so paths resolve identically whether
the code runs from the host, a container, or pytest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the project root (the nearest ancestor containing ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: two levels up from this file (src/readmitrisk/paths.py -> root).
    return here.parents[2]


def _data_dir() -> Path:
    root = project_root()
    raw = os.environ.get("READMIT_DATA_DIR", "data")
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


@dataclass(frozen=True)
class Paths:
    """Resolved project paths. Use :func:`get_paths` to construct."""

    root: Path
    config: Path
    sql: Path
    data: Path
    raw: Path
    cohort: Path
    artifacts: Path
    reports: Path
    sample: Path

    @property
    def cohort_parquet(self) -> Path:
        return self.cohort / "cohort.parquet"

    @property
    def sample_cohort_parquet(self) -> Path:
        return self.sample / "cohort.parquet"

    @property
    def duckdb_file(self) -> Path:
        return self.data / "readmitrisk.duckdb"

    def ensure(self) -> Paths:
        """Create the writable output directories if they do not yet exist."""
        for d in (self.data, self.raw, self.cohort, self.artifacts, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    root = project_root()
    data = _data_dir()
    return Paths(
        root=root,
        config=root / "config",
        sql=root / "sql",
        data=data,
        raw=data / "raw",
        cohort=data / "cohort",
        artifacts=data / "artifacts",
        reports=data / "reports",
        sample=root / "data" / "sample",
    )
