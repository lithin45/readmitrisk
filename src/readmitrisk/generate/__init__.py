"""Synthetic data generation: dispatch between real Synthea and the fallback.

``run_generation`` is the single entry point used by ``make generate`` / the CLI. It
honors the configured backend (``synthea`` | ``fallback`` | ``auto``) and always writes
the same four Synthea-schema CSV tables to the raw data dir, so everything downstream is
backend-agnostic.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..config import Config, load_config
from ..paths import get_paths
from . import fallback
from .synthea import SyntheaUnavailable, run_synthea

REQUIRED_TABLES = ("patients", "encounters", "conditions", "observations")


def _all_tables_present(raw_dir: Path) -> bool:
    return all((raw_dir / f"{t}.csv").exists() for t in REQUIRED_TABLES)


def run_generation(config: Config | None = None, force_backend: str | None = None) -> dict:
    """Generate synthetic EHR into the raw data dir and return a summary dict.

    The summary always contains ``backend`` plus per-table presence; for the fallback
    backend it also includes record counts useful for the smoke test.
    """
    config = config or load_config()
    paths = get_paths().ensure()
    raw = paths.raw
    backend = force_backend or config.generation.backend

    # Idempotency / Synthea-profile handoff: if data already exists and the caller did
    # not force a backend, reuse it (this is how the pipeline container consumes CSVs the
    # Synthea one-shot container already staged into the shared volume).
    if force_backend is None and backend == "auto" and _all_tables_present(raw):
        return {
            "backend": "existing",
            "raw_dir": str(raw),
            "tables_present": True,
            "note": "reused existing raw tables (run `make clean` or pass --backend to regenerate)",
        }

    used = None
    summary: dict = {}

    if backend in ("synthea", "auto"):
        try:
            syn = config.synthea_raw.get("synthea", {})
            run_synthea(
                out_dir=raw,
                population=config.generation.population,
                seed=config.generation.seed,
                state=config.generation.state,
                city=config.generation.city,
                jar_path=syn.get("jar_path", "/opt/synthea/synthea-with-dependencies.jar"),
                exporter_flags=syn.get("exporter_flags"),
            )
            used = "synthea"
        except SyntheaUnavailable as exc:
            if backend == "synthea":
                raise
            summary["synthea_fallback_reason"] = str(exc)

    if used is None:
        gen = fallback.generate(
            out_dir=raw,
            population=config.generation.population,
            seed=config.generation.seed,
            reference_date=config.generation.reference_date,
            followup_days=config.features.followup_days,
        )
        used = "fallback"
        summary.update(asdict(gen))

    summary["backend"] = used
    summary["raw_dir"] = str(raw)
    summary["tables_present"] = _all_tables_present(raw)
    return summary


__all__ = ["run_generation", "fallback", "REQUIRED_TABLES"]
