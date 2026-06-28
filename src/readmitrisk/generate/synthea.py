"""Invoke the real Synthea generator (Java fat-jar) to produce synthetic EHR CSVs.

This is the *primary* generation backend, used by ``make generate`` when Docker/Java
and the Synthea jar are available (e.g. inside the ``synthea`` compose service). It runs
Synthea with a fixed seed and CSV export, then copies the four tables we consume into
the project's raw data dir using the same filenames the fallback generator emits.

If Java or the jar are unavailable this raises :class:`SyntheaUnavailable`, which the
dispatcher in :mod:`readmitrisk.generate` catches to fall back to the deterministic
Python generator.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REQUIRED_TABLES = ("patients", "encounters", "conditions", "observations")


class SyntheaUnavailable(RuntimeError):
    """Raised when the real Synthea backend cannot run (no Java or no jar)."""


def _java_available() -> bool:
    return shutil.which("java") is not None


def _resolve_jar(jar_path: str) -> Path | None:
    candidate = Path(os.environ.get("READMIT_SYNTHEA_JAR", jar_path))
    return candidate if candidate.exists() else None


def run_synthea(
    out_dir: Path,
    population: int,
    seed: int,
    state: str,
    city: str,
    jar_path: str,
    exporter_flags: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Run Synthea and copy the required CSV tables into ``out_dir``.

    Returns a mapping ``{table_name: written_path}``. Raises :class:`SyntheaUnavailable`
    if the backend is not runnable in this environment.
    """
    if not _java_available():
        raise SyntheaUnavailable("Java runtime not found on PATH.")
    jar = _resolve_jar(jar_path)
    if jar is None:
        raise SyntheaUnavailable(f"Synthea jar not found at {jar_path}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    synthea_out = out_dir / "_synthea_run"
    synthea_out.mkdir(parents=True, exist_ok=True)

    flags = {
        "exporter.csv.export": "true",
        "exporter.fhir.export": "false",
        "exporter.baseDirectory": str(synthea_out),
    }
    flags.update(exporter_flags or {})

    cmd = [
        "java",
        "-jar",
        str(jar),
        "-p",
        str(population),
        "-s",
        str(seed),
        "-cs",
        str(seed),  # clinician seed -> full determinism
        "-r",
        "20200101",  # fixed reference date
    ]
    for key, value in flags.items():
        cmd += [f"--{key}", str(value)]
    cmd += [state, city]

    # Synthea is verbose; surface failures but keep stdout out of the way.
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SyntheaUnavailable(
            f"Synthea exited with code {result.returncode}:\n{result.stderr[-2000:]}"
        )

    csv_dir = synthea_out / "csv"
    if not csv_dir.exists():
        raise SyntheaUnavailable(f"Synthea produced no csv/ output under {synthea_out}.")

    written: dict[str, Path] = {}
    for table in REQUIRED_TABLES:
        src = csv_dir / f"{table}.csv"
        if not src.exists():
            raise SyntheaUnavailable(f"Expected Synthea table missing: {src}")
        dst = out_dir / f"{table}.csv"
        shutil.copyfile(src, dst)
        written[table] = dst
    return written
