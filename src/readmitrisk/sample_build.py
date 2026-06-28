"""Build the small committed cached sample under ``data/sample/``.

CI and the ``--sample`` Make targets run against this committed cohort instead of
regenerating the full population, so the eval gate is fast and fully reproducible. Run
via ``make sample`` after changing the generator or cohort SQL, then commit the result.
"""

from __future__ import annotations

import json
from pathlib import Path

from .cohort import build_cohort_from_raw
from .config import load_config
from .generate import fallback
from .paths import get_paths

SAMPLE_POPULATION = 1500
SAMPLE_SEED = 7


def build_sample() -> Path:
    paths = get_paths()
    sample_dir = paths.sample
    raw_dir = sample_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    followup = cfg.features.followup_days

    gen = fallback.generate(
        out_dir=raw_dir,
        population=SAMPLE_POPULATION,
        seed=SAMPLE_SEED,
        reference_date=cfg.generation.reference_date,
        followup_days=followup,
    )
    cohort = build_cohort_from_raw(raw_dir, followup_days=followup)
    out = sample_dir / "cohort.parquet"
    cohort.to_parquet(out, index=False)

    meta = {
        "population": SAMPLE_POPULATION,
        "seed": SAMPLE_SEED,
        "followup_days": followup,
        "n_index_encounters": int(len(cohort)),
        "n_events": int(cohort[cfg.features.event_col].sum()),
        "event_rate": round(float(cohort[cfg.features.event_col].mean()), 4),
        "generator": "fallback",
        "raw_counts": gen.__dict__,
    }
    (sample_dir / "sample_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote sample cohort -> {out}")
    print(json.dumps(meta, indent=2))
    return out


if __name__ == "__main__":
    build_sample()
