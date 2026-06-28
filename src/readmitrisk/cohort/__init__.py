"""Cohort construction, run the DuckDB clinical SQL and return a tidy cohort.

The heavy lifting lives in ``sql/cohort.sql`` (index selection, the discharge->next-
admission clock, right-censoring, comorbidity burden, prior utilization, latest labs).
This module just parameterizes and executes it, then wraps the result with light
post-processing and a human-readable summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import load_config
from ..duckdb_util import connect, register_raw_views
from ..paths import get_paths

# Columns the SQL is contracted to produce. Tests + the modeling layer rely on these.
TARGET_COLUMNS = ["duration_days", "event_observed"]
SUBGROUP_COLUMNS = ["sex", "age_band", "race", "ethnicity"]


def _load_sql(followup_days: int, lookback_days: int) -> str:
    sql_path = get_paths().sql / "cohort.sql"
    sql = sql_path.read_text(encoding="utf-8")
    return sql.replace("__FOLLOWUP__", str(int(followup_days))).replace(
        "__LOOKBACK__", str(int(lookback_days))
    )


def build_cohort_from_raw(
    raw_dir: Path,
    followup_days: int = 30,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Execute the cohort SQL over the raw CSVs in ``raw_dir`` and return a DataFrame."""
    sql = _load_sql(followup_days, lookback_days)
    con = connect()
    try:
        register_raw_views(con, raw_dir)
        df = con.execute(sql).fetchdf()
    finally:
        con.close()

    # Integer-typed counts/flags (DuckDB returns them as int64 already, but be explicit).
    for col in [
        "event_observed",
        "n_conditions",
        "comorbidity_score",
        "n_prior_encounters",
        "n_prior_inpatient",
        "n_prior_emergency",
    ]:
        df[col] = df[col].astype("int64")
    return df


@dataclass
class CohortResult:
    df: pd.DataFrame
    followup_days: int
    path: Path | None

    def describe(self) -> str:
        df = self.df
        n = len(df)
        events = int(df["event_observed"].sum())
        rate = events / n if n else 0.0
        dur = df["duration_days"]
        lines = [
            f"Index inpatient encounters : {n:,}",
            f"Distinct patients          : {df['patient_id'].nunique():,}",
            f"Readmission events (<= {self.followup_days}d): {events:,}  ({rate:.1%})",
            f"Right-censored             : {n - events:,}  ({1 - rate:.1%})",
            f"Duration days  min/median/max: {dur.min():.2f} / {dur.median():.2f} / {dur.max():.2f}",
            "",
            "Subgroup coverage:",
        ]
        for col in SUBGROUP_COLUMNS:
            counts = df[col].value_counts(dropna=False).to_dict()
            pretty = ", ".join(f"{k}={v}" for k, v in counts.items())
            lines.append(f"  {col:<10}: {pretty}")
        missing = {c: int(df[c].isna().sum()) for c in ["bmi", "systolic_bp", "hba1c", "glucose"]}
        lines.append("")
        lines.append(f"Lab missingness (median-imputed downstream): {missing}")
        if self.path is not None:
            lines.append("")
            lines.append(f"Written to: {self.path}")
        return "\n".join(lines)


def build_cohort(write: bool = True, use_sample: bool = False) -> CohortResult:
    """Build the cohort from the configured raw data (or the committed sample's raw)."""
    cfg = load_config()
    paths = get_paths()
    followup = cfg.features.followup_days
    lookback = cfg.features.lookback_days

    if use_sample:
        # Prefer the prebuilt sample cohort parquet; fall back to its raw CSVs.
        if paths.sample_cohort_parquet.exists():
            df = pd.read_parquet(paths.sample_cohort_parquet)
            return CohortResult(df=df, followup_days=followup, path=paths.sample_cohort_parquet)
        raw_dir = paths.sample / "raw"
    else:
        raw_dir = paths.raw

    df = build_cohort_from_raw(raw_dir, followup_days=followup, lookback_days=lookback)

    out_path = None
    if write and not use_sample:
        paths.ensure()
        out_path = paths.cohort_parquet
        df.to_parquet(out_path, index=False)
    return CohortResult(df=df, followup_days=followup, path=out_path)


def load_cohort(use_sample: bool = False) -> pd.DataFrame:
    """Load a previously built cohort (sample or full); build it if missing."""
    paths = get_paths()
    if use_sample:
        if paths.sample_cohort_parquet.exists():
            return pd.read_parquet(paths.sample_cohort_parquet)
        return build_cohort(write=False, use_sample=True).df
    if paths.cohort_parquet.exists():
        return pd.read_parquet(paths.cohort_parquet)
    return build_cohort(write=True, use_sample=False).df


__all__ = [
    "build_cohort",
    "build_cohort_from_raw",
    "load_cohort",
    "CohortResult",
    "TARGET_COLUMNS",
    "SUBGROUP_COLUMNS",
]
