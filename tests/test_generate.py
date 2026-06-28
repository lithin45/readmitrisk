"""Phase 1 acceptance tests: synthetic generation + DuckDB readability + determinism."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from readmitrisk.duckdb_util import connect, raw_counts, register_raw_views
from readmitrisk.generate import REQUIRED_TABLES, fallback

EXPECTED_PATIENT_COLS = {"Id", "BIRTHDATE", "DEATHDATE", "RACE", "ETHNICITY", "GENDER"}
EXPECTED_ENCOUNTER_COLS = {"Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS"}


def test_all_required_tables_present(raw_dir: Path) -> None:
    for table in REQUIRED_TABLES:
        path = raw_dir / f"{table}.csv"
        assert path.exists(), f"missing {table}.csv"
        assert path.stat().st_size > 0, f"empty {table}.csv"


def test_duckdb_reads_raw_and_counts(raw_dir: Path) -> None:
    counts = raw_counts(raw_dir)
    assert counts["distinct_patients"] == 600
    assert counts["patients"] == 600
    assert counts["inpatient_encounters"] > 0
    # Realistic admission burden: not every encounter is inpatient, but every patient
    # has at least one index admission on average.
    assert counts["inpatient_encounters"] >= counts["distinct_patients"]
    assert counts["inpatient_encounters"] < 3 * counts["distinct_patients"]


def test_synthea_schema_columns(raw_dir: Path) -> None:
    patients = pd.read_csv(raw_dir / "patients.csv")
    encounters = pd.read_csv(raw_dir / "encounters.csv")
    assert EXPECTED_PATIENT_COLS.issubset(patients.columns)
    assert EXPECTED_ENCOUNTER_COLS.issubset(encounters.columns)
    # Encounter classes include the Synthea vocabulary we rely on.
    classes = set(encounters["ENCOUNTERCLASS"].str.lower().unique())
    assert "inpatient" in classes


def test_reproducible_under_fixed_seed(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    fallback.generate(out_dir=a, population=400, seed=99, reference_date="2020-01-01")
    fallback.generate(out_dir=b, population=400, seed=99, reference_date="2020-01-01")
    for table in REQUIRED_TABLES:
        assert (a / f"{table}.csv").read_bytes() == (b / f"{table}.csv").read_bytes(), (
            f"{table}.csv differs across identical-seed runs"
        )


def test_different_seed_changes_output(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    fallback.generate(out_dir=a, population=400, seed=1, reference_date="2020-01-01")
    fallback.generate(out_dir=b, population=400, seed=2, reference_date="2020-01-01")
    assert (a / "patients.csv").read_bytes() != (b / "patients.csv").read_bytes()


def test_observations_use_expected_loinc_codes(raw_dir: Path) -> None:
    con = connect()
    try:
        register_raw_views(con, raw_dir)
        codes = {row[0] for row in con.execute("SELECT DISTINCT CODE FROM observations").fetchall()}
    finally:
        con.close()
    # The four lab/vital LOINC codes the cohort SQL pulls must be present.
    assert {"39156-5", "8480-6", "4548-4", "2339-0"}.issubset(codes)
