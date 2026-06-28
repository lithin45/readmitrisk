"""Small DuckDB helpers shared by the cohort build and the smoke test.

DuckDB reads the raw Synthea CSVs directly (no load step), which is what makes the
clinical cohort construction a pure-SQL exercise. These helpers register the four raw
tables as views so any query can reference ``patients``/``encounters``/``conditions``/
``observations`` by name.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

RAW_TABLES = ("patients", "encounters", "conditions", "observations")


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection (the cohort build is ephemeral)."""
    return duckdb.connect(database=":memory:", read_only=read_only)


def register_raw_views(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    """Register each raw CSV as a DuckDB view named after the table.

    ``all_varchar=true`` keeps every column as text so we control type coercion
    explicitly in the cohort SQL (Synthea's empty STOP/DEATHDATE cells otherwise trip
    up type inference).
    """
    raw_dir = Path(raw_dir)
    for table in RAW_TABLES:
        path = raw_dir / f"{table}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw table: {path}")
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"SELECT * FROM read_csv_auto('{path.as_posix()}', header=true, all_varchar=true)"
        )


def raw_counts(raw_dir: Path) -> dict[str, int]:
    """Return record counts for the raw tables plus an inpatient-encounter count.

    Used as the Phase 1 smoke test: proves Synthea output is present and DuckDB reads it.
    """
    con = connect()
    try:
        register_raw_views(con, raw_dir)
        counts = {}
        for table in RAW_TABLES:
            counts[table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        counts["inpatient_encounters"] = con.execute(
            "SELECT count(*) FROM encounters WHERE lower(ENCOUNTERCLASS) = 'inpatient'"
        ).fetchone()[0]
        counts["distinct_patients"] = con.execute(
            "SELECT count(DISTINCT Id) FROM patients"
        ).fetchone()[0]
        return counts
    finally:
        con.close()
