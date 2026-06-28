"""Golden test: exact duration/event logic on a tiny hand-crafted dataset.

This pins down the censoring semantics the rest of the project depends on:
  * readmission within the window  -> event=1, duration = gap
  * no readmission, full follow-up -> event=0, duration = horizon (administrative censor)
  * death before readmission       -> event=0, duration = days-to-death
  * readmission beyond the window  -> event=0, duration = horizon
  * death DURING the index stay    -> row excluded (cannot be readmitted)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from readmitrisk.cohort import build_cohort_from_raw

FOLLOWUP = 30


def _ts(s: str) -> str:
    return f"{s}T12:00:00Z"


def _write_raw(tmp: Path) -> None:
    # Anchor end-of-data far in the future so administrative censoring = horizon for the
    # cases that should reach full follow-up.
    encounters = pd.DataFrame(
        [
            # patient A: index discharged 2020-01-05, readmitted 10 days later -> event, dur=10
            ("A_idx", _ts("2020-01-01"), _ts("2020-01-05"), "A", "inpatient"),
            ("A_re", _ts("2020-01-15"), _ts("2020-01-18"), "A", "inpatient"),
            # patient B: index, no readmission, full follow-up -> censored at 30
            ("B_idx", _ts("2020-02-01"), _ts("2020-02-05"), "B", "inpatient"),
            # patient C: index, dies 12 days after discharge -> censored at 12
            ("C_idx", _ts("2020-01-01"), _ts("2020-01-05"), "C", "inpatient"),
            # patient D: index, readmitted 45 days later (beyond window) -> censored at 30
            ("D_idx", _ts("2020-04-01"), _ts("2020-04-05"), "D", "inpatient"),
            ("D_re", _ts("2020-05-20"), _ts("2020-05-23"), "D", "inpatient"),
            # patient E: dies DURING the index stay -> excluded entirely
            ("E_idx", _ts("2020-03-01"), _ts("2020-03-10"), "E", "inpatient"),
            # patient Z: late stay that anchors end-of-data well past everyone else
            ("Z_anchor", _ts("2021-05-01"), _ts("2021-06-01"), "Z", "inpatient"),
            # a non-inpatient prior encounter for A (prior utilization, not an index)
            ("A_prior", _ts("2019-12-01"), _ts("2019-12-01"), "A", "emergency"),
        ],
        columns=["Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS"],
    )
    patients = pd.DataFrame(
        [
            ("A", "1950-01-01", "", "M", "white", "nonhispanic"),
            ("B", "1955-06-01", "", "F", "black", "nonhispanic"),
            ("C", "1948-03-01", "2020-01-17", "M", "white", "hispanic"),  # death = discharge+12
            ("D", "1960-09-01", "", "F", "asian", "nonhispanic"),
            ("E", "1952-01-01", "2020-03-05", "M", "white", "nonhispanic"),  # death during stay
            ("Z", "1970-01-01", "", "F", "white", "nonhispanic"),
        ],
        columns=["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE", "ETHNICITY"],
    )
    conditions = pd.DataFrame(
        [
            ("A", "44054006", "2018-01-01"),
            ("A", "84114007", "2018-06-01"),
            ("D", "59621000", "2019-01-01"),
        ],
        columns=["PATIENT", "CODE", "START"],
    )
    observations = pd.DataFrame(
        [
            ("A", "39156-5", _ts("2020-01-04"), "31.0", "kg/m2"),
            ("A", "8480-6", _ts("2020-01-04"), "145.0", "mm[Hg]"),
            ("D", "4548-4", _ts("2020-04-04"), "7.2", "%"),
        ],
        columns=["PATIENT", "CODE", "DATE", "VALUE", "UNITS"],
    )
    encounters.to_csv(tmp / "encounters.csv", index=False)
    patients.to_csv(tmp / "patients.csv", index=False)
    conditions.to_csv(tmp / "conditions.csv", index=False)
    observations.to_csv(tmp / "observations.csv", index=False)


def test_golden_durations_and_events(tmp_path: Path) -> None:
    _write_raw(tmp_path)
    df = build_cohort_from_raw(tmp_path, followup_days=FOLLOWUP).set_index("index_encounter_id")

    # A_idx: readmitted within 10 days -> event, duration 10
    assert df.loc["A_idx", "event_observed"] == 1
    assert df.loc["A_idx", "duration_days"] == 10

    # B_idx: no readmission, full follow-up -> censored at horizon
    assert df.loc["B_idx", "event_observed"] == 0
    assert df.loc["B_idx", "duration_days"] == FOLLOWUP

    # C_idx: death 12 days after discharge, no readmission -> censored at 12
    assert df.loc["C_idx", "event_observed"] == 0
    assert df.loc["C_idx", "duration_days"] == 12

    # D_idx: readmission beyond the window -> censored at horizon
    assert df.loc["D_idx", "event_observed"] == 0
    assert df.loc["D_idx", "duration_days"] == FOLLOWUP

    # E_idx: patient died during the index stay -> excluded entirely
    assert "E_idx" not in df.index


def test_golden_features(tmp_path: Path) -> None:
    _write_raw(tmp_path)
    df = build_cohort_from_raw(tmp_path, followup_days=FOLLOWUP).set_index("index_encounter_id")

    # A has diabetes (w=1) + heart failure (w=3) active before index -> score 4, 2 conditions.
    assert df.loc["A_idx", "n_conditions"] == 2
    assert df.loc["A_idx", "comorbidity_score"] == 4
    # A's latest BMI/SBP before discharge are pulled via the ASOF join.
    assert df.loc["A_idx", "bmi"] == 31.0
    assert df.loc["A_idx", "systolic_bp"] == 145.0
    # A had one prior emergency encounter within the lookback window.
    assert df.loc["A_idx", "n_prior_emergency"] == 1
    # Age at index for A (born 1950-01-01, discharged 2020-01-05) ~ 70.
    assert 69.5 < df.loc["A_idx", "age_at_index"] < 70.5
