"""Phase 2 acceptance tests: the time-to-event cohort is well-formed and censoring is sane."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from readmitrisk.cohort import SUBGROUP_COLUMNS, build_cohort_from_raw

FOLLOWUP = 30

NUMERIC_FEATURES = [
    "age_at_index",
    "length_of_stay_days",
    "n_conditions",
    "comorbidity_score",
    "n_prior_encounters",
    "n_prior_inpatient",
    "n_prior_emergency",
    "bmi",
    "systolic_bp",
    "hba1c",
    "glucose",
]


def test_required_columns_present(cohort_df: pd.DataFrame) -> None:
    required = {"patient_id", "index_encounter_id", "duration_days", "event_observed"}
    required |= set(NUMERIC_FEATURES) | set(SUBGROUP_COLUMNS)
    assert required.issubset(cohort_df.columns), required - set(cohort_df.columns)


def test_one_row_per_index_encounter(cohort_df: pd.DataFrame) -> None:
    assert cohort_df["index_encounter_id"].is_unique
    # Multi-index cohort: at least as many index rows as patients.
    assert len(cohort_df) >= cohort_df["patient_id"].nunique()


def test_durations_positive_and_within_horizon(cohort_df: pd.DataFrame) -> None:
    dur = cohort_df["duration_days"]
    assert (dur > 0).all(), "durations must be strictly positive (degenerate rows dropped)"
    assert (dur <= FOLLOWUP + 1e-6).all(), "durations cannot exceed the follow-up horizon"
    assert dur.notna().all()


def test_event_is_binary(cohort_df: pd.DataFrame) -> None:
    assert set(cohort_df["event_observed"].unique()).issubset({0, 1})
    assert cohort_df["event_observed"].notna().all()


def test_event_implies_readmission_within_horizon(cohort_df: pd.DataFrame) -> None:
    events = cohort_df[cohort_df["event_observed"] == 1]
    assert (events["duration_days"] <= FOLLOWUP + 1e-6).all()
    # Censored rows have non-negative durations too.
    censored = cohort_df[cohort_df["event_observed"] == 0]
    assert (censored["duration_days"] > 0).all()


def test_meaningful_fraction_censored(cohort_df: pd.DataFrame) -> None:
    rate = cohort_df["event_observed"].mean()
    # A meaningful fraction must be censored, and there must be enough events to model.
    assert 0.05 < rate < 0.6, f"event rate {rate:.3f} outside sane range"
    censored_frac = 1 - rate
    assert censored_frac > 0.4, f"censored fraction {censored_frac:.3f} too low"


def test_subgroup_attributes_present_and_nonnull(cohort_df: pd.DataFrame) -> None:
    for col in SUBGROUP_COLUMNS:
        assert col in cohort_df.columns
        assert cohort_df[col].notna().all(), f"{col} has nulls"
        assert cohort_df[col].nunique() >= 2, f"{col} has no variation"


def test_age_band_consistent_with_age(cohort_df: pd.DataFrame) -> None:
    def band(age: float) -> str:
        if age < 40:
            return "<40"
        if age < 65:
            return "40-64"
        if age < 80:
            return "65-79"
        return "80+"

    derived = cohort_df["age_at_index"].map(band)
    assert (derived == cohort_df["age_band"]).all()


def test_feature_ranges_are_plausible(cohort_df: pd.DataFrame) -> None:
    assert (cohort_df["age_at_index"] > 0).all()
    assert (cohort_df["age_at_index"] < 110).all()
    assert (cohort_df["length_of_stay_days"] >= 0).all()
    for col in ["n_conditions", "n_prior_encounters", "n_prior_inpatient", "n_prior_emergency"]:
        assert (cohort_df[col] >= 0).all()
    # Prior inpatient/emergency counts cannot exceed total prior encounters.
    assert (cohort_df["n_prior_inpatient"] <= cohort_df["n_prior_encounters"]).all()
    assert (cohort_df["n_prior_emergency"] <= cohort_df["n_prior_encounters"]).all()


def test_cohort_build_is_deterministic(raw_dir: Path) -> None:
    a = build_cohort_from_raw(raw_dir, followup_days=FOLLOWUP)
    b = build_cohort_from_raw(raw_dir, followup_days=FOLLOWUP)
    pd.testing.assert_frame_equal(a, b)


def test_labs_present_for_most_rows(cohort_df: pd.DataFrame) -> None:
    # ~15% missingness by construction; the vast majority should have a value.
    for col in ["bmi", "systolic_bp", "hba1c", "glucose"]:
        frac_present = cohort_df[col].notna().mean()
        assert frac_present > 0.6, f"{col} present in only {frac_present:.2f} of rows"
