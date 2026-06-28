"""Phase 6 tests: the Streamlit demo's data/logic layer (no Streamlit runtime needed)."""

from __future__ import annotations

import numpy as np
import pytest

from readmitrisk.ui import backend


@pytest.fixture(scope="module")
def bundle():
    # Train on the cached sample so the demo backend is exercised end to end.
    return backend.load_bundle(prefer_sample=True)


def test_bundle_has_models_and_cohort(bundle) -> None:
    assert "cox" in bundle.models
    assert len(bundle.test) > 0
    assert bundle.cox_explainer is not None


def test_patient_table_aligned_with_test(bundle) -> None:
    table = backend.patient_table(bundle)
    assert len(table) == len(bundle.test)
    assert table["index_encounter_id"].is_unique


def test_risk_curves_monotone_in_unit_interval(bundle) -> None:
    enc = bundle.test["index_encounter_id"].iloc[0]
    curves = backend.risk_curves(bundle, enc, max_day=30)
    assert curves["times"][0] == 0 and curves["times"][-1] == 30
    for name, surv in curves.items():
        if name == "times":
            continue
        assert len(surv) == len(curves["times"])
        assert np.all((surv >= -1e-9) & (surv <= 1 + 1e-9))
        assert np.all(np.diff(surv) <= 1e-9), f"{name} survival curve not non-increasing"


def test_patient_risk_summary(bundle) -> None:
    enc = bundle.test["index_encounter_id"].iloc[0]
    summary = backend.patient_risk_summary(bundle, enc, horizon=30.0)
    assert summary["horizon"] == 30.0
    for risk in summary["predicted_risk"].values():
        assert 0.0 <= risk <= 1.0
    assert summary["actual"]["event_observed"] in (0, 1)


def test_patient_drivers_present(bundle) -> None:
    enc = bundle.test["index_encounter_id"].iloc[0]
    expl = backend.patient_drivers(bundle, enc)
    assert expl is not None
    assert len(expl.contributions) > 0
