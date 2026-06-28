"""Phase 3 acceptance tests: Cox PH fits, produces valid survival curves, C-index works."""

from __future__ import annotations

import numpy as np

from readmitrisk.evaluation.metrics import harrell_concordance


def test_cox_fits_and_has_features(cox_model) -> None:
    assert cox_model.feature_names_, "Cox model has no fitted features"
    # The protected/subgroup columns must NOT be among model inputs.
    forbidden = {"race", "ethnicity", "age_band"}
    assert not (set(cox_model.feature_names_) & forbidden)


def test_risk_prediction_is_finite_positive(cox_model, survival_split) -> None:
    risk = cox_model.predict_risk(survival_split.test)
    assert risk.shape == (len(survival_split.test),)
    assert np.all(np.isfinite(risk))
    assert np.all(risk > 0), "partial hazard must be strictly positive"


def test_survival_curves_monotonic_non_increasing(cox_model, survival_split) -> None:
    sample = survival_split.test.head(25)
    times, surv = cox_model.survival_function(sample)
    assert surv.shape[0] == len(sample)
    assert np.all(surv >= -1e-9) and np.all(surv <= 1 + 1e-9), "survival probs must be in [0,1]"
    # Each row must be non-increasing in time.
    diffs = np.diff(surv, axis=1)
    assert np.all(diffs <= 1e-9), "survival curves must be monotonic non-increasing"


def test_predict_survival_at_times(cox_model, survival_split) -> None:
    times = np.array([5.0, 10.0, 20.0, 30.0])
    surv = cox_model.predict_survival_at(survival_split.test.head(40), times)
    assert surv.shape == (40, 4)
    assert np.all(surv >= -1e-9) and np.all(surv <= 1 + 1e-9)
    assert np.all(np.diff(surv, axis=1) <= 1e-9)


def test_cumulative_risk_is_complement_of_survival(cox_model, survival_split) -> None:
    risk30 = cox_model.predict_risk_at(survival_split.test.head(10), 30.0)
    surv30 = cox_model.predict_survival_at(survival_split.test.head(10), np.array([30.0]))[:, 0]
    assert np.allclose(risk30, 1.0 - surv30)
    assert np.all((risk30 >= 0) & (risk30 <= 1))


def test_cindex_computed_and_better_than_random(cox_model, survival_split) -> None:
    fc = survival_split.feature_cfg
    risk = cox_model.predict_risk(survival_split.test)
    ci = harrell_concordance(
        survival_split.test[fc.event_col], survival_split.test[fc.duration_col], risk
    )
    assert 0.5 < ci <= 1.0
    # On this synthetic signal the model should be clearly better than chance.
    assert ci > 0.65, f"C-index unexpectedly low: {ci:.3f}"


def test_higher_risk_means_shorter_survival(cox_model, survival_split) -> None:
    """Sanity on the risk convention: the highest-risk patients have lower S(30)."""
    test = survival_split.test
    risk = cox_model.predict_risk(test)
    order = np.argsort(risk)
    low, high = test.iloc[order[:50]], test.iloc[order[-50:]]
    s_low = cox_model.predict_survival_at(low, np.array([30.0]))[:, 0].mean()
    s_high = cox_model.predict_survival_at(high, np.array([30.0]))[:, 0].mean()
    assert s_high < s_low, "higher predicted risk must correspond to lower survival"
