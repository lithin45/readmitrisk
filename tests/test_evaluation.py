"""Phase 4 acceptance tests: RSF, integrated Brier, time-dependent calibration, gate."""

from __future__ import annotations

import numpy as np

from readmitrisk.evaluation.evaluate import EvalResult


def test_rsf_survival_curves_valid(rsf_model, survival_split) -> None:
    sample = survival_split.test.head(20)
    times, surv = rsf_model.survival_function(sample)
    assert surv.shape[0] == len(sample)
    assert np.all(surv >= -1e-9) and np.all(surv <= 1 + 1e-9)
    assert np.all(np.diff(surv, axis=1) <= 1e-9), "RSF survival curves must be non-increasing"


def test_rsf_predict_survival_at_times(rsf_model, survival_split) -> None:
    surv = rsf_model.predict_survival_at(survival_split.test.head(15), np.array([5.0, 15.0, 25.0]))
    assert surv.shape == (15, 3)
    assert np.all((surv >= -1e-9) & (surv <= 1 + 1e-9))


def test_both_models_evaluated(eval_result: EvalResult) -> None:
    names = {m.name for m in eval_result.metrics}
    assert {"Cox PH", "Random Survival Forest"} <= names, "Cox and RSF must both be evaluated"


def test_integrated_brier_in_valid_range(eval_result: EvalResult) -> None:
    for m in eval_result.metrics:
        # 0 = perfect, 0.25 = uninformative coin-flip; a useful model is well under 0.25.
        assert 0.0 < m.integrated_brier_score < 0.25, f"{m.name} IBS {m.integrated_brier_score}"


def test_time_dependent_auc_computed(eval_result: EvalResult) -> None:
    for m in eval_result.metrics:
        assert m.time_auc, "time-dependent AUC must be computed at each eval time"
        assert all(0.0 <= a <= 1.0 for a in m.time_auc.values())
        assert 0.5 < m.mean_time_auc <= 1.0


def test_concordance_present(eval_result: EvalResult) -> None:
    for m in eval_result.metrics:
        assert 0.5 < m.c_index <= 1.0
        assert 0.5 < m.ipcw_c_index <= 1.0


def test_calibration_curve_well_formed(eval_result: EvalResult) -> None:
    cal = eval_result.best().calibration
    assert len(cal.bin_predicted) == len(cal.bin_observed) == len(cal.bin_count)
    assert len(cal.bin_predicted) >= 3, "need several bins for a calibration curve"
    assert np.all((cal.bin_predicted >= 0) & (cal.bin_predicted <= 1))
    assert np.all((cal.bin_observed >= -1e-9) & (cal.bin_observed <= 1 + 1e-9))
    assert np.isfinite(cal.calibration_error) and cal.calibration_error >= 0


def test_calibration_observed_tracks_predicted(eval_result: EvalResult) -> None:
    """Observed risk should rise with predicted risk across bins (positive association)."""
    cal = eval_result.best().calibration
    corr = np.corrcoef(cal.bin_predicted, cal.bin_observed)[0, 1]
    assert corr > 0.5, f"calibration bins poorly ordered (corr={corr:.2f})"


def test_gate_passes_on_test_population(eval_result: EvalResult) -> None:
    assert eval_result.best().c_index >= eval_result.min_c_index
    assert eval_result.passed


def test_gate_fails_when_threshold_unreachable(eval_result: EvalResult) -> None:
    impossible = EvalResult(
        metrics=eval_result.metrics,
        eval_times=eval_result.eval_times,
        calibration_time=eval_result.calibration_time,
        min_c_index=0.999,
        config=eval_result.config,
        report=eval_result.report,
    )
    assert not impossible.passed


def test_rsf_competitive_with_cox(eval_result: EvalResult) -> None:
    """Sanity: both models should be in a similar, strong concordance band on this signal."""
    by_name = {m.name: m for m in eval_result.metrics}
    assert abs(by_name["Cox PH"].c_index - by_name["Random Survival Forest"].c_index) < 0.1
