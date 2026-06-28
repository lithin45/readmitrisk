"""Censoring / metric-correctness INVARIANTS (a hard gate for this project).

These tests enforce that:
  1. evaluation uses survival metrics and never plain classification accuracy;
  2. concordance is computed only over *comparable* (censoring-aware) pairs — proven by
     matching an independent brute-force Harrell concordance, and by showing a naive
     "treat censored as negative" accuracy gives a different, wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from readmitrisk.config import load_config
from readmitrisk.evaluation.metrics import (
    FORBIDDEN_METRIC_NAMES,
    SURVIVAL_METRIC_NAMES,
    assert_survival_metrics,
    harrell_concordance,
)


def _brute_force_concordance(events, times, risk) -> float:
    """Reference Harrell C-index over comparable pairs (assumes distinct times).

    A pair contributes iff the subject with the *smaller* time had an event (otherwise the
    later subject's outcome is unknown — that is exactly what censoring means).
    """
    events = np.asarray(events).astype(int)
    times = np.asarray(times, dtype=float)
    risk = np.asarray(risk, dtype=float)
    num = den = 0.0
    n = len(times)
    for i in range(n):
        for j in range(n):
            if i == j or not (times[i] < times[j]):
                continue
            if events[i] != 1:  # earlier subject censored -> not comparable
                continue
            den += 1
            if risk[i] > risk[j]:
                num += 1.0
            elif risk[i] == risk[j]:
                num += 0.5
    return num / den


def test_concordance_matches_censoring_aware_bruteforce() -> None:
    rng = np.random.default_rng(0)
    n = 120
    # Distinct times so tie-handling can't explain any discrepancy.
    times = rng.permutation(np.arange(1, n + 1)).astype(float)
    events = (rng.random(n) < 0.4).astype(int)
    risk = rng.normal(size=n)
    ours = harrell_concordance(events, times, risk)
    ref = _brute_force_concordance(events, times, risk)
    assert ours == pytest.approx(ref, abs=1e-9)


def test_concordance_ignores_incomparable_censored_pairs() -> None:
    """A censored subject with a short follow-up is NOT comparable to later events, so
    mis-ranking it must not change the concordance — unlike a naive accuracy."""
    # X censored very early (high risk but unknown outcome); Y and Z are real events.
    events = [0, 1, 1]
    times = [2.0, 5.0, 9.0]
    risk = [99.0, 2.0, 1.0]  # X has absurdly high risk but is censored at t=2
    # Only comparable pair: (Y earlier event at 5) vs (Z at 9) -> concordant if r_Y > r_Z.
    assert harrell_concordance(events, times, risk) == pytest.approx(1.0)
    # The early-censored X never enters a comparable pair; its extreme risk is irrelevant.


def test_naive_accuracy_would_disagree_with_concordance() -> None:
    """Demonstrate WHY accuracy is invalid: treating censored as 'not readmitted' gives a
    different number than the censoring-aware concordance."""
    events = np.array([1, 0, 1, 0, 1])
    times = np.array([3.0, 4.0, 7.0, 8.0, 11.0])
    risk = np.array([0.9, 0.8, 0.6, 0.2, 0.5])
    c = harrell_concordance(events, times, risk)
    # Naive binary accuracy: label = event, predict positive if risk >= 0.5.
    pred = (risk >= 0.5).astype(int)
    naive_acc = float((pred == events).mean())
    assert c != pytest.approx(naive_acc), "concordance must not collapse to naive accuracy"


def test_assert_survival_metrics_rejects_accuracy() -> None:
    with pytest.raises(ValueError, match="Forbidden"):
        assert_survival_metrics(["concordance_index", "accuracy"], ["concordance_index"])


def test_assert_survival_metrics_requires_present() -> None:
    with pytest.raises(ValueError, match="missing"):
        assert_survival_metrics(
            ["concordance_index"], ["concordance_index", "integrated_brier_score"]
        )
    # The happy path does not raise.
    assert_survival_metrics(
        ["concordance_index", "integrated_brier_score", "time_dependent_auc"],
        ["concordance_index", "integrated_brier_score"],
    )


def test_eval_config_requires_survival_metrics_not_accuracy() -> None:
    cfg = load_config()
    required = set(cfg.evaluation.require_survival_metrics)
    assert required, "eval gate must require at least one survival metric"
    assert required & SURVIVAL_METRIC_NAMES, "required metrics must be survival metrics"
    assert not (required & FORBIDDEN_METRIC_NAMES), "eval must not require classification metrics"
