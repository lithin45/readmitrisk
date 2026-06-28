"""Survival metrics — the ONLY metrics this project evaluates with.

Discrimination uses concordance (Harrell's + IPCW); overall fit uses the integrated
Brier score; time-resolved discrimination uses cumulative/dynamic AUC. Plain
classification accuracy is intentionally absent and is rejected by
:func:`assert_survival_metrics` — treating censored patients as negatives would be wrong.
"""

from __future__ import annotations

import numpy as np
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

# Metric-name vocabulary used by the eval gate to enforce the survival-only invariant.
SURVIVAL_METRIC_NAMES = frozenset(
    {"concordance_index", "ipcw_concordance_index", "integrated_brier_score", "time_dependent_auc"}
)
FORBIDDEN_METRIC_NAMES = frozenset(
    {"accuracy", "classification_accuracy", "precision", "recall", "f1", "roc_auc_binary"}
)


def assert_survival_metrics(metric_names: list[str], required: list[str]) -> None:
    """Raise if a required survival metric is missing or a forbidden metric is present."""
    present = set(metric_names)
    forbidden = present & FORBIDDEN_METRIC_NAMES
    if forbidden:
        raise ValueError(
            f"Forbidden (non-survival) metrics in evaluation: {sorted(forbidden)}. "
            "Readmission is right-censored — classification accuracy is invalid."
        )
    missing = set(required) - present
    if missing:
        raise ValueError(f"Required survival metrics missing from evaluation: {sorted(missing)}")


def harrell_concordance(events: np.ndarray, times: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index. ``risk`` higher = higher risk (shorter survival)."""
    events = np.asarray(events).astype(bool)
    times = np.asarray(times, dtype=float)
    risk = np.asarray(risk, dtype=float)
    return float(concordance_index_censored(events, times, risk)[0])


def ipcw_concordance(
    y_train: np.ndarray, y_test: np.ndarray, risk: np.ndarray, tau: float | None = None
) -> float:
    """Uno's IPCW-adjusted C-index (corrects for censoring distribution)."""
    return float(concordance_index_ipcw(y_train, y_test, np.asarray(risk, dtype=float), tau=tau)[0])


def restrict_times(y_train: np.ndarray, y_test: np.ndarray, times: list[float]) -> np.ndarray:
    """Filter evaluation times to the range valid for IPCW-based metrics.

    scikit-survival requires evaluation times to lie strictly inside the observed
    follow-up of both train and test (so the censoring distribution is estimable). We
    keep times in ``(max(min_test_time, ...), min(max_train_time, max_test_time))``.
    """
    lo = max(float(y_test["time"].min()), float(y_train["time"].min()))
    hi = min(float(y_train["time"].max()), float(y_test["time"].max()))
    valid = [float(t) for t in times if lo < t < hi]
    return np.asarray(sorted(set(valid)), dtype=float)


def time_dependent_auc(
    y_train: np.ndarray, y_test: np.ndarray, risk: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, float]:
    """Cumulative/dynamic AUC at each time plus the time-averaged AUC."""
    aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, np.asarray(risk, dtype=float), times)
    return np.asarray(aucs, dtype=float), float(mean_auc)


def integrated_brier(
    y_train: np.ndarray, y_test: np.ndarray, surv_prob: np.ndarray, times: np.ndarray
) -> float:
    """Integrated Brier score over ``times`` (lower is better; 0.25 = uninformative)."""
    return float(integrated_brier_score(y_train, y_test, surv_prob, times))


def brier_at_times(
    y_train: np.ndarray, y_test: np.ndarray, surv_prob: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    t, bs = brier_score(y_train, y_test, surv_prob, times)
    return np.asarray(t, dtype=float), np.asarray(bs, dtype=float)
