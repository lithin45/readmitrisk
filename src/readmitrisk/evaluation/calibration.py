"""Time-dependent calibration for survival models.

A model is *calibrated* at horizon ``t`` if, among patients it assigns ~p% readmission
risk by ``t``, about p% are actually readmitted by ``t``. The catch: some patients are
right-censored before ``t``, so we cannot just count events. Within each predicted-risk
bin we estimate the **observed** risk with a Kaplan-Meier curve (which is censoring-aware)
evaluated at ``t``. This is the correct way to build a survival calibration curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


@dataclass
class CalibrationCurve:
    time: float
    bin_predicted: np.ndarray  # mean predicted risk per bin
    bin_observed: np.ndarray  # KM observed risk per bin
    bin_lower: np.ndarray  # KM 95% CI lower (observed risk)
    bin_upper: np.ndarray  # KM 95% CI upper (observed risk)
    bin_count: np.ndarray  # patients per bin
    calibration_error: float  # sample-weighted mean |predicted - observed| (ECE-like)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "predicted_risk": self.bin_predicted,
                "observed_risk": self.bin_observed,
                "observed_lower": self.bin_lower,
                "observed_upper": self.bin_upper,
                "n": self.bin_count,
            }
        )


def time_dependent_calibration(
    durations: np.ndarray,
    events: np.ndarray,
    predicted_risk: np.ndarray,
    time: float,
    n_bins: int = 10,
) -> CalibrationCurve:
    """Build a KM-based calibration curve at ``time``.

    ``predicted_risk`` is the model's predicted cumulative incidence by ``time`` (i.e.
    ``1 - S(time)``). Patients are grouped into quantile bins of predicted risk; observed
    risk per bin is ``1 - KM(time)``.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events).astype(int)
    predicted_risk = np.asarray(predicted_risk, dtype=float)

    df = pd.DataFrame({"dur": durations, "evt": events, "risk": predicted_risk})
    # Quantile bins; fall back to fewer bins if there are many tied risk values.
    try:
        df["bin"] = pd.qcut(df["risk"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["risk"], bins=min(n_bins, df["risk"].nunique()), labels=False)

    preds, obs, lo, hi, counts = [], [], [], [], []
    for _, grp in df.groupby("bin"):
        kmf = KaplanMeierFitter()
        kmf.fit(grp["dur"].to_numpy(), grp["evt"].to_numpy())
        surv_t = float(kmf.predict(time))
        observed = 1.0 - surv_t
        # KM confidence interval -> observed-risk CI (1 - survival CI, bounds swap).
        ci = kmf.confidence_interval_
        col_lower = [c for c in ci.columns if "lower" in c][0]
        col_upper = [c for c in ci.columns if "upper" in c][0]
        idx = ci.index[ci.index <= time]
        if len(idx) > 0:
            at = idx[-1]
            obs_lower = 1.0 - float(ci.loc[at, col_upper])
            obs_upper = 1.0 - float(ci.loc[at, col_lower])
        else:
            obs_lower = obs_upper = observed
        preds.append(float(grp["risk"].mean()))
        obs.append(observed)
        lo.append(obs_lower)
        hi.append(obs_upper)
        counts.append(int(len(grp)))

    preds = np.asarray(preds)
    obs = np.asarray(obs)
    counts = np.asarray(counts)
    cal_err = (
        float(np.sum(counts * np.abs(preds - obs)) / counts.sum()) if counts.sum() else float("nan")
    )
    return CalibrationCurve(
        time=float(time),
        bin_predicted=preds,
        bin_observed=obs,
        bin_lower=np.asarray(lo),
        bin_upper=np.asarray(hi),
        bin_count=counts,
        calibration_error=cal_err,
    )
