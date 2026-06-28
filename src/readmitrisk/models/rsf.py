"""Random Survival Forest (scikit-survival), wrapped in the common interface.

RSF is an ensemble of survival trees split on the log-rank statistic. It captures
non-linearities and interactions the linear Cox model cannot, and is scale-free (no
standardization), so it is a strong non-parametric comparison point for Cox PH.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest

from ..config import FeatureConfig
from .base import SurvivalModel
from .preprocess import FeaturePreprocessor, to_structured_y


def _step_interpolate(event_times: np.ndarray, surv: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Right-continuous step interpolation of survival curves onto ``query`` times.

    ``surv`` is ``(n_rows, len(event_times))``; returns ``(n_rows, len(query))``. For a
    query time before the first event time the survival probability is 1.0.
    """
    idx = np.searchsorted(event_times, query, side="right") - 1
    out = np.ones((surv.shape[0], len(query)), dtype=float)
    valid = idx >= 0
    if valid.any():
        out[:, valid] = surv[:, idx[valid]]
    return np.clip(out, 0.0, 1.0)


class RSFModel(SurvivalModel):
    name = "Random Survival Forest"

    def __init__(self, feature_cfg: FeatureConfig, rsf_cfg: dict):
        self.feature_cfg = feature_cfg
        self.pre = FeaturePreprocessor(feature_cfg, standardize=False)
        self.rsf = RandomSurvivalForest(
            n_estimators=int(rsf_cfg.get("n_estimators", 200)),
            min_samples_leaf=int(rsf_cfg.get("min_samples_leaf", 15)),
            max_features=rsf_cfg.get("max_features", "sqrt"),
            n_jobs=-1,
            random_state=int(rsf_cfg.get("seed", 42)),
        )
        self.feature_names_: list[str] = []
        self.event_times_: np.ndarray = np.array([])

    def fit(self, train_df: pd.DataFrame) -> RSFModel:
        self.pre.fit(train_df)
        X = self.pre.transform(train_df).to_numpy()
        y = to_structured_y(
            train_df[self.feature_cfg.event_col], train_df[self.feature_cfg.duration_col]
        )
        self.rsf.fit(X, y)
        self.feature_names_ = list(self.pre.feature_names_)
        # scikit-survival exposes the survival-function time grid as ``unique_times_``.
        self.event_times_ = np.asarray(self.rsf.unique_times_, dtype=float)
        return self

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        X = self.pre.transform(df).to_numpy()
        # RSF risk score = expected number of events (cumulative hazard); higher = riskier.
        return np.asarray(self.rsf.predict(X), dtype=float)

    def predict_survival_at(self, df: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        X = self.pre.transform(df).to_numpy()
        surv = self.rsf.predict_survival_function(X, return_array=True)
        return _step_interpolate(self.event_times_, surv, np.asarray(times, dtype=float))

    def survival_function(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self.pre.transform(df).to_numpy()
        surv = np.asarray(self.rsf.predict_survival_function(X, return_array=True), dtype=float)
        return self.event_times_, np.clip(surv, 0.0, 1.0)
