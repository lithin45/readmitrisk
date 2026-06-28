"""Common interface for the survival models so evaluation, fairness, and the UI can
treat Cox PH and Random Survival Forest interchangeably.

Risk convention: ``predict_risk`` returns a score where **higher = higher risk = shorter
expected time to readmission** (so concordance treats higher risk + earlier event as a
concordant pair).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SurvivalModel(ABC):
    """Abstract survival model with a uniform predict surface."""

    name: str = "survival-model"
    feature_names_: list[str]

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> SurvivalModel: ...

    @abstractmethod
    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Higher = higher readmission risk."""

    @abstractmethod
    def predict_survival_at(self, df: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """Survival probability S(t) for each row at each requested time.

        Returns an array of shape ``(n_rows, len(times))`` with values in [0, 1].
        """

    @abstractmethod
    def survival_function(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Full step survival curves: ``(times[T], surv[n_rows, T])``."""

    def predict_risk_at(self, df: pd.DataFrame, time: float) -> np.ndarray:
        """Cumulative incidence (risk) of readmission by ``time``: 1 - S(time)."""
        surv = self.predict_survival_at(df, np.asarray([time], dtype=float))
        return 1.0 - surv[:, 0]
