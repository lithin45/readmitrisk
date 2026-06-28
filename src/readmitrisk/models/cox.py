"""Cox Proportional Hazards model (lifelines), wrapped in the common interface.

Cox PH is the survival-analysis workhorse: a semi-parametric model of the hazard with
interpretable hazard ratios per feature, proper handling of right-censoring in the
partial-likelihood fit, and individual survival curves via the Breslow baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from ..config import FeatureConfig
from .base import SurvivalModel
from .preprocess import FeaturePreprocessor


class CoxModel(SurvivalModel):
    name = "Cox PH"

    def __init__(self, feature_cfg: FeatureConfig, penalizer: float = 0.05, l1_ratio: float = 0.0):
        self.feature_cfg = feature_cfg
        self.pre = FeaturePreprocessor(feature_cfg, standardize=feature_cfg.standardize_for_cox)
        self.cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self.feature_names_: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> CoxModel:
        self.pre.fit(train_df)
        X = self.pre.transform(train_df)
        data = X.copy()
        data[self.feature_cfg.duration_col] = train_df[self.feature_cfg.duration_col].to_numpy()
        data[self.feature_cfg.event_col] = train_df[self.feature_cfg.event_col].to_numpy()
        self.cph.fit(
            data,
            duration_col=self.feature_cfg.duration_col,
            event_col=self.feature_cfg.event_col,
        )
        self.feature_names_ = list(X.columns)
        return self

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        X = self.pre.transform(df)
        # Partial hazard: proportional to the hazard, higher = higher risk.
        return np.asarray(self.cph.predict_partial_hazard(X)).ravel()

    def predict_survival_at(self, df: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        X = self.pre.transform(df)
        times = np.asarray(times, dtype=float)
        sf = self.cph.predict_survival_function(X, times=times)
        # lifelines returns a DataFrame indexed by time, one column per row.
        return np.clip(sf.to_numpy().T, 0.0, 1.0)

    def survival_function(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self.pre.transform(df)
        sf = self.cph.predict_survival_function(X)
        return np.asarray(sf.index, dtype=float), np.clip(sf.to_numpy().T, 0.0, 1.0)

    # -- interpretability -------------------------------------------------------
    def hazard_ratios(self) -> pd.DataFrame:
        """Hazard ratios with 95% CIs and p-values, sorted by effect size."""
        s = self.cph.summary
        out = s[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
        out.columns = ["coef", "hazard_ratio", "hr_lower95", "hr_upper95", "p_value"]
        return out.reindex(out["hazard_ratio"].sub(1).abs().sort_values(ascending=False).index)
