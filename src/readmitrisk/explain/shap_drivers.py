"""Explainability: SHAP for the Cox model + model-agnostic permutation importance.

The Cox partial log-hazard is linear in the (standardized) design matrix, so SHAP values
are exact and cheap via ``shap.LinearExplainer``, they decompose each patient's risk
score into additive per-feature contributions. For the non-linear Random Survival Forest
(which SHAP's TreeExplainer does not support for scikit-survival forests) we fall back to
permutation importance measured as the drop in C-index when a feature is shuffled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from ..config import FeatureConfig
from ..evaluation.metrics import harrell_concordance
from ..models.base import SurvivalModel
from ..models.cox import CoxModel


@dataclass
class PatientExplanation:
    base_value: float
    prediction: float  # log partial hazard (sum of base + contributions)
    contributions: pd.Series  # per-feature SHAP value (signed; + = raises risk)
    feature_values: pd.Series  # the patient's (display) feature values


class CoxExplainer:
    """Exact SHAP explanations for the linear Cox model via ``shap.LinearExplainer``."""

    def __init__(self, cox: CoxModel, background_df: pd.DataFrame):
        self.cox = cox
        Xbg = cox.pre.transform(background_df)
        self.columns = list(Xbg.columns)
        coef = cox.cph.params_.reindex(self.columns).fillna(0.0).to_numpy()
        self.coef = coef
        self._bg = Xbg.to_numpy()
        self.explainer = shap.LinearExplainer((coef, 0.0), self._bg)
        self.base_value = float(self.explainer.expected_value)

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.cox.pre.transform(df)[self.columns]

    def global_importance(self, df: pd.DataFrame) -> pd.Series:
        """Mean |SHAP| per feature over ``df`` (global importance ranking)."""
        X = self._design(df).to_numpy()
        sv = np.asarray(self.explainer.shap_values(X))
        return pd.Series(np.abs(sv).mean(axis=0), index=self.columns).sort_values(ascending=False)

    def explain_patient(self, patient_df: pd.DataFrame) -> PatientExplanation:
        """SHAP contributions for a single-row DataFrame."""
        X = self._design(patient_df)
        sv = np.asarray(self.explainer.shap_values(X.to_numpy()))[0]
        contributions = pd.Series(sv, index=self.columns)
        return PatientExplanation(
            base_value=self.base_value,
            prediction=float(self.base_value + sv.sum()),
            contributions=contributions.reindex(
                contributions.abs().sort_values(ascending=False).index
            ),
            feature_values=X.iloc[0],
        )


def permutation_cindex_importance(
    model: SurvivalModel,
    df: pd.DataFrame,
    feature_cfg: FeatureConfig,
    n_repeats: int = 3,
    seed: int = 0,
) -> tuple[pd.Series, float]:
    """Model-agnostic global importance: mean drop in C-index when a feature is shuffled.

    Works for any model (used for the RSF, where SHAP TreeExplainer is unavailable).
    """
    rng = np.random.default_rng(seed)
    dur = df[feature_cfg.duration_col].to_numpy()
    evt = df[feature_cfg.event_col].to_numpy()
    base = harrell_concordance(evt, dur, model.predict_risk(df))

    cols = list(feature_cfg.numeric_features) + list(feature_cfg.categorical_features)
    importances: dict[str, float] = {}
    for col in cols:
        drops = []
        for _ in range(n_repeats):
            shuffled = df.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            c = harrell_concordance(evt, dur, model.predict_risk(shuffled))
            drops.append(base - c)
        importances[col] = float(np.mean(drops))
    series = pd.Series(importances).sort_values(ascending=False)
    return series, float(base)
