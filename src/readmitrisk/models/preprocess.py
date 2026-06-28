"""Feature preprocessing shared by the Cox and RSF models.

Turns the tidy cohort DataFrame into a numeric design matrix:
  * missing-lab indicators (``{lab}_missing``), missingness can be informative,
  * median imputation of numeric features (medians fit on TRAIN only, no leakage),
  * one-hot encoding of categoricals (first level dropped),
  * optional standardization of numeric features (on for Cox, off for the scale-free RSF).

Subgroup/protected attributes are deliberately excluded, they are used only by the
fairness audit, never as model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureConfig


class FeaturePreprocessor:
    """Fit-on-train / transform feature builder. One instance per model."""

    def __init__(self, feature_cfg: FeatureConfig, standardize: bool):
        self.numeric = list(feature_cfg.numeric_features)
        self.categorical = list(feature_cfg.categorical_features)
        # The lab columns get explicit missingness indicators.
        self.lab_cols = list(feature_cfg.observations.keys())
        self.standardize = standardize
        self.medians_: dict[str, float] = {}
        self.cat_levels_: dict[str, list] = {}
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> FeaturePreprocessor:
        self.medians_ = {c: float(df[c].median()) for c in self.numeric}
        self.cat_levels_ = {
            c: sorted(str(v) for v in df[c].dropna().unique()) for c in self.categorical
        }
        imputed = self._impute_numeric(df)
        if self.standardize:
            self.mean_ = imputed[self.numeric].mean()
            self.std_ = imputed[self.numeric].std(ddof=0).replace(0.0, 1.0)
        # Establish the canonical column order from a transform of the training frame.
        self.feature_names_ = list(self._assemble(df, imputed).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed = self._impute_numeric(df)
        out = self._assemble(df, imputed)
        # Guarantee identical columns/order to fit time (missing one-hot levels -> 0).
        return out.reindex(columns=self.feature_names_, fill_value=0.0)

    # -- internals --------------------------------------------------------------
    def _impute_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for c in self.numeric:
            cols[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.medians_[c]).astype(float)
        return pd.DataFrame(cols, index=df.index)

    def _assemble(self, df: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
        parts: dict[str, pd.Series] = {}
        # Missing indicators (computed from the RAW frame, before imputation).
        for c in self.lab_cols:
            parts[f"{c}_missing"] = df[c].isna().astype(float)
        numeric = imputed[self.numeric].copy()
        if self.standardize and self.mean_ is not None:
            numeric = (numeric - self.mean_) / self.std_
        X = pd.concat([pd.DataFrame(parts, index=df.index), numeric], axis=1)
        # One-hot categoricals (drop first level for identifiability in Cox).
        for c in self.categorical:
            levels = self.cat_levels_[c]
            for lev in levels[1:]:
                X[f"{c}_{lev}"] = (df[c].astype(str) == lev).astype(float)
        return X

    @property
    def numeric_feature_names(self) -> list[str]:
        return list(self.numeric)


def to_structured_y(events: np.ndarray, times: np.ndarray) -> np.ndarray:
    """scikit-survival structured target: ``[('event', bool), ('time', float)]``."""
    y = np.empty(len(events), dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = np.asarray(events).astype(bool)
    y["time"] = np.asarray(times, dtype=float)
    return y
