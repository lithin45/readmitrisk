"""Train/test splitting for the survival cohort.

The split is **group-aware** (by ``patient_id``) so the same patient never appears in
both train and test, without this, multiple index encounters from one patient leak risk
across the split and inflate the C-index. It is also stratified on the event indicator so
the readmission rate is preserved across folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from ..config import Config, FeatureConfig
from .preprocess import to_structured_y


@dataclass
class SurvivalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_cfg: FeatureConfig

    @property
    def y_train(self) -> np.ndarray:
        return to_structured_y(
            self.train[self.feature_cfg.event_col], self.train[self.feature_cfg.duration_col]
        )

    @property
    def y_test(self) -> np.ndarray:
        return to_structured_y(
            self.test[self.feature_cfg.event_col], self.test[self.feature_cfg.duration_col]
        )


def split_cohort(df: pd.DataFrame, config: Config) -> SurvivalSplit:
    """Return a group-aware (optionally event-stratified) train/test split."""
    fc = config.features
    ec = config.evaluation
    id_col = fc.id_col
    event_col = fc.event_col
    groups = df[id_col].to_numpy()

    if ec.stratify_on_event:
        n_splits = max(2, round(1.0 / ec.test_size))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=ec.split_seed)
        train_idx, test_idx = next(sgkf.split(df, df[event_col].to_numpy(), groups))
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=ec.test_size, random_state=ec.split_seed)
        train_idx, test_idx = next(gss.split(df, groups=groups))

    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    return SurvivalSplit(train=train, test=test, feature_cfg=fc)
