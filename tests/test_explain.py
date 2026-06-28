"""Phase 6 tests: SHAP drivers (Cox) + permutation importance (model-agnostic)."""

from __future__ import annotations

import numpy as np

from readmitrisk.explain.shap_drivers import (
    CoxExplainer,
    PatientExplanation,
    permutation_cindex_importance,
)


def test_cox_global_importance_covers_features(survival_split, cox_model) -> None:
    exp = CoxExplainer(cox_model, survival_split.train)
    imp = exp.global_importance(survival_split.test)
    assert set(imp.index) == set(cox_model.feature_names_)
    assert (imp >= 0).all() and imp.sum() > 0


def test_cox_shap_additivity(survival_split, cox_model) -> None:
    """SHAP for a linear model is exact: base + sum(contributions) == coef*x."""
    exp = CoxExplainer(cox_model, survival_split.train)
    row = survival_split.test.iloc[[3]]
    pe = exp.explain_patient(row)
    assert isinstance(pe, PatientExplanation)
    design = exp._design(row).to_numpy()[0]
    linear = float(np.dot(exp.coef, design))
    assert abs(pe.prediction - linear) < 1e-6
    assert abs((pe.base_value + pe.contributions.sum()) - linear) < 1e-6


def test_cox_contributions_one_per_feature(survival_split, cox_model) -> None:
    exp = CoxExplainer(cox_model, survival_split.train)
    pe = exp.explain_patient(survival_split.test.iloc[[0]])
    assert set(pe.contributions.index) == set(cox_model.feature_names_)


def test_permutation_importance_runs_for_rsf(survival_split, rsf_model) -> None:
    from readmitrisk.config import load_config

    fc = load_config().features
    imp, base = permutation_cindex_importance(
        rsf_model, survival_split.test, fc, n_repeats=2, seed=0
    )
    expected = set(fc.numeric_features) | set(fc.categorical_features)
    assert set(imp.index) == expected
    assert 0.5 < base <= 1.0
    # At least one clinically-meaningful feature should carry positive importance.
    assert imp.max() > 0
