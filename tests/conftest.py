"""Shared pytest fixtures.

Tests run against a small, fast, deterministically-generated population in a temp dir
(never the user's real ``data/``). Fixtures are session-scoped so generation/cohort
build happen once across the whole suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from readmitrisk.generate import fallback

# Small but large enough for stable survival metrics in tests.
TEST_POPULATION = 600
TEST_SEED = 1234
FOLLOWUP_DAYS = 30


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A deterministically generated Synthea-schema dataset in a temp dir."""
    d = tmp_path_factory.mktemp("raw")
    fallback.generate(
        out_dir=d,
        population=TEST_POPULATION,
        seed=TEST_SEED,
        reference_date="2020-01-01",
        followup_days=FOLLOWUP_DAYS,
    )
    return d


@pytest.fixture(scope="session")
def cohort_df(raw_dir: Path):
    """The built time-to-event cohort for the test population (Phase 2+)."""
    from readmitrisk.cohort import build_cohort_from_raw

    return build_cohort_from_raw(raw_dir, followup_days=FOLLOWUP_DAYS)


@pytest.fixture(scope="session")
def survival_split(cohort_df):
    """Group-aware train/test split of the test cohort (Phase 3+)."""
    from readmitrisk.config import load_config
    from readmitrisk.models.dataset import split_cohort

    return split_cohort(cohort_df, load_config())


@pytest.fixture(scope="session")
def cox_model(survival_split):
    """A Cox PH model fit on the training split (Phase 3+)."""
    from readmitrisk.config import load_config
    from readmitrisk.models.cox import CoxModel

    cfg = load_config()
    model = CoxModel(cfg.features, penalizer=0.05)
    model.fit(survival_split.train)
    return model


@pytest.fixture(scope="session")
def rsf_model(survival_split):
    """A small Random Survival Forest fit on the training split (Phase 4+)."""
    from readmitrisk.config import load_config
    from readmitrisk.models.rsf import RSFModel

    cfg = load_config()
    model = RSFModel(cfg.features, {**cfg.evaluation.rsf, "n_estimators": 80})
    model.fit(survival_split.train)
    return model


@pytest.fixture(scope="session")
def eval_result(survival_split, cox_model, rsf_model):
    """Full evaluation (Cox + RSF, metrics + calibration) on the test population."""
    from readmitrisk.config import load_config
    from readmitrisk.evaluation.evaluate import evaluate_models
    from readmitrisk.evaluation.metrics import harrell_concordance
    from readmitrisk.models.pipeline import ModelResult, TrainReport

    cfg = load_config()
    fc = cfg.features
    results = []
    for key, model in [("cox", cox_model), ("rsf", rsf_model)]:
        risk = model.predict_risk(survival_split.test)
        ci = harrell_concordance(
            survival_split.test[fc.event_col], survival_split.test[fc.duration_col], risk
        )
        results.append(ModelResult(key=key, name=model.name, c_index=ci, model=model))
    report = TrainReport(
        results=results,
        split=survival_split,
        models={"cox": cox_model, "rsf": rsf_model},
        config=cfg,
    )
    return evaluate_models(report, cfg)
