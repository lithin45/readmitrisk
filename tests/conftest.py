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
