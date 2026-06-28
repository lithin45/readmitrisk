"""Training orchestration: fit the survival models on a group-aware split and report
discrimination. Fitted models + the split are persisted so evaluation, the fairness
audit, and the Streamlit demo can reuse them without retraining.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

from ..config import Config, load_config
from ..evaluation.metrics import harrell_concordance
from ..paths import get_paths
from .base import SurvivalModel
from .cox import CoxModel
from .dataset import SurvivalSplit, split_cohort

ARTIFACT_NAME = "models.pkl"


def build_models(config: Config) -> dict[str, SurvivalModel]:
    """Construct the (unfitted) model registry from config."""
    fc = config.features
    cox_cfg = config.evaluation.cox
    models: dict[str, SurvivalModel] = {
        "cox": CoxModel(
            fc,
            penalizer=float(cox_cfg.get("penalizer", 0.05)),
            l1_ratio=float(cox_cfg.get("l1_ratio", 0.0)),
        ),
    }
    # Random Survival Forest is added in Phase 4.
    try:
        from .rsf import RSFModel

        rsf_cfg = config.evaluation.rsf
        models["rsf"] = RSFModel(fc, rsf_cfg)
    except ImportError:
        pass
    return models


@dataclass
class ModelResult:
    key: str
    name: str
    c_index: float
    model: SurvivalModel


@dataclass
class TrainReport:
    results: list[ModelResult]
    split: SurvivalSplit
    models: dict[str, SurvivalModel]
    config: Config
    metadata: dict = field(default_factory=dict)

    def best(self) -> ModelResult:
        return max(self.results, key=lambda r: r.c_index)

    def summary(self) -> str:
        tr, te = self.split.train, self.split.test
        ec = self.config.features.event_col
        lines = [
            f"Train: {len(tr):,} index encounters ({tr[ec].mean():.1%} events)",
            f"Test : {len(te):,} index encounters ({te[ec].mean():.1%} events)  [group-aware split]",
            "",
            f"{'Model':<22}{'Harrell C-index':>16}",
            f"{'-' * 38}",
        ]
        for r in sorted(self.results, key=lambda x: -x.c_index):
            lines.append(f"{r.name:<22}{r.c_index:>16.4f}")
        lines.append("")
        lines.append(f"Best model: {self.best().name} (C-index {self.best().c_index:.4f})")
        return "\n".join(lines)


def train_models(use_sample: bool = False, persist: bool = True) -> TrainReport:
    """Load the cohort, split (group-aware), fit all models, report concordance."""
    from ..cohort import load_cohort

    config = load_config()
    fc = config.features
    df = load_cohort(use_sample=use_sample)
    split = split_cohort(df, config)
    models = build_models(config)

    results: list[ModelResult] = []
    for key, model in models.items():
        model.fit(split.train)
        risk = model.predict_risk(split.test)
        ci = harrell_concordance(split.test[fc.event_col], split.test[fc.duration_col], risk)
        results.append(ModelResult(key=key, name=model.name, c_index=ci, model=model))

    report = TrainReport(
        results=results,
        split=split,
        models=models,
        config=config,
        metadata={
            "use_sample": use_sample,
            "n_train": len(split.train),
            "n_test": len(split.test),
        },
    )
    if persist and not use_sample:
        save_artifacts(report)
    return report


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_artifacts(report: TrainReport) -> None:
    paths = get_paths().ensure()
    payload = {
        "models": report.models,
        "train": report.split.train,
        "test": report.split.test,
        "metadata": report.metadata,
    }
    with (paths.artifacts / ARTIFACT_NAME).open("wb") as fh:
        pickle.dump(payload, fh)


def load_artifacts() -> dict | None:
    path = get_paths().artifacts / ARTIFACT_NAME
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)
