"""Data/logic layer for the Streamlit demo (kept separate so it is unit-testable).

Loads the trained models + cohort (from persisted artifacts, or by training on the cached
sample as a fallback so the demo always works), and exposes per-patient risk curves,
30-day risk, and SHAP drivers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..explain.shap_drivers import CoxExplainer, PatientExplanation
from ..models.base import SurvivalModel
from ..paths import get_paths


@dataclass
class DemoBundle:
    models: dict[str, SurvivalModel]
    train: pd.DataFrame
    test: pd.DataFrame
    config: Config
    cox_explainer: CoxExplainer
    source: str
    fairness: dict | None = None
    metrics: dict | None = None


def _read_json(path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_bundle(prefer_sample: bool = False) -> DemoBundle:
    """Load models + cohort for the demo.

    Uses persisted full-data artifacts when present; otherwise trains on the cached
    sample so the demo runs with zero prior setup.
    """
    from ..models.pipeline import load_artifacts, train_models

    paths = get_paths()
    art = None if prefer_sample else load_artifacts()
    if art is not None:
        models, train, test, source = (
            art["models"],
            art["train"],
            art["test"],
            "full-data artifacts",
        )
        config = load_config()
    else:
        report = train_models(use_sample=True, persist=False)
        models, train, test = report.models, report.split.train, report.split.test
        config, source = report.config, "cached sample (trained on the fly)"

    cox = models.get("cox")
    cox_explainer = CoxExplainer(cox, train) if cox is not None else None
    return DemoBundle(
        models=models,
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        config=config,
        cox_explainer=cox_explainer,
        source=source,
        fairness=_read_json(paths.reports / "fairness.json"),
        metrics=_read_json(paths.reports / "metrics.json"),
    )


def patient_label(row: pd.Series, fc) -> str:
    return (
        f"{row['index_encounter_id'][:8]} · {row['sex']}, {row['age_at_index']:.0f}y, "
        f"{int(row['n_conditions'])} cond, LOS {row['length_of_stay_days']:.0f}d"
    )


def patient_table(bundle: DemoBundle) -> pd.DataFrame:
    fc = bundle.config.features
    df = bundle.test
    labels = df.apply(lambda r: patient_label(r, fc), axis=1)
    return pd.DataFrame({"label": labels, "index_encounter_id": df["index_encounter_id"]})


def get_patient_row(bundle: DemoBundle, encounter_id: str) -> pd.DataFrame:
    return bundle.test[bundle.test["index_encounter_id"] == encounter_id]


def risk_curves(bundle: DemoBundle, encounter_id: str, max_day: int = 30) -> dict[str, np.ndarray]:
    """Survival probability over [0, max_day] for each model, plus the shared time grid."""
    row = get_patient_row(bundle, encounter_id)
    grid = np.arange(0, max_day + 1, dtype=float)
    out: dict[str, np.ndarray] = {"times": grid}
    for model in bundle.models.values():
        out[model.name] = model.predict_survival_at(row, grid)[0]
    return out


def patient_risk_summary(bundle: DemoBundle, encounter_id: str, horizon: float = 30.0) -> dict:
    row = get_patient_row(bundle, encounter_id)
    fc = bundle.config.features
    risks = {
        model.name: float(model.predict_risk_at(row, horizon)[0])
        for model in bundle.models.values()
    }
    actual = {
        "duration_days": float(row[fc.duration_col].iloc[0]),
        "event_observed": int(row[fc.event_col].iloc[0]),
    }
    return {"horizon": horizon, "predicted_risk": risks, "actual": actual}


def patient_drivers(bundle: DemoBundle, encounter_id: str) -> PatientExplanation | None:
    if bundle.cox_explainer is None:
        return None
    return bundle.cox_explainer.explain_patient(get_patient_row(bundle, encounter_id))
