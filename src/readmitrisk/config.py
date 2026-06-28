"""Configuration loading for ReadmitRisk.

YAML config files under ``config/`` are the source of truth; a small set of knobs can
be overridden by environment variables (see ``.env.example``) so the same code runs
unchanged under Docker, CI, and local dev. Env overrides always win over YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .paths import get_paths


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw not in (None, "") else default


@dataclass(frozen=True)
class GenerationConfig:
    backend: str  # "synthea" | "fallback" | "auto"
    population: int
    seed: int
    state: str
    city: str
    reference_date: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureConfig:
    id_col: str
    duration_col: str
    event_col: str
    followup_days: int
    numeric_features: list[str]
    categorical_features: list[str]
    subgroup_attributes: list[dict[str, str]]
    observations: dict[str, dict[str, str]]
    lookback_days: int
    numeric_impute: str
    standardize_for_cox: bool

    @property
    def subgroup_columns(self) -> list[str]:
        return [s["column"] for s in self.subgroup_attributes]


@dataclass(frozen=True)
class EvalConfig:
    test_size: float
    split_seed: int
    stratify_on_event: bool
    min_c_index: float
    require_survival_metrics: list[str]
    eval_times: list[int]
    ibs_max_time: int
    calibration_bins: int
    calibration_time: int
    max_subgroup_gap: float
    min_subgroup_n: int
    subgroup_metrics: list[str]
    cox: dict[str, Any]
    rsf: dict[str, Any]


@dataclass(frozen=True)
class Config:
    generation: GenerationConfig
    features: FeatureConfig
    evaluation: EvalConfig
    synthea_raw: dict[str, Any]


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load and merge all config files, applying environment overrides."""
    cfg_dir = get_paths().config
    synthea = _load_yaml(cfg_dir / "synthea.yaml")
    features = _load_yaml(cfg_dir / "features.yaml")
    evalc = _load_yaml(cfg_dir / "eval.yaml")

    pop = synthea.get("population", {})
    generation = GenerationConfig(
        backend=_env_str("READMIT_GENERATOR", synthea.get("generator", {}).get("backend", "auto")),
        population=_env_int("READMIT_POPULATION", int(pop.get("size", 4000))),
        seed=_env_int("READMIT_SEED", int(pop.get("seed", 42))),
        state=pop.get("state", "Massachusetts"),
        city=pop.get("city", "Bedford"),
        reference_date=str(pop.get("reference_date", "2020-01-01")),
        raw=synthea,
    )

    tgt = features.get("target", {})
    prep = features.get("preprocessing", {})
    feature_cfg = FeatureConfig(
        id_col=features.get("id_col", "patient_id"),
        duration_col=tgt.get("duration_col", "duration_days"),
        event_col=tgt.get("event_col", "event_observed"),
        followup_days=_env_int("READMIT_FOLLOWUP_DAYS", int(tgt.get("followup_days", 30))),
        numeric_features=list(features.get("numeric_features", [])),
        categorical_features=list(features.get("categorical_features", [])),
        subgroup_attributes=list(features.get("subgroup_attributes", [])),
        observations=dict(features.get("observations", {})),
        lookback_days=int(prep.get("lookback_days", 365)),
        numeric_impute=str(prep.get("numeric_impute", "median")),
        standardize_for_cox=bool(prep.get("standardize_for_cox", True)),
    )

    split = evalc.get("split", {})
    gate = evalc.get("gate", {})
    metrics = evalc.get("metrics", {})
    fairness = evalc.get("fairness", {})
    models = evalc.get("models", {})
    eval_cfg = EvalConfig(
        test_size=_env_float("READMIT_TEST_SIZE", float(split.get("test_size", 0.25))),
        split_seed=int(split.get("seed", 42)),
        stratify_on_event=bool(split.get("stratify_on_event", True)),
        min_c_index=float(gate.get("min_c_index", 0.70)),
        require_survival_metrics=list(gate.get("require_survival_metrics", [])),
        eval_times=list(metrics.get("eval_times", [7, 14, 21, 30])),
        ibs_max_time=int(metrics.get("ibs_max_time", 30)),
        calibration_bins=int(metrics.get("calibration_bins", 10)),
        calibration_time=int(metrics.get("calibration_time", 30)),
        max_subgroup_gap=float(fairness.get("max_subgroup_gap", 0.05)),
        min_subgroup_n=int(fairness.get("min_subgroup_n", 50)),
        subgroup_metrics=list(fairness.get("subgroup_metrics", ["concordance_index"])),
        cox=dict(models.get("cox", {})),
        rsf=dict(models.get("rsf", {})),
    )

    return Config(
        generation=generation,
        features=feature_cfg,
        evaluation=eval_cfg,
        synthea_raw=synthea,
    )
