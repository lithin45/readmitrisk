"""Fairness audit — per-subgroup discrimination + calibration, with gap flagging.

A model can be strongly discriminating overall yet perform unevenly across demographic
subgroups. For each protected attribute (sex, age band, race, ethnicity) we compute, on
the held-out test set, the within-subgroup Harrell C-index and the within-subgroup
calibration error at the horizon, then report the *gap* (best minus worst) and flag any
gap exceeding the configured threshold. Small subgroups are reported but flagged as
low-confidence, because subgroup metrics on tiny samples are noisy — a caveat, not a pass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..evaluation.calibration import time_dependent_calibration
from ..evaluation.metrics import harrell_concordance
from ..models.base import SurvivalModel
from ..paths import get_paths

# The standard error of a C-index scales like ~0.5/sqrt(n_events), so a subgroup needs a
# minimum number of *events* (not just rows) before its concordance is worth comparing.
MIN_EVENTS_FOR_CINDEX = 10


@dataclass
class SubgroupMetric:
    attribute: str
    level: str
    n: int
    n_events: int
    c_index: float | None
    calibration_error: float | None
    low_confidence: bool


@dataclass
class AttributeAudit:
    attribute: str
    subgroups: list[SubgroupMetric]
    c_index_gap: float | None
    calibration_gap: float | None
    worst_c_level: str | None
    flagged: bool


@dataclass
class FairnessReport:
    model_name: str
    overall_c_index: float
    threshold: float
    min_subgroup_n: int
    calibration_time: float
    attributes: list[AttributeAudit] = field(default_factory=list)

    @property
    def flagged_attributes(self) -> list[str]:
        return [a.attribute for a in self.attributes if a.flagged]


def _safe_cindex(events: np.ndarray, durations: np.ndarray, risk: np.ndarray) -> float | None:
    if events.sum() < 1:
        return None
    try:
        return harrell_concordance(events, durations, risk)
    except (ValueError, ZeroDivisionError):
        return None


def _subgroup_calibration_error(
    durations: np.ndarray, events: np.ndarray, pred_risk: np.ndarray, time: float, n: int
) -> float | None:
    if n < 20 or events.sum() < 3:
        return None  # too few to estimate a calibration curve meaningfully
    n_bins = max(3, min(5, n // 20))
    try:
        cal = time_dependent_calibration(durations, events, pred_risk, time, n_bins=n_bins)
        return cal.calibration_error
    except (ValueError, ZeroDivisionError):
        return None


def audit_model(
    model: SurvivalModel,
    test: pd.DataFrame,
    config: Config,
    model_name: str | None = None,
) -> FairnessReport:
    """Run the subgroup audit for one fitted model on the test set."""
    fc = config.features
    ec = config.evaluation
    cal_time = float(ec.calibration_time)

    durations_all = test[fc.duration_col].to_numpy()
    events_all = test[fc.event_col].to_numpy()
    risk_all = model.predict_risk(test)
    pred_cal_all = 1.0 - model.predict_survival_at(test, np.array([cal_time]))[:, 0]
    overall_c = harrell_concordance(events_all, durations_all, risk_all)

    attributes: list[AttributeAudit] = []
    for spec in fc.subgroup_attributes:
        col = spec["column"]
        if col not in test.columns:
            continue
        subgroups: list[SubgroupMetric] = []
        for level, idx in test.groupby(col).groups.items():
            mask = test.index.isin(idx)
            n = int(mask.sum())
            ev = events_all[mask]
            n_events = int(ev.sum())
            sub = SubgroupMetric(
                attribute=spec["name"],
                level=str(level),
                n=n,
                n_events=n_events,
                c_index=_safe_cindex(ev, durations_all[mask], risk_all[mask]),
                calibration_error=_subgroup_calibration_error(
                    durations_all[mask], ev, pred_cal_all[mask], cal_time, n
                ),
                low_confidence=(n < ec.min_subgroup_n) or (n_events < MIN_EVENTS_FOR_CINDEX),
            )
            subgroups.append(sub)

        # Gaps are computed only over sufficiently-large subgroups with valid metrics.
        reliable = [s for s in subgroups if not s.low_confidence]
        c_vals = [(s.level, s.c_index) for s in reliable if s.c_index is not None]
        cal_vals = [s.calibration_error for s in reliable if s.calibration_error is not None]

        c_gap = (
            (max(v for _, v in c_vals) - min(v for _, v in c_vals)) if len(c_vals) >= 2 else None
        )
        cal_gap = (max(cal_vals) - min(cal_vals)) if len(cal_vals) >= 2 else None
        worst_c_level = min(c_vals, key=lambda kv: kv[1])[0] if c_vals else None
        flagged = bool(
            (c_gap is not None and c_gap > ec.max_subgroup_gap)
            or (cal_gap is not None and cal_gap > ec.max_subgroup_gap)
        )
        attributes.append(
            AttributeAudit(
                attribute=spec["name"],
                subgroups=subgroups,
                c_index_gap=c_gap,
                calibration_gap=cal_gap,
                worst_c_level=worst_c_level,
                flagged=flagged,
            )
        )

    return FairnessReport(
        model_name=model_name or model.name,
        overall_c_index=overall_c,
        threshold=ec.max_subgroup_gap,
        min_subgroup_n=ec.min_subgroup_n,
        calibration_time=cal_time,
        attributes=attributes,
    )


def render_report(report: FairnessReport) -> str:
    lines = [
        f"Model audited: {report.model_name}  |  overall C-index = {report.overall_c_index:.4f}",
        f"Flag threshold: subgroup gap > {report.threshold:.2f}  |  "
        f"low-confidence if n < {report.min_subgroup_n}",
        "",
    ]
    for attr in report.attributes:
        cg = "n/a" if attr.c_index_gap is None else f"{attr.c_index_gap:.3f}"
        kg = "n/a" if attr.calibration_gap is None else f"{attr.calibration_gap:.3f}"
        flag = "  ⚠️ FLAGGED" if attr.flagged else ""
        lines.append(f"[{attr.attribute}]  C-index gap={cg}  calib gap={kg}{flag}")
        lines.append(f"  {'level':<14}{'n':>6}{'events':>8}{'C-index':>10}{'calib err':>11}  note")
        for s in attr.subgroups:
            ci = "  —  " if s.c_index is None else f"{s.c_index:.3f}"
            ce = "  —  " if s.calibration_error is None else f"{s.calibration_error:.3f}"
            note = "low-N/events (caveat)" if s.low_confidence else ""
            lines.append(f"  {s.level:<14}{s.n:>6}{s.n_events:>8}{ci:>10}{ce:>11}  {note}")
        lines.append("")
    flagged = report.flagged_attributes
    if flagged:
        lines.append(f"⚠️  Flagged attributes (gap > {report.threshold:.2f}): {', '.join(flagged)}")
    else:
        lines.append(
            f"✅  No subgroup gap exceeds {report.threshold:.2f} (among reliable subgroups)."
        )
    lines.append(
        f"Caveat: subgroup C-index SE ~ 0.5/sqrt(events), so subgroups with < "
        f"{MIN_EVENTS_FOR_CINDEX} events or n < {report.min_subgroup_n} are reported for "
        "transparency but excluded from gap flagging; even 'reliable' subgroup metrics "
        "carry wide confidence intervals on a single test split."
    )
    return "\n".join(lines)


def write_reports(report: FairnessReport) -> dict:
    from .plots import plot_subgroup_cindex

    paths = get_paths().ensure()
    rdir = paths.reports
    payload = {
        "model": report.model_name,
        "overall_c_index": report.overall_c_index,
        "threshold": report.threshold,
        "min_subgroup_n": report.min_subgroup_n,
        "flagged_attributes": report.flagged_attributes,
        "attributes": [
            {
                "attribute": a.attribute,
                "c_index_gap": a.c_index_gap,
                "calibration_gap": a.calibration_gap,
                "worst_c_level": a.worst_c_level,
                "flagged": a.flagged,
                "subgroups": [asdict(s) for s in a.subgroups],
            }
            for a in report.attributes
        ],
    }
    (rdir / "fairness.json").write_text(json.dumps(payload, indent=2))
    (rdir / "fairness_report.txt").write_text(render_report(report))
    png = plot_subgroup_cindex(report, rdir / "fairness_cindex.png")
    return {"fairness_json": rdir / "fairness.json", "fairness_png": png}


def run_fairness(use_sample: bool = False) -> FairnessReport:
    """Train (group-aware), pick the best model, run the subgroup audit, write the report."""
    from ..models.pipeline import train_models

    config = load_config()
    report = train_models(use_sample=use_sample, persist=not use_sample)
    best = report.best()
    audit = audit_model(best.model, report.split.test, config, model_name=best.name)

    print(render_report(audit))
    artifacts = write_reports(audit)
    print(f"\nFairness report + plot written to: {artifacts['fairness_json'].parent}")
    return audit
