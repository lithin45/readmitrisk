"""Evaluation gate: survival metrics, Cox vs RSF comparison, calibration, and the
C-index threshold. ``make eval`` runs this and exits non-zero if the gate fails or if
survival metrics are not used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..config import Config, load_config
from ..models.pipeline import TrainReport, train_models
from ..paths import get_paths
from .calibration import CalibrationCurve, time_dependent_calibration
from .metrics import (
    assert_survival_metrics,
    harrell_concordance,
    integrated_brier,
    ipcw_concordance,
    restrict_times,
    time_dependent_auc,
)

# Metric keys produced per model, mapped to the survival-metric vocabulary used by the gate.
PRODUCED_METRIC_NAMES = [
    "concordance_index",
    "ipcw_concordance_index",
    "integrated_brier_score",
    "time_dependent_auc",
]


@dataclass
class ModelMetrics:
    key: str
    name: str
    c_index: float
    ipcw_c_index: float
    integrated_brier_score: float
    mean_time_auc: float
    time_auc: dict[float, float]
    calibration_error: float
    calibration: CalibrationCurve


@dataclass
class EvalResult:
    metrics: list[ModelMetrics]
    eval_times: np.ndarray
    calibration_time: float
    min_c_index: float
    config: Config
    report: TrainReport
    extra: dict = field(default_factory=dict)

    def best(self) -> ModelMetrics:
        return max(self.metrics, key=lambda m: m.c_index)

    @property
    def passed(self) -> bool:
        return self.best().c_index >= self.min_c_index


def evaluate_models(report: TrainReport, config: Config | None = None) -> EvalResult:
    config = config or report.config
    fc = config.features
    mc = config.evaluation
    split = report.split
    y_train, y_test = split.y_train, split.y_test
    durations = split.test[fc.duration_col].to_numpy()
    events = split.test[fc.event_col].to_numpy()

    times = restrict_times(y_train, y_test, [float(t) for t in mc.eval_times])
    cal_time = float(mc.calibration_time)
    tau = float(times.max()) if len(times) else float(mc.ibs_max_time)

    metrics: list[ModelMetrics] = []
    for res in report.results:
        model = res.model
        risk = model.predict_risk(split.test)
        c = harrell_concordance(events, durations, risk)
        ipcw = ipcw_concordance(y_train, y_test, risk, tau=tau)
        surv_prob = model.predict_survival_at(split.test, times)
        ibs = integrated_brier(y_train, y_test, surv_prob, times)
        aucs, mean_auc = time_dependent_auc(y_train, y_test, risk, times)
        pred_risk_cal = 1.0 - model.predict_survival_at(split.test, np.array([cal_time]))[:, 0]
        cal = time_dependent_calibration(
            durations, events, pred_risk_cal, cal_time, n_bins=mc.calibration_bins
        )
        metrics.append(
            ModelMetrics(
                key=res.key,
                name=res.name,
                c_index=c,
                ipcw_c_index=ipcw,
                integrated_brier_score=ibs,
                mean_time_auc=mean_auc,
                time_auc={float(t): float(a) for t, a in zip(times, aucs, strict=False)},
                calibration_error=cal.calibration_error,
                calibration=cal,
            )
        )
    return EvalResult(
        metrics=metrics,
        eval_times=times,
        calibration_time=cal_time,
        min_c_index=mc.min_c_index,
        config=config,
        report=report,
    )


def render_metrics_table(result: EvalResult) -> str:
    header = f"{'Model':<24}{'C-index':>9}{'IPCW-C':>9}{'IBS':>8}{'mean tAUC':>11}{'Calib err':>11}"
    lines = [header, "-" * len(header)]
    for m in sorted(result.metrics, key=lambda x: -x.c_index):
        lines.append(
            f"{m.name:<24}{m.c_index:>9.4f}{m.ipcw_c_index:>9.4f}"
            f"{m.integrated_brier_score:>8.4f}{m.mean_time_auc:>11.4f}{m.calibration_error:>11.4f}"
        )
    return "\n".join(lines)


def metrics_payload(result: EvalResult) -> dict:
    """Serialize an evaluation result to the dict the report JSON + UI consume."""
    best = result.best()
    split = result.report.split
    fc = result.config.features
    return {
        "passed": result.passed,
        "min_c_index": result.min_c_index,
        "best_model": best.name,
        "best_c_index": best.c_index,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "test_event_rate": float(split.test[fc.event_col].mean()),
        "eval_times": [float(t) for t in result.eval_times],
        "calibration_time": result.calibration_time,
        "models": [
            {
                "name": m.name,
                "concordance_index": m.c_index,
                "ipcw_concordance_index": m.ipcw_c_index,
                "integrated_brier_score": m.integrated_brier_score,
                "time_dependent_auc_mean": m.mean_time_auc,
                "time_dependent_auc": m.time_auc,
                "calibration_error": m.calibration_error,
                "calibration_curve": m.calibration.to_frame().to_dict(orient="list"),
            }
            for m in result.metrics
        ],
    }


def write_reports(result: EvalResult) -> dict:
    from .plots import plot_calibration, plot_cindex_comparison, plot_example_survival_curves

    paths = get_paths().ensure()
    rdir = paths.reports
    best = result.best()
    split = result.report.split
    fc = result.config.features

    # Plots from the best model.
    model = next(m.model for m in result.report.results if m.key == best.key)
    times, surv = model.survival_function(split.test.head(400))
    risk = model.predict_risk(split.test.head(400))
    sc = plot_example_survival_curves(
        times,
        surv,
        risk,
        rdir / "survival_curves.png",
        f"Readmission survival curves for {best.name}",
    )
    cal_png = plot_calibration(
        best.calibration,
        rdir / "calibration.png",
        f"Day {result.calibration_time:.0f} calibration for {best.name}",
    )
    cmp_png = plot_cindex_comparison(
        [m.name for m in result.metrics],
        [m.c_index for m in result.metrics],
        rdir / "cindex_comparison.png",
        result.min_c_index,
    )

    payload = metrics_payload(result)
    (rdir / "metrics.json").write_text(json.dumps(payload, indent=2))

    md = [
        "# ReadmitRisk — evaluation report",
        "",
        f"- **Best model:** {best.name} (Harrell C-index **{best.c_index:.4f}**)",
        f"- **Gate:** C-index >= {result.min_c_index:.2f} -> "
        f"{'PASS ✅' if result.passed else 'FAIL ❌'}",
        f"- Test set: {len(split.test):,} index encounters "
        f"({split.test[fc.event_col].mean():.1%} readmitted)",
        "",
        "| Model | C-index | IPCW-C | IBS | mean tAUC | Calib err |",
        "|---|---|---|---|---|---|",
    ]
    for m in sorted(result.metrics, key=lambda x: -x.c_index):
        md.append(
            f"| {m.name} | {m.c_index:.4f} | {m.ipcw_c_index:.4f} | "
            f"{m.integrated_brier_score:.4f} | {m.mean_time_auc:.4f} | {m.calibration_error:.4f} |"
        )
    md += [
        "",
        "Lower IBS is better (0.25 = uninformative). Calibration error is the sample-weighted",
        "mean gap between predicted and Kaplan-Meier observed risk at the horizon.",
        "",
        "![C-index](cindex_comparison.png)",
        "![Survival curves](survival_curves.png)",
        "![Calibration](calibration.png)",
    ]
    (rdir / "eval_report.md").write_text("\n".join(md))
    return {
        "metrics_json": rdir / "metrics.json",
        "survival_png": sc,
        "calibration_png": cal_png,
        "comparison_png": cmp_png,
    }


def run_eval(use_sample: bool = False) -> bool:
    """Train, evaluate with survival metrics, write reports, enforce the gate."""
    config = load_config()
    report = train_models(use_sample=use_sample, persist=not use_sample)
    result = evaluate_models(report, config)

    print(render_metrics_table(result))
    print()

    # Enforce the survival-metric invariant (rejects accuracy; requires the survival set).
    try:
        assert_survival_metrics(PRODUCED_METRIC_NAMES, config.evaluation.require_survival_metrics)
    except ValueError as exc:
        print(f"GATE FAIL ❌  survival-metric invariant violated: {exc}")
        return False

    artifacts = write_reports(result)
    best = result.best()
    print(
        f"Best model: {best.name}  |  Harrell C-index = {best.c_index:.4f}  "
        f"(gate >= {result.min_c_index:.2f})"
    )
    print(
        f"Integrated Brier score = {best.integrated_brier_score:.4f}  |  "
        f"calibration error = {best.calibration_error:.4f}"
    )
    print(f"Reports + plots written to: {artifacts['metrics_json'].parent}")

    if not result.passed:
        print(
            f"\nGATE FAIL ❌  best C-index {best.c_index:.4f} < required {result.min_c_index:.2f}"
        )
        return False
    print(
        f"\nGATE PASS ✅  C-index {best.c_index:.4f} >= {result.min_c_index:.2f}; "
        "survival metrics used (no classification accuracy)."
    )
    return True
