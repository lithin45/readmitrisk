"""Matplotlib figures for the evaluation report (saved as PNGs under data/reports/).

Headless by design (Agg backend) so it runs in CI and Docker without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .calibration import CalibrationCurve  # noqa: E402


def plot_example_survival_curves(
    times: np.ndarray, surv: np.ndarray, risk: np.ndarray, out_path: Path, title: str
) -> Path:
    """Plot survival curves for low/median/high-risk example patients."""
    order = np.argsort(risk)
    picks = {
        "low risk (10th pct)": order[max(0, int(0.10 * len(order)) - 1)],
        "median risk": order[len(order) // 2],
        "high risk (90th pct)": order[min(len(order) - 1, int(0.90 * len(order)))],
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, idx in picks.items():
        ax.step(times, surv[idx], where="post", label=label, linewidth=2)
    ax.set_xlabel("Days since discharge")
    ax.set_ylabel("Survival probability S(t)\n(not readmitted)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_calibration(curve: CalibrationCurve, out_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim = max(0.05, float(np.nanmax([curve.bin_predicted.max(), curve.bin_observed.max()])) * 1.1)
    ax.plot([0, lim], [0, lim], "--", color="gray", label="perfect calibration")
    yerr = np.vstack(
        [
            np.clip(curve.bin_observed - curve.bin_lower, 0, None),
            np.clip(curve.bin_upper - curve.bin_observed, 0, None),
        ]
    )
    ax.errorbar(
        curve.bin_predicted,
        curve.bin_observed,
        yerr=yerr,
        fmt="o-",
        color="#1f77b4",
        capsize=3,
        label="model (KM observed)",
    )
    ax.set_xlabel(f"Predicted readmission risk by day {curve.time:.0f}")
    ax.set_ylabel("Observed (Kaplan Meier) risk")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(
        f"{title}\ncalibration error (ECE) = {curve.calibration_error:.3f}", fontsize=10.5
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_cindex_comparison(
    names: list[str], cindices: list[float], out_path: Path, threshold: float
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#2ca02c" if c >= threshold else "#d62728" for c in cindices]
    bars = ax.barh(names, cindices, color=colors)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label=f"gate {threshold:.2f}")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Harrell concordance (test)")
    for bar, c in zip(bars, cindices, strict=False):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{c:.3f}", va="center"
        )
    ax.set_title("Discrimination: Cox PH vs Random Survival Forest")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
