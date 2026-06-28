"""Fairness plot: per-subgroup C-index by attribute (headless / Agg)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .audit import FairnessReport  # noqa: E402


def plot_subgroup_cindex(report: FairnessReport, out_path: Path) -> Path:
    attrs = [a for a in report.attributes if any(s.c_index is not None for s in a.subgroups)]
    n = max(1, len(attrs))
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.2 * n + 0.6), squeeze=False)
    axes = axes[:, 0]

    for ax, attr in zip(axes, attrs, strict=False):
        levels = [s for s in attr.subgroups if s.c_index is not None]
        names = [f"{s.level} (n={s.n})" for s in levels]
        vals = [s.c_index for s in levels]
        colors = [
            "#cccccc" if s.low_confidence else ("#d62728" if attr.flagged else "#2ca02c")
            for s in levels
        ]
        hatches = ["//" if s.low_confidence else "" for s in levels]
        bars = ax.barh(names, vals, color=colors)
        for bar, h, v in zip(bars, hatches, vals, strict=False):
            bar.set_hatch(h)
            ax.text(
                v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8
            )
        ax.axvline(report.overall_c_index, color="black", linestyle="--", linewidth=1)
        gap = "n/a" if attr.c_index_gap is None else f"{attr.c_index_gap:.3f}"
        title = f"{attr.attribute}  (C-index gap = {gap}{'  ⚠' if attr.flagged else ''})"
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlim(0.5, 1.0)
        ax.grid(axis="x", alpha=0.25)

    axes[-1].set_xlabel(
        f"Within-subgroup Harrell C-index  (dashed = overall {report.overall_c_index:.3f}; "
        "hatched/grey = low-N caveat)"
    )
    fig.suptitle(f"Fairness audit — {report.model_name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
