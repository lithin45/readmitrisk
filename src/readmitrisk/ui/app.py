"""ReadmitRisk: Streamlit risk-curve demo.

Select a synthetic patient and see their predicted readmission survival curve (Cox PH vs
Random Survival Forest), the SHAP drivers behind the Cox risk score, the model-evaluation
report, and the subgroup fairness audit.

Run with:  streamlit run src/readmitrisk/ui/app.py   (or `make demo`).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from readmitrisk.paths import get_paths
from readmitrisk.ui import backend

st.set_page_config(page_title="ReadmitRisk", page_icon="🏥", layout="wide")


@st.cache_resource(show_spinner="Loading models + cohort...")
def _bundle():
    return backend.load_bundle()


def _report_image(name: str) -> str | None:
    """Find a report plot on disk, falling back to the committed docs/assets copies."""
    for base in (get_paths().reports, get_paths().root / "docs" / "assets"):
        p = base / name
        if p.exists():
            return str(p)
    return None


def _risk_curve_figure(curves: dict[str, np.ndarray]) -> go.Figure:
    times = curves["times"]
    fig = go.Figure()
    palette = {"Cox PH": "#1f77b4", "Random Survival Forest": "#ff7f0e"}
    for name, surv in curves.items():
        if name == "times":
            continue
        fig.add_trace(
            go.Scatter(
                x=times,
                y=surv,
                mode="lines",
                name=name,
                line_shape="hv",
                line={"width": 3, "color": palette.get(name)},
                hovertemplate="day %{x}: S(t)=%{y:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        xaxis_title="Days since discharge",
        yaxis_title="Survival probability  S(t) = P(not readmitted)",
        yaxis_range=[0, 1.02],
        legend={"orientation": "h", "y": -0.25},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=380,
    )
    return fig


def _drivers_figure(explanation) -> go.Figure:
    contrib = explanation.contributions.head(8)[::-1]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in contrib.values]
    fig = go.Figure(
        go.Bar(
            x=contrib.values,
            y=list(contrib.index),
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="SHAP contribution to log-hazard  (red ↑ risk, blue ↓ risk)",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=320,
    )
    return fig


def _patient_tab(bundle) -> None:
    fc = bundle.config.features
    table = backend.patient_table(bundle)

    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        choice = st.selectbox(
            "Select a synthetic index hospitalization",
            table["label"],
            index=int(st.session_state.get("pidx", 0)),
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🎲 Random patient"):
            st.session_state["pidx"] = int(np.random.default_rng().integers(0, len(table)))
            st.rerun()
    enc_id = table.loc[table["label"] == choice, "index_encounter_id"].iloc[0]
    row = backend.get_patient_row(bundle, enc_id).iloc[0]

    # Patient snapshot.
    st.markdown("##### Patient snapshot")
    c = st.columns(6)
    c[0].metric("Age", f"{row['age_at_index']:.0f}")
    c[1].metric("Sex", row["sex"])
    c[2].metric("LOS (days)", f"{row['length_of_stay_days']:.0f}")
    c[3].metric("Comorbidities", int(row["n_conditions"]))
    c[4].metric("Prior ED", int(row["n_prior_emergency"]))
    c[5].metric("HbA1c", f"{row['hba1c']:.1f}" if not np.isnan(row["hba1c"]) else "n/a")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Predicted readmission survival curve")
        curves = backend.risk_curves(bundle, enc_id, max_day=fc.followup_days)
        st.plotly_chart(_risk_curve_figure(curves), use_container_width=True)
        summary = backend.patient_risk_summary(bundle, enc_id, horizon=fc.followup_days)
        rc = st.columns(len(summary["predicted_risk"]) + 1)
        for i, (name, risk) in enumerate(summary["predicted_risk"].items()):
            rc[i].metric(f"{name}: {fc.followup_days}d risk", f"{risk:.1%}")
        actual = summary["actual"]
        outcome = (
            f"readmitted on day {actual['duration_days']:.0f}"
            if actual["event_observed"] == 1
            else f"censored at day {actual['duration_days']:.0f} (no readmission observed)"
        )
        rc[-1].metric("Actual outcome", "readmitted" if actual["event_observed"] else "censored")
        st.caption(f"Ground truth for this synthetic patient: {outcome}.")

    with right:
        st.markdown("##### Why this prediction? SHAP drivers (Cox PH)")
        expl = backend.patient_drivers(bundle, enc_id)
        if expl is not None:
            st.plotly_chart(_drivers_figure(expl), use_container_width=True)
            st.caption(
                "Contributions to this patient's log-hazard relative to the average patient. "
                "Positive (red) features increase readmission risk."
            )
        else:
            st.info("SHAP drivers require the Cox model.")


def _evaluation_tab(bundle) -> None:
    st.markdown("##### Survival-model evaluation (held-out test set)")
    metrics = bundle.metrics
    if metrics:
        rows = [
            {
                "Model": m["name"],
                "C-index": round(m["concordance_index"], 4),
                "IPCW-C": round(m["ipcw_concordance_index"], 4),
                "Integrated Brier": round(m["integrated_brier_score"], 4),
                "mean time-AUC": round(m["time_dependent_auc_mean"], 4),
                "Calibration err": round(m["calibration_error"], 4),
            }
            for m in metrics["models"]
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            f"Best model: **{metrics['best_model']}** (C-index {metrics['best_c_index']:.4f}); "
            f"gate ≥ {metrics['min_c_index']:.2f}. Lower integrated Brier is better."
        )
    else:
        st.info("Run `make eval` to generate the evaluation report.")

    cols = st.columns(3)
    for col, (img, cap) in zip(
        cols,
        [
            ("cindex_comparison.png", "Discrimination: Cox vs RSF"),
            ("survival_curves.png", "Example survival curves"),
            ("calibration.png", "Time-dependent calibration"),
        ],
        strict=False,
    ):
        path = _report_image(img)
        if path:
            col.image(path, caption=cap, use_container_width=True)


def _fairness_tab(bundle) -> None:
    st.markdown("##### Fairness audit, per-subgroup C-index & calibration")
    fair = bundle.fairness
    if not fair:
        st.info("Run `make fairness` to generate the subgroup audit.")
        return
    flagged = fair.get("flagged_attributes", [])
    if flagged:
        st.warning(
            f"⚠️ Subgroup gap > {fair['threshold']:.2f} flagged for: **{', '.join(flagged)}**"
        )
    else:
        st.success(f"✅ No subgroup C-index/calibration gap exceeds {fair['threshold']:.2f}.")

    for attr in fair["attributes"]:
        gap = attr["c_index_gap"]
        gap_s = "n/a" if gap is None else f"{gap:.3f}"
        flag = " ⚠️" if attr["flagged"] else ""
        st.markdown(f"**{attr['attribute']}**, C-index gap: {gap_s}{flag}")
        st.dataframe(
            [
                {
                    "subgroup": s["level"],
                    "n": s["n"],
                    "events": s["n_events"],
                    "C-index": None if s["c_index"] is None else round(s["c_index"], 3),
                    "calib err": None
                    if s["calibration_error"] is None
                    else round(s["calibration_error"], 3),
                    "note": "low-N/events" if s["low_confidence"] else "",
                }
                for s in attr["subgroups"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Subgroup C-index standard error ~ 0.5/√events; low-N/events subgroups are shown for "
        "transparency but excluded from gap flagging."
    )
    fp = _report_image("fairness_cindex.png")
    if fp:
        st.image(fp, use_container_width=True)


def main() -> None:
    st.title("🏥 ReadmitRisk: time-to-readmission risk explorer")
    st.caption(
        "Survival analysis (right-censoring handled), calibrated risk, and a subgroup "
        "fairness audit, on fully synthetic Synthea-schema EHR data."
    )
    bundle = _bundle()
    st.caption(f"Model source: {bundle.source} | {len(bundle.test):,} test index encounters.")
    st.info(
        "Safe, self contained demo on fully synthetic data. There are no real patients, "
        "no external APIs, no keys, and nothing to configure or pay for. Everything you "
        "see is generated and computed locally.",
        icon="🔒",
    )

    tab1, tab2, tab3 = st.tabs(["🧑‍⚕️ Patient risk", "📊 Model evaluation", "⚖️ Fairness audit"])
    with tab1:
        _patient_tab(bundle)
    with tab2:
        _evaluation_tab(bundle)
    with tab3:
        _fairness_tab(bundle)


main()
