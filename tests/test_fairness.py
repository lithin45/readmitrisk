"""Phase 5 acceptance tests: subgroup audit covers attributes, computes + flags gaps."""

from __future__ import annotations

from readmitrisk.evaluation.metrics import harrell_concordance
from readmitrisk.fairness.audit import (
    MIN_EVENTS_FOR_CINDEX,
    FairnessReport,
    render_report,
)


def test_audit_covers_all_subgroup_attributes(fairness_report: FairnessReport) -> None:
    audited = {a.attribute for a in fairness_report.attributes}
    assert {"sex", "age_band", "race", "ethnicity"} <= audited


def test_subgroups_have_counts_and_events(fairness_report: FairnessReport) -> None:
    for attr in fairness_report.attributes:
        assert attr.subgroups, f"{attr.attribute} has no subgroups"
        for s in attr.subgroups:
            assert s.n >= 0
            assert 0 <= s.n_events <= s.n


def test_low_confidence_flag_uses_n_and_events(fairness_report: FairnessReport) -> None:
    min_n = fairness_report.min_subgroup_n
    for attr in fairness_report.attributes:
        for s in attr.subgroups:
            expected = (s.n < min_n) or (s.n_events < MIN_EVENTS_FOR_CINDEX)
            assert s.low_confidence == expected


def test_gaps_computed_over_reliable_subgroups_only(fairness_report: FairnessReport) -> None:
    for attr in fairness_report.attributes:
        reliable = [s for s in attr.subgroups if not s.low_confidence and s.c_index is not None]
        if len(reliable) >= 2:
            vals = [s.c_index for s in reliable]
            assert attr.c_index_gap is not None
            assert abs(attr.c_index_gap - (max(vals) - min(vals))) < 1e-9
        else:
            assert attr.c_index_gap is None


def test_flag_matches_threshold(fairness_report: FairnessReport) -> None:
    thr = fairness_report.threshold
    for attr in fairness_report.attributes:
        expected = bool(
            (attr.c_index_gap is not None and attr.c_index_gap > thr)
            or (attr.calibration_gap is not None and attr.calibration_gap > thr)
        )
        assert attr.flagged == expected


def test_overall_cindex_matches_full_test(fairness_report, survival_split, cox_model) -> None:
    fc = survival_split.feature_cfg
    risk = cox_model.predict_risk(survival_split.test)
    ci = harrell_concordance(
        survival_split.test[fc.event_col], survival_split.test[fc.duration_col], risk
    )
    assert abs(fairness_report.overall_c_index - ci) < 1e-9


def test_report_renders_with_all_attributes(fairness_report: FairnessReport) -> None:
    text = render_report(fairness_report)
    for attr in ["sex", "age_band", "race", "ethnicity"]:
        assert attr in text
    assert "Caveat" in text


def test_flagged_attributes_are_consistent(fairness_report: FairnessReport) -> None:
    flagged = set(fairness_report.flagged_attributes)
    assert flagged == {a.attribute for a in fairness_report.attributes if a.flagged}
