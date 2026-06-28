"""Deterministic, Synthea-schema-compatible synthetic EHR generator.

Why this exists
---------------
The *primary* data source is the real Synthea generator (run via the Docker one-shot,
``make generate``). But Synthea is a ~100&nbsp;MB Java fat-jar that takes minutes to run a
full population — unacceptable for CI and awkward for a first-clone "it just works"
experience. This module produces output with the **exact same CSV column schema** that
Synthea's CSV exporter emits (``patients``, ``encounters``, ``conditions``,
``observations``), so the downstream DuckDB cohort SQL is byte-for-byte the same code
path regardless of which backend produced the files.

How the readmission signal is encoded
--------------------------------------
Crucially, we do **not** write durations or event labels. We simulate a realistic
hospitalization *process* per patient. Every patient has at least one (index) inpatient
admission; each discharge is followed by a near-term readmission with a probability that
is logistic in a latent per-patient risk (a "mixture / cure" formulation — most
discharges are simply not followed by a near-term admission). When a readmission does
occur, its timing is drawn from a Weibull so events spread across the follow-up window.
Sicker patients (older, more comorbidities, longer stays, abnormal labs) have higher
readmission odds. The observable features are correlated with — but a noisy view of —
that latent risk, with an irreducible unobserved-frailty component that caps the
achievable C-index at a realistic level (~0.75). The DuckDB SQL then *re-derives* the
time-to-event records from the raw timestamps, exactly as it would for real Synthea.

Determinism: everything is driven by a single ``numpy`` ``Generator`` seeded from the
configured master seed, so a fixed (population, seed) reproduces identical output.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Clinical catalogs (real SNOMED / LOINC codes so the cohort SQL's weight map and
# observation pulls apply equally to real Synthea output).
# ---------------------------------------------------------------------------

# (SNOMED code, description, Charlson-style weight, base prevalence)
CONDITION_CATALOG: list[tuple[str, str, int, float]] = [
    ("44054006", "Diabetes mellitus type 2 (disorder)", 1, 0.22),
    ("59621000", "Essential hypertension (disorder)", 1, 0.34),
    ("55822004", "Hyperlipidemia (disorder)", 1, 0.30),
    ("195967001", "Asthma (disorder)", 1, 0.10),
    ("49436004", "Atrial fibrillation (disorder)", 2, 0.08),
    ("53741008", "Coronary arteriosclerosis (disorder)", 2, 0.12),
    ("13645005", "Chronic obstructive lung disease (disorder)", 2, 0.09),
    ("84114007", "Heart failure (disorder)", 3, 0.07),
    ("431857002", "Chronic kidney disease stage 4 (disorder)", 2, 0.05),
    ("22298006", "Myocardial infarction (disorder)", 2, 0.06),
    ("363346000", "Malignant neoplastic disease (disorder)", 3, 0.05),
    ("230690007", "Cerebrovascular accident (disorder)", 2, 0.05),
]

# LOINC code -> (description, units, healthy mean, sd, diabetes/hypertension shift)
LAB_SPECS: dict[str, dict] = {
    "39156-5": {
        "name": "bmi",
        "desc": "Body mass index (BMI) [Ratio]",
        "unit": "kg/m2",
        "mean": 27.5,
        "sd": 5.0,
    },
    "8480-6": {
        "name": "systolic_bp",
        "desc": "Systolic Blood Pressure",
        "unit": "mm[Hg]",
        "mean": 126.0,
        "sd": 14.0,
    },
    "4548-4": {
        "name": "hba1c",
        "desc": "Hemoglobin A1c/Hemoglobin.total in Blood",
        "unit": "%",
        "mean": 5.6,
        "sd": 0.7,
    },
    "2339-0": {
        "name": "glucose",
        "desc": "Glucose [Mass/volume] in Blood",
        "unit": "mg/dL",
        "mean": 95.0,
        "sd": 18.0,
    },
}

RACES = ["white"] * 64 + ["black"] * 13 + ["asian"] * 7 + ["native"] * 2 + ["other"] * 14
ENCOUNTER_INPATIENT_CODE = ("1505002", "Hospital admission (procedure)")
ENCOUNTER_AMBULATORY_CODE = ("162673000", "General examination of patient (procedure)")
ENCOUNTER_EMERGENCY_CODE = ("50849002", "Emergency room admission (procedure)")
ENCOUNTER_WELLNESS_CODE = ("410620009", "Well child visit (procedure)")


@dataclass
class GenSummary:
    n_patients: int
    n_encounters: int
    n_inpatient: int
    n_conditions: int
    n_observations: int
    n_deaths: int


def _iso(ref: datetime, day_offset: float) -> str:
    """Format a day offset (relative to the reference date) as a Synthea ISO timestamp."""
    dt = ref + timedelta(days=float(day_offset))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_date(ref: datetime, day_offset: float) -> str:
    dt = ref + timedelta(days=float(day_offset))
    return dt.strftime("%Y-%m-%d")


def _uuid(rng: np.random.Generator) -> str:
    return str(uuid.UUID(bytes=rng.bytes(16), version=4))


def generate(
    out_dir: Path,
    population: int,
    seed: int,
    reference_date: str = "2020-01-01",
    followup_days: int = 30,
) -> GenSummary:
    """Generate a synthetic population to ``out_dir`` as Synthea-schema CSV files.

    Parameters
    ----------
    out_dir:
        Directory to write ``patients.csv``, ``encounters.csv``, ``conditions.csv``,
        ``observations.csv`` into.
    population:
        Number of synthetic patients.
    seed:
        Master seed; fixes the entire generation.
    reference_date:
        Anchor "today"; the encounter window ends here so censoring is deterministic.
    followup_days:
        Readmission horizon (only used to lightly tune the event rate, NOT to label).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ref = datetime.fromisoformat(reference_date)

    n = int(population)

    # --- Stage A: patient-level latent attributes (vectorized) --------------------
    # Age skewed older (inpatient populations are older); clip to a plausible range.
    age = np.clip(np.round(rng.normal(63, 17, n)), 18, 98).astype(int)
    sex = rng.choice(["M", "F"], size=n)
    race = rng.choice(RACES, size=n)
    ethnicity = rng.choice(["hispanic", "nonhispanic"], size=n, p=[0.16, 0.84])

    # Unobserved frailty — the irreducible component the model can NEVER see. Its weight
    # (NOISE_WEIGHT below) sets the ceiling on the achievable concordance.
    frailty = rng.normal(0, 1, n)

    # Comorbidity burden grows with age and frailty.
    cond_lam = 0.5 + 0.045 * np.clip(age - 40, 0, None) + 0.6 * np.clip(frailty, 0, None)
    n_conditions = rng.poisson(cond_lam).clip(0, len(CONDITION_CATALOG))

    # Per-patient comorbidity set + weighted score (and disease flags driving labs).
    cond_codes: list[list[str]] = []
    comorbidity_score = np.zeros(n)
    diabetes_flag = np.zeros(n)
    htn_flag = np.zeros(n)
    base_prev = np.array([c[3] for c in CONDITION_CATALOG])
    weights = np.array([c[2] for c in CONDITION_CATALOG])
    codes = [c[0] for c in CONDITION_CATALOG]
    for i in range(n):
        k = int(n_conditions[i])
        if k == 0:
            cond_codes.append([])
            continue
        # Sample without replacement, weighted by prevalence scaled by this patient's risk.
        p = base_prev * (1.0 + 0.4 * frailty[i])
        p = np.clip(p, 1e-3, None)
        p = p / p.sum()
        chosen = rng.choice(len(CONDITION_CATALOG), size=k, replace=False, p=p)
        cond_codes.append([codes[j] for j in chosen])
        comorbidity_score[i] = weights[chosen].sum()
        diabetes_flag[i] = 1.0 if "44054006" in cond_codes[i] else 0.0
        htn_flag[i] = 1.0 if "59621000" in cond_codes[i] else 0.0

    # Labs correlated with risk + disease flags (with realistic spread).
    risk_z = 0.5 * frailty + 0.02 * (age - 63) + 0.15 * comorbidity_score
    bmi = rng.normal(LAB_SPECS["39156-5"]["mean"] + 2.0 * risk_z, LAB_SPECS["39156-5"]["sd"])
    systolic = rng.normal(
        LAB_SPECS["8480-6"]["mean"] + 8.0 * htn_flag + 3.0 * risk_z, LAB_SPECS["8480-6"]["sd"]
    )
    hba1c = rng.normal(
        LAB_SPECS["4548-4"]["mean"] + 1.4 * diabetes_flag + 0.3 * risk_z, LAB_SPECS["4548-4"]["sd"]
    )
    glucose = rng.normal(
        LAB_SPECS["2339-0"]["mean"] + 28.0 * diabetes_flag + 8.0 * risk_z,
        LAB_SPECS["2339-0"]["sd"],
    )
    bmi = np.clip(bmi, 15, 55)
    systolic = np.clip(systolic, 85, 210)
    hba1c = np.clip(hba1c, 4.0, 14.0)
    glucose = np.clip(glucose, 60, 350)
    labs = {"39156-5": bmi, "8480-6": systolic, "4548-4": hba1c, "2339-0": glucose}
    lab_present = {code: rng.random(n) < 0.85 for code in labs}  # ~15% missing per lab

    # Per-patient length-of-stay scale (sicker -> longer stays).
    los_factor = np.exp(0.30 * risk_z)

    # --- Stage B: latent linear predictor for the readmission hazard --------------
    def z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-9)

    # Observed (model-visible) component — built from standardized observed features.
    obs_lp = (
        0.55 * z(age.astype(float))
        + 0.70 * z(comorbidity_score)
        + 0.45 * z(n_conditions.astype(float))
        + 0.55 * z(hba1c)
        + 0.35 * z(glucose)
        + 0.30 * z(systolic)
        + 0.20 * z(bmi)
        + 0.40 * z(los_factor)
    )
    obs_lp = z(obs_lp)  # standardize the aggregate observed score
    # NOISE_WEIGHT (frailty) and READMIT_SIGNAL together set the achievable C-index. They
    # are env-overridable so the signal-to-noise can be tuned without code changes; the
    # committed defaults land the honest test C-index at a realistic ~0.78.
    NOISE_WEIGHT = float(os.environ.get("READMIT_SIM_NOISE", "1.10"))
    eta = obs_lp + NOISE_WEIGHT * frailty

    # Per-discharge readmission probability (logistic in the latent risk). This is a
    # "mixture / cure" formulation: most discharges are NOT followed by a near-term
    # admission, while a risk-dependent fraction are. It keeps admissions-per-patient
    # clinically realistic (~1-2) while producing a ~15-20% 30-day readmission rate.
    READMIT_LOGIT0 = float(os.environ.get("READMIT_SIM_LOGIT0", "-3.05"))  # baseline log-odds
    READMIT_SIGNAL = float(os.environ.get("READMIT_SIM_SIGNAL", "1.35"))
    readmit_prob = 1.0 / (1.0 + np.exp(-(READMIT_LOGIT0 + READMIT_SIGNAL * eta)))
    SHORT_GAP_SHAPE = 1.1  # Weibull shape for time-to-readmission (days)
    # Readmission timing also depends on risk: higher-risk discharges are readmitted
    # SOONER (shorter scale), which makes survival times — not just event incidence —
    # informative. Kept modest so the honest C-index stays in the realistic 0.74-0.78
    # band rather than the implausibly-easy 0.85+ a strong timing signal would produce.
    timing_coef = float(os.environ.get("READMIT_SIM_TIMING", "0.10"))
    short_gap_scale = 22.0 * np.exp(-timing_coef * eta)

    # Mortality: small per-patient probability over the window, higher for frail patients.
    death_prob = np.clip(0.02 + 0.03 * np.clip(eta, 0, None), 0, 0.25)
    will_die = rng.random(n) < death_prob

    # --- Stage C: per-patient encounter timelines ---------------------------------
    pat_rows = []
    enc_rows = []
    cond_rows = []
    obs_rows = []
    n_inpatient = 0
    n_deaths = 0

    for i in range(n):
        pid = _uuid(rng)
        active_days = int(rng.uniform(380, 720))  # length of this patient's record window
        window_start = -active_days  # day offsets relative to reference_date (0 = today)
        birth_offset = -int(age[i] * 365.25) - int(rng.uniform(0, 365))

        # Death (if any) somewhere in the active window.
        death_offset = None
        if will_die[i]:
            death_offset = window_start + int(rng.uniform(0.3, 1.0) * active_days)

        # Chronic conditions diagnosed near the start of the record.
        for code in cond_codes[i]:
            desc = next(c[1] for c in CONDITION_CATALOG if c[0] == code)
            dx_off = window_start + int(rng.uniform(-200, 30))
            cond_rows.append(
                {
                    "START": _iso_date(ref, dx_off),
                    "STOP": "",
                    "PATIENT": pid,
                    "ENCOUNTER": "",
                    "CODE": code,
                    "DESCRIPTION": desc,
                }
            )

        # Inpatient admission process. Every patient has at least one (index) admission;
        # each discharge is followed by a near-term readmission with risk-dependent
        # probability. The DuckDB SQL re-derives durations/events from these timestamps.
        admit = int(rng.uniform(window_start + 30, -8))
        chain = 0
        while admit < 0 and chain < 6:  # encounters only up to "today" (reference_date)
            if death_offset is not None and admit >= death_offset:
                break  # patient already deceased
            los = int(np.clip(np.ceil(rng.gamma(2.0, 1.6 * los_factor[i])), 1, 30))
            discharge = admit + los
            # Truncate a stay that runs past death.
            if death_offset is not None and discharge > death_offset:
                discharge = death_offset
            enc_id = _uuid(rng)
            enc_rows.append(_encounter_row(ref, enc_id, pid, admit, discharge, "inpatient"))
            n_inpatient += 1
            # Labs recorded at this admission (drives the latest-value feature pull).
            for code, arr in labs.items():
                if not lab_present[code][i]:
                    continue
                spec = LAB_SPECS[code]
                val = float(arr[i] + rng.normal(0, spec["sd"] * 0.15))
                obs_rows.append(
                    {
                        "DATE": _iso(ref, admit),
                        "PATIENT": pid,
                        "ENCOUNTER": enc_id,
                        "CATEGORY": "vital-signs"
                        if spec["name"] in {"bmi", "systolic_bp"}
                        else "laboratory",
                        "CODE": code,
                        "DESCRIPTION": spec["desc"],
                        "VALUE": round(val, 1),
                        "UNITS": spec["unit"],
                        "TYPE": "numeric",
                    }
                )
            # Decide whether a near-term readmission follows this discharge.
            if death_offset is not None and discharge >= death_offset:
                break
            if rng.random() < readmit_prob[i]:
                gap = float(rng.weibull(SHORT_GAP_SHAPE) * short_gap_scale[i])
                admit = discharge + max(1, int(round(gap)))
                chain += 1
            else:
                break

        # Outpatient / ED utilization (prior-utilization features) via a Poisson process.
        n_outpatient = rng.poisson(3 + 2 * np.clip(eta[i], 0, None))
        for _ in range(int(n_outpatient)):
            off = window_start + int(rng.uniform(0, active_days))
            if off >= 0 or (death_offset is not None and off >= death_offset):
                continue
            kind = rng.choice(["ambulatory", "emergency", "wellness"], p=[0.6, 0.25, 0.15])
            enc_rows.append(_encounter_row(ref, _uuid(rng), pid, off, off + 1, kind))

        deathdate = _iso_date(ref, death_offset) if death_offset is not None else ""
        if deathdate:
            n_deaths += 1
        pat_rows.append(
            {
                "Id": pid,
                "BIRTHDATE": _iso_date(ref, birth_offset),
                "DEATHDATE": deathdate,
                "RACE": race[i],
                "ETHNICITY": ethnicity[i],
                "GENDER": sex[i],
                "FIRST": f"Synthetic{i}",
                "LAST": "Patient",
                "CITY": "Bedford",
                "STATE": "Massachusetts",
                "ZIP": "01730",
                "INCOME": int(rng.uniform(20000, 140000)),
            }
        )

    # --- Write CSVs (Synthea exporter schema) -------------------------------------
    pat_df = pd.DataFrame(pat_rows)
    enc_df = pd.DataFrame(enc_rows)
    cond_df = pd.DataFrame(cond_rows)
    obs_df = pd.DataFrame(obs_rows)

    pat_df.to_csv(out_dir / "patients.csv", index=False)
    enc_df.to_csv(out_dir / "encounters.csv", index=False)
    cond_df.to_csv(out_dir / "conditions.csv", index=False)
    obs_df.to_csv(out_dir / "observations.csv", index=False)

    return GenSummary(
        n_patients=len(pat_df),
        n_encounters=len(enc_df),
        n_inpatient=n_inpatient,
        n_conditions=len(cond_df),
        n_observations=len(obs_df),
        n_deaths=n_deaths,
    )


def _encounter_row(
    ref: datetime, enc_id: str, pid: str, start_off: float, stop_off: float, enc_class: str
) -> dict:
    base_cost = {
        "inpatient": 9500.0,
        "emergency": 1400.0,
        "ambulatory": 180.0,
        "wellness": 130.0,
    }.get(enc_class, 200.0)
    code, desc = {
        "inpatient": ENCOUNTER_INPATIENT_CODE,
        "emergency": ENCOUNTER_EMERGENCY_CODE,
        "ambulatory": ENCOUNTER_AMBULATORY_CODE,
        "wellness": ENCOUNTER_WELLNESS_CODE,
    }.get(enc_class, ENCOUNTER_AMBULATORY_CODE)
    return {
        "Id": enc_id,
        "START": _iso(ref, start_off),
        "STOP": _iso(ref, stop_off),
        "PATIENT": pid,
        "ENCOUNTERCLASS": enc_class,
        "CODE": code,
        "DESCRIPTION": desc,
        "REASONCODE": "",
        "REASONDESCRIPTION": "",
        "BASE_ENCOUNTER_COST": base_cost,
        "TOTAL_CLAIM_COST": round(base_cost * 1.4, 2),
        "PAYER_COVERAGE": round(base_cost * 0.8, 2),
    }
