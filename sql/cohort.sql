-- =====================================================================================
-- ReadmitRisk — cohort construction (DuckDB)
-- =====================================================================================
-- Builds tidy *time-to-event* records for survival analysis directly from the raw
-- Synthea-schema EHR tables (patients / encounters / conditions / observations). This is
-- the clinical SQL centerpiece of the project: index selection, the discharge->next-
-- admission clock, right-censoring (administrative + death), comorbidity burden, prior
-- utilization, and latest labs are all computed here.
--
-- The Python builder substitutes two parameters before execution:
--   __FOLLOWUP__  readmission horizon in days (the event window, e.g. 30)
--   __LOOKBACK__  lookback window in days for prior-utilization + latest-lab features
--
-- Output: one row per index inpatient encounter with
--   patient_id, index_encounter_id, dates, duration_days, event_observed (1=readmitted
--   within horizon, 0=right-censored), model features, and subgroup attributes.
-- =====================================================================================

-- Robust ISO timestamp parsing: handles both 'Z' (UTC) and numeric-offset Synthea
-- timestamps. Raw tables are read as all-VARCHAR so type coercion is explicit here.
CREATE OR REPLACE TEMP MACRO parse_ts(x) AS
    COALESCE(TRY_CAST(x AS TIMESTAMP), TRY_CAST(x AS TIMESTAMPTZ)::TIMESTAMP);

WITH
-- 1) ----------------------------------------------------------------- typed source rows
enc AS (
    SELECT
        Id                          AS encounter_id,
        PATIENT                     AS patient_id,
        lower(ENCOUNTERCLASS)       AS encounter_class,
        parse_ts(START)             AS start_ts,
        parse_ts(STOP)              AS stop_ts
    FROM encounters
    WHERE PATIENT IS NOT NULL AND START IS NOT NULL
),
pat AS (
    SELECT
        Id                              AS patient_id,
        TRY_CAST(BIRTHDATE AS DATE)     AS birthdate,
        TRY_CAST(NULLIF(DEATHDATE, '') AS DATE) AS deathdate,
        GENDER                          AS sex,
        lower(RACE)                     AS race,
        lower(ETHNICITY)                AS ethnicity
    FROM patients
),
cond AS (
    SELECT
        PATIENT                     AS patient_id,
        CODE                        AS code,
        TRY_CAST(START AS DATE)     AS onset_date
    FROM conditions
    WHERE PATIENT IS NOT NULL AND START IS NOT NULL
),
obs AS (
    SELECT
        PATIENT                     AS patient_id,
        CODE                        AS code,
        parse_ts(DATE)              AS obs_ts,
        TRY_CAST(VALUE AS DOUBLE)   AS value
    FROM observations
    WHERE PATIENT IS NOT NULL AND TRY_CAST(VALUE AS DOUBLE) IS NOT NULL
),

-- Charlson-style comorbidity weights for a curated set of chronic conditions. Conditions
-- outside this map still count toward n_conditions but contribute 0 to the weighted score.
weight_map(code, weight) AS (
    VALUES
        ('44054006', 1), ('59621000', 1), ('55822004', 1), ('195967001', 1),
        ('49436004', 2), ('53741008', 2), ('13645005', 2), ('84114007', 3),
        ('431857002', 2), ('22298006', 2), ('363346000', 3), ('230690007', 2)
),

-- Administrative end-of-observation: the latest discharge seen in the data is our proxy
-- for "today". Index stays discharged close to this date have < horizon follow-up and are
-- administratively right-censored accordingly.
data_bounds AS (
    SELECT max(stop_ts) AS end_of_data FROM enc WHERE stop_ts IS NOT NULL
),

-- 2) ------------------------------------------------------------- index inpatient stays
-- Every completed inpatient stay is a candidate index. The next inpatient admission for
-- the same patient (chronologically) starts the readmission clock.
inpatient AS (
    SELECT
        e.encounter_id,
        e.patient_id,
        e.start_ts AS admit_ts,
        e.stop_ts  AS discharge_ts,
        LEAD(e.start_ts) OVER (
            PARTITION BY e.patient_id ORDER BY e.start_ts, e.stop_ts
        ) AS next_admit_ts
    FROM enc e
    WHERE e.encounter_class = 'inpatient'
      AND e.stop_ts IS NOT NULL
      AND e.stop_ts >= e.start_ts
),
index_enc AS (
    SELECT
        i.encounter_id,
        i.patient_id,
        i.admit_ts,
        i.discharge_ts,
        i.next_admit_ts,
        p.birthdate, p.deathdate, p.sex, p.race, p.ethnicity,
        b.end_of_data,
        -- Length of stay (days, fractional from seconds).
        GREATEST(date_diff('second', i.admit_ts, i.discharge_ts) / 86400.0, 0.0)
            AS length_of_stay_days,
        -- Age at index discharge (years).
        date_diff('day', p.birthdate, CAST(i.discharge_ts AS DATE)) / 365.25
            AS age_at_index,
        -- Days from discharge to the next admission (NULL if none).
        CASE WHEN i.next_admit_ts IS NOT NULL
             THEN date_diff('second', i.discharge_ts, i.next_admit_ts) / 86400.0 END
            AS gap_to_next_days,
        -- Days from discharge to death, only if death strictly after discharge.
        CASE WHEN p.deathdate IS NOT NULL AND p.deathdate > CAST(i.discharge_ts AS DATE)
             THEN date_diff('day', CAST(i.discharge_ts AS DATE), p.deathdate) END
            AS gap_to_death_days,
        -- Administrative follow-up available before end-of-data (days).
        GREATEST(date_diff('second', i.discharge_ts, b.end_of_data) / 86400.0, 0.0)
            AS gap_to_eod_days
    FROM inpatient i
    JOIN pat p USING (patient_id)
    CROSS JOIN data_bounds b
    -- Exclude index stays where the patient died during/at the stay (cannot be readmitted).
    WHERE p.deathdate IS NULL OR p.deathdate > CAST(i.discharge_ts AS DATE)
),

-- 3) --------------------------------------------------- survival target (duration/event)
-- Right-censoring horizon = min(readmission window, days-to-death, days-to-end-of-data).
target AS (
    SELECT
        *,
        LEAST(
            CAST(__FOLLOWUP__ AS DOUBLE),
            COALESCE(gap_to_death_days, CAST(__FOLLOWUP__ AS DOUBLE)),
            gap_to_eod_days
        ) AS censor_horizon
    FROM index_enc
),
labeled AS (
    SELECT
        *,
        -- Event: a readmission strictly within the (death/eod-truncated) horizon.
        CASE
            WHEN gap_to_next_days IS NOT NULL
                 AND gap_to_next_days >= 0
                 AND gap_to_next_days <= censor_horizon
            THEN 1 ELSE 0
        END AS event_observed,
        CASE
            WHEN gap_to_next_days IS NOT NULL
                 AND gap_to_next_days >= 0
                 AND gap_to_next_days <= censor_horizon
            THEN gap_to_next_days
            ELSE censor_horizon
        END AS duration_days
    FROM target
),

-- 4) ------------------------------------------------------- comorbidity burden at index
-- Conditions whose onset is on/before the index discharge are "active" at index.
active_cond AS (
    SELECT DISTINCT l.encounter_id, c.code
    FROM labeled l
    JOIN cond c
      ON c.patient_id = l.patient_id
     AND c.onset_date <= CAST(l.discharge_ts AS DATE)
),
comorbid AS (
    SELECT
        ac.encounter_id,
        count(*)                       AS n_conditions,
        COALESCE(sum(w.weight), 0)     AS comorbidity_score
    FROM active_cond ac
    LEFT JOIN weight_map w USING (code)
    GROUP BY ac.encounter_id
),

-- 5) ------------------------------------------------------------ prior utilization (lookback)
-- Encounters strictly before this admission, within the lookback window.
prior_util AS (
    SELECT
        l.encounter_id,
        count(*)                                                      AS n_prior_encounters,
        count(*) FILTER (WHERE e2.encounter_class = 'inpatient')      AS n_prior_inpatient,
        count(*) FILTER (WHERE e2.encounter_class = 'emergency')      AS n_prior_emergency
    FROM labeled l
    JOIN enc e2
      ON e2.patient_id = l.patient_id
     AND e2.start_ts <  l.admit_ts
     AND e2.start_ts >= l.admit_ts - INTERVAL '1 day' * __LOOKBACK__
    GROUP BY l.encounter_id
),

-- 6) ----------------------------------------------------------- latest labs/vitals (ASOF)
-- ASOF LEFT JOIN finds, per index, the most recent observation of each LOINC code at/before
-- discharge; the CASE enforces the lookback window (older values are treated as missing).
lab_bmi AS (
    SELECT l.encounter_id,
           CASE WHEN o.obs_ts >= l.discharge_ts - INTERVAL '1 day' * __LOOKBACK__
                THEN o.value END AS bmi
    FROM labeled l
    ASOF LEFT JOIN (SELECT patient_id, obs_ts, value FROM obs WHERE code = '39156-5') o
      ON l.patient_id = o.patient_id AND l.discharge_ts >= o.obs_ts
),
lab_sbp AS (
    SELECT l.encounter_id,
           CASE WHEN o.obs_ts >= l.discharge_ts - INTERVAL '1 day' * __LOOKBACK__
                THEN o.value END AS systolic_bp
    FROM labeled l
    ASOF LEFT JOIN (SELECT patient_id, obs_ts, value FROM obs WHERE code = '8480-6') o
      ON l.patient_id = o.patient_id AND l.discharge_ts >= o.obs_ts
),
lab_hba1c AS (
    SELECT l.encounter_id,
           CASE WHEN o.obs_ts >= l.discharge_ts - INTERVAL '1 day' * __LOOKBACK__
                THEN o.value END AS hba1c
    FROM labeled l
    ASOF LEFT JOIN (SELECT patient_id, obs_ts, value FROM obs WHERE code = '4548-4') o
      ON l.patient_id = o.patient_id AND l.discharge_ts >= o.obs_ts
),
lab_glucose AS (
    SELECT l.encounter_id,
           CASE WHEN o.obs_ts >= l.discharge_ts - INTERVAL '1 day' * __LOOKBACK__
                THEN o.value END AS glucose
    FROM labeled l
    ASOF LEFT JOIN (SELECT patient_id, obs_ts, value FROM obs WHERE code = '2339-0') o
      ON l.patient_id = o.patient_id AND l.discharge_ts >= o.obs_ts
)

-- 7) ------------------------------------------------------------------- tidy output table
SELECT
    l.patient_id,
    l.encounter_id                              AS index_encounter_id,
    CAST(l.admit_ts AS DATE)                    AS index_admit_date,
    CAST(l.discharge_ts AS DATE)                AS index_discharge_date,
    round(l.duration_days, 4)                   AS duration_days,
    l.event_observed,
    -- numeric features
    round(l.age_at_index, 2)                    AS age_at_index,
    round(l.length_of_stay_days, 3)             AS length_of_stay_days,
    COALESCE(cm.n_conditions, 0)                AS n_conditions,
    COALESCE(cm.comorbidity_score, 0)           AS comorbidity_score,
    COALESCE(pu.n_prior_encounters, 0)          AS n_prior_encounters,
    COALESCE(pu.n_prior_inpatient, 0)           AS n_prior_inpatient,
    COALESCE(pu.n_prior_emergency, 0)           AS n_prior_emergency,
    lb.bmi,
    ls.systolic_bp,
    lh.hba1c,
    lg.glucose,
    -- subgroup attributes (fairness audit only; never model inputs)
    l.sex,
    CASE
        WHEN l.age_at_index < 40 THEN '<40'
        WHEN l.age_at_index < 65 THEN '40-64'
        WHEN l.age_at_index < 80 THEN '65-79'
        ELSE '80+'
    END                                         AS age_band,
    l.race,
    l.ethnicity
FROM labeled l
LEFT JOIN comorbid    cm ON cm.encounter_id = l.encounter_id
LEFT JOIN prior_util  pu ON pu.encounter_id = l.encounter_id
LEFT JOIN lab_bmi     lb ON lb.encounter_id = l.encounter_id
LEFT JOIN lab_sbp     ls ON ls.encounter_id = l.encounter_id
LEFT JOIN lab_hba1c   lh ON lh.encounter_id = l.encounter_id
LEFT JOIN lab_glucose lg ON lg.encounter_id = l.encounter_id
-- Drop degenerate rows with no usable follow-up (discharge at end-of-data, etc.).
WHERE l.duration_days > 0
ORDER BY l.patient_id, l.admit_ts;
