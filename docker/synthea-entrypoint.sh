#!/usr/bin/env bash
# Run Synthea deterministically and stage the four CSV tables the pipeline consumes
# into <DATA_DIR>/raw using the same filenames the fallback generator emits.
set -euo pipefail

POP="${READMIT_POPULATION:-4000}"
SEED="${READMIT_SEED:-42}"
STATE="${READMIT_STATE:-Massachusetts}"
CITY="${READMIT_CITY:-Bedford}"
DATA_DIR="${READMIT_DATA_DIR:-/app/data}"
OUT="${DATA_DIR}/raw"
RUN_DIR="${OUT}/_synthea_run"

mkdir -p "${RUN_DIR}"

echo ">> Synthea: population=${POP} seed=${SEED} state='${STATE}' city='${CITY}'"
java -jar /opt/synthea/synthea-with-dependencies.jar \
  -p "${POP}" -s "${SEED}" -cs "${SEED}" -r 20200101 \
  --exporter.csv.export true \
  --exporter.fhir.export false \
  --exporter.hospital.fhir.export false \
  --exporter.practitioner.fhir.export false \
  --exporter.baseDirectory "${RUN_DIR}" \
  "${STATE}" "${CITY}"

for table in patients encounters conditions observations; do
  src="${RUN_DIR}/csv/${table}.csv"
  if [[ ! -f "${src}" ]]; then
    echo "ERROR: expected Synthea output missing: ${src}" >&2
    exit 1
  fi
  cp "${src}" "${OUT}/${table}.csv"
done

echo ">> Synthea CSVs staged in ${OUT}"
