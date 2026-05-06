#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python3}"

# Edit these paths for a fresh run, or override via environment variables.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${MODULE_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"

RAW_TRANSCRIPT_DIR="${RAW_TRANSCRIPT_DIR:-${DATA_ROOT}/transcripts_jsonl}"
TRANSCRIPT_OUT_DIR="${TRANSCRIPT_OUT_DIR:-${OUTPUT_ROOT}/transcript_panel}"
SENTIMENT_OUT_DIR="${SENTIMENT_OUT_DIR:-${OUTPUT_ROOT}/sentiment_panel}"

mkdir -p \
  "${TRANSCRIPT_OUT_DIR}" \
  "${SENTIMENT_OUT_DIR}/pre" \
  "${SENTIMENT_OUT_DIR}/qa_score"

"${PYTHON}" "${MODULE_DIR}/src_preprocessing/text_embedding_parse_transcript.py" \
  --input_dir "${RAW_TRANSCRIPT_DIR}" \
  --output_dir "${TRANSCRIPT_OUT_DIR}"

"${PYTHON}" "${MODULE_DIR}/src_preprocessing/text_embedding_compute_finbert_tone.py" \
  --panel pre \
  --input_dir "${TRANSCRIPT_OUT_DIR}/pre" \
  --output_dir "${SENTIMENT_OUT_DIR}/pre"

"${PYTHON}" "${MODULE_DIR}/src_preprocessing/text_embedding_compute_subjective_qa.py" \
  --input_dir "${TRANSCRIPT_OUT_DIR}/qa" \
  --output_dir "${SENTIMENT_OUT_DIR}/qa_score"

"${PYTHON}" "${MODULE_DIR}/src_preprocessing/text_embedding_compute_changepoint.py" \
  --panel pre \
  --input_dir "${SENTIMENT_OUT_DIR}/pre"
