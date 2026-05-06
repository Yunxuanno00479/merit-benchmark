#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE="${MODULE_DIR}/src/text_embedding_b4_pipeline.py"

PYTHON="${PYTHON:-python3}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${MODULE_DIR}/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
BENCHMARK_SPLIT_DIR="${BENCHMARK_SPLIT_DIR:-${OUTPUT_ROOT}/benchmark_split}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${OUTPUT_ROOT}/benchmark}"
MPNET_CACHE_DIR="${MPNET_CACHE_DIR:-${BENCHMARK_OUTPUT_DIR}/embedding_cache}"
OPENAI_CACHE_DIR="${OPENAI_CACHE_DIR:-${BENCHMARK_OUTPUT_DIR}/embedding_cache_openai_small512}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

mkdir -p "${BENCHMARK_OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

"${PYTHON}" -u "$PIPELINE" \
  --benchmark_split_dir "${BENCHMARK_SPLIT_DIR}" \
  --output_csv "${BENCHMARK_OUTPUT_DIR}/b4_text_embedding_results_allwindows_mpnet_absq75_lr.csv" \
  --cache_dir "${MPNET_CACHE_DIR}" \
  --embedding_models mpnet \
  --tasks pre qa \
  --post_windows 30 60 120 300 \
  --batch_size 32 \
  --run_mode full \
  --classifier_models LR \
  --train_subsample_frac 1.0 \
  --classifier_parallel_jobs 4 \
  --target_label_mode q75_abs \
  --checkpoint_prefix b4_mpnet_absq75_lr_allwindows \
  2>&1 | tee "${BENCHMARK_OUTPUT_DIR}/b4_mpnet_absq75_lr_allwindows.log"

"${PYTHON}" -u "$PIPELINE" \
  --benchmark_split_dir "${BENCHMARK_SPLIT_DIR}" \
  --output_csv "${BENCHMARK_OUTPUT_DIR}/b4_text_embedding_results_allwindows_openai_small512_absq75_lr.csv" \
  --cache_dir "${OPENAI_CACHE_DIR}" \
  --cache_model_alias openai_small512 \
  --embedding_models openai_small512 \
  --openai_embedding_model text-embedding-3-small \
  --openai_embedding_dimensions 512 \
  --tasks pre qa \
  --post_windows 30 60 120 300 \
  --batch_size 32 \
  --run_mode full \
  --classifier_models LR \
  --train_subsample_frac 1.0 \
  --classifier_parallel_jobs 4 \
  --target_label_mode q75_abs \
  --checkpoint_prefix b4_openai_small512_absq75_lr_allwindows \
  2>&1 | tee "${BENCHMARK_OUTPUT_DIR}/b4_openai_small512_absq75_lr_allwindows.log"
