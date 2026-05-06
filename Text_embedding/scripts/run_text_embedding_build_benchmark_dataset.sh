#!/bin/bash
# run_build_benchmark_dataset.sh
#
# Build the final benchmark dataset from Layer 2 quote window features
# and the released sentiment panel.
#
# Run after aggregate_quote_windows.py has completed.
#
# Usage:
#   bash run_build_benchmark_dataset.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Configuration: edit these paths before running.
# ---------------------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${MODULE_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"

LAYER2_DIR="${LAYER2_DIR:-${OUTPUT_ROOT}/layer2}"
SENTIMENT_DIR="${SENTIMENT_DIR:-${OUTPUT_ROOT}/sentiment_panel}"
CALENDAR="${CALENDAR:-${DATA_ROOT}/ec_calendar.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/benchmark}"

# Minimum post-window ticks required to include an anchor.
# Anchors below this threshold are excluded and written to excluded_anchors_*.csv.
# Recommended: 3. Reduce to 1 to maximize coverage in after_hours/pre_market sessions.
MIN_TICKS_POST=3

PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------
# End of configuration.
# ---------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"

"$PYTHON" "${MODULE_DIR}/src/text_embedding_build_benchmark_dataset.py" \
    --layer2_dir     "$LAYER2_DIR" \
    --sentiment_dir  "$SENTIMENT_DIR" \
    --calendar       "$CALENDAR" \
    --output_dir     "$OUTPUT_DIR" \
    --min_ticks_post "$MIN_TICKS_POST" \
    --anchor_type    all
