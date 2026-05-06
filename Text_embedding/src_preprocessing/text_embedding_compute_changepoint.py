"""
compute_changepoint.py

Apply PELT change point detection to FinBERT-tone score files.

This script reads the finberttone_expected_value column from score files
produced by compute_finbert_tone.py and overwrites the finberttone_change_point
column with binary flags (0 or 1).

Algorithm:
    PELT (Pruned Exact Linear Time)
    Killick, Fearnhead & Eckley (2012). Optimal detection of changepoints
    with a linear computational cost. JASA, 107(500), 1590-1598.

Penalty:
    AIC: pen = 2
    Akaike (1974). A new look at the statistical model identification.
    IEEE Transactions on Automatic Control, 19(6), 716-723.

    AIC penalizes each additional change point by 2 (one parameter per
    segment mean). Compared to BIC (pen = log(n)), AIC is less conservative
    and better suited to detecting subtle tone shifts in short sequences,
    which is the goal of this benchmark.

Cost model:
    rbf (radial basis function / Gaussian kernel)
    Appropriate for sentiment scores, which are bounded and non-Gaussian.
    Does not assume a parametric distribution for the signal.

Change point flag:
    1 marks the first sentence of each new sentiment regime.
    0 otherwise.

Supported panels:
    pre         -- reads *_pre_score.csv, updates finberttone_change_point
    qa_sentence -- reads *_qa_sentence_score.csv, updates finberttone_change_point

Usage:
    pip install ruptures

    # Presentation panel
    python compute_changepoint.py \
        --panel      pre \
        --input_dir  /path/to/sentiment_panel/pre

    # QA sentence panel
    python compute_changepoint.py \
        --panel      qa_sentence \
        --input_dir  /path/to/sentiment_panel/qa_sentence

    # Preview without writing (dry run)
    python compute_changepoint.py \
        --panel      pre \
        --input_dir  /path/to/sentiment_panel/pre \
        --dry_run
"""

import argparse
import csv
import glob
import logging
import os
import sys

import numpy as np


# AIC penalty: 2 per additional change point (one new segment mean parameter).
# Reference: Akaike (1974), IEEE Transactions on Automatic Control, 19(6), 716-723.
AIC_PENALTY = 2.0

PANEL_CONFIGS = {
    "pre": {
        "file_pattern": "*_pre_score.csv",
        "ev_col": "finberttone_expected_value",
        "cp_col": "finberttone_change_point",
    },
    "qa_sentence": {
        "file_pattern": "*_qa_sentence_score.csv",
        "ev_col": "finberttone_expected_value",
        "cp_col": "finberttone_change_point",
    },
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def load_ruptures():
    try:
        import ruptures as rpt
        return rpt
    except ImportError:
        print("ERROR: ruptures is not installed.\nInstall with: pip install ruptures")
        sys.exit(1)


def detect_changepoints(ev_list, cp_model, cp_min_size, rpt):
    """
    Apply PELT with AIC penalty to a sequence of expected values.

    Parameters
    ----------
    ev_list : list of float
        Sequence of expected values for one earnings call.
    cp_model : str
        PELT cost function. 'rbf' is recommended for sentiment scores.
    cp_min_size : int
        Minimum number of sentences between consecutive change points.
    rpt : module
        The ruptures module.

    Returns
    -------
    list of int
        Binary flags of the same length as ev_list.
        Flag 1 marks the first sentence of each new sentiment regime.
    """
    n = len(ev_list)
    if n < cp_min_size * 2:
        return [0] * n

    signal = np.array(ev_list, dtype=float).reshape(-1, 1)

    # ruptures.Pelt returns breakpoints as the index of the last element
    # in each segment, ending with n as a terminal sentinel.
    # We mark the start of each new segment (breakpoint index) with flag 1.
    algo = rpt.Pelt(model=cp_model, min_size=cp_min_size, jump=1).fit(signal)
    breakpoints = algo.predict(pen=AIC_PENALTY)

    flags = [0] * n
    for bp in breakpoints[:-1]:
        if 0 < bp < n:
            flags[bp] = 1
    return flags


def process_file(input_path, config, cp_model, cp_min_size, dry_run, rpt, log):
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        log.warning("%s: empty file, skipping.", input_path)
        return 0, 0

    ev_list = [float(row[config["ev_col"]]) for row in rows]
    flags   = detect_changepoints(ev_list, cp_model, cp_min_size, rpt)

    for row, flag in zip(rows, flags):
        row[config["cp_col"]] = flag

    if not dry_run:
        fieldnames = list(rows[0].keys())
        with open(input_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    n_cp = sum(flags)
    rate = n_cp / len(flags) if flags else 0.0
    log.info(
        "%s: n=%d, change_points=%d (rate=%.3f)%s",
        os.path.basename(input_path),
        len(rows), n_cp, rate,
        " [dry_run]" if dry_run else "",
    )
    return len(rows), n_cp


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Apply PELT + AIC change point detection to FinBERT-tone score files. "
            "Updates the finberttone_change_point column in place."
        )
    )
    p.add_argument(
        "--panel", required=True, choices=["pre", "qa_sentence"],
        help="Panel to process: 'pre' or 'qa_sentence'.",
    )
    p.add_argument("--input_dir", required=True,
                   help="Directory containing *_score.csv files.")
    p.add_argument("--cp_model", default="rbf", choices=["rbf", "l1", "l2"],
                   help="PELT cost model (default: rbf).")
    p.add_argument("--cp_min_size", type=int, default=3,
                   help="Minimum segment length between change points (default: 3).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print statistics without modifying any files.")
    return p.parse_args()


def main():
    args   = parse_args()
    log    = setup_logging()
    rpt    = load_ruptures()
    config = PANEL_CONFIGS[args.panel]

    log.info(
        "PELT + AIC: pen=%.1f, model=%s, min_size=%d, dry_run=%s",
        AIC_PENALTY, args.cp_model, args.cp_min_size, args.dry_run,
    )

    files = sorted(
        glob.glob(os.path.join(args.input_dir, config["file_pattern"]))
    )
    log.info("Found %d files for panel '%s'.", len(files), args.panel)

    total_n, total_cp = 0, 0
    for input_path in files:
        n, cp = process_file(
            input_path, config,
            args.cp_model, args.cp_min_size,
            args.dry_run, rpt, log,
        )
        total_n  += n
        total_cp += cp

    log.info(
        "Summary: total_sentences=%d, total_change_points=%d, avg_rate=%.3f",
        total_n, total_cp,
        total_cp / total_n if total_n > 0 else 0.0,
    )
    if args.dry_run:
        log.info("dry_run=True: no files were modified.")


if __name__ == "__main__":
    main()
