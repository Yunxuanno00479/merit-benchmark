"""
text_embedding_split_benchmark_dataset.py

Apply session filtering and time-based train/test split to the benchmark
dataset produced by text_embedding_build_benchmark_dataset.py.

Split design:
    Train : 2021 Q1 - 2022 Q4
    Test  : 2023 Q1 - 2023 Q3

    A time-based split is used to prevent look-ahead bias. Test set companies
    appear in the training set (same companies, different time periods), but
    the model cannot use future information to predict past outcomes.

Session filtering:
    By default, pre_market and non_trading ECs are excluded from the main
    benchmark because NBBO quote data is systematically sparse during these
    sessions, making the market microstructure response (Y) unreliable.

    Excluded sessions are written to a separate file so that researchers can
    run robustness checks on the full dataset if needed.

    Session definitions (Eastern Time):
        regular      : 09:30 - 15:59  (regular trading, high quote density)
        after_hours  : 16:00 - 19:59  (post-close, moderate quote density)
        pre_market   : 04:00 - 09:29  (pre-open, low quote density)
        non_trading  : other times    (very low or no quote activity)

Output layout:
    {output_dir}/
        train/
            benchmark_pre_{W}s.csv
            benchmark_qa_{W}s.csv
        test/
            benchmark_pre_{W}s.csv
            benchmark_qa_{W}s.csv
        robustness/
            pre_market_pre_{W}s.csv
            pre_market_qa_{W}s.csv
        split_summary.csv

Usage:
    python text_embedding_split_benchmark_dataset.py \\
        --benchmark_dir  /path/to/benchmark \\
        --output_dir     /path/to/benchmark_split \\
        [--train_years   2021 2022] \\
        [--test_years    2023] \\
        [--main_sessions regular after_hours]
"""

import argparse
import glob
import logging
import os
import sys

import pandas as pd


POST_WINDOWS  = [30, 60, 120, 300]
ANCHOR_TYPES  = ["pre", "qa"]


def setup_logging(output_dir):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "split_benchmark_dataset.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def split_and_write(df, train_years, test_years, main_sessions,
                    anchor_type, post_w, output_dir, log):
    """
    Apply session filter and time split to one benchmark file.
    Writes train, test, and robustness subsets.
    Returns a summary dict.
    """
    filename = f"benchmark_{anchor_type}_{post_w}s.csv"

    # Session split: main vs robustness (pre_market).
    main  = df[df["ec_session"].isin(main_sessions)].copy()
    extra = df[~df["ec_session"].isin(main_sessions)].copy()

    # Time split on main sessions.
    train = main[main["year"].isin(train_years)].copy()
    test  = main[main["year"].isin(test_years)].copy()

    # Write train.
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    train.to_csv(os.path.join(train_dir, filename), index=False, encoding="utf-8")

    # Write test.
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    test.to_csv(os.path.join(test_dir, filename), index=False, encoding="utf-8")

    # Write robustness (pre_market only, not split by time).
    robust_dir = os.path.join(output_dir, "robustness")
    os.makedirs(robust_dir, exist_ok=True)
    robust_name = f"pre_market_{anchor_type}_{post_w}s.csv"
    extra.to_csv(os.path.join(robust_dir, robust_name),
                 index=False, encoding="utf-8")

    def ec_count(frame):
        return frame[["tic", "year", "quarter"]].drop_duplicates().__len__()

    log.info(
        "%s post=%ds: train=%d anchors (%d ECs), test=%d anchors (%d ECs), "
        "robustness=%d anchors (%d ECs)",
        anchor_type, post_w,
        len(train), ec_count(train),
        len(test),  ec_count(test),
        len(extra), ec_count(extra),
    )

    return {
        "anchor_type":    anchor_type,
        "post_window_sec": post_w,
        "n_train_anchors": len(train),
        "n_train_ecs":     ec_count(train),
        "n_test_anchors":  len(test),
        "n_test_ecs":      ec_count(test),
        "n_robustness_anchors": len(extra),
        "n_robustness_ecs":     ec_count(extra),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Apply session filtering and time-based train/test split "
            "to the benchmark dataset."
        )
    )
    p.add_argument("--benchmark_dir",  required=True,
                   help="Directory containing benchmark_*.csv files "
                        "(output of text_embedding_build_benchmark_dataset.py).")
    p.add_argument("--output_dir",     required=True,
                   help="Directory for split output files.")
    p.add_argument("--train_years",    nargs="+", type=int,
                   default=[2021, 2022],
                   help="Years assigned to the training set (default: 2021 2022).")
    p.add_argument("--test_years",     nargs="+", type=int,
                   default=[2023],
                   help="Years assigned to the test set (default: 2023).")
    p.add_argument("--main_sessions",  nargs="+",
                   default=["regular", "after_hours"],
                   help=(
                       "EC sessions included in the main benchmark "
                       "(default: regular after_hours). "
                       "pre_market and non_trading are written to "
                       "robustness/ for supplementary analysis."
                   ))
    return p.parse_args()


def main():
    args = parse_args()
    log  = setup_logging(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("benchmark_dir : %s", args.benchmark_dir)
    log.info("output_dir    : %s", args.output_dir)
    log.info("train_years   : %s", args.train_years)
    log.info("test_years    : %s", args.test_years)
    log.info("main_sessions : %s", args.main_sessions)

    summary_rows = []

    for anchor_type in ANCHOR_TYPES:
        for post_w in POST_WINDOWS:
            filename = f"benchmark_{anchor_type}_{post_w}s.csv"
            path     = os.path.join(args.benchmark_dir, filename)

            if not os.path.exists(path):
                log.warning("File not found, skipping: %s", path)
                continue

            df = pd.read_csv(path)
            df["year"] = df["year"].astype(int)

            row = split_and_write(
                df, args.train_years, args.test_years, args.main_sessions,
                anchor_type, post_w, args.output_dir, log,
            )
            summary_rows.append(row)

    # Write split summary.
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "split_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    log.info("Split summary saved: %s", summary_path)

    # Print headline numbers (post_window=30 only).
    ref = summary[summary["post_window_sec"] == 30]
    if not ref.empty:
        log.info("=== Headline statistics (post_window=30s) ===")
        for _, r in ref.iterrows():
            log.info(
                "  %s: train=%d anchors / %d ECs  |  test=%d anchors / %d ECs",
                r["anchor_type"],
                r["n_train_anchors"], r["n_train_ecs"],
                r["n_test_anchors"],  r["n_test_ecs"],
            )

    log.info("Done.")


if __name__ == "__main__":
    main()