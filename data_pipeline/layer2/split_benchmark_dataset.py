"""
split_benchmark_dataset.py

Apply session filtering and time-based train/test split to the benchmark
dataset produced by build_benchmark_dataset.py.

Split design:
    Train : 2021 Q1 - 2022 Q4  (N=729 EC events)
    Test  : 2023 Q1 - 2023 Q3  (N=389 EC events)

    A chronological split is used to test temporal generalisation.
    Both splits share 143 companies; performance differences reflect
    transfer across time periods rather than firm-level distribution shift.

Session filtering:
    The main benchmark retains regular-session and after-hours ECs only.
    Pre-market events (EC start before 09:30 ET) are excluded due to
    differences in pre-open quote liquidity and quoting behaviour.
    They are written to robustness/ for reference.

    Session definitions (Eastern Time):
        regular      : 09:30 - 15:59
        after_hours  : 16:00 - 19:59
        pre_market   : before 09:30  (excluded from main benchmark)
        non_trading  : other times   (excluded from main benchmark)

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
    python split_benchmark_dataset.py \
        --benchmark_dir  /path/to/benchmark \
        --output_dir     /path/to/benchmark_split \
        [--train_years   2021 2022] \
        [--test_years    2023] \
        [--main_sessions regular after_hours]
"""

import argparse
import logging
import os
import sys

import pandas as pd


POST_WINDOWS = [30, 60, 120, 300]
ANCHOR_TYPES = ["pre", "qa"]


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
    filename = f"benchmark_{anchor_type}_{post_w}s.csv"

    main  = df[df["ec_session"].isin(main_sessions)].copy()
    extra = df[~df["ec_session"].isin(main_sessions)].copy()

    train = main[main["year"].isin(train_years)].copy()
    test  = main[main["year"].isin(test_years)].copy()

    for subdir, data in [("train", train), ("test", test)]:
        d = os.path.join(output_dir, subdir)
        os.makedirs(d, exist_ok=True)
        data.to_csv(os.path.join(d, filename), index=False, encoding="utf-8")

    # Pre-market events written to robustness/ for reference only.
    robust_dir = os.path.join(output_dir, "robustness")
    os.makedirs(robust_dir, exist_ok=True)
    extra.to_csv(
        os.path.join(robust_dir, f"pre_market_{anchor_type}_{post_w}s.csv"),
        index=False, encoding="utf-8",
    )

    def ec_count(frame):
        return frame[["tic", "year", "quarter"]].drop_duplicates().__len__()

    log.info(
        "%s post=%ds: train=%d anchors (%d ECs), test=%d anchors (%d ECs)",
        anchor_type, post_w,
        len(train), ec_count(train),
        len(test),  ec_count(test),
    )

    return {
        "anchor_type":       anchor_type,
        "post_window_sec":   post_w,
        "n_train_anchors":   len(train),
        "n_train_ecs":       ec_count(train),
        "n_test_anchors":    len(test),
        "n_test_ecs":        ec_count(test),
        "n_robustness_anchors": len(extra),
        "n_robustness_ecs":     ec_count(extra),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply session filtering and train/test split to MERIT benchmark."
    )
    p.add_argument("--benchmark_dir",  required=True,
                   help="Directory containing benchmark_*.csv files.")
    p.add_argument("--output_dir",     required=True,
                   help="Directory for split output files.")
    p.add_argument("--train_years",    nargs="+", type=int, default=[2021, 2022])
    p.add_argument("--test_years",     nargs="+", type=int, default=[2023])
    p.add_argument("--main_sessions",  nargs="+",
                   default=["regular", "after_hours"],
                   help="Sessions included in the main benchmark (default: regular after_hours).")
    return p.parse_args()


def main():
    args = parse_args()
    log  = setup_logging(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("train_years  : %s", args.train_years)
    log.info("test_years   : %s", args.test_years)
    log.info("main_sessions: %s", args.main_sessions)

    summary_rows = []
    for anchor_type in ANCHOR_TYPES:
        for post_w in POST_WINDOWS:
            path = os.path.join(
                args.benchmark_dir, f"benchmark_{anchor_type}_{post_w}s.csv"
            )
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

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, "split_summary.csv"),
        index=False, encoding="utf-8",
    )
    log.info("Done.")


if __name__ == "__main__":
    main()