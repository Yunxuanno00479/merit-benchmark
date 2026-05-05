"""
aggregate_quote_windows.py

Aggregate tick-level quote metrics (Layer 1) into pre/post anchor windows
(Layer 2) for each earnings call event.

Two anchor types are supported:

    pre  -- each sentence in the Presentation section.
             Anchor timestamp: timestamp_p (seconds from EC start).

    qa   -- each QA pair.
             Anchor timestamp: Q_Timestamp (question start, seconds from EC start).

For each anchor, two windows are computed:

    pre-window  : [timestamp_anchor - 30, timestamp_anchor)    fixed at 30 seconds
    post-window : [timestamp_anchor, timestamp_anchor + W)     W in {30, 60, 120, 300}

Each anchor produces four output rows (one per post-window width).

Note on output columns:
    obi_mean_* and total_depth_mean_* are retained in the output for
    completeness but are not used as benchmark targets in the released
    evaluation (see add_labels.py).

Tick filtering (Section 3.3 of the paper):
    - Crossed-NBBO ticks (bid > ask, approx. 1.2% of ticks) are discarded.
    - Extreme-spread ticks (relative spread > 0.50, approx. 1.3% of ticks)
      are excluded following standard TAQ cleaning heuristics.

Note: This script requires access to the raw Layer 1 NBBO parquet files,
which are not publicly released due to NYSE TAQ licensing restrictions.
It is provided for reproducibility documentation. The released benchmark
was produced using build_benchmark_dataset.py.

Usage:
    python aggregate_quote_windows.py \
        --quote_dir     /path/to/layer1/quote \
        --sentiment_dir /path/to/released_panel \
        --calendar      /path/to/ec_calendar.csv \
        --output_dir    /path/to/layer2 \
        [--anchor_type  pre|qa|all]

Requirements:
    pandas >= 2.0, numpy, pyarrow >= 12.0
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRE_WINDOW_SEC = 30
POST_WINDOWS   = [30, 60, 120, 300]

# Note: obi_mean_* and total_depth_mean_* are retained for completeness
# but excluded from benchmark targets (see add_labels.py).
OUTPUT_COLS = [
    "tic", "year", "quarter",
    "anchor_type", "anchor_id", "post_window_sec",
    "timestamp_anchor",
    # pre-window
    "bid_ask_spread_mean_pre", "bid_ask_spread_std_pre",
    "obi_mean_pre",
    "total_depth_mean_pre",
    "qrf_mean_pre",
    "quote_volatility_mean_pre",
    "n_ticks_pre",
    # post-window
    "bid_ask_spread_mean_post", "bid_ask_spread_std_post",
    "obi_mean_post",
    "total_depth_mean_post",
    "qrf_mean_post",
    "quote_volatility_mean_post",
    "n_ticks_post",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "aggregate_quote_windows.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_checkpoint(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(path, key):
    with open(path, "a") as f:
        f.write(key + "\n")


# ---------------------------------------------------------------------------
# Quote panel loading
# ---------------------------------------------------------------------------

def load_quote_panel(quote_dir, tic, year, quarter):
    """
    Load the Layer 1 quote panel for one EC and compute seconds_to_ec.
    Returns a DataFrame with seconds_to_ec added, or None if not found.
    """
    path = os.path.join(quote_dir, f"{tic}_{year}_{quarter}.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)

    if "seconds_to_ec" not in df.columns:
        ec_utc = pd.to_datetime(df["ec_timestamp_utc"].iloc[0], utc=True)
        ts = df["timestamp_utc"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        df["seconds_to_ec"] = (ts - ec_utc).dt.total_seconds()

    return df


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------

def aggregate_window(df, t_start, t_end):
    """
    Aggregate quote metrics for ticks in [t_start, t_end).
    Applies tick filters: removes crossed-NBBO and extreme-spread ticks.
    """
    win = df[(df["seconds_to_ec"] >= t_start) & (df["seconds_to_ec"] < t_end)].copy()

    # Remove crossed-NBBO ticks (bid_ask_spread <= 0) and
    # extreme-spread ticks (relative spread > 0.50).
    if "bid_ask_spread" in win.columns:
        win = win[(win["bid_ask_spread"] > 0) & (win["bid_ask_spread"] <= 0.5)]

    def safe_mean(col):
        return win[col].mean() if col in win.columns and len(win) > 0 else np.nan

    def safe_std(col):
        return win[col].std() if col in win.columns and len(win) > 1 else np.nan

    return {
        "bid_ask_spread_mean":   safe_mean("bid_ask_spread"),
        "bid_ask_spread_std":    safe_std("bid_ask_spread"),
        "obi_mean":              safe_mean("obi"),
        "total_depth_mean":      safe_mean("total_depth"),
        "qrf_mean":              safe_mean("qrf"),
        "quote_volatility_mean": safe_mean("quote_volatility"),
        "n_ticks":               len(win),
    }


# ---------------------------------------------------------------------------
# Per-EC processing
# ---------------------------------------------------------------------------

def process_anchors(tic, year, quarter, quote_df, anchor_df,
                    anchor_type, id_col, timestamp_col):
    """
    Compute pre/post window aggregates for all anchors in anchor_df.
    """
    records = []
    for _, row in anchor_df.iterrows():
        t         = float(row[timestamp_col])
        anchor_id = int(row[id_col])

        for post_w in POST_WINDOWS:
            pre_agg  = aggregate_window(quote_df, t - PRE_WINDOW_SEC, t)
            post_agg = aggregate_window(quote_df, t, t + post_w)

            # Require at least one valid tick in the pre-window.
            if pre_agg["n_ticks"] == 0:
                continue

            record = {
                "tic":             tic,
                "year":            year,
                "quarter":         quarter,
                "anchor_type":     anchor_type,
                "anchor_id":       anchor_id,
                "post_window_sec": post_w,
                "timestamp_anchor": t,
            }
            for k, v in pre_agg.items():
                record[f"{k}_pre"] = v
            for k, v in post_agg.items():
                record[f"{k}_post"] = v
            records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Aggregate Layer 1 quote metrics into pre/post anchor windows. "
            "Requires access to raw NYSE TAQ parquet files."
        )
    )
    p.add_argument("--quote_dir",     required=True,
                   help="Directory containing Layer 1 quote panel parquet files.")
    p.add_argument("--sentiment_dir", required=True,
                   help="Root directory of the released sentiment panel "
                        "(contains pre/ and qa_score/ subdirectories).")
    p.add_argument("--calendar",      required=True,
                   help="Earnings call calendar CSV. Required columns: "
                        "tic, year, quarter.")
    p.add_argument("--output_dir",    required=True,
                   help="Directory for Layer 2 output CSV files.")
    p.add_argument("--anchor_type",   default="all",
                   choices=["pre", "qa", "all"],
                   help="Which anchor type to process (default: all).")
    p.add_argument("--checkpoint",    default=None,
                   help="Checkpoint file path for resuming interrupted runs.")
    return p.parse_args()


def main():
    args = parse_args()
    log  = setup_logging(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = args.checkpoint or os.path.join(
        args.output_dir, "logs", "checkpoint.txt"
    )
    done = load_checkpoint(ckpt_path)

    calendar = pd.read_csv(args.calendar)
    log.info("Calendar loaded: %d earnings calls.", len(calendar))

    total, processed = len(calendar), 0

    for _, cal_row in calendar.iterrows():
        tic     = str(cal_row["tic"])
        year    = int(cal_row["year"])
        quarter = str(cal_row["quarter"])
        key     = f"{tic}|{year}|{quarter}"

        if key in done:
            continue

        log.info("[%d/%d] %s %d %s", processed + 1, total, tic, year, quarter)

        quote_df = load_quote_panel(args.quote_dir, tic, year, quarter)
        if quote_df is None or quote_df.empty:
            log.warning("No quote data for %s %d %s, skipping.", tic, year, quarter)
            mark_done(ckpt_path, key)
            continue

        frames = []
        try:
            if args.anchor_type in ("pre", "all"):
                sent_path = os.path.join(
                    args.sentiment_dir, "pre",
                    f"{tic}_{year}_{quarter}_pre_score.csv"
                )
                if os.path.exists(sent_path):
                    sent = pd.read_csv(sent_path)
                    frames.append(process_anchors(
                        tic, year, quarter, quote_df, sent,
                        "pre", "section_id", "timestamp_p",
                    ))
                else:
                    log.warning("pre sentiment not found: %s", sent_path)

            if args.anchor_type in ("qa", "all"):
                sent_path = os.path.join(
                    args.sentiment_dir, "qa_score",
                    f"{tic}_{year}_{quarter}_qa_score.csv"
                )
                if os.path.exists(sent_path):
                    sent = pd.read_csv(sent_path)
                    frames.append(process_anchors(
                        tic, year, quarter, quote_df, sent,
                        "qa", "qa_index", "Q_Timestamp",
                    ))
                else:
                    log.warning("qa_score sentiment not found: %s", sent_path)

            if not frames:
                mark_done(ckpt_path, key)
                continue

            result  = pd.concat(frames, ignore_index=True)
            present = [c for c in OUTPUT_COLS if c in result.columns]
            extra   = [c for c in result.columns if c not in OUTPUT_COLS]
            result  = result[present + extra]

            out_path = os.path.join(
                args.output_dir, f"{tic}_{year}_{quarter}_layer2.csv"
            )
            result.to_csv(out_path, index=False, encoding="utf-8")
            mark_done(ckpt_path, key)
            processed += 1
            log.info("Saved %s (%d rows).", out_path, len(result))

        except Exception:
            log.exception("Error processing %s %d %s.", tic, year, quarter)

    log.info("Done. Processed %d / %d earnings calls.", processed, total)


if __name__ == "__main__":
    main()