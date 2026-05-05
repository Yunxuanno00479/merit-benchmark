"""
build_benchmark_dataset.py

Construct the final benchmark dataset from Layer 2 quote window features
and the released sentiment panel.

This script performs four steps:

    1. Load and merge:
       For each earnings call, join the Layer 2 quote window aggregates
       with the corresponding sentiment panel (pre or qa_score).

    2. Compute trajectory statistics (CT features):
       For each anchor in the Presentation panel, compute expanding-window
       statistics over all sentences up to and including that anchor.
       These statistics capture the cumulative language trajectory (CT)
       as opposed to the cumulative level (CL, expanding mean only).

       CT features added to the pre panel:
           ev_expanding_mean   mean of expected_value over sentences 1..k
           ev_expanding_std    std  of expected_value over sentences 1..k
           ev_expanding_max    max  of expected_value over sentences 1..k
           ev_expanding_min    min  of expected_value over sentences 1..k
           n_change_points     cumulative count of change point flags up to k

       Note: obi_mean_* and total_depth_mean_* are retained for completeness
       but are not used as benchmark targets (see add_labels.py).

    3. Annotate sessions:
       Classify each earnings call into one of four trading sessions based
       on the EC start time in Eastern Time:

           pre_market   : before 09:30 ET
           regular      : 09:30 - 15:59 ET
           after_hours  : 16:00 - 19:59 ET
           non_trading  : all other times

    4. Filter anchors:
       Remove anchors whose pre-window contains no valid ticks.

Usage:
    python build_benchmark_dataset.py \
        --layer2_dir    /path/to/layer2_output \
        --sentiment_dir /path/to/released_panel \
        --calendar      /path/to/ec_calendar.csv \
        --output_dir    /path/to/benchmark \
        [--anchor_type  pre|qa|all]

Requirements:
    pandas >= 2.0, numpy, pytz
"""

import argparse
import glob
import logging
import os
import sys
from datetime import time as dtime

import numpy as np
import pandas as pd
import pytz


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ET_TZ        = pytz.timezone("America/New_York")
POST_WINDOWS = [30, 60, 120, 300]

PRE_SENTIMENT_COLS = [
    "section_id", "timestamp_p", "section",
    "finberttone_expected_value",
    "finberttone_cumulative_tone",
    "finberttone_change_point",
]

# Trajectory feature columns computed by compute_trajectory_features_for_ec().
TRAJECTORY_COLS = [
    "ev_expanding_mean",
    "ev_expanding_std",
    "ev_expanding_max",
    "ev_expanding_min",
    "n_change_points",
]

QA_SENTIMENT_COLS = [
    "qa_index", "Q_Timestamp", "A_Timestamp",
    "assertive_negative_score",  "assertive_neutral_score",  "assertive_positive_score",
    "cautious_negative_score",   "cautious_neutral_score",   "cautious_positive_score",
    "optimistic_negative_score", "optimistic_neutral_score", "optimistic_positive_score",
    "specific_negative_score",   "specific_neutral_score",   "specific_positive_score",
    "clear_negative_score",      "clear_neutral_score",      "clear_positive_score",
    "relevant_negative_score",   "relevant_neutral_score",   "relevant_positive_score",
]

PRE_OUTPUT_COLS = [
    "tic", "year", "quarter", "anchor_type", "anchor_id",
    "post_window_sec", "timestamp_anchor", "ec_session",
    "section_id", "timestamp_p", "section",
    # Pre-window quote features
    "bid_ask_spread_mean_pre", "bid_ask_spread_std_pre",
    "obi_mean_pre", "total_depth_mean_pre",      # retained for completeness
    "qrf_mean_pre", "quote_volatility_mean_pre",
    "n_ticks_pre",
    # Post-window quote features
    "bid_ask_spread_mean_post", "bid_ask_spread_std_post",
    "obi_mean_post", "total_depth_mean_post",    # retained for completeness
    "qrf_mean_post", "quote_volatility_mean_post",
    "n_ticks_post",
    # Inst.: static sentiment feature (current sentence)
    "finberttone_expected_value",
    "finberttone_cumulative_tone",
    "finberttone_change_point",
    # CT: trajectory features (expanding window up to this anchor)
    "ev_expanding_mean",
    "ev_expanding_std",
    "ev_expanding_max",
    "ev_expanding_min",
    "n_change_points",
]

QA_OUTPUT_COLS = [
    "tic", "year", "quarter", "anchor_type", "anchor_id",
    "post_window_sec", "timestamp_anchor", "ec_session",
    "qa_index", "Q_Timestamp", "A_Timestamp",
    # Pre-window quote features
    "bid_ask_spread_mean_pre", "bid_ask_spread_std_pre",
    "obi_mean_pre", "total_depth_mean_pre",
    "qrf_mean_pre", "quote_volatility_mean_pre",
    "n_ticks_pre",
    # Post-window quote features
    "bid_ask_spread_mean_post", "bid_ask_spread_std_post",
    "obi_mean_post", "total_depth_mean_post",
    "qrf_mean_post", "quote_volatility_mean_post",
    "n_ticks_post",
    # SubjECTive-QA scores
    "assertive_negative_score",  "assertive_neutral_score",  "assertive_positive_score",
    "cautious_negative_score",   "cautious_neutral_score",   "cautious_positive_score",
    "optimistic_negative_score", "optimistic_neutral_score", "optimistic_positive_score",
    "specific_negative_score",   "specific_neutral_score",   "specific_positive_score",
    "clear_negative_score",      "clear_neutral_score",      "clear_positive_score",
    "relevant_negative_score",   "relevant_neutral_score",   "relevant_positive_score",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "build_benchmark_dataset.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session classification
# ---------------------------------------------------------------------------

def classify_ec_session(timestamp_et_str):
    """
    Classify an EC start time into a trading session (Eastern Time).
        pre_market  : before 09:30
        regular     : 09:30 - 15:59
        after_hours : 16:00 - 19:59
        non_trading : other
    """
    ts = pd.to_datetime(timestamp_et_str, utc=True).tz_convert(ET_TZ)
    t  = dtime(ts.hour, ts.minute)
    if   dtime(4,  0) <= t < dtime(9, 30):  return "pre_market"
    elif dtime(9, 30) <= t < dtime(16, 0):  return "regular"
    elif dtime(16, 0) <= t < dtime(20, 0):  return "after_hours"
    else:                                    return "non_trading"


def build_session_map(calendar_path):
    cal = pd.read_csv(calendar_path)
    session_map = {}
    for _, row in cal.iterrows():
        key = (str(row["tic"]), int(row["year"]), str(row["quarter"]))
        session_map[key] = classify_ec_session(row["timestamp_start_et"])
    return session_map


# ---------------------------------------------------------------------------
# Trajectory statistics (CT features)
# ---------------------------------------------------------------------------

def compute_trajectory_features_for_ec(sent_df):
    """
    Compute the five expanding-window CT features for one EC's
    Presentation panel. Features are ordered by section_id.

    Returns DataFrame with TRAJECTORY_COLS added.
    """
    df = sent_df.sort_values("section_id").reset_index(drop=True).copy()
    ev = df["finberttone_expected_value"]
    cp = df["finberttone_change_point"]

    df["ev_expanding_mean"] = ev.expanding().mean()
    df["ev_expanding_std"]  = ev.expanding().std().fillna(0.0)
    df["ev_expanding_max"]  = ev.expanding().max()
    df["ev_expanding_min"]  = ev.expanding().min()
    df["n_change_points"]   = cp.cumsum()

    return df


# ---------------------------------------------------------------------------
# Sentiment loading
# ---------------------------------------------------------------------------

def load_pre_sentiment(sentiment_dir, tic, year, quarter):
    path = os.path.join(
        sentiment_dir, "pre", f"{tic}_{year}_{quarter}_pre_score.csv"
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["tic", "year", "quarter"] + PRE_SENTIMENT_COLS)
    df["section_id"] = df["section_id"].astype(int)
    df = compute_trajectory_features_for_ec(df)
    return df


def load_qa_sentiment(sentiment_dir, tic, year, quarter):
    path = os.path.join(
        sentiment_dir, "qa_score", f"{tic}_{year}_{quarter}_qa_score.csv"
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["tic", "year", "quarter"] + QA_SENTIMENT_COLS)
    df["qa_index"] = df["qa_index"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Per-EC processing
# ---------------------------------------------------------------------------

def process_ec(layer2_path, sentiment_dir, session_map,
               min_ticks_pre, anchor_type, log):
    stem  = os.path.basename(layer2_path).replace("_layer2.csv", "")
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        log.warning("Unexpected filename: %s", layer2_path)
        return {}, {}

    tic, year, quarter = parts[0], int(parts[1]), parts[2]
    session = session_map.get((tic, year, quarter), "unknown")

    layer2 = pd.read_csv(layer2_path)
    layer2["ec_session"] = session
    layer2["year"]       = layer2["year"].astype(int)
    layer2["quarter"]    = layer2["quarter"].astype(str)

    valid_frames = {}

    for post_w in POST_WINDOWS:
        sub = layer2[
            (layer2["post_window_sec"] == post_w) &
            (layer2["anchor_type"].isin(
                ["pre", "qa"] if anchor_type == "all" else [anchor_type]
            ))
        ].copy()
        if sub.empty:
            continue

        # Anchor validity: require at least one tick in the pre-window.
        valid = sub[sub["n_ticks_pre"] >= min_ticks_pre].copy()
        if valid.empty:
            continue

        merged_parts = []

        pre_rows = valid[valid["anchor_type"] == "pre"]
        qa_rows  = valid[valid["anchor_type"] == "qa"]

        if not pre_rows.empty:
            sent = load_pre_sentiment(sentiment_dir, tic, year, quarter)
            if sent is not None:
                sent_join = sent.drop(columns=["tic", "year", "quarter"])
                sent_join = sent_join.rename(columns={"section_id": "anchor_id"})
                merged_parts.append(pre_rows.merge(sent_join, on="anchor_id", how="left"))
            else:
                merged_parts.append(pre_rows)

        if not qa_rows.empty:
            sent = load_qa_sentiment(sentiment_dir, tic, year, quarter)
            if sent is not None:
                sent_join = sent.drop(columns=["tic", "year", "quarter"])
                sent_join = sent_join.rename(columns={"qa_index": "anchor_id"})
                merged_parts.append(qa_rows.merge(sent_join, on="anchor_id", how="left"))
            else:
                merged_parts.append(qa_rows)

        if merged_parts:
            valid_frames[post_w] = pd.concat(merged_parts, ignore_index=True)

    return valid_frames


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build the MERIT benchmark dataset from Layer 2 quote window "
            "features and the released sentiment panel."
        )
    )
    p.add_argument("--layer2_dir",     required=True,
                   help="Directory containing *_layer2.csv files.")
    p.add_argument("--sentiment_dir",  required=True,
                   help="Root of the released sentiment panel "
                        "(contains pre/ and qa_score/).")
    p.add_argument("--calendar",       required=True,
                   help="Earnings call calendar CSV. Required columns: "
                        "tic, year, quarter, timestamp_start_et.")
    p.add_argument("--output_dir",     required=True,
                   help="Directory for benchmark output CSV files.")
    p.add_argument("--min_ticks_pre",  type=int, default=1,
                   help="Minimum pre-window ticks to include an anchor (default: 1).")
    p.add_argument("--anchor_type",    default="all",
                   choices=["pre", "qa", "all"])
    return p.parse_args()


def main():
    args = parse_args()
    log  = setup_logging(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    session_map = build_session_map(args.calendar)
    log.info("Session map built for %d earnings calls.", len(session_map))

    files = sorted(glob.glob(os.path.join(args.layer2_dir, "*_layer2.csv")))
    log.info("Found %d Layer 2 files.", len(files))

    valid_acc    = {w: [] for w in POST_WINDOWS}
    anchor_types = ["pre", "qa"] if args.anchor_type == "all" else [args.anchor_type]
    output_cols  = {"pre": PRE_OUTPUT_COLS, "qa": QA_OUTPUT_COLS}

    for path in files:
        valid_frames = process_ec(
            path, args.sentiment_dir, session_map,
            args.min_ticks_pre, args.anchor_type, log,
        )
        for w in POST_WINDOWS:
            if w in valid_frames and not valid_frames[w].empty:
                valid_acc[w].append(valid_frames[w])

    for w in POST_WINDOWS:
        if not valid_acc[w]:
            continue
        combined = pd.concat(valid_acc[w], ignore_index=True)
        for atype in anchor_types:
            sub = combined[combined["anchor_type"] == atype]
            if sub.empty:
                continue
            cols    = [c for c in output_cols[atype] if c in sub.columns]
            outpath = os.path.join(args.output_dir, f"benchmark_{atype}_{w}s.csv")
            sub[cols].to_csv(outpath, index=False, encoding="utf-8")
            log.info("Saved %s (%d rows).", outpath, len(sub))

    log.info("Done.")


if __name__ == "__main__":
    main()