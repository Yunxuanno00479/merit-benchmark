"""
text_embedding_build_benchmark_dataset.py

Construct the final benchmark dataset from Layer 2 quote window features
and the released sentiment panel.

This script performs four steps:

    1. Load and merge:
       For each earnings call, join the Layer 2 quote window aggregates
       with the corresponding sentiment panel (pre or qa_score).

    2. Compute trajectory statistics (B2 features):
       For each anchor in the Presentation panel, compute expanding-window
       statistics over all sentences up to and including that anchor.
       These statistics capture the cumulative language trajectory (B2)
       as opposed to the static score of a single sentence (B1).

       B2 features added to the pre panel:
           ev_expanding_mean   mean of expected_value over sentences 1..k
           ev_expanding_std    std  of expected_value over sentences 1..k
           ev_expanding_max    max  of expected_value over sentences 1..k
           ev_expanding_min    min  of expected_value over sentences 1..k
           n_change_points     cumulative count of change point flags up to k

       B1 feature (already present):
           finberttone_expected_value   score for sentence k only

    3. Annotate sessions:
       Classify each earnings call into one of four trading sessions based
       on the EC start time in Eastern Time:

           pre_market   : 04:00 - 09:29 ET
           regular      : 09:30 - 15:59 ET
           after_hours  : 16:00 - 19:59 ET
           non_trading  : all other times (rare)

    4. Filter anchors:
       Remove anchors whose post-window contains too few quote ticks.
       The default threshold is n_ticks_post >= MIN_TICKS_POST.
       Excluded anchors are written separately for transparency.

Output:
    {output_dir}/benchmark_pre_{W}s.csv     Presentation anchors
    {output_dir}/benchmark_qa_{W}s.csv      QA-pair anchors
    {output_dir}/excluded_anchors_{W}s.csv  Filtered-out anchors
    {output_dir}/benchmark_coverage.csv     Per-EC anchor counts

Usage:
    python text_embedding_build_benchmark_dataset.py \\
        --layer2_dir    /path/to/layer2_output \\
        --sentiment_dir /path/to/released_panel \\
        --calendar      /path/to/ec_calendar.csv \\
        --output_dir    /path/to/benchmark \\
        [--min_ticks_post 3] \\
        [--anchor_type pre|qa|all]

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

# Columns loaded from released_panel/pre/.
PRE_SENTIMENT_COLS = [
    "section_id", "timestamp_p", "section",
    "finberttone_expected_value",
    "finberttone_cumulative_tone",
    "finberttone_change_point",
]

# Trajectory feature columns computed by compute_trajectory_features().
TRAJECTORY_COLS = [
    "ev_expanding_mean",
    "ev_expanding_std",
    "ev_expanding_max",
    "ev_expanding_min",
    "n_change_points",
]

# Columns loaded from released_panel/qa_score/.
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
    # Identity
    "tic", "year", "quarter", "anchor_type", "anchor_id",
    "post_window_sec", "timestamp_anchor", "ec_session",
    # Preserved transcript text
    "speaker_name_P", "speaker_title_P", "presentation_text",
    # Structural
    "section_id", "timestamp_p", "section",
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
    # B1: static sentiment feature
    "finberttone_expected_value",
    "finberttone_cumulative_tone",
    "finberttone_change_point",
    # B2: trajectory features (expanding window up to this anchor)
    "ev_expanding_mean",
    "ev_expanding_std",
    "ev_expanding_max",
    "ev_expanding_min",
    "n_change_points",
]

QA_OUTPUT_COLS = [
    # Identity
    "tic", "year", "quarter", "anchor_type", "anchor_id",
    "post_window_sec", "timestamp_anchor", "ec_session",
    # Preserved transcript text
    "Question", "Answer",
    "q_speaker_name", "q_speaker_title",
    "a_speaker_name", "a_speaker_title",
    # Structural
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
    # SubjECTive-QA scores (B1 and B2 both use these at pair level)
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
    Classify an EC start time into a trading session.

    Session boundaries (Eastern Time):
        pre_market  : 04:00 - 09:29
        regular     : 09:30 - 15:59
        after_hours : 16:00 - 19:59
        non_trading : all other times
    """
    ts = pd.to_datetime(timestamp_et_str, utc=True).tz_convert(ET_TZ)
    t  = dtime(ts.hour, ts.minute)

    if   dtime(4,  0) <= t < dtime(9, 30):  return "pre_market"
    elif dtime(9, 30) <= t < dtime(16, 0):  return "regular"
    elif dtime(16, 0) <= t < dtime(20, 0):  return "after_hours"
    else:                                     return "non_trading"


def build_session_map(calendar_path):
    """Build a dict mapping (tic, year, quarter) -> ec_session."""
    cal = pd.read_csv(calendar_path)
    session_map = {}
    for _, row in cal.iterrows():
        key = (str(row["tic"]), int(row["year"]), str(row["quarter"]))
        session_map[key] = classify_ec_session(row["timestamp_start_et"])
    return session_map


# ---------------------------------------------------------------------------
# Trajectory statistics (B2 features)
# ---------------------------------------------------------------------------

def compute_trajectory_features(sent_df):
    """
    Compute expanding-window trajectory statistics for each anchor
    in a Presentation panel DataFrame.

    For anchor at row index k (0-based), the statistics are computed over
    all rows 0..k (inclusive), ordered by section_id.

    Parameters
    ----------
    sent_df : DataFrame
        Presentation panel for one EC, with columns:
        finberttone_expected_value, finberttone_change_point.

    Returns
    -------
    DataFrame with TRAJECTORY_COLS added.
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
    """Load and compute trajectory features for one EC's pre panel."""
    path = os.path.join(
        sentiment_dir, "pre", f"{tic}_{year}_{quarter}_pre_score.csv"
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["tic", "year", "quarter"] + PRE_SENTIMENT_COLS)
    df["section_id"] = df["section_id"].astype(int)
    df = compute_trajectory_features(df)
    return df


def load_qa_sentiment(sentiment_dir, tic, year, quarter):
    """Load qa_score panel for one EC."""
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
               min_ticks_post, anchor_type, log):
    """
    Load Layer 2 output for one EC, annotate session, join sentiment
    (including trajectory features), and filter by n_ticks_post.

    Returns
    -------
    dict : {post_window_sec -> DataFrame} for valid anchors
    dict : {post_window_sec -> DataFrame} for excluded anchors
    """
    stem   = os.path.basename(layer2_path).replace("_layer2.csv", "")
    parts  = stem.rsplit("_", 2)
    if len(parts) != 3:
        log.warning("Unexpected filename: %s", layer2_path)
        return {}, {}

    tic, year, quarter = parts[0], int(parts[1]), parts[2]
    session = session_map.get((tic, year, quarter), "unknown")

    layer2 = pd.read_csv(layer2_path)
    layer2["ec_session"] = session
    layer2["year"]       = layer2["year"].astype(int)
    layer2["quarter"]    = layer2["quarter"].astype(str)

    valid_frames    = {}
    excluded_frames = {}

    for post_w in POST_WINDOWS:
        sub = layer2[
            (layer2["post_window_sec"] == post_w) &
            (layer2["anchor_type"].isin(
                ["pre", "qa"] if anchor_type == "all" else [anchor_type]
            ))
        ].copy()

        if sub.empty:
            continue

        valid    = sub[sub["n_ticks_post"] >= min_ticks_post].copy()
        excluded = sub[sub["n_ticks_post"] <  min_ticks_post].copy()

        if not valid.empty:
            merged_parts = []

            pre_rows = valid[valid["anchor_type"] == "pre"]
            qa_rows  = valid[valid["anchor_type"] == "qa"]

            if not pre_rows.empty:
                sent = load_pre_sentiment(sentiment_dir, tic, year, quarter)
                if sent is not None:
                    # Join on section_id (anchor_id in layer2 = section_id in sent).
                    sent_join = sent.drop(columns=["tic", "year", "quarter"])
                    sent_join = sent_join.rename(columns={"section_id": "anchor_id"})
                    joined = pre_rows.merge(sent_join, on="anchor_id", how="left")
                    merged_parts.append(joined)
                else:
                    merged_parts.append(pre_rows)

            if not qa_rows.empty:
                sent = load_qa_sentiment(sentiment_dir, tic, year, quarter)
                if sent is not None:
                    sent_join = sent.drop(columns=["tic", "year", "quarter"])
                    sent_join = sent_join.rename(columns={"qa_index": "anchor_id"})
                    joined = qa_rows.merge(sent_join, on="anchor_id", how="left")
                    merged_parts.append(joined)
                else:
                    merged_parts.append(qa_rows)

            if merged_parts:
                valid_frames[post_w] = pd.concat(merged_parts, ignore_index=True)

        if not excluded.empty:
            excluded_frames[post_w] = excluded

    return valid_frames, excluded_frames


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build the benchmark dataset from Layer 2 quote window features "
            "and the released sentiment panel, including B2 trajectory features."
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
    p.add_argument("--min_ticks_post", type=int, default=3,
                   help="Minimum post-window ticks to include an anchor "
                        "(default: 3).")
    p.add_argument("--anchor_type",    default="all",
                   choices=["pre", "qa", "all"],
                   help="Which anchor type to include (default: all).")
    return p.parse_args()


def main():
    args = parse_args()
    log  = setup_logging(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("layer2_dir    : %s", args.layer2_dir)
    log.info("sentiment_dir : %s", args.sentiment_dir)
    log.info("calendar      : %s", args.calendar)
    log.info("output_dir    : %s", args.output_dir)
    log.info("min_ticks_post: %d", args.min_ticks_post)
    log.info("anchor_type   : %s", args.anchor_type)

    session_map = build_session_map(args.calendar)
    log.info("Session map built for %d earnings calls.", len(session_map))

    files = sorted(glob.glob(os.path.join(args.layer2_dir, "*_layer2.csv")))
    log.info("Found %d Layer 2 files.", len(files))

    valid_acc    = {w: [] for w in POST_WINDOWS}
    excluded_acc = {w: [] for w in POST_WINDOWS}
    coverage     = []

    for path in files:
        valid_frames, excluded_frames = process_ec(
            path, args.sentiment_dir, session_map,
            args.min_ticks_post, args.anchor_type, log,
        )
        stem  = os.path.basename(path).replace("_layer2.csv", "")
        parts = stem.rsplit("_", 2)
        tic, year, quarter = (
            (parts[0], int(parts[1]), parts[2]) if len(parts) == 3
            else ("unknown", 0, "unknown")
        )
        session = session_map.get((tic, year, quarter), "unknown")

        for w in POST_WINDOWS:
            if w in valid_frames and not valid_frames[w].empty:
                valid_acc[w].append(valid_frames[w])
            if w in excluded_frames and not excluded_frames[w].empty:
                excluded_acc[w].append(excluded_frames[w])

        n_valid    = len(valid_frames.get(30, pd.DataFrame()))
        n_excluded = len(excluded_frames.get(30, pd.DataFrame()))
        coverage.append({
            "tic": tic, "year": year, "quarter": quarter,
            "ec_session": session,
            "n_valid_30s": n_valid,
            "n_excluded_30s": n_excluded,
        })

    anchor_types = ["pre", "qa"] if args.anchor_type == "all" else [args.anchor_type]
    output_cols  = {"pre": PRE_OUTPUT_COLS, "qa": QA_OUTPUT_COLS}

    for w in POST_WINDOWS:
        if not valid_acc[w]:
            continue
        combined = pd.concat(valid_acc[w], ignore_index=True)

        for atype in anchor_types:
            sub = combined[combined["anchor_type"] == atype]
            if sub.empty:
                continue
            cols    = [c for c in output_cols[atype] if c in sub.columns]
            out     = sub[cols]
            outpath = os.path.join(
                args.output_dir, f"benchmark_{atype}_{w}s.csv"
            )
            out.to_csv(outpath, index=False, encoding="utf-8")
            log.info("Saved %s (%d rows, %d cols).",
                     outpath, len(out), len(cols))

    for w in POST_WINDOWS:
        if not excluded_acc[w]:
            continue
        excl      = pd.concat(excluded_acc[w], ignore_index=True)
        excl_path = os.path.join(
            args.output_dir, f"excluded_anchors_{w}s.csv"
        )
        excl.to_csv(excl_path, index=False, encoding="utf-8")
        log.info("Excluded anchors: %s (%d rows).", excl_path, len(excl))

    cov_df = pd.DataFrame(coverage)
    cov_df.to_csv(
        os.path.join(args.output_dir, "benchmark_coverage.csv"),
        index=False, encoding="utf-8",
    )

    if valid_acc[30]:
        n_v = len(pd.concat(valid_acc[30], ignore_index=True))
        n_e = len(pd.concat(excluded_acc[30], ignore_index=True)) if excluded_acc[30] else 0
        log.info("Done. Valid anchors (post=30s): %d  Excluded: %d", n_v, n_e)


if __name__ == "__main__":
    main()
