"""
compute_trajectory_features.py

Compute sentence-level trajectory features for earnings call presentations.

NOTE: This script computes an extended set of trajectory features for
exploratory analysis (EMA, momentum, distributional statistics, etc.).
The MERIT benchmark evaluation uses only the five expanding-window features
computed within build_benchmark_dataset.py:
    ev_expanding_mean, ev_expanding_std, ev_expanding_max,
    ev_expanding_min, n_change_points.

This script is provided for researchers who wish to experiment with
richer trajectory representations beyond the released benchmark features.

Input:
    sentiment_dir : sentiment_panel/pre/   -- TIC_YEAR_Q{Q}_pre_score.csv
    transcript_dir: transcript_panel/pre/  -- TIC_YEAR_Q{Q}_pre.csv

Output:
    output_dir    : trajectory_features/pre/ -- TIC_YEAR_Q{Q}_trajectory.csv

Usage:
    python compute_trajectory_features.py \
        --sentiment_dir  /path/to/sentiment_panel/pre \
        --transcript_dir /path/to/transcript_panel/pre \
        --output_dir     /path/to/trajectory_features/pre \
        --years          2021-2023
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _ema_last(values: np.ndarray, alpha: float) -> float:
    v = values[0]
    for x in values[1:]:
        v = alpha * x + (1 - alpha) * v
    return v


def _stats(values: np.ndarray) -> dict:
    if len(values) == 0:
        return dict(mean=0.0, std=0.0, max=0.0, min=0.0, range=0.0)
    return dict(
        mean=float(np.mean(values)),
        std=float(np.std(values)) if len(values) > 1 else 0.0,
        max=float(np.max(values)),
        min=float(np.min(values)),
        range=float(np.max(values) - np.min(values)),
    )


def compute_features_for_ec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trajectory features for one earnings call.
    Requires: finberttone_expected_value, finberttone_change_point,
              tic, year, quarter, section_id.
    Optional: timestamp_p, speaker_title_P.
    """
    scores = df["finberttone_expected_value"].values.astype(float)
    cps    = df["finberttone_change_point"].values.astype(int)
    n      = len(scores)

    has_ts  = "timestamp_p" in df.columns
    has_spk = "speaker_title_P" in df.columns
    timestamps = df["timestamp_p"].values if has_ts else None
    speakers   = df["speaker_title_P"].values if has_spk else None

    rows = []
    for i in range(n):
        f = {
            "tic":           df.iloc[i]["tic"],
            "year":          int(df.iloc[i]["year"]),
            "quarter":       df.iloc[i]["quarter"],
            "section_id":    int(df.iloc[i]["section_id"]),
            "sentence_idx":  i,
            "current_score": float(scores[i]),
        }
        if has_ts:
            f["timestamp_p"] = float(timestamps[i])
        if has_spk:
            title = str(speakers[i]).upper()
            f["is_ceo"] = int("CEO" in title)
            f["is_cfo"] = int("CFO" in title)

        # Recent history (zero-padded)
        for w in (5, 10, 20):
            hist = scores[max(0, i - w + 1): i + 1]
            pad  = [0.0] * (w - len(hist))
            for lag, val in enumerate(pad + list(hist)):
                f[f"recent_{w}_lag{lag}"] = float(val)

        # Multi-scale summary statistics
        for w in (5, 10):
            window = scores[max(0, i - w + 1): i + 1]
            for k, v in _stats(window).items():
                f[f"summary_{w}_{k}"] = v
        for k, v in _stats(scores[: i + 1]).items():
            f[f"summary_all_{k}"] = v

        # Cross-scale
        f["short_vs_long_mean"]  = f["summary_5_mean"]  - f["summary_all_mean"]
        f["medium_vs_long_mean"] = f["summary_10_mean"] - f["summary_all_mean"]
        f["short_vs_long_std"]   = f["summary_5_std"]   - f["summary_all_std"]

        # Momentum
        for lag in (1, 5, 10):
            delta = float(scores[i] - scores[i - lag]) if i >= lag else 0.0
            f[f"delta_{lag}"] = delta
            f[f"rate_{lag}"]  = delta / lag if lag > 0 else 0.0
        if i > 0:
            ema5  = _ema_last(scores[: i + 1], 2 / 6)
            ema20 = _ema_last(scores[: i + 1], 2 / 21)
        else:
            ema5 = ema20 = float(scores[i])
        f["ema_5"]  = ema5
        f["ema_20"] = ema20
        f["macd"]   = ema5 - ema20

        # Volatility
        for w in (5, 10, 20):
            window = scores[max(0, i - w + 1): i + 1]
            f[f"rolling_std_{w}"] = float(np.std(window)) if len(window) > 1 else 0.0
        f["vol_change_5to10"] = (
            f["rolling_std_5"] - f["rolling_std_10"] if i >= 10 else 0.0
        )
        for w in (5, 10):
            start = max(1, i - w + 1)
            rv    = sum((scores[j] - scores[j - 1]) ** 2 for j in range(start, i + 1))
            f[f"realized_vol_{w}"] = float(rv)

        # Change point
        f["changepoint_count"]       = int(np.sum(cps[: i + 1]))
        cp_idx = np.where(cps[: i + 1] == 1)[0]
        f["sentences_since_last_cp"] = int(i - cp_idx[-1]) if len(cp_idx) > 0 else i

        # Distributional (last 20 sentences)
        window = scores[max(0, i - 19): i + 1]
        if len(window) >= 4:
            f["skewness"] = float(skew(window))
            f["kurtosis"] = float(kurtosis(window))
            f["q25"]      = float(np.percentile(window, 25))
            f["q75"]      = float(np.percentile(window, 75))
            f["iqr"]      = f["q75"] - f["q25"]
        else:
            f.update(skewness=0.0, kurtosis=0.0, q25=0.0, q75=0.0, iqr=0.0)

        # Context
        f["history_length"]      = i
        f["is_early_anchor"]     = int(i < 10)
        f["normalized_position"] = i / n if n > 0 else 0.0
        if has_ts and i > 0:
            f["time_since_start"] = float(timestamps[i] - timestamps[0])
        else:
            f["time_since_start"] = 0.0

        rows.append(f)

    return pd.DataFrame(rows)


def process_file(sentiment_path: Path, transcript_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(sentiment_path)

    required = {
        "tic", "year", "quarter", "section_id",
        "finberttone_expected_value", "finberttone_change_point",
    }
    if not required.issubset(df.columns):
        tqdm.write(f"  missing columns in {sentiment_path.name}, skipping")
        return None

    transcript_name = sentiment_path.name.replace("_pre_score.csv", "_pre.csv")
    transcript_path = transcript_dir / transcript_name
    if transcript_path.exists():
        tr   = pd.read_csv(transcript_path)
        cols = ["section_id"] + [
            c for c in ("timestamp_p", "speaker_title_P") if c in tr.columns
        ]
        df = df.merge(tr[cols], on="section_id", how="left")

    return compute_features_for_ec(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment_dir",  required=True)
    parser.add_argument("--transcript_dir", required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--years",          default="2021-2023")
    args = parser.parse_args()

    if "-" in args.years:
        s, e  = map(int, args.years.split("-"))
        years = set(range(s, e + 1))
    else:
        years = {int(args.years)}

    sentiment_dir  = Path(args.sentiment_dir)
    transcript_dir = Path(args.transcript_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(sentiment_dir.glob("*_pre_score.csv"))
    files = [f for f in files if int(f.stem.split("_")[1]) in years]
    print(f"Files to process: {len(files)}")

    ok, err = 0, 0
    for path in tqdm(files, desc="Computing features"):
        out_name = path.name.replace("_pre_score.csv", "_trajectory.csv")
        out_path = output_dir / out_name
        if out_path.exists():
            ok += 1
            continue
        result = process_file(path, transcript_dir)
        if result is not None:
            result.to_csv(out_path, index=False)
            ok += 1
        else:
            err += 1

    print(f"Done. success={ok}, errors={err}")


if __name__ == "__main__":
    main()