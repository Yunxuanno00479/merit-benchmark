"""
add_labels.py

Add binary labels to benchmark CSV files (both pre and qa).
Three label schemes per target:
  - label_<target>_binary : |delta| > 0  -> 1, else 0
  - label_<target>_q75    : |delta| > global Q75 -> 1, else 0
  - label_<target>_q90    : |delta| > global Q90 -> 1, else 0

Labels are based on the ABSOLUTE value of delta to capture extreme
microstructure movements in either direction, consistent with the
research question of predicting extreme market reactions rather than
directional changes.

Global quantile thresholds are computed from the training split only
(years 2021-2022) to prevent data leakage, then applied to all files.

Note: OBI (order book imbalance) is excluded from benchmark targets.
Only BAS, QRF, and QVol are evaluated in the MERIT benchmark.

Usage:
    python add_labels.py \
        --benchmark_dir /path/to/benchmark \
        --out_dir       /path/to/benchmark/benchmark_labeled
"""

import argparse
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# OBI (order book imbalance) excluded from benchmark targets;
# only BAS, QRF, and QVol are evaluated.
TARGETS = {
    'bas':  ('bid_ask_spread_mean_pre',   'bid_ask_spread_mean_post'),
    'qrf':  ('qrf_mean_pre',              'qrf_mean_post'),
    'qvol': ('quote_volatility_mean_pre', 'quote_volatility_mean_post'),
}

POST_WINDOWS = [30, 60, 120, 300]
TYPES        = ['pre', 'qa']
TRAIN_YEARS  = [2021, 2022]  # thresholds computed on train split only to prevent leakage

# ---------------------------------------------------------------------------
# Step 1: compute global quantile thresholds from training split only
# ---------------------------------------------------------------------------

def compute_global_thresholds(benchmark_dir: str) -> dict:
    """
    Load all benchmark CSVs from the training split (2021-2022),
    compute |delta| for each target, pool across post windows and
    anchor types, return {target: {'q75': float, 'q90': float}}.
    """
    all_deltas = {key: [] for key in TARGETS}

    for t in TYPES:
        for s in POST_WINDOWS:
            fpath = os.path.join(benchmark_dir, f'benchmark_{t}_{s}s.csv')
            if not os.path.exists(fpath):
                print(f'  [skip] {fpath} not found')
                continue
            df = pd.read_csv(fpath)

            # restrict to training years only
            if 'year' in df.columns:
                df = df[df['year'].isin(TRAIN_YEARS)]

            for key, (pre_col, post_col) in TARGETS.items():
                if pre_col in df.columns and post_col in df.columns:
                    abs_delta = (df[post_col] - df[pre_col]).abs().dropna().values
                    all_deltas[key].append(abs_delta)

    thresholds = {}
    print('\n=== Global |delta| quantile thresholds (train split 2021-2022) ===\n')
    print(f'{"target":<8} {"n_total":>10} {"Q75":>12} {"Q90":>12}')
    print('-' * 46)
    for key, arrays in all_deltas.items():
        if not arrays:
            continue
        pooled = np.concatenate(arrays)
        q75 = float(np.percentile(pooled, 75))
        q90 = float(np.percentile(pooled, 90))
        thresholds[key] = {'q75': q75, 'q90': q90}
        print(f'{key:<8} {len(pooled):>10,} {q75:>12.6f} {q90:>12.6f}')

    return thresholds

# ---------------------------------------------------------------------------
# Step 2: add label columns to each file
# ---------------------------------------------------------------------------

def add_labels_to_file(fpath: str, out_path: str, thresholds: dict):
    df = pd.read_csv(fpath)

    for key, (pre_col, post_col) in TARGETS.items():
        if pre_col not in df.columns or post_col not in df.columns:
            continue

        delta     = df[post_col] - df[pre_col]
        abs_delta = delta.abs()

        df[f'label_{key}_binary'] = (abs_delta > 0).astype(int)
        df[f'label_{key}_q75']    = (abs_delta > thresholds[key]['q75']).astype(int)
        df[f'label_{key}_q90']    = (abs_delta > thresholds[key]['q90']).astype(int)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    label_cols = [c for c in df.columns if c.startswith('label_')]
    print(f'\n  {os.path.basename(fpath)}  (n={len(df):,})')
    print(f'  {"label":<30} {"pos_rate":>9}')
    print(f'  {"-"*41}')
    for col in label_cols:
        print(f'  {col:<30} {df[col].mean():>9.1%}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Add Q75 extreme-event labels to MERIT benchmark files.'
    )
    parser.add_argument('--benchmark_dir', required=True,
                        help='Directory containing benchmark_pre_*.csv and '
                             'benchmark_qa_*.csv files.')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for labeled benchmark files.')
    args = parser.parse_args()

    print('Step 1: computing global thresholds from training split...')
    thresholds = compute_global_thresholds(args.benchmark_dir)

    print('\nStep 2: adding labels to all benchmark files...')
    for t in TYPES:
        for s in POST_WINDOWS:
            fname    = f'benchmark_{t}_{s}s.csv'
            fpath    = os.path.join(args.benchmark_dir, fname)
            if not os.path.exists(fpath):
                print(f'  [skip] {fname} not found')
                continue
            out_path = os.path.join(args.out_dir, fname)
            add_labels_to_file(fpath, out_path, thresholds)

    print('\nDone. Labeled files saved to:', args.out_dir)

if __name__ == '__main__':
    main()