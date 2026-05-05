"""
run_baselines_b0b1b2.py

Main baseline experiments for the MERIT benchmark.

Three representation regimes:
    Inst. (B0): instantaneous utterance — current sentence tone only
                Pre:  finberttone_expected_value
                Q&A:  {dim}_positive_score for each SubjECTive-QA dimension

    CL    (B1): cumulative level — expanding mean, no trajectory shape
                Pre:  ev_expanding_mean
                Q&A:  {dim}_expanding_mean

    CT    (B2): cumulative trajectory — Inst. + CL + shape statistics
                Pre:  finberttone_expected_value, ev_expanding_mean/std/max/min,
                      n_change_points, finberttone_cumulative_tone  (7 features)
                Q&A:  {dim}_positive_score + {dim}_expanding_mean/std  (18 features)

The document-level aggregate (Agg) regime is evaluated separately via
run_static_ec_level.py and achieves BAcc = 0.500 across all configurations.

Usage:
    # Presentation segment
    python experiment/run_baselines_b0b1b2.py \
        --task        pre \
        --split_dir   /path/to/benchmark_split \
        --labeled_dir /path/to/benchmark/benchmark_labeled \
        --output      results/task1_results.csv

    # Q&A segment
    python experiment/run_baselines_b0b1b2.py \
        --task        qa \
        --split_dir   /path/to/benchmark_split \
        --labeled_dir /path/to/benchmark/benchmark_labeled \
        --output      results/task2_results.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

DIMS = ['assertive', 'cautious', 'optimistic', 'specific', 'clear', 'relevant']

# Presentation segment (pre)
B0_PRE = ['finberttone_expected_value']
B1_PRE = ['ev_expanding_mean']
B2_PRE = [
    'finberttone_expected_value',
    'ev_expanding_mean', 'ev_expanding_std',
    'ev_expanding_max',  'ev_expanding_min',
    'n_change_points',   'finberttone_cumulative_tone',
]

# Q&A segment (qa)
B0_QA = [f'{d}_positive_score' for d in DIMS]
B1_QA = [f'{d}_expanding_mean' for d in DIMS]
B2_QA = (
    [f'{d}_positive_score' for d in DIMS] +
    [f'{d}_expanding_mean'  for d in DIMS] +
    [f'{d}_expanding_std'   for d in DIMS]
)

TARGETS = {
    'BAS':  'label_bas_q75',
    'QRF':  'label_qrf_q75',
    'QVol': 'label_qvol_q75',
}

WINDOWS   = [30, 60, 120, 300]
MODELS    = ['LR', 'SVM', 'RF', 'XGB']
FEAT_SETS = ['B0', 'B1', 'B2']

# Skip the first two QA pairs per call: expanding statistics require
# at least two prior observations to be meaningful.
QA_MIN_RANK = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_non_overlapping(df: pd.DataFrame, W: int) -> pd.DataFrame:
    """
    Retain anchors such that no two anchors share any portion of their
    post-window, preventing target autocorrelation.
    """
    selected = []
    for _, grp in df.groupby(['tic', 'year', 'quarter'], sort=False):
        ts     = grp.sort_values('timestamp_anchor')['timestamp_anchor'].values
        idx    = grp.sort_values('timestamp_anchor').index.values
        last_t = -np.inf
        for t, i in zip(ts, idx):
            if t - last_t >= W:
                selected.append(i)
                last_t = t
    return df.loc[selected].reset_index(drop=True)


def add_qa_expanding(df: pd.DataFrame) -> pd.DataFrame:
    """Compute expanding mean and std of SubjECTive-QA scores per call."""
    df = df.copy()
    for d in DIMS:
        col = f'{d}_positive_score'
        df[f'{d}_expanding_mean'] = df.groupby(
            ['tic', 'year', 'quarter'])[col].transform(
            lambda x: x.expanding().mean())
        df[f'{d}_expanding_std'] = df.groupby(
            ['tic', 'year', 'quarter'])[col].transform(
            lambda x: x.expanding().std().fillna(0))
    return df

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(name: str, spw: float = 1.0):
    """
    All classifiers use fixed hyperparameters across all tasks and windows.
    LR (ℓ2, C=1.0, max_iter=1000, class_weight=balanced);
    SVM (RBF kernel, class_weight=balanced);
    RF  (300 trees, max_depth=4, min_samples_leaf=10, class_weight=balanced);
    XGB (300 rounds, max_depth=3, learning_rate=0.05, subsample=0.8,
         scale_pos_weight set to train-set negative-to-positive ratio).
    """
    if name == 'LR':
        return LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, class_weight='balanced'
        )
    if name == 'SVM':
        return SVC(
            kernel='rbf', probability=True, random_state=42, class_weight='balanced'
        )
    if name == 'RF':
        return RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=1,
        )
    return XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42, n_jobs=1,
        eval_metric='logloss', verbosity=0,
        scale_pos_weight=spw,
    )

# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_one(args):
    task, W, tname, label_col, model_name, feat_set, \
        b0_feats, b1_feats, b2_feats, split_dir, labeled_dir = args

    try:
        train_split = pd.read_csv(f'{split_dir}/train/benchmark_{task}_{W}s.csv')
        test_split  = pd.read_csv(f'{split_dir}/test/benchmark_{task}_{W}s.csv')
        label_df    = pd.read_csv(f'{labeled_dir}/benchmark_{task}_{W}s.csv')

        key           = ['tic', 'year', 'quarter', 'anchor_id']
        label_cols    = [c for c in label_df.columns if c.startswith('label_')]

        train_raw = train_split.merge(label_df[key + label_cols], on=key, how='left')
        test_raw  = test_split.merge( label_df[key + label_cols], on=key, how='left')

        if task == 'qa':
            train_raw = add_qa_expanding(train_raw)
            test_raw  = add_qa_expanding(test_raw)
            train_raw['_qa_rank'] = (
                train_raw.groupby(['tic', 'year', 'quarter']).cumcount() + 1
            )
            test_raw['_qa_rank'] = (
                test_raw.groupby(['tic', 'year', 'quarter']).cumcount() + 1
            )
            train_raw = train_raw[
                train_raw['_qa_rank'] >= QA_MIN_RANK
            ].drop(columns=['_qa_rank'])
            test_raw = test_raw[
                test_raw['_qa_rank'] >= QA_MIN_RANK
            ].drop(columns=['_qa_rank'])

        train_noo = apply_non_overlapping(train_raw, W)
        test_noo  = apply_non_overlapping(test_raw,  W)

        feats = {'B0': b0_feats, 'B1': b1_feats, 'B2': b2_feats}[feat_set]

        cols = feats + [label_col]
        tr   = train_noo[cols].dropna()
        te   = test_noo[cols].dropna()

        if len(tr) < 50 or len(te) < 20:
            return None
        if len(np.unique(tr[label_col].values)) < 2:
            return None

        X_tr = tr[feats].values
        X_te = te[feats].values
        y_tr = tr[label_col].values
        y_te = te[label_col].values

        if model_name in ('LR', 'SVM'):
            sc   = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        spw   = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)
        model = build_model(model_name, spw=spw)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc  = accuracy_score(y_te, y_pred)
        bacc = balanced_accuracy_score(y_te, y_pred)
        try:
            auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        except Exception:
            auc = np.nan

        return {
            'task':     task,
            'W':        W,
            'target':   tname,
            'model':    model_name,
            'feat_set': feat_set,
            'label':    'q75',
            'n_train':  len(tr),
            'n_test':   len(te),
            'pos_rate': round(float(y_te.mean()), 6),
            'acc':      round(acc,  6),
            'bacc':     round(bacc, 6),
            'auc':      round(auc,  6) if not np.isnan(auc) else np.nan,
        }

    except Exception as e:
        return {
            'error': str(e), 'task': task, 'W': W,
            'target': tname, 'model': model_name, 'feat_set': feat_set,
        }

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    print(f"\n=== Mean BAcc & AUC across 3 targets (Q75, non-overlapping) ===\n")
    print(
        f"{'Model':<6} {'W':>5}  "
        f"{'B0_bacc':>8} {'B1_bacc':>8} {'B2_bacc':>8} {'B2-B1':>7}  "
        f"{'B0_auc':>8} {'B1_auc':>8} {'B2_auc':>8} {'B2-B1':>7}"
    )
    print('-' * 88)

    for W in WINDOWS:
        for m in MODELS:
            b0 = df[(df['W'] == W) & (df['model'] == m) & (df['feat_set'] == 'B0')]
            b1 = df[(df['W'] == W) & (df['model'] == m) & (df['feat_set'] == 'B1')]
            b2 = df[(df['W'] == W) & (df['model'] == m) & (df['feat_set'] == 'B2')]
            if len(b0) == 0 or len(b1) == 0 or len(b2) == 0:
                continue
            b0b = b0['bacc'].mean(); b1b = b1['bacc'].mean(); b2b = b2['bacc'].mean()
            b0a = b0['auc'].mean();  b1a = b1['auc'].mean();  b2a = b2['auc'].mean()
            print(
                f"{m:<6} {W:>5}  "
                f"{b0b:>8.4f} {b1b:>8.4f} {b2b:>8.4f} {100*(b2b-b1b):>+7.2f}pp  "
                f"{b0a:>8.4f} {b1a:>8.4f} {b2a:>8.4f} {100*(b2a-b1a):>+7.2f}pp"
            )
        print()

    print("=== LR per-target @ W=300 ===")
    print(
        f"  {'Target':<6} "
        f"{'B0_bacc':>8} {'B1_bacc':>8} {'B2_bacc':>8} {'B2-B1':>7}  "
        f"{'B0_auc':>8} {'B1_auc':>8} {'B2_auc':>8} {'B2-B1':>7}"
    )
    print(f"  {'-'*85}")
    for tname in TARGETS:
        b0 = df[(df['W']==300)&(df['model']=='LR')&(df['feat_set']=='B0')&(df['target']==tname)]
        b1 = df[(df['W']==300)&(df['model']=='LR')&(df['feat_set']=='B1')&(df['target']==tname)]
        b2 = df[(df['W']==300)&(df['model']=='LR')&(df['feat_set']=='B2')&(df['target']==tname)]
        if len(b0) == 0 or len(b1) == 0 or len(b2) == 0:
            continue
        print(
            f"  {tname:<6} "
            f"{b0['bacc'].values[0]:>8.4f} {b1['bacc'].values[0]:>8.4f} "
            f"{b2['bacc'].values[0]:>8.4f} "
            f"{100*(b2['bacc'].values[0]-b1['bacc'].values[0]):>+7.2f}pp  "
            f"{b0['auc'].values[0]:>8.4f} {b1['auc'].values[0]:>8.4f} "
            f"{b2['auc'].values[0]:>8.4f} "
            f"{100*(b2['auc'].values[0]-b1['auc'].values[0]):>+7.2f}pp  "
            f"n_tr={b2['n_train'].values[0]} n_te={b2['n_test'].values[0]}"
        )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run MERIT baseline experiments (Inst. / CL / CT regimes).'
    )
    parser.add_argument('--task',        required=True, choices=['pre', 'qa'])
    parser.add_argument('--split_dir',   required=True,
                        help='Path to benchmark_split/ directory '
                             '(contains train/ and test/ subdirectories).')
    parser.add_argument('--labeled_dir', required=True,
                        help='Path to benchmark/benchmark_labeled/ directory.')
    parser.add_argument('--output',      required=True,
                        help='Output CSV path for results.')
    parser.add_argument('--workers',     type=int, default=min(8, cpu_count()))
    args = parser.parse_args()

    task  = args.task
    b0_f  = B0_PRE if task == 'pre' else B0_QA
    b1_f  = B1_PRE if task == 'pre' else B1_QA
    b2_f  = B2_PRE if task == 'pre' else B2_QA

    done = set()
    if os.path.exists(args.output):
        done_df = pd.read_csv(args.output)
        done    = set(zip(done_df['W'], done_df['target'],
                          done_df['model'], done_df['feat_set']))
        print(f"Resuming: {len(done)} configs already done.")
    else:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        print("Starting fresh.")

    configs = [
        (task, W, tname, label_col, model_name, feat_set,
         b0_f, b1_f, b2_f, args.split_dir, args.labeled_dir)
        for W                in WINDOWS
        for tname, label_col in TARGETS.items()
        for model_name       in MODELS
        for feat_set         in FEAT_SETS
        if (W, tname, model_name, feat_set) not in done
    ]

    total = len(WINDOWS) * len(TARGETS) * len(MODELS) * len(FEAT_SETS)
    print(f"Total configs: {total}  Remaining: {len(configs)}  Workers: {args.workers}")

    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_one, configs)):
            if result is None:
                continue
            if 'error' in result:
                print(f"  ERROR [{i+1}]: {result}")
                continue

            row_df       = pd.DataFrame([result])
            write_header = not os.path.exists(args.output)
            row_df.to_csv(args.output, mode='a', header=write_header, index=False)

            print(
                f"[{i+1}/{len(configs)}] "
                f"W={result['W']:3d} {result['target']:5s} "
                f"{result['model']:4s} {result['feat_set']}  "
                f"bacc={result['bacc']:.4f} "
                f"auc={result.get('auc', float('nan')):.4f} "
                f"pos={result['pos_rate']:.1%} "
                f"n_tr={result['n_train']} n_te={result['n_test']}"
            )

    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        print_summary(df)


if __name__ == '__main__':
    main()