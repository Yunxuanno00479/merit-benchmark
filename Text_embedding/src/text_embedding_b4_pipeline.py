#!/usr/bin/env python3
"""
text_embedding_b4_pipeline.py

B4 text-embedding baseline on benchmark_split train/test files.

What this script does:
1) Load benchmark_split/{train,test}/benchmark_{pre,qa}_{W}s.csv
2) Build text input:
   - pre: presentation_text
   - qa : "Question: ... Answer: ..."
3) Generate embeddings with:
   - all-mpnet-base-v2
   - LLaMA-3-8B (mean pooling on last hidden states)
4) Train classifiers:
   - Logistic Regression (L2)
   - SVM (RBF kernel)
   - Random Forest (300 trees, max_depth=4)
   - XGBoost (300 rounds, max_depth=3, lr=0.05, subsample=0.8)
5) Evaluate metrics on test:
   - Accuracy
   - Balanced Accuracy
   - AUC
6) Save one CSV with all experiments.

Example:
python text_embedding_b4_pipeline.py \
  --benchmark_split_dir /path/to/benchmark_split \
  --output_csv /path/to/results/b4_text_embedding_results.csv
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


TARGET_SPECS = [
    ("bid_ask_spread_mean_pre", "bid_ask_spread_mean_post", "BAS"),
    ("obi_mean_pre", "obi_mean_post", "OBI"),
    ("qrf_mean_pre", "qrf_mean_post", "QRF"),
    ("quote_volatility_mean_pre", "quote_volatility_mean_post", "QVol"),
]
TARGET_LABEL_QUANTILE = 0.75

TASKS = ["pre", "qa"]
POST_WINDOWS = [30, 60, 120, 300]

EMBEDDING_MODELS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "openai_small512": "text-embedding-3-small",
}
MODEL_ALIASES = {}


@dataclass
class ExperimentRow:
    embedding_model: str
    embedding_hf_model: str
    task: str
    post_window_sec: int
    target: str
    classifier: str
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    auc: float


RESULT_FIELDS = [f.name for f in fields(ExperimentRow)]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_text_series(df: pd.DataFrame, task: str) -> pd.Series:
    if task == "pre":
        if "presentation_text" not in df.columns:
            raise KeyError("Missing column 'presentation_text' for task=pre")
        return df["presentation_text"].fillna("").astype(str)

    if task == "qa":
        if "Question" not in df.columns or "Answer" not in df.columns:
            raise KeyError("Missing 'Question' or 'Answer' for task=qa")
        q = df["Question"].fillna("").astype(str)
        a = df["Answer"].fillna("").astype(str)
        return ("Question: " + q + " Answer: " + a).astype(str)

    raise ValueError(f"Unknown task: {task}")


def compute_target_thresholds(train_df: pd.DataFrame, target_label_mode: str) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for pre_col, post_col, short_name in TARGET_SPECS:
        if pre_col not in train_df.columns or post_col not in train_df.columns:
            raise KeyError(f"Missing target columns: {pre_col}, {post_col}")
        delta = train_df[post_col] - train_df[pre_col]
        if target_label_mode == "q75_abs":
            delta = delta.abs()
        thresholds[short_name] = float(delta.quantile(TARGET_LABEL_QUANTILE))
    return thresholds


def add_targets(df: pd.DataFrame, thresholds: Dict[str, float], target_label_mode: str) -> pd.DataFrame:
    out = df.copy()
    for pre_col, post_col, short_name in TARGET_SPECS:
        if pre_col not in out.columns or post_col not in out.columns:
            raise KeyError(f"Missing target columns: {pre_col}, {post_col}")
        delta = out[post_col] - out[pre_col]
        if target_label_mode == "q75_abs":
            delta = delta.abs()
        out[f"Y_{short_name}"] = (delta > thresholds[short_name]).astype(int)
    return out


def load_train_test(
    benchmark_split_dir: str,
    task: str,
    post_window_sec: int,
    target_label_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(
        benchmark_split_dir, "train", f"benchmark_{task}_{post_window_sec}s.csv"
    )
    test_path = os.path.join(
        benchmark_split_dir, "test", f"benchmark_{task}_{post_window_sec}s.csv"
    )

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    thresholds = compute_target_thresholds(train_df, target_label_mode)
    train_df = add_targets(train_df, thresholds, target_label_mode)
    test_df = add_targets(test_df, thresholds, target_label_mode)
    return train_df, test_df


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def embed_with_mpnet(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def embed_with_llama(
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    hf_token: str,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            hidden = outputs.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)

            # Mean pooling over non-padding tokens.
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / counts

            emb = pooled.detach().cpu().numpy()
            all_emb.append(emb)

    out = np.vstack(all_emb).astype(np.float32)
    out = l2_normalize(out)
    return out


def embed_texts(
    texts: List[str],
    model_key: str,
    batch_size: int,
    max_length: int,
    hf_token: str,
    openai_api_key: str,
    mpnet_device: str,
    openai_max_retries: int,
    openai_retry_sleep_sec: float,
    openai_dimensions: Optional[int] = None,
    batch_ckpt_json: Optional[str] = None,
    batch_parts_dir: Optional[str] = None,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    model_key = MODEL_ALIASES.get(model_key, model_key)
    model_name = EMBEDDING_MODELS[model_key]
    if model_key == "mpnet":
        return embed_with_mpnet(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            device=mpnet_device,
        )
    if model_key == "llama3":
        return embed_with_llama(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            hf_token=hf_token,
        )
    if model_key == "openai_small512":
        return embed_with_openai(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            openai_api_key=openai_api_key,
            max_retries=openai_max_retries,
            retry_sleep_sec=openai_retry_sleep_sec,
            openai_dimensions=openai_dimensions,
            batch_ckpt_json=batch_ckpt_json,
            batch_parts_dir=batch_parts_dir,
            progress_desc=progress_desc,
        )
    raise ValueError(f"Unsupported embedding model: {model_key}")


def embed_with_openai(
    texts: List[str],
    model_name: str,
    batch_size: int,
    openai_api_key: str,
    max_retries: int,
    retry_sleep_sec: float,
    openai_dimensions: Optional[int] = None,
    batch_ckpt_json: Optional[str] = None,
    batch_parts_dir: Optional[str] = None,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    if not openai_api_key:
        raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or --openai_api_key")

    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key)
    all_rows = []
    total_texts = len(texts)
    batch_starts = list(range(0, total_texts, batch_size))
    total_batches = len(batch_starts)
    desc = progress_desc or f"OpenAI Embedding ({model_name})"

    completed_batches: Set[int] = set()
    if batch_ckpt_json and batch_parts_dir:
        os.makedirs(os.path.dirname(batch_ckpt_json), exist_ok=True)
        os.makedirs(batch_parts_dir, exist_ok=True)

        if os.path.exists(batch_ckpt_json):
            prev = load_json(batch_ckpt_json, {})
            same_shape = (
                prev.get("total_texts") == total_texts
                and prev.get("batch_size") == batch_size
                and prev.get("total_batches") == total_batches
            )
            if same_shape:
                completed_batches = set(prev.get("completed_batches", []))

        for b in list(completed_batches):
            part_path = os.path.join(batch_parts_dir, f"batch_{b:06d}.npy")
            if not os.path.exists(part_path):
                completed_batches.remove(b)

        save_json(
            batch_ckpt_json,
            {
                "total_texts": total_texts,
                "batch_size": batch_size,
                "total_batches": total_batches,
                "completed_batches": sorted(completed_batches),
                "updated_at": now_iso(),
            },
        )

    pbar = tqdm(
        batch_starts,
        total=total_batches,
        desc=desc,
        unit="batch",
        initial=len(completed_batches),
    )
    for batch_idx, i in enumerate(batch_starts):
        if batch_idx in completed_batches:
            continue

        batch = texts[i : i + batch_size]
        attempt = 0
        while True:
            try:
                req = {
                    "model": model_name,
                    "input": batch,
                }
                if openai_dimensions is not None and model_name.startswith("text-embedding-3"):
                    req["dimensions"] = int(openai_dimensions)
                res = client.embeddings.create(**req)
                batch_rows = [
                    np.asarray(item.embedding, dtype=np.float32)
                    for item in res.data
                ]
                batch_arr = np.vstack(batch_rows).astype(np.float32)
                if batch_parts_dir:
                    part_path = os.path.join(batch_parts_dir, f"batch_{batch_idx:06d}.npy")
                    np.save(part_path, batch_arr)
                else:
                    all_rows.extend(batch_rows)

                completed_batches.add(batch_idx)
                if batch_ckpt_json:
                    save_json(
                        batch_ckpt_json,
                        {
                            "total_texts": total_texts,
                            "batch_size": batch_size,
                            "total_batches": total_batches,
                            "completed_batches": sorted(completed_batches),
                            "updated_at": now_iso(),
                        },
                    )
                pbar.update(1)
                break
            except Exception as exc:
                msg = str(exc)
                is_rate_limit = (
                    ("429" in msg)
                    or ("rate limit" in msg.lower())
                    or ("too many requests" in msg.lower())
                )
                if not is_rate_limit:
                    raise

                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"OpenAI embedding failed after {max_retries} retries "
                        f"for batch starting at index {i}"
                    ) from exc

                wait_sec = retry_sleep_sec
                m = re.search(r"retry in\\s+([0-9]+(?:\\.[0-9]+)?)s", msg, re.IGNORECASE)
                if m:
                    wait_sec = max(wait_sec, float(m.group(1)))

                logging.warning(
                    "OpenAI embedding rate-limited at batch index %d (attempt %d/%d). "
                    "Sleep %.1fs then retry.",
                    i,
                    attempt,
                    max_retries,
                    wait_sec,
                )
                time.sleep(wait_sec)
    pbar.close()

    if batch_parts_dir:
        for b in range(total_batches):
            part_path = os.path.join(batch_parts_dir, f"batch_{b:06d}.npy")
            if not os.path.exists(part_path):
                raise RuntimeError(f"Missing batch part file: {part_path}")
            all_rows.append(np.load(part_path))

    out = np.vstack(all_rows)
    out = l2_normalize(out)
    return out


def openai_batch_paths(
    checkpoint_dir: str,
    checkpoint_prefix: str,
    task: str,
    window: int,
    split_name: str,
) -> Tuple[str, str]:
    safe = f"{checkpoint_prefix}_openai_small512_{task}_{window}s_{split_name}"
    ckpt_json = os.path.join(checkpoint_dir, f"{safe}_batch_ckpt.json")
    parts_dir = os.path.join(checkpoint_dir, f"{safe}_parts")
    return ckpt_json, parts_dir


def build_classifiers(seed: int):
    models = {
        "LR": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        max_iter=3000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
            ]
        ),
        "RF": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    try:
        import xgboost as xgb

        models["XGB"] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    except Exception as exc:
        logging.warning("XGBoost unavailable, skip XGB model: %s", exc)

    return models


def stratified_subsample_indices(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac >= 1.0:
        return np.arange(len(y))
    if frac <= 0:
        raise ValueError("train_subsample_frac must be > 0")

    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_keep = max(1, int(round(len(cls_idx) * frac)))
        n_keep = min(n_keep, len(cls_idx))
        keep.append(rng.choice(cls_idx, size=n_keep, replace=False))

    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def safe_auc(model, x_test: np.ndarray, y_test: np.ndarray) -> float:
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_test)[:, 1]
            return float(roc_auc_score(y_test, prob))
        if hasattr(model, "decision_function"):
            score = model.decision_function(x_test)
            return float(roc_auc_score(y_test, score))
    except Exception:
        return float("nan")
    return float("nan")


def run_one_setting(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: str,
    window: int,
    model_key: str,
    emb_train: np.ndarray,
    emb_test: np.ndarray,
    seed: int,
    completed_result_keys: Set[str],
    on_result_done,
    enabled_classifiers: List[str],
    train_subsample_frac: float,
    classifier_parallel_jobs: int,
) -> List[ExperimentRow]:
    rows: List[ExperimentRow] = []
    models = build_classifiers(seed)
    model_order = [m for m in ["LR", "SVM", "RF", "XGB"] if m in enabled_classifiers and m in models]
    jobs = []

    for _, _, target_short in TARGET_SPECS:
        target_col = f"Y_{target_short}"

        y_train_full = train_df[target_col].values
        y_test = test_df[target_col].values

        if len(np.unique(y_train_full)) < 2 or len(np.unique(y_test)) < 2:
            logging.warning(
                "Skip task=%s W=%s target=%s due to single class in train/test",
                task,
                window,
                target_col,
            )
            continue

        train_idx = stratified_subsample_indices(y_train_full, train_subsample_frac, seed)
        y_train = y_train_full[train_idx]
        x_train = emb_train[train_idx]
        if len(np.unique(y_train)) < 2:
            logging.warning(
                "Skip task=%s W=%s target=%s due to single class after subsample",
                task,
                window,
                target_col,
            )
            continue

        for clf_name in model_order:
            result_key = f"{model_key}|{task}|{window}|{target_col}|{clf_name}"
            if result_key in completed_result_keys:
                continue
            jobs.append((clf_name, result_key, target_col, x_train, y_train, y_test))

    def fit_one(job):
        clf_name, result_key, target_col, x_train, y_train, y_test = job
        clf = build_classifiers(seed)[clf_name]
        clf.fit(x_train, y_train)
        pred = clf.predict(emb_test)

        acc = float(accuracy_score(y_test, pred))
        bacc = float(balanced_accuracy_score(y_test, pred))
        auc = safe_auc(clf, emb_test, y_test)

        row = ExperimentRow(
            embedding_model=model_key,
            embedding_hf_model=EMBEDDING_MODELS[model_key],
            task=task,
            post_window_sec=window,
            target=target_col,
            classifier=clf_name,
            n_train=int(len(y_train)),
            n_test=int(len(y_test)),
            accuracy=acc,
            balanced_accuracy=bacc,
            auc=auc,
        )
        return row, result_key

    if classifier_parallel_jobs > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=classifier_parallel_jobs) as executor:
            futures = [executor.submit(fit_one, job) for job in jobs]
            for future in as_completed(futures):
                row, result_key = future.result()
                rows.append(row)
                on_result_done(row, result_key)
    else:
        for job in jobs:
            row, result_key = fit_one(job)
            rows.append(row)
            on_result_done(row, result_key)

    return rows


def maybe_load_cache(path: str) -> np.ndarray:
    if os.path.exists(path):
        return np.load(path)
    return None


def save_cache(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def cache_paths(cache_dir: str, model_key: str, task: str, window: int) -> Tuple[str, str]:
    train_path = os.path.join(cache_dir, f"{model_key}_{task}_{window}s_train.npy")
    test_path = os.path.join(cache_dir, f"{model_key}_{task}_{window}s_test.npy")
    return train_path, test_path


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_json(path: str, default_obj: dict) -> dict:
    if not os.path.exists(path):
        return default_obj
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_result_row(csv_path: str, row: ExperimentRow) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B4 text embedding baselines")
    parser.add_argument(
        "--benchmark_split_dir",
        type=str,
        required=True,
        help="Path to benchmark_split directory containing train/ and test/",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output results CSV",
    )
    parser.add_argument(
        "--embedding_models",
        nargs="+",
        default=["mpnet", "llama3"],
        choices=["mpnet", "llama3", "openai_small512"],
        help="Embedding model keys to run",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS,
        choices=TASKS,
        help="Tasks to run",
    )
    parser.add_argument(
        "--post_windows",
        nargs="+",
        type=int,
        default=POST_WINDOWS,
        help="Post-window seconds to run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--llama_max_length",
        type=int,
        default=512,
        help="Max token length for LLaMA embedding",
    )
    parser.add_argument(
        "--mpnet_device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for all-mpnet-base-v2 embedding",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace token (or use HF_TOKEN env var)",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (or use OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--openai_embedding_model",
        type=str,
        default="text-embedding-3-small",
        choices=["text-embedding-3-small", "text-embedding-3-large"],
        help="OpenAI embedding model used by embedding_models=openai_small512",
    )
    parser.add_argument(
        "--openai_embedding_dimensions",
        type=int,
        default=512,
        help="Embedding dimensions for OpenAI text-embedding-3 models",
    )
    parser.add_argument(
        "--openai_max_retries",
        type=int,
        default=1000,
        help="Max retries per OpenAI batch when rate-limited (429)",
    )
    parser.add_argument(
        "--openai_retry_sleep_sec",
        type=float,
        default=20.0,
        help="Base sleep seconds for OpenAI rate-limit retries",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional cache dir for embeddings (npy files)",
    )
    parser.add_argument(
        "--cache_model_alias",
        type=str,
        default=None,
        help="Optional model name used only for embedding cache filenames",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="full",
        choices=["full", "embedding_only"],
        help="Run full pipeline or embedding stage only",
    )
    parser.add_argument(
        "--classifier_models",
        nargs="+",
        default=["LR", "SVM", "RF", "XGB"],
        choices=["LR", "SVM", "RF", "XGB"],
        help="Classifier models to train in Stage 2",
    )
    parser.add_argument(
        "--train_subsample_frac",
        type=float,
        default=1.0,
        help="Fraction of training rows used for classifier fitting (stratified by label)",
    )
    parser.add_argument(
        "--classifier_parallel_jobs",
        type=int,
        default=1,
        help="Number of target/classifier jobs to train in parallel within each setting",
    )
    parser.add_argument(
        "--target_label_mode",
        type=str,
        default="q75_positive",
        choices=["q75_positive", "q75_abs"],
        help="Target label rule: q75_positive uses delta > Q75(delta); q75_abs uses abs(delta) > Q75(abs(delta))",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory for checkpoint files (default: output_csv/checkpoints)",
    )
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default=None,
        help="Checkpoint file prefix (default: output_csv basename)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)
    args.embedding_models = [MODEL_ALIASES.get(m, m) for m in args.embedding_models]
    EMBEDDING_MODELS["openai_small512"] = args.openai_embedding_model

    out_dir = os.path.dirname(args.output_csv) or "."
    ckpt_dir = args.checkpoint_dir or os.path.join(out_dir, "checkpoints")
    prefix = args.checkpoint_prefix or os.path.splitext(os.path.basename(args.output_csv))[0]
    emb_ckpt_path = os.path.join(ckpt_dir, f"{prefix}_embedding_ckpt.json")
    cls_ckpt_path = os.path.join(ckpt_dir, f"{prefix}_classifier_ckpt.json")
    partial_csv_path = os.path.join(ckpt_dir, f"{prefix}_partial_results.csv")

    emb_ckpt = load_json(
        emb_ckpt_path,
        {"completed_keys": [], "updated_at": None, "total_planned": 0},
    )
    cls_ckpt = load_json(
        cls_ckpt_path,
        {"completed_keys": [], "updated_at": None, "total_planned": 0},
    )
    completed_embedding_keys: Set[str] = set(emb_ckpt.get("completed_keys", []))
    completed_result_keys: Set[str] = set(cls_ckpt.get("completed_keys", []))

    if "llama3" in args.embedding_models and not args.hf_token:
        logging.warning(
            "No HF token provided. LLaMA-3-8B may fail if access is gated."
        )
    if "openai_small512" in args.embedding_models and not args.openai_api_key:
        logging.warning(
            "No OpenAI API key provided. OpenAI embedding will fail if cache is unavailable."
        )
    if args.mpnet_device == "cuda":
        logging.info("MPNet embedding device forced to CUDA.")
    if args.train_subsample_frac < 1.0:
        logging.info("Classifier train_subsample_frac=%.3f", args.train_subsample_frac)
    if args.classifier_parallel_jobs > 1:
        logging.info("Classifier parallel jobs=%d", args.classifier_parallel_jobs)
    logging.info("Target label mode=%s", args.target_label_mode)

    results: List[ExperimentRow] = []
    embedding_total = len(args.embedding_models) * len(args.tasks) * len(args.post_windows)
    emb_ckpt["total_planned"] = embedding_total
    save_json(emb_ckpt_path, emb_ckpt)

    # Stage 1: build all embeddings first.
    logging.info("=== Stage 1/2: Build embedding cache ===")
    plan: List[Tuple[str, str, int]] = []
    for task in args.tasks:
        for window in args.post_windows:
            logging.info("Load data for embedding: task=%s window=%ss", task, window)
            train_df, test_df = load_train_test(
                args.benchmark_split_dir,
                task,
                window,
                args.target_label_mode,
            )
            train_texts = build_text_series(train_df, task).tolist()
            test_texts = build_text_series(test_df, task).tolist()

            for req_model_key in args.embedding_models:
                model_key = req_model_key
                embedding_key = f"{model_key}|{task}|{window}"
                emb_train = None
                emb_test = None

                c_train, c_test = None, None
                cache_model_key = args.cache_model_alias or model_key
                if args.cache_dir:
                    c_train, c_test = cache_paths(args.cache_dir, cache_model_key, task, window)
                    emb_train = maybe_load_cache(c_train)
                    emb_test = maybe_load_cache(c_test)
                    if emb_train is not None and emb_test is not None:
                        logging.info("Use embedding cache: %s / %s", c_train, c_test)

                if (
                    emb_train is not None
                    and emb_test is not None
                    and embedding_key in completed_embedding_keys
                ):
                    logging.info(
                        "Embedding checkpoint hit: %s (%d/%d)",
                        embedding_key,
                        len(completed_embedding_keys),
                        embedding_total,
                    )
                elif emb_train is None or emb_test is None:
                    logging.info(
                        "Embedding start: model=%s task=%s W=%s (train=%d test=%d)",
                        model_key,
                        task,
                        window,
                        len(train_texts),
                        len(test_texts),
                    )
                    train_batch_ckpt_json, train_batch_parts_dir = None, None
                    test_batch_ckpt_json, test_batch_parts_dir = None, None
                    if model_key == "openai_small512":
                        train_batch_ckpt_json, train_batch_parts_dir = openai_batch_paths(
                            checkpoint_dir=ckpt_dir,
                            checkpoint_prefix=prefix,
                            task=task,
                            window=window,
                            split_name="train",
                        )
                        test_batch_ckpt_json, test_batch_parts_dir = openai_batch_paths(
                            checkpoint_dir=ckpt_dir,
                            checkpoint_prefix=prefix,
                            task=task,
                            window=window,
                            split_name="test",
                        )
                    try:
                        emb_train = embed_texts(
                            train_texts,
                            model_key=model_key,
                            batch_size=args.batch_size,
                            max_length=args.llama_max_length,
                            hf_token=args.hf_token,
                            openai_api_key=args.openai_api_key,
                            mpnet_device=args.mpnet_device,
                            openai_max_retries=args.openai_max_retries,
                            openai_retry_sleep_sec=args.openai_retry_sleep_sec,
                            openai_dimensions=args.openai_embedding_dimensions,
                            batch_ckpt_json=train_batch_ckpt_json,
                            batch_parts_dir=train_batch_parts_dir,
                            progress_desc=f"OpenAI Embedding {task}_{window}s train",
                        )
                        emb_test = embed_texts(
                            test_texts,
                            model_key=model_key,
                            batch_size=args.batch_size,
                            max_length=args.llama_max_length,
                            hf_token=args.hf_token,
                            openai_api_key=args.openai_api_key,
                            mpnet_device=args.mpnet_device,
                            openai_max_retries=args.openai_max_retries,
                            openai_retry_sleep_sec=args.openai_retry_sleep_sec,
                            openai_dimensions=args.openai_embedding_dimensions,
                            batch_ckpt_json=test_batch_ckpt_json,
                            batch_parts_dir=test_batch_parts_dir,
                            progress_desc=f"OpenAI Embedding {task}_{window}s test",
                        )
                    except Exception as exc:
                        if model_key == "llama3" and args.openai_api_key:
                            logging.warning(
                                "LLaMA embedding failed (%s). Fallback to OpenAI embedding.",
                                exc,
                            )
                            model_key = "openai_small512"
                            train_batch_ckpt_json, train_batch_parts_dir = openai_batch_paths(
                                checkpoint_dir=ckpt_dir,
                                checkpoint_prefix=prefix,
                                task=task,
                                window=window,
                                split_name="train",
                            )
                            test_batch_ckpt_json, test_batch_parts_dir = openai_batch_paths(
                                checkpoint_dir=ckpt_dir,
                                checkpoint_prefix=prefix,
                                task=task,
                                window=window,
                                split_name="test",
                            )
                            emb_train = embed_texts(
                                train_texts,
                                model_key=model_key,
                                batch_size=args.batch_size,
                                max_length=args.llama_max_length,
                                hf_token=args.hf_token,
                                openai_api_key=args.openai_api_key,
                                mpnet_device=args.mpnet_device,
                                openai_max_retries=args.openai_max_retries,
                                openai_retry_sleep_sec=args.openai_retry_sleep_sec,
                                openai_dimensions=args.openai_embedding_dimensions,
                                batch_ckpt_json=train_batch_ckpt_json,
                                batch_parts_dir=train_batch_parts_dir,
                                progress_desc=f"OpenAI Embedding {task}_{window}s train",
                            )
                            emb_test = embed_texts(
                                test_texts,
                                model_key=model_key,
                                batch_size=args.batch_size,
                                max_length=args.llama_max_length,
                                hf_token=args.hf_token,
                                openai_api_key=args.openai_api_key,
                                mpnet_device=args.mpnet_device,
                                openai_max_retries=args.openai_max_retries,
                                openai_retry_sleep_sec=args.openai_retry_sleep_sec,
                                openai_dimensions=args.openai_embedding_dimensions,
                                batch_ckpt_json=test_batch_ckpt_json,
                                batch_parts_dir=test_batch_parts_dir,
                                progress_desc=f"OpenAI Embedding {task}_{window}s test",
                            )
                        else:
                            raise

                    if args.cache_dir:
                        c_train, c_test = cache_paths(args.cache_dir, cache_model_key, task, window)
                        save_cache(c_train, emb_train)
                        save_cache(c_test, emb_test)

                    logging.info(
                        "Embedding done: model=%s shape(train=%s, test=%s)",
                        model_key,
                        emb_train.shape,
                        emb_test.shape,
                    )
                else:
                    logging.info(
                        "Use embedding cache (no checkpoint key yet): %s", embedding_key
                    )

                if embedding_key not in completed_embedding_keys:
                    completed_embedding_keys.add(embedding_key)
                    emb_ckpt["completed_keys"] = sorted(completed_embedding_keys)
                    emb_ckpt["updated_at"] = now_iso()
                    save_json(emb_ckpt_path, emb_ckpt)
                    logging.info(
                        "Embedding progress: %d/%d complete",
                        len(completed_embedding_keys),
                        embedding_total,
                    )

                plan.append((model_key, task, window))

    if args.run_mode == "embedding_only":
        logging.info("Embedding-only mode complete. Skip classifier training.")
        return

    # Stage 2: train classifiers from cached embeddings.
    logging.info("=== Stage 2/2: Train classifiers ===")
    available = build_classifiers(args.seed)
    enabled_classifier_names = [m for m in args.classifier_models if m in available]
    n_classifier_models = len(enabled_classifier_names)
    cls_total = len(set(plan)) * len(TARGET_SPECS) * n_classifier_models
    cls_ckpt["total_planned"] = cls_total
    save_json(cls_ckpt_path, cls_ckpt)

    def on_result_done(row: ExperimentRow, key: str) -> None:
        append_result_row(partial_csv_path, row)
        completed_result_keys.add(key)
        cls_ckpt["completed_keys"] = sorted(completed_result_keys)
        cls_ckpt["updated_at"] = now_iso()
        save_json(cls_ckpt_path, cls_ckpt)
        logging.info(
            "Classifier progress: %d/%d complete (%s)",
            len(completed_result_keys),
            cls_total,
            key,
        )
    seen = set()
    for model_key, task, window in plan:
        key = (model_key, task, window)
        if key in seen:
            continue
        seen.add(key)

        logging.info("Train start: model=%s task=%s window=%ss", model_key, task, window)
        train_df, test_df = load_train_test(
            args.benchmark_split_dir,
            task,
            window,
            args.target_label_mode,
        )

        if not args.cache_dir:
            raise RuntimeError("cache_dir is required for two-stage pipeline.")
        cache_model_key = args.cache_model_alias or model_key
        c_train, c_test = cache_paths(args.cache_dir, cache_model_key, task, window)
        emb_train = maybe_load_cache(c_train)
        emb_test = maybe_load_cache(c_test)
        if emb_train is None or emb_test is None:
            raise RuntimeError(
                f"Missing cached embeddings for model={model_key}, task={task}, window={window}"
            )

        rows = run_one_setting(
            train_df=train_df,
            test_df=test_df,
            task=task,
            window=window,
            model_key=model_key,
            emb_train=emb_train,
            emb_test=emb_test,
            seed=args.seed,
            completed_result_keys=completed_result_keys,
            on_result_done=on_result_done,
            enabled_classifiers=enabled_classifier_names,
            train_subsample_frac=args.train_subsample_frac,
            classifier_parallel_jobs=args.classifier_parallel_jobs,
        )
        results.extend(rows)

    if not results:
        raise RuntimeError("No experiment results generated.")

    if os.path.exists(partial_csv_path):
        out_df = pd.read_csv(partial_csv_path)
    else:
        out_df = pd.DataFrame([r.__dict__ for r in results])

    key_cols = ["embedding_model", "task", "post_window_sec", "target", "classifier"]
    if not out_df.empty:
        out_df = out_df.drop_duplicates(subset=key_cols, keep="last")
    out_df = out_df.sort_values(
        ["embedding_model", "task", "post_window_sec", "target", "classifier"]
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    logging.info("Saved results to %s", args.output_csv)


if __name__ == "__main__":
    main()
