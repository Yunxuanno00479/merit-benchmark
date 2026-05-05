"""
compute_subjective_qa.py

Compute SubjECTive-QA subjectivity scores for QA-pair panels.

Model:
    gtfintechlab/SubjECTiveQA-{FEATURE}  (Pardawala et al., 2024)
    Access: gated repository. Request access at:
            https://huggingface.co/gtfintechlab/SubjECTiveQA-ASSERTIVE

Six subjectivity dimensions:
    assertive, cautious, optimistic, specific, clear, relevant

Each dimension is a three-class classifier:
    0 = negatively demonstrative (negative_score)
    1 = neutral demonstration    (neutral_score)
    2 = positively demonstrative (positive_score)

The positive-class probability is used as the feature value in the
MERIT benchmark (B0 = current pair's positive_score per dimension).

Input format:
    CSV with columns: tic, year, quarter, qa_index, Question, Answer

    Input text per row: "Question: {question} Answer: {answer}"

Output columns:
    tic, year, quarter, qa_index,
    assertive_negative_score, assertive_neutral_score, assertive_positive_score,
    cautious_negative_score,  cautious_neutral_score,  cautious_positive_score,
    ...

Usage:
    huggingface-cli login   # required for gated model access

    python compute_subjective_qa.py \
        --input_dir  /path/to/transcript_panel/qa \
        --output_dir /path/to/sentiment_panel/qa_score \
        --device     0
"""

import argparse
import csv
import glob
import logging
import os
import sys

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


FEATURES = ["assertive", "cautious", "optimistic", "specific", "clear", "relevant"]

MODEL_IDS = {
    feature: f"gtfintechlab/SubjECTiveQA-{feature.upper()}"
    for feature in FEATURES
}

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

ID_COLS = ["tic", "year", "quarter", "qa_index"]

OUTPUT_COLS = ID_COLS + [
    f"{feature}_{label}_score"
    for feature in FEATURES
    for label in ("negative", "neutral", "positive")
]

MAX_LENGTH = 512


def setup_logging(output_dir):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "compute_subjective_qa.log")
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
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                stem = line.strip()
                if stem:
                    done.add(stem)
    return done


def mark_done(path, stem):
    with open(path, "a") as f:
        f.write(stem + "\n")


def build_input_text(question, answer):
    return f"Question: {question} Answer: {answer}"


def load_classifier(feature, device, log):
    model_id  = MODEL_IDS[feature]
    log.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, do_lower_case=True, do_basic_tokenize=True
    )
    model  = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
    config = AutoConfig.from_pretrained(model_id)
    clf = pipeline(
        "text-classification",
        model=model, tokenizer=tokenizer, config=config,
        framework="pt", top_k=None, device=device,
    )
    return clf


def score_feature(texts, clf, batch_size, log):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            predictions = clf(batch, truncation=True, max_length=MAX_LENGTH)
        except RuntimeError as exc:
            log.error("Inference error at batch %d: %s", i, exc)
            predictions = [[]] * len(batch)
        for pred in predictions:
            if pred:
                scores = {
                    LABEL_MAP.get(item["label"], item["label"]): round(item["score"], 6)
                    for item in pred
                }
            else:
                scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
            results.append(scores)
    return results


def process_file(input_path, output_path, device, batch_size, log):
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        log.warning("%s: empty file, skipping.", input_path)
        return

    texts = [build_input_text(row["Question"], row["Answer"]) for row in rows]

    # Score each feature dimension sequentially; load and release each model
    # to keep peak GPU memory to one model at a time.
    feature_scores = {}
    for feature in FEATURES:
        clf = load_classifier(feature, device, log)
        feature_scores[feature] = score_feature(texts, clf, batch_size, log)
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_rows = []
    for i, row in enumerate(rows):
        out_row = {col: row[col] for col in ID_COLS if col in row}
        for feature in FEATURES:
            scores = feature_scores[feature][i]
            for label in ("negative", "neutral", "positive"):
                out_row[f"{feature}_{label}_score"] = scores.get(label, 0.0)
        out_rows.append(out_row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        writer.writerows(out_rows)

    log.info("%s: %d QA pairs scored.", os.path.basename(input_path), len(out_rows))


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute SubjECTive-QA scores for QA-pair panels."
    )
    p.add_argument("--input_dir",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Inference batch size per model (default: 16).")
    p.add_argument("--device",     type=int, default=0,
                   help="GPU index. Use -1 for CPU (default: 0).")
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    log  = setup_logging(args.output_dir)

    checkpoint_path = args.checkpoint or os.path.join(
        args.output_dir, "checkpoint_subjective_qa.txt"
    )
    done   = load_checkpoint(checkpoint_path)
    device = args.device if torch.cuda.is_available() else -1
    log.info("Using device=%d.", device)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*_qa.csv")))
    log.info("Found %d QA-pair CSV files.", len(files))

    for input_path in files:
        stem = os.path.basename(input_path)
        if stem in done:
            log.info("Skip (already done): %s", stem)
            continue
        output_stem = stem.replace("_qa.csv", "_qa_score.csv")
        output_path = os.path.join(args.output_dir, output_stem)
        try:
            process_file(input_path, output_path, device, args.batch_size, log)
            mark_done(checkpoint_path, stem)
        except Exception:
            log.exception("Error processing %s.", stem)

    log.info("Done.")


if __name__ == "__main__":
    main()