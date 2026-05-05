"""
compute_finbert_tone.py

Compute FinBERT-tone sentiment scores for earnings call transcript panels.

Model:
    yiyanghkust/finbert-tone (Huang et al., 2023)
    Labels: Positive, Neutral, Negative
    ev_i = P(Positive) - P(Negative)  in [-1, 1]

Supported panels:
    pre         -- Presentation section, one row per sentence
    qa_sentence -- QA section, one row per sentence

Output columns (pre panel):
    tic, year, quarter, section_id,
    finberttone_expected_value,     # ev_i = P(Pos) - P(Neg)
    finberttone_cumulative_tone,    # running sum of ev_i within the EC
    finberttone_change_point        # placeholder (-1); set by compute_changepoint.py

Usage:
    python compute_finbert_tone.py \
        --panel      pre \
        --input_dir  /path/to/transcript_panel/pre \
        --output_dir /path/to/sentiment_panel/pre \
        --device     0

Note: This script requires access to the raw transcript panel, which is not
publicly released due to commercial licensing restrictions on the transcript
data source.
"""

import argparse
import csv
import glob
import logging
import os
import sys

import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    pipeline,
)


FINBERT_MODEL = "yiyanghkust/finbert-tone"

PANEL_CONFIGS = {
    "pre": {
        "file_pattern":   "*_pre.csv",
        "input_suffix":   "_pre.csv",
        "output_suffix":  "_pre_score.csv",
        "text_col":       "presentation_text",
        "id_cols":        ["tic", "year", "quarter", "section_id"],
        "output_cols":    [
            "tic", "year", "quarter", "section_id",
            "finberttone_expected_value",
            "finberttone_cumulative_tone",
            "finberttone_change_point",
        ],
    },
    "qa_sentence": {
        "file_pattern":   "*_qa_sentence.csv",
        "input_suffix":   "_qa_sentence.csv",
        "output_suffix":  "_qa_sentence_score.csv",
        "text_col":       "sentence",
        "id_cols":        ["tic", "year", "quarter", "qa_index", "sentence_index"],
        "output_cols":    [
            "tic", "year", "quarter", "qa_index", "sentence_index",
            "finberttone_expected_value",
            "finberttone_cumulative_tone",
            "finberttone_change_point",
        ],
    },
}


def setup_logging(output_dir):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "compute_finbert_tone.log")
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


def load_model(device):
    """
    Load FinBERT-tone using explicit BertTokenizer and
    BertForSequenceClassification to avoid model_type lookup errors.
    """
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
    model     = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=device,
    )
    return clf


def run_inference(texts, clf, batch_size):
    """
    Run FinBERT-tone inference. Returns list of ev values: P(Pos) - P(Neg).
    """
    ev_list = []
    for i in range(0, len(texts), batch_size):
        batch       = texts[i : i + batch_size]
        predictions = clf(batch, truncation=True, max_length=512)
        for pred in predictions:
            score_map = {item["label"]: item["score"] for item in pred}
            pos = score_map.get("Positive", 0.0)
            neg = score_map.get("Negative", 0.0)
            ev_list.append(pos - neg)
    return ev_list


def process_file(input_path, output_path, clf, config, batch_size, log):
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        log.warning("%s: empty file, skipping.", input_path)
        return

    texts   = [row[config["text_col"]] for row in rows]
    ev_list = run_inference(texts, clf, batch_size)

    cumulative = 0.0
    cum_list   = []
    for ev in ev_list:
        cumulative += ev
        cum_list.append(cumulative)

    out_rows = []
    for row, ev, cum_val in zip(rows, ev_list, cum_list):
        out_row = {col: row[col] for col in config["id_cols"]}
        out_row["finberttone_expected_value"]  = round(ev, 9)
        out_row["finberttone_cumulative_tone"] = round(cum_val, 9)
        out_row["finberttone_change_point"]    = -1   # filled by compute_changepoint.py
        out_rows.append(out_row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config["output_cols"])
        writer.writeheader()
        writer.writerows(out_rows)

    log.info("%s: %d sentences scored.", os.path.basename(input_path), len(out_rows))


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute FinBERT-tone scores for transcript panels."
    )
    p.add_argument("--panel",      required=True, choices=["pre", "qa_sentence"])
    p.add_argument("--input_dir",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Inference batch size (default: 32).")
    p.add_argument("--device",     type=int, default=0,
                   help="GPU index. Use -1 for CPU (default: 0).")
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    config = PANEL_CONFIGS[args.panel]
    log    = setup_logging(args.output_dir)

    checkpoint_path = args.checkpoint or os.path.join(
        args.output_dir, f"checkpoint_finbert_{args.panel}.txt"
    )
    done = load_checkpoint(checkpoint_path)

    device = args.device if torch.cuda.is_available() else -1
    log.info("Loading %s on device=%d ...", FINBERT_MODEL, device)
    clf = load_model(device)

    files = sorted(
        glob.glob(os.path.join(args.input_dir, config["file_pattern"]))
    )
    log.info("Found %d input files.", len(files))

    for input_path in files:
        stem = os.path.basename(input_path)
        if stem in done:
            log.info("Skip (already done): %s", stem)
            continue
        output_stem = stem.replace(config["input_suffix"], config["output_suffix"])
        output_path = os.path.join(args.output_dir, output_stem)
        try:
            process_file(input_path, output_path, clf, config, args.batch_size, log)
            mark_done(checkpoint_path, stem)
        except Exception:
            log.exception("Error processing %s.", stem)

    log.info("Done.")


if __name__ == "__main__":
    main()