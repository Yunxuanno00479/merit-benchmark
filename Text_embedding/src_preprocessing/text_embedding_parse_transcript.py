"""
parse_transcripts.py

Parse raw earnings call transcript JSONL files into two panel formats:

  1. Presentation panel (sentence-level):
     One row per sentence spoken during the Presentation section.
     Moderator/Operator boilerplate is excluded.
     Output: {output_dir}/pre/{TICKER}_{YEAR}_{QUARTER}_pre.csv

  2. QA panel (two outputs per file):

     a. QA-pair level:
        One row per analyst question + executive answer pair.
        Moderator turn announcements are excluded.
        Output: {output_dir}/qa/{TICKER}_{YEAR}_{QUARTER}_qa.csv

     b. QA sentence-level:
        One row per sentence within each QA exchange, with qa_index and
        sentence_role (question / answer) to link back to the pair level.
        Output: {output_dir}/qa_sentence/{TICKER}_{YEAR}_{QUARTER}_qa_sentence.csv

Input:
    JSONL files where each line is one JSON object:
    {
        "ticker": str,
        "year": int,
        "quarter": int,
        "transcript": [
            {
                "speaker_id": str,
                "speaker_name": str,
                "speaker_title": str,
                "text": [
                    {"sentence": str, "timestamp": float, "pre_or_qa": "Pre" | "QA"},
                    ...
                ]
            },
            ...
        ]
    }

Moderator identification:
    A speaker is classified as Moderator/Operator if their title contains any of
    MODERATOR_TITLE_KEYWORDS. These speakers are excluded from the Presentation
    panel and from QA pairing (their turn announcements are not anchor sentences).

QA pairing logic:
    Within the QA section, sentences are sorted by timestamp and grouped into
    exchanges. An exchange boundary is defined as: a new ANALYST speaker begins
    speaking after at least one EXEC sentence has been seen since the last boundary.
    Within each exchange, all ANALYST sentences concatenated form the Question
    and all EXEC sentences concatenated form the Answer.

Usage:
    python parse_transcripts.py \\
        --input_dir  /path/to/jsonl_files \\
        --output_dir /path/to/transcript_panel \\
        [--ec_calendar /path/to/ec_calendar.csv]

Requirements:
    pandas >= 2.0
"""

import argparse
import csv
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODERATOR_TITLE_KEYWORDS = [
    "moderator",
    "operator",
    "facilitator",
    "host",
]

EXEC_TITLE_KEYWORDS = [
    "ceo",
    "cfo",
    "president",
    "vice president",
    "director",
    "investor relations",
    "universal relations",
    "chief",
]

# Columns for each output file, matching the existing _pre.csv and _qa.csv schemas.
PRE_COLUMNS = [
    "tic", "year", "quarter",
    "timestamp_p", "section", "section_id",
    "speaker_name_P", "speaker_title_P",
    "presentation_text",
]

QA_PAIR_COLUMNS = [
    "tic", "year", "quarter",
    "qa_index",
    "Question", "Answer",
    "Q_Timestamp", "A_Timestamp",
    "q_speaker_name", "q_speaker_title",
    "a_speaker_name", "a_speaker_title",
]

QA_SENTENCE_COLUMNS = [
    "tic", "year", "quarter",
    "qa_index", "sentence_index",
    "sentence_role",        # "question" or "answer"
    "timestamp",
    "speaker_name", "speaker_title",
    "sentence",
]


# ---------------------------------------------------------------------------
# Speaker role classification
# ---------------------------------------------------------------------------

def classify_speaker(title: str) -> str:
    """
    Return "moderator", "exec", "analyst", or "unknown" based on speaker title.

    "unknown" is returned when title is empty or None; callers decide how to
    handle this case depending on section context (Pre vs QA).
    """
    t = (title or "").strip().lower()
    if not t:
        return "unknown"
    if any(k in t for k in MODERATOR_TITLE_KEYWORDS):
        return "moderator"
    if any(k in t for k in EXEC_TITLE_KEYWORDS):
        return "exec"
    return "analyst"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Sentence:
    timestamp: float
    sentence: str
    speaker_name: str
    speaker_title: str
    role: str           # "moderator" | "exec" | "analyst"
    section: str        # "Pre" | "QA"


@dataclass
class QAPair:
    qa_index: int
    q_sentences: list = field(default_factory=list)
    a_sentences: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_sentences(record: dict) -> list:
    """
    Flatten all sentences from a transcript record into a sorted list of
    Sentence objects.
    """
    sentences = []
    for spk in record.get("transcript", []):
        name  = spk.get("speaker_name") or ""
        title = spk.get("speaker_title") or ""
        role  = classify_speaker(title)
        for item in spk.get("text", []):
            sentences.append(Sentence(
                timestamp    = float(item["timestamp"]),
                sentence     = item["sentence"].strip(),
                speaker_name = name,
                speaker_title= title,
                role         = role,
                section      = item.get("pre_or_qa", ""),
            ))
    sentences.sort(key=lambda s: s.timestamp)
    return sentences


# ---------------------------------------------------------------------------
# Presentation panel
# ---------------------------------------------------------------------------

def build_pre_rows(
    sentences: list,
    tic: str,
    year: int,
    quarter: int,
) -> list:
    """
    Return presentation panel rows (sentence-level, moderator excluded).
    section_id is a 1-based integer index over non-moderator Pre sentences.
    """
    rows = []
    section_id = 0
    for s in sentences:
        if s.section != "Pre":
            continue
        # Exclude moderators and unknown-title speakers from the Pre panel.
        if s.role in ("moderator", "unknown"):
            continue
        section_id += 1
        rows.append({
            "tic":              tic,
            "year":             year,
            "quarter":          quarter,
            "timestamp_p":      s.timestamp,
            "section":          "Pre",
            "section_id":       section_id,
            "speaker_name_P":   s.speaker_name,
            "speaker_title_P":  s.speaker_title,
            "presentation_text": s.sentence,
        })
    return rows


# ---------------------------------------------------------------------------
# QA pairing
# ---------------------------------------------------------------------------

def build_qa_pairs(sentences: list) -> list:
    """
    Group QA sentences into QAPair objects.

    Pairing rule:
        - Moderator sentences are skipped entirely.
        - An exchange starts when an ANALYST sentence is encountered.
        - The exchange ends when a new ANALYST sentence appears after at least
          one EXEC sentence has been collected (i.e., after the answer has begun).
        - This handles multi-sentence questions and multi-sentence answers,
          including cases where multiple executives respond.
    """
    qa_sents = [s for s in sentences if s.section == "QA" and s.role != "moderator"]

    pairs = []
    current_pair = None
    answer_started = False

    for s in qa_sents:
        if s.role in ("analyst", "unknown"):
            if current_pair is None:
                # Start of first exchange.
                current_pair = QAPair(qa_index=1)
                answer_started = False
            elif answer_started:
                # New analyst question after an answer: close current, open new.
                pairs.append(current_pair)
                current_pair = QAPair(qa_index=len(pairs) + 1)
                answer_started = False
            # Append to question.
            current_pair.q_sentences.append(s)

        elif s.role == "exec":
            if current_pair is None:
                # Exec speaks before any analyst question (e.g. opening remark
                # or supplementary comment). Attach to the last completed pair
                # if one exists, otherwise skip.
                if pairs:
                    pairs[-1].a_sentences.append(s)
                continue
            current_pair.a_sentences.append(s)
            answer_started = True

    # Close the last open pair.
    if current_pair is not None and (current_pair.q_sentences or current_pair.a_sentences):
        pairs.append(current_pair)

    return pairs


def build_qa_pair_rows(
    pairs: list,
    tic: str,
    year: int,
    quarter: int,
) -> list:
    """
    Convert QAPair objects into QA-pair level rows matching the _qa.csv schema.

    Q_Timestamp: timestamp of the first analyst sentence in the question.
    A_Timestamp: timestamp of the first exec sentence in the answer.
    """
    rows = []
    for pair in pairs:
        q_text = " ".join(s.sentence for s in pair.q_sentences)
        a_text = " ".join(s.sentence for s in pair.a_sentences)

        q_ts = pair.q_sentences[0].timestamp if pair.q_sentences else None
        a_ts = pair.a_sentences[0].timestamp if pair.a_sentences else None

        q_name  = pair.q_sentences[0].speaker_name  if pair.q_sentences else ""
        q_title = pair.q_sentences[0].speaker_title if pair.q_sentences else ""
        a_name  = pair.a_sentences[0].speaker_name  if pair.a_sentences else ""
        a_title = pair.a_sentences[0].speaker_title if pair.a_sentences else ""

        rows.append({
            "tic":             tic,
            "year":            year,
            "quarter":         quarter,
            "qa_index":        pair.qa_index,
            "Question":        q_text,
            "Answer":          a_text,
            "Q_Timestamp":     q_ts,
            "A_Timestamp":     a_ts,
            "q_speaker_name":  q_name,
            "q_speaker_title": q_title,
            "a_speaker_name":  a_name,
            "a_speaker_title": a_title,
        })
    return rows


def build_qa_sentence_rows(
    pairs: list,
    tic: str,
    year: int,
    quarter: int,
) -> list:
    """
    Convert QAPair objects into sentence-level QA rows.
    sentence_index is 1-based within each qa_index.
    sentence_role is "question" for analyst sentences, "answer" for exec sentences.
    """
    rows = []
    for pair in pairs:
        # Merge question and answer sentences, sorted by timestamp.
        all_sents = (
            [(s, "question") for s in pair.q_sentences] +
            [(s, "answer")   for s in pair.a_sentences]
        )
        all_sents.sort(key=lambda x: x[0].timestamp)

        for idx, (s, role) in enumerate(all_sents, start=1):
            rows.append({
                "tic":           tic,
                "year":          year,
                "quarter":       quarter,
                "qa_index":      pair.qa_index,
                "sentence_index": idx,
                "sentence_role": role,
                "timestamp":     s.timestamp,
                "speaker_name":  s.speaker_name,
                "speaker_title": s.speaker_title,
                "sentence":      s.sentence,
            })
    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_csv(rows: list, columns: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record(
    record: dict,
    output_dir: str,
    log: logging.Logger,
) -> None:
    tic     = record["ticker"]
    year    = int(record["year"])
    quarter = int(record["quarter"])
    stem    = f"{tic}_{year}_Q{quarter}"

    sentences = extract_sentences(record)

    # Warn if this transcript has no recognisable Pre or QA labels.
    # This indicates an upstream data quality issue (e.g. pre_or_qa = "Unknown").
    known_sections = {s.section for s in sentences}
    if "Pre" not in known_sections and "QA" not in known_sections:
        log.warning(
            "%s: no sentences with section 'Pre' or 'QA' found "
            "(found sections: %s). Skipping.",
            stem, known_sections,
        )
        return

    # --- Presentation panel ---
    pre_rows = build_pre_rows(sentences, tic, year, quarter)
    pre_path = os.path.join(output_dir, "pre", f"{stem}_pre.csv")
    write_csv(pre_rows, PRE_COLUMNS, pre_path)

    # --- QA pair panel ---
    pairs = build_qa_pairs(sentences)
    qa_rows = build_qa_pair_rows(pairs, tic, year, quarter)
    qa_path = os.path.join(output_dir, "qa", f"{stem}_qa.csv")
    write_csv(qa_rows, QA_PAIR_COLUMNS, qa_path)

    # --- QA sentence panel ---
    qa_sent_rows = build_qa_sentence_rows(pairs, tic, year, quarter)
    qa_sent_path = os.path.join(output_dir, "qa_sentence", f"{stem}_qa_sentence.csv")
    write_csv(qa_sent_rows, QA_SENTENCE_COLUMNS, qa_sent_path)

    log.info(
        "%s  pre=%d sentences  qa=%d pairs  qa_sentence=%d rows",
        stem, len(pre_rows), len(qa_rows), len(qa_sent_rows),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse earnings call transcript JSONL files into panel CSVs."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing *.jsonl transcript files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root directory for output CSV files.",
    )
    return parser.parse_args()


def setup_logging(output_dir: str) -> logging.Logger:
    log_path = os.path.join(output_dir, "logs", "parse_transcripts.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def main() -> None:
    args    = parse_args()
    log     = setup_logging(args.output_dir)
    files   = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.jsonl"), recursive=True))

    if not files:
        log.error("No .jsonl files found in %s", args.input_dir)
        sys.exit(1)

    log.info("Found %d JSONL files.", len(files))
    total_records = 0

    for path in files:
        records = load_jsonl(path)
        for record in records:
            try:
                process_record(record, args.output_dir, log)
                total_records += 1
            except Exception:
                log.exception(
                    "Error processing %s year=%s quarter=%s",
                    record.get("ticker"), record.get("year"), record.get("quarter"),
                )

    log.info("Done. Processed %d transcript records from %d files.",
             total_records, len(files))


if __name__ == "__main__":
    main()