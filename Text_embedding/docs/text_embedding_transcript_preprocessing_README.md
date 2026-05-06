# Transcript Processing

This directory parses raw earnings call transcript files into three panel
formats used by the benchmark.

## Directory Structure

```
transcript/
    parse_transcripts.py    Parse JSONL transcripts into CSV panels
    sample_transcript.jsonl Example of the required input format
    README.md               This file
```

## Prerequisites

```bash
pip install pandas>=2.0
```

## Input Format

Input files are JSONL files (one JSON object per line). Each object represents
one earnings call and must follow the schema below. A minimal example is
provided in `sample_transcript.jsonl`.

```json
{
    "ticker":   "AAPL",
    "year":     2021,
    "quarter":  1,
    "transcript": [
        {
            "speaker_id":    "spk_0",
            "speaker_name":  "Operator",
            "speaker_title": "Conference Call Operator",
            "text": [
                {
                    "sentence":   "Welcome to the earnings call.",
                    "timestamp":  4.7,
                    "pre_or_qa":  "Pre"
                }
            ]
        }
    ]
}
```

The `pre_or_qa` field must be `"Pre"`, `"QA"`, or `"Unknown"`. Earnings calls
where all sentences have `"Unknown"` are skipped with a warning in the log.

Since raw transcripts cannot be released due to licensing restrictions, users
must supply their own transcript data in the format above.

## Usage

```bash
python parse_transcripts.py \
    --input_dir  /path/to/jsonl_files \
    --output_dir /path/to/transcript_panel
```

The `--input_dir` is searched recursively for `*.jsonl` files, so year-based
subdirectories (e.g. `2021/`, `2022/`, `2023/`) are handled automatically.

## Output

Three subdirectories are created under `--output_dir`:

```
transcript_panel/
    pre/
        {TICKER}_{YEAR}_Q{Q}_pre.csv
    qa/
        {TICKER}_{YEAR}_Q{Q}_qa.csv
    qa_sentence/
        {TICKER}_{YEAR}_Q{Q}_qa_sentence.csv
    logs/
        parse_transcripts.log
```

### pre CSV columns

| Column | Description |
|--------|-------------|
| `tic` | Ticker symbol |
| `year` | Earnings call year |
| `quarter` | Quarter (integer) |
| `timestamp_p` | Sentence start time in seconds from call start |
| `section` | Always `"Pre"` |
| `section_id` | 1-based index within this earnings call's presentation |
| `speaker_name_P` | Speaker name |
| `speaker_title_P` | Speaker title |
| `presentation_text` | Sentence text |

### qa CSV columns

| Column | Description |
|--------|-------------|
| `tic` | Ticker symbol |
| `year` | Year |
| `quarter` | Quarter |
| `qa_index` | 1-based QA pair index within this earnings call |
| `Question` | Concatenated analyst sentences |
| `Answer` | Concatenated executive sentences |
| `Q_Timestamp` | Timestamp of the first analyst sentence (seconds) |
| `A_Timestamp` | Timestamp of the first executive sentence (seconds) |
| `q_speaker_name` | Name of the analyst |
| `q_speaker_title` | Title of the analyst |
| `a_speaker_name` | Name of the first executive to respond |
| `a_speaker_title` | Title of the first executive to respond |

### qa_sentence CSV columns

| Column | Description |
|--------|-------------|
| `tic` | Ticker symbol |
| `year` | Year |
| `quarter` | Quarter |
| `qa_index` | QA pair index (links back to qa CSV) |
| `sentence_index` | 1-based sentence index within the QA pair |
| `sentence_role` | `"question"` or `"answer"` |
| `timestamp` | Sentence start time in seconds |
| `speaker_name` | Speaker name |
| `speaker_title` | Speaker title |
| `sentence` | Sentence text |

## Speaker Classification Rules

| Role | Identified by |
|------|---------------|
| Moderator | Title contains: `moderator`, `operator`, `facilitator`, `host` |
| Executive | Title contains: `ceo`, `cfo`, `president`, `vice president`, `director`, `investor relations`, `chief` |
| Analyst | Any speaker not classified as moderator or executive |
| Unknown title | Excluded from Pre panel; treated as analyst in QA panel |

Moderator sentences are excluded from all output panels.
Sentences containing financial disclaimer keywords (`forward-looking`,
`non-gaap`, `reconciliation`, `safe harbor`, etc.) are excluded from the
Pre panel regardless of speaker.

## QA Pairing Rules

- An exchange starts when an analyst sentence is encountered.
- The exchange closes when a new analyst speaks after at least one executive
  sentence has been collected.
- Executive sentences before any analyst question are merged into the
  preceding pair's Answer.
- Multi-turn exchanges (Q -> A -> Q -> A) produce one pair per analyst turn.