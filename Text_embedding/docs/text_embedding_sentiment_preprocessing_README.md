# Sentiment Scoring

This directory computes sentiment scores for earnings call transcript panels
produced by `transcript/parse_transcripts.py`.

Three scripts cover three separate tasks and must be run in order.

## Directory Structure

```
sentiment/
    compute_finbert_tone.py     Step 1: FinBERT-tone scores for Pre and QA-sentence panels
    compute_subjective_qa.py    Step 2: SubjECTive-QA scores for QA-pair panel
    compute_changepoint.py      Step 3: PELT change point detection on FinBERT scores
    README.md                   This file
```

## Prerequisites

```bash
pip install torch>=2.0 transformers>=4.35 ruptures
```

SubjECTive-QA models are hosted in a gated repository on Hugging Face.
Request access before running Step 2:

- https://huggingface.co/gtfintechlab/SubjECTiveQA-ASSERTIVE
- https://huggingface.co/gtfintechlab/SubjECTiveQA-CAUTIOUS
- https://huggingface.co/gtfintechlab/SubjECTiveQA-OPTIMISTIC
- https://huggingface.co/gtfintechlab/SubjECTiveQA-SPECIFIC
- https://huggingface.co/gtfintechlab/SubjECTiveQA-CLEAR
- https://huggingface.co/gtfintechlab/SubjECTiveQA-RELEVANT

After access is granted, authenticate:

```bash
huggingface-cli login
```

## Step 1: FinBERT-tone (Pre and QA-sentence panels)

Model: `yiyanghkust/finbert-tone` (Huang et al., 2022)

Computes `expected_value = P(Positive) - P(Negative)` for each sentence and
a running `cumulative_tone` within each earnings call. The `change_point`
column is set to -1 as a placeholder and filled in Step 3.

```bash
# Presentation panel (GPU 0)
python compute_finbert_tone.py \
    --panel      pre \
    --input_dir  /path/to/transcript_panel/pre \
    --output_dir /path/to/sentiment_panel/pre \
    --device     0

# QA sentence panel (GPU 1, run in parallel)
python compute_finbert_tone.py \
    --panel      qa_sentence \
    --input_dir  /path/to/transcript_panel/qa_sentence \
    --output_dir /path/to/sentiment_panel/qa_sentence \
    --device     1
```

Output files: `{TICKER}_{YEAR}_Q{Q}_{panel}_score.csv`

Output columns (pre panel):

| Column | Description |
|--------|-------------|
| `tic`, `year`, `quarter`, `section_id` | Identity |
| `finberttone_expected_value` | P(Positive) - P(Negative) for this sentence |
| `finberttone_cumulative_tone` | Running sum of expected_value within this EC |
| `finberttone_change_point` | -1 until Step 3 is run |

Output columns (qa_sentence panel): same, with `qa_index` and `sentence_index`
replacing `section_id`.

## Step 2: SubjECTive-QA (QA-pair panel)

Model: `gtfintechlab/SubjECTiveQA-{FEATURE}` (Pardawala et al., NeurIPS 2024)

Scores each QA pair across six subjectivity dimensions. Each dimension has
three classes (0 = negative, 1 = neutral, 2 = positive); all three class
probabilities are recorded.

```bash
python compute_subjective_qa.py \
    --input_dir  /path/to/transcript_panel/qa \
    --output_dir /path/to/sentiment_panel/qa_score \
    --device     0
```

Output files: `{TICKER}_{YEAR}_Q{Q}_qa_score.csv`

Output columns:

| Column | Description |
|--------|-------------|
| `tic`, `year`, `quarter`, `qa_index` | Identity |
| `{dim}_negative_score` | P(class 0) for dimension `dim` |
| `{dim}_neutral_score` | P(class 1) for dimension `dim` |
| `{dim}_positive_score` | P(class 2) for dimension `dim` |

Where `dim` is one of: `assertive`, `cautious`, `optimistic`, `specific`,
`clear`, `relevant`.

## Step 3: Change point detection (Pre and QA-sentence panels)

Algorithm: PELT (Killick, Fearnhead & Eckley, 2012, JASA 107:1590-1598)
Penalty: AIC, pen = 2 (Akaike, 1974, IEEE Trans. Autom. Control 19:716-723)
Cost model: rbf (radial basis function kernel; distribution-free)

AIC is used instead of BIC because the goal is to capture subtle tone shifts.
AIC penalizes each additional change point by 2 (one segment mean parameter),
making it less conservative than BIC (pen = log n) and better suited to
short sentiment sequences.

This script reads the score files produced in Step 1 and overwrites the
`finberttone_change_point` column in place. Run `--dry_run` first to verify
the change point density before committing.

```bash
# Dry run first (recommended)
python compute_changepoint.py \
    --panel    pre \
    --input_dir /path/to/sentiment_panel/pre \
    --dry_run

# Write change points
python compute_changepoint.py \
    --panel    pre \
    --input_dir /path/to/sentiment_panel/pre

python compute_changepoint.py \
    --panel    qa_sentence \
    --input_dir /path/to/sentiment_panel/qa_sentence
```

The `finberttone_change_point` column is updated to 0 or 1.
Flag 1 marks the first sentence of each new sentiment regime.

## Checkpoint and Resume

Steps 1 and 2 write a checkpoint file to the output directory:

```
{output_dir}/checkpoint_finbert_{panel}.txt
{output_dir}/checkpoint_subjective_qa.txt
```

Re-running either script will skip files already listed in the checkpoint.
To reprocess a file, remove its entry from the checkpoint.

## References

- Huang, A. H., Wang, H., and Yang, Y. (2022). FinBERT: A large language
  model for extracting information from financial text. Contemporary
  Accounting Research.

- Pardawala, H., et al. (2024). SubjECTive-QA: Measuring subjectivity in
  earnings call transcripts QA through six-dimensional feature analysis.
  NeurIPS 2024 Datasets and Benchmarks Track.

- Killick, R., Fearnhead, P., and Eckley, I. A. (2012). Optimal detection
  of changepoints with a linear computational cost. Journal of the American
  Statistical Association, 107(500), 1590-1598.

- Akaike, H. (1974). A new look at the statistical model identification.
  IEEE Transactions on Automatic Control, 19(6), 716-723.