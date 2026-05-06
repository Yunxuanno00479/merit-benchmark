---
title: 'MERIT: A Sentence-Level Benchmark for Earnings Call Language Trajectories and Sub-Minute Market Microstructure'

---

# MERIT: A Sentence-Level Benchmark for Earnings Call Language Trajectories and Sub-Minute Market Microstructure

MERIT (**M**icrostructure and **E**arnings call **R**eal-time **I**nformation **T**rajectories) is a benchmark that links sentence-level earnings-call language to sub-minute market microstructure responses. It pairs FinBERT-tone sentiment and SubjECTive-QA scores with tick-level NBBO panels across 1,118 earnings call events from 267 S&P 500 firms (2021–2023).

---

## Repository Structure

```text
merit-benchmark/
├── README.md
├── requirements.txt
├── data_pipeline/
│   ├── layer2/
│   │   ├── add_labels.py              # Add Q75 extreme-event labels to benchmark files
│   │   ├── aggregate_quote_windows.py # Aggregate NBBO ticks into pre/post windows
│   │   ├── build_benchmark_dataset.py # Build final benchmark CSVs from Layer 2 + sentiment
│   │   └── split_benchmark_dataset.py # Apply session filter and train/test split
│   └── sentiment/
│       ├── compute_changepoint.py         # PELT + AIC change point detection
│       ├── compute_finbert_tone.py        # FinBERT-tone sentence scoring
│       ├── compute_subjective_qa.py       # SubjECTive-QA pair scoring
│       └── compute_trajectory_features.py # Extended trajectory feature computation (exploratory)
├── experiment/
│   └── run_baselines_b0b1b2.py        # Main baseline experiments (Inst. / CL / CT regimes)
└── text_embedding/
    ├── README.md                                      # Overview of the appendix text-embedding baseline
    ├── docs/
    │   ├── text_embedding_sentiment_preprocessing_README.md   # Notes for FinBERT, SubjECTive-QA, and change-point preprocessing
    │   └── text_embedding_transcript_preprocessing_README.md  # Notes for parsing raw transcripts into text-preserving panels
    ├── scripts/
    │   ├── run_text_embedding_absq75_lr_allwindows.sh         # Run final MPNet/OpenAI embedding + LR experiments for all horizons
    │   ├── run_text_embedding_build_benchmark_dataset.sh      # Build benchmark CSVs while preserving raw text fields
    │   └── run_text_embedding_preprocessing.sh                # Run transcript parsing and sentiment preprocessing pipeline
    ├── src/
    │   ├── text_embedding_b4_pipeline.py                      # Main B4 pipeline: embed text, cache vectors, train classifiers, report metrics
    │   ├── text_embedding_build_benchmark_dataset.py          # Join quote-window features with sentiment panels and raw text fields
    │   └── text_embedding_split_benchmark_dataset.py          # Apply session filtering and time-based train/test split
    ├── src_preprocessing/
    │   ├── text_embedding_parse_transcript.py                 # Parse transcript JSONL files into presentation, QA-pair, and QA-sentence panels
    │   ├── text_embedding_compute_finbert_tone.py             # Compute FinBERT-tone scores for presentation and QA-sentence text
    │   ├── text_embedding_compute_subjective_qa.py            # Compute six-dimensional SubjECTive-QA scores for QA pairs
    │   └── text_embedding_compute_changepoint.py              # Detect FinBERT-tone regime shifts using PELT with AIC penalty
    └── reference_upstream/
        ├── run_text_embedding_build_benchmark_dataset_upstream.sh # Original upstream runner for benchmark construction
        ├── text_embedding_build_benchmark_dataset_upstream.py     # Upstream version of benchmark construction script
        └── text_embedding_split_benchmark_dataset_upstream.py     # Upstream version of benchmark split script
```

---

## Released Data

The benchmark data is available on Hugging Face Datasets:

```text
https://anonymous-hf.up.railway.app/a/qohkf1cu3j5t/
```

The dataset comprises three components:

| Component | Description |
|---|---|
| `sentiment_panel/` | FinBERT-tone scores and SubjECTive-QA scores for all 1,118 EC events |
| `benchmark/` | Pre-computed pre/post window aggregates for Presentation and Q&A segments |
| `benchmark_split/` | Train (2021–2022) and test (2023) splits with Inst., CL, and CT features |

**Note:** Raw NYSE TAQ tick streams and earnings call transcripts are not released due to commercial licensing restrictions.

---

## Pipeline Overview

The full pipeline proceeds in four stages. Stages 1–2 require access to proprietary data sources (NYSE TAQ and a commercial transcript provider) and are provided for reproducibility documentation only. Stages 3–4 operate on the released benchmark data.

Appendix experiments, including the text-embedding baseline, are provided in `text_embedding/` and are not required for reproducing the main MERIT benchmark pipeline.

```text
Stage 1: Sentiment scoring
    compute_finbert_tone.py       →  sentiment_panel/pre/
    compute_subjective_qa.py      →  sentiment_panel/qa_score/
    compute_changepoint.py        →  (updates sentiment_panel/pre/ in place)

Stage 2: Quote window aggregation  [requires NYSE TAQ]
    aggregate_quote_windows.py    →  layer2/
    build_benchmark_dataset.py    →  benchmark/

Stage 3: Labeling                  [operates on released data]
    add_labels.py                 →  benchmark/benchmark_labeled/

Stage 4: Train/test split          [operates on released data]
    split_benchmark_dataset.py    →  benchmark_split/

Stage 5: Baseline experiments      [operates on released data]
    run_baselines_b0b1b2.py       →  results/
```

---

## Quick Start (from released data)

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the benchmark data

```bash
# Install Hugging Face datasets
pip install datasets

python -c "
from datasets import load_dataset
ds = load_dataset('[ANONYMOUS]/MERIT')
"
```

Or download manually from the Hugging Face repository and place files under `data/`:

```text
data/
├── benchmark/
│   ├── benchmark_pre_30s.csv
│   ├── benchmark_pre_60s.csv
│   ├── benchmark_pre_120s.csv
│   ├── benchmark_pre_300s.csv
│   ├── benchmark_qa_30s.csv
│   ├── benchmark_qa_60s.csv
│   ├── benchmark_qa_120s.csv
│   └── benchmark_qa_300s.csv
└── benchmark_split/
    ├── train/
    └── test/
```

### 3. Generate labels

```bash
python data_pipeline/layer2/add_labels.py \
    --benchmark_dir data/benchmark \
    --out_dir       data/benchmark/benchmark_labeled
```

### 4. Apply train/test split

```bash
python data_pipeline/layer2/split_benchmark_dataset.py \
    --benchmark_dir data/benchmark \
    --output_dir    data/benchmark_split
```

### 5. Run baseline experiments

```bash
# Presentation segment (Task 1)
python experiment/run_baselines_b0b1b2.py \
    --task        pre \
    --split_dir   data/benchmark_split \
    --labeled_dir data/benchmark/benchmark_labeled \
    --output      results/task1_results.csv

# Q&A segment (Task 2)
python experiment/run_baselines_b0b1b2.py \
    --task        qa \
    --split_dir   data/benchmark_split \
    --labeled_dir data/benchmark/benchmark_labeled \
    --output      results/task2_results.csv
```

---

## Representation Regimes

The benchmark evaluates four feature regimes for the Presentation segment:

| Regime | Abbreviation | Features | Description |
|---|---|---|---|
| Document-level aggregate | Agg | Mean FinBERT tone over full call | Whole-call compression baseline |
| Instantaneous utterance | Inst. | `finberttone_expected_value` | Current sentence tone only |
| Cumulative level | CL | `ev_expanding_mean` | Expanding mean tone up to current anchor |
| Cumulative trajectory | CT | Inst. + CL + `ev_expanding_std/max/min` + `n_change_points` + `finberttone_cumulative_tone` | Full trajectory shape |

For the Q&A segment, the analogous features use SubjECTive-QA positive-class probabilities across six dimensions (Assertive, Cautious, Optimistic, Specific, Clear, Relevant).

---

## Appendix: Text Embedding Baseline

The `text_embedding/` directory contains the implementation of the appendix text-embedding baseline. This experiment evaluates whether generic semantic representations of earnings-call text can directly predict subsequent sub-minute market microstructure responses.

The baseline constructs two text views:

| View | Description |
|---|---|
| Presentation text | The management presentation sentence associated with the anchor |
| Question-answer text | The analyst question concatenated with the corresponding management answer |

For each view and horizon, the text is embedded using MPNet and OpenAI's small embedding model with 512 output dimensions. A logistic regression classifier is then trained on the resulting embeddings under the same train/test split and absolute Q75 extreme-event labeling protocol used in the main benchmark.

To run the appendix experiment:

```bash
bash text_embedding/scripts/run_text_embedding_absq75_lr_allwindows.sh
```

To use OpenAI embeddings, set an API key before running:

```bash
export OPENAI_API_KEY="your-api-key"
```

See `text_embedding/README.md` for the full script layout and preprocessing details. Raw transcript access is required only if users want to rerun the text-preserving preprocessing pipeline from scratch.

---

## Evaluation Protocol

- **Targets:** ΔBAS (bid-ask spread), ΔQRF (quote revision frequency), ΔQVol (quote volatility)
- **Label:** `Y = 1[|Δmetric| > τ₇₅]`, where τ₇₅ is the 75th percentile of the pooled absolute delta distribution computed on the training split only
- **Positive-class rate:** approximately 20–25% depending on target and horizon
- **Primary metric:** Balanced Accuracy (BAcc); secondary: AUC-ROC
- **Split:** Train 2021–2022 (N=729 events), Test 2023 (N=389 events)
- **Post-windows:** W ∈ {30, 60, 120, 300} seconds
- **Non-overlapping anchor protocol:** anchors are retained only if they start at least W seconds after the previously retained anchor, preventing target autocorrelation

**Meaningful progress** on MERIT is defined as statistically significant improvement over the Cumulative trajectory (CT) baseline under temporal generalisation.

---

## Citation

```bibtex
@inproceedings{anonymous2026merit,
  title   = {{MERIT}: A Sentence-Level Benchmark for Earnings Call Language
             Trajectories and Sub-Minute Market Microstructure},
  author  = {Anonymous},
  booktitle = {},
  year    = {2026},
  note    = {Datasets and Benchmarks Track}
}
```

---

## License

- **Pipeline code and baselines:** MIT License
- **Benchmark data (sentiment panel and benchmark split):** CC BY 4.0
- **Raw transcripts:** Not released (copyright restrictions)
- **Raw NYSE TAQ data:** Not released (commercial license)