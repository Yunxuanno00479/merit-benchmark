# Text Embedding Experiment Code Snapshot

This folder is a copy of the code and shell scripts used for the B4 text embedding experiment.

Final label definition:

```text
Y = 1[ |delta| > Q75(|delta|) ]
```

The Q75 threshold is computed from the training split only for each task, horizon, and target, then applied to both train and test splits.

Main experiment script:

```text
src/text_embedding_b4_pipeline.py
```

Preprocessing scripts that preserve original transcript text:

```text
src_preprocessing/text_embedding_parse_transcript.py
src_preprocessing/text_embedding_compute_finbert_tone.py
src_preprocessing/text_embedding_compute_subjective_qa.py
src_preprocessing/text_embedding_compute_changepoint.py
scripts/run_text_embedding_preprocessing.sh
```

Final classifier runner:

```text
scripts/run_text_embedding_absq75_lr_allwindows.sh
```

Benchmark dataset helpers:

```text
src/text_embedding_build_benchmark_dataset.py
src/text_embedding_split_benchmark_dataset.py
scripts/run_text_embedding_build_benchmark_dataset.sh
```

Embedding models used:

```text
mpnet: sentence-transformers/all-mpnet-base-v2
openai_small512: text-embedding-3-small with 512 dimensions
```

Classifier used in the final abs-Q75 run:

```text
Logistic Regression with StandardScaler, L2 regularization, C=1.0, max_iter=3000
```

Folder layout:

```text
src_preprocessing/
  text_embedding_parse_transcript.py
  text_embedding_compute_finbert_tone.py
  text_embedding_compute_subjective_qa.py
  text_embedding_compute_changepoint.py

src/
  text_embedding_b4_pipeline.py
  text_embedding_build_benchmark_dataset.py
  text_embedding_split_benchmark_dataset.py

scripts/
  run_text_embedding_preprocessing.sh
  run_text_embedding_absq75_lr_allwindows.sh
  run_text_embedding_build_benchmark_dataset.sh

docs/
  text_embedding_transcript_preprocessing_README.md
  text_embedding_sentiment_preprocessing_README.md

reference_upstream/
  text_embedding_build_benchmark_dataset_upstream.py
  text_embedding_split_benchmark_dataset_upstream.py
  run_text_embedding_build_benchmark_dataset_upstream.sh
```
