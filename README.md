# Conversation Logs NLP Pipeline

A small, reproducible demo built to demonstrate practical skills commonly required in research
and data roles: working with JSON-structured logs, building repeatable NLP pipelines,
and exporting analysis-ready datasets.

This demo uses **StudyChat** (`wmcnicho/StudyChat`), a public dataset of real student–LLM assistant conversations from a university course. It converts raw JSONL logs into:
1) turn-level predictions (intent + sentiment), and
2) conversation-level aggregates (engagement and behavior proxies),
ready for downstream stats and modeling.

## Features

1. **Load + normalize labels**
   - Reads JSONL and maps hierarchical dialogue-act labels into 8 prompt types:
     `conceptual_questions`, `contextual_questions`, `provide_context`, `verification`,
     `editing_request`, `writing_request`, `off_topic`, `misc`.

2. **Turn-level text classification**
   - Trains (or loads a cached) **TF-IDF + Linear SVM** model to predict a main label per prompt.
   - Saves the model to `outputs/help_classifier.joblib` for reproducible reruns.

3. **Turn-level sentiment**
   - Adds `NEGATIVE / NEUTRAL / POSITIVE` using either:
     - Transformers sentiment models (`multilingual` by default, `RoBERTa`/`BERTweet` available), or
     - VADER fallback

4. **Conversation-level aggregation**
   - Groups by `chatId`, orders by `interactionCount`, and exports features such as:
     - Engagement: `n_turns`, `duration_minutes`, `total_words`
     - Effort and persistence: `label_switch_count`, `unique_help_types`
     - Help-seeking behavior: `dominant_help_type`, `question_ratio`, `avg_unique_ratio`
     - Sentiment: `dominant_sentiment`

## Quickstart

Setup and run the project:
```bash
# install dependencies
pip install -U scikit-learn joblib tqdm vaderSentiment transformers torch

# download StudyChat as data/data.jsonl from Hugging Face:
# https://huggingface.co/datasets/wmcnicho/StudyChat

# create train/test splits by conversation
cd utils && python split_dataset.py && cd ..

# run the main pipeline with the default transformer model
python conversation_features.py
```
Outputs are written to the `outputs/` folder.

Optional benchmarks:
```bash
cd nlp_tests
python nlp_sentiment.py
python nlp_labels.py
```

# Data formats

## Inputs

* The input is the **StudyChat** conversation-log: JSONL conversation logs with structured identifiers, timestamps, and an intent-style label per turn, enabling both turn-level NLP and conversation-level aggregation.

## Outputs

- `outputs/turn_level_predictions.csv`: Includes identifiers (chatId, interactionCount, timestamp), the prompt, true label, predicted label, and sentiment label.
- `outputs/conversation_level_features.csv`: One row per conversation (chatId) with aggregated metrics ready for downstream statistics, monitoring, or modeling.

## Repository layout

- `conversation_features.py` main end-to-end pipeline (train/load, sentiment, exports)
- `data/` raw and split JSONL files
  - `data.jsonl`
  - `train_13k.jsonl`, `test_3851.jsonl`
- `outputs/`
  - `turn_level_predictions.csv`
  - `conversation_level_features.csv`
  - `help_classifier.joblib`
- `utils/` small helpers (`split_dataset.py`, `count_lines.py`, `analyze_labels.py`)
- `nlp_tests/` quick experiments/benchmarks (`nlp_labels.py`, `nlp_sentiment.py`)
 

# Notes

Train/test split is done by conversation (chatId) to reduce leakage across turns from the same chat.

The sentiment component is model-pluggable to balance accuracy vs runtime, and supports multilingual text (useful when logs are not English).
