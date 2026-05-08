# Fake News Detection System

A dual-model fake news detection system combining TF-IDF + Logistic Regression with fine-tuned DistilBERT, served via a Streamlit UI with word-level explainability and smart ensemble routing.

---

## Demo

| Real News (Federal Reserve) | Fake News (Clickbait) |
|---|---|
| Real News — 97.2% confidence | Fake News — 98.6% confidence |
| Decision by: TF-IDF (high confidence) | Decision by: TF-IDF (high confidence) |

**Detailed Decision Analysis** shows which words pushed the prediction and by how much — e.g. `said` (0.9691), `monday` (0.8574) toward Real vs `hiding` (0.8910), `shocking` (0.7500) toward Fake.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Models](#models)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Performance](#performance)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## Overview

This project implements a binary fake news classifier (Real vs Fake) with two complementary models:

- **TF-IDF + Logistic Regression** — fast, interpretable, with per-word explainability
- **DistilBERT (fine-tuned)** — semantic understanding, handles out-of-distribution vocabulary

A **Smart Ensemble** mode automatically routes predictions: TF-IDF handles high-confidence cases (≥85%), BERT handles uncertain ones via weighted combination (LR×0.25 + BERT×0.75).

The system detects **journalistic format patterns**, not factual truth. A well-formatted false story may score as Real — this is a documented limitation, not a bug.

---

## Architecture

```
User Input (Text / Article)
        ↓
DataProcessor.clean_text()
  └── URL removal → lowercase → punctuation strip
      → stopword removal → lemmatization
        ↓
┌─────────────────────────────────────────┐
│           Smart Ensemble Router         │
│                                         │
│  TF-IDF + LR → confidence ≥ 85%? ──YES─→ prediction
│                     ↓ NO               │
│              DistilBERT                 │
│           LR×0.25 + BERT×0.75 ─────────→ prediction
└─────────────────────────────────────────┘
        ↓
Verdict + Confidence + Word-level Explainability
```

---

## Dataset

**WELFake** — aggregated from four sources:

| Source | Description |
|---|---|
| Kaggle Fake News | Political news articles |
| McIntire Dataset | US political content |
| ISOT Dataset | Real/fake news mix |
| GossipCop | Celebrity and general news |

| Metric | Value |
|---|---|
| Total rows | 72,134 |
| Real (label 0) | 34,970 |
| Fake (label 1) | 37,030 |
| Time period | 2015–2018 |
| Primary domain | US political news |

Labels: `0 = Real`, `1 = Fake`

---

## Models

### Model 1 — TF-IDF + Logistic Regression

| Parameter | Value | Reason |
|---|---|---|
| `max_features` | 50,000 | Captures domain-specific vocabulary |
| `ngram_range` | (1, 2) | Unigrams + bigrams for phrase detection |
| `sublinear_tf` | True | Log-scale TF, better for LR |
| `max_iter` | 1,000 | Full convergence on 72k rows |
| `class_weight` | balanced | Removes intercept bias from class imbalance |
| `solver` | lbfgs | Memory-efficient for high-dimensional sparse data |

**What it learned:** Journalistic format signals — `said` (0.97), `monday/tuesday` (0.85), `citing` (0.48) → Real. `hiding` (0.89), `shocking` (0.75), `government concealing` (0.51) → Fake.

**Explainability:** Per-word impact = TF-IDF weight × LR coefficient, displayed in the UI for every prediction.

**Data leakage fix:** Vectorizer fitted on training split only. Test set never influences IDF scores.

### Model 2 — DistilBERT (Fine-tuned)

Pre-trained `distilbert-base-uncased` fine-tuned on WELFake for 3 epochs on Google Colab T4 GPU.

| Parameter | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Parameters | 66,955,010 |
| Epochs | 3 |
| Batch size | 16 |
| Max sequence length | 512 tokens |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Training time | ~35 minutes (T4 GPU) |

**Why BERT over TF-IDF on hard cases:**
- Subword tokenization — `Artemis` → `["art", "##em", "##is"]`, never truly unknown
- Contextual attention — `said` in formal attribution ≠ `said` in conspiracy framing
- Wikipedia pretraining — understands OPEC as an energy institution, not a conspiracy token
- Short input robustness — extracts semantic meaning from 2 sentences where TF-IDF fails

**Model weights** hosted on HuggingFace Hub (too large for GitHub at 268MB).

---

## Project Structure

```
Fake_News_detection/
├── src/
│   ├── __init__.py
│   ├── data_processor.py       # Text cleaning pipeline
│   ├── model_trainer.py        # TF-IDF + LR training
│   └── bert_predictor.py       # DistilBERT inference wrapper
├── scripts/
│   └── download_model.py       # Downloads BERT weights from HuggingFace
├── notebooks/
│   └── bert_finetuning.ipynb   # Colab training notebook
├── models/
│   ├── README.md               # Model documentation
│   ├── logistic_model.joblib   # Trained LR model (generated by main.py)
│   └── tfidf_vectorized.joblib # Fitted TF-IDF vectorizer (generated by main.py)
├── data/
│   └── final_processed.csv     # Preprocessed WELFake dataset
├── app_ui.py                   # Streamlit application
├── main.py                     # Training entry point
├── pyproject.toml              # Dependencies (managed by uv)
├── .env                        # API keys (never committed)
└── .gitignore
```

---

## Setup & Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- WELFake dataset (`WELFake_Dataset.csv`)
- HuggingFace account + access token (for BERT download)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Fake_News_detection.git
cd Fake_News_detection
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

### 4. Download BERT model weights

```bash
uv run scripts/download_model.py
```

This downloads the fine-tuned DistilBERT weights from HuggingFace Hub into `models/distilbert_finetuned/`.

### 5. Prepare and train TF-IDF model

Place `WELFake_Dataset.csv` in the project root, then run:

```bash
uv run main.py
```

This preprocesses the dataset, trains TF-IDF + LR, and saves artifacts to `models/`.

### 6. Launch the app

```bash
uv run streamlit run app_ui.py
```

---

## Usage

### Prediction Modes

The sidebar offers three modes:

| Mode | Description | When to use |
|---|---|---|
| **Smart Ensemble** | Auto-routes based on TF-IDF confidence | Default — best overall accuracy |
| **TF-IDF Only** | Fast, fully explainable | When you need word-level reasoning |
| **DistilBERT Only** | Maximum accuracy, semantic understanding | Short inputs, OOD vocabulary |

### Input Guidelines

- **Best results:** Full articles (100+ words) with clear journalistic formatting
- **Short headlines:** May trigger low-confidence warning — model routes to BERT automatically
- **Non-English or highly technical text:** Expect lower confidence — model trained on English political news

### Understanding the Output

```
✅ Prediction: Real News
Confidence: 87.30%
Decision by: TF-IDF (high confidence)

Detailed Decision Analysis:
  Pushing toward Real:      Pushing toward Fake:
  said       ████ 0.9691   world    ███ 0.2022
  tuesday    ███  0.3593   war      ██  0.1102
  citing     ██   0.1989
```

- **Confidence < 65%:** Low confidence warning shown — treat result with caution
- **Impact score:** TF-IDF weight × LR coefficient — higher means stronger influence on decision
- **Negative coefficient** → pushes Real, **Positive coefficient** → pushes Fake

---

## Performance

### TF-IDF + Logistic Regression

| Metric | Value |
|---|---|
| Training Accuracy | 94.66% |
| Test Accuracy | 93.5% |
| Overfit Gap | 1.16% |
| PR-AUC | 0.99 |
| F1 Score (weighted) | 0.935 |
| Intercept (balanced) | 1.76 |

### DistilBERT (Fine-tuned)

| Epoch | Train Loss | Val Loss | Accuracy | F1 |
|---|---|---|---|---|
| 1 | 0.022987 | 0.053586 | 98.78% | 0.9878 |
| 2 | 0.025779 | 0.030534 | 99.13% | 0.9913 |
| 3 | 0.000141 | 0.033311 | 99.36% | 0.9936 |
| **Test** | — | — | **99.27%** | **0.9927** |

### Stress Test Results

| Input | TF-IDF | DistilBERT | Correct? |
|---|---|---|---|
| Federal Reserve (formal) | Real 80% | Real 97% | ✅ |
| SHOCKING clickbait | Fake 98% | Fake 99% | ✅ |
| NASA Artemis (OOD vocab) | Real 57% | Real ~78% | ✅ |
| UAE/OPEC article | Real 59% | Real ~80% | ✅ |
| AP-format disinformation | Real 85% | Real ~82% | ❌ Both fooled |
| Vague conspiracy framing | Fake 97% | Fake 90% | ✅ |

---

## Known Limitations

### Architecture Ceiling

This system detects **journalistic format patterns**, not factual truth:

```
Real journalism markers:  named sources, weekday attribution,
                          specific figures, institutional language
Fake news markers:        emotional vocabulary, vague attribution,
                          all-caps urgency, conspiratorial framing
```

A well-formatted false story mimicking AP wire style will score as Real News with high confidence. This is the fundamental ceiling of supervised text classification without external fact-checking.

### Specific Failure Modes

| Limitation | Impact | Status |
|---|---|---|
| Temporal gap (2015-2018 data) | Recent events/vocabulary score lower confidence | Documented |
| US-centric domain bias | Non-US sources (SEBI India, etc.) underperform | Documented |
| Format-mimicking disinformation | AP-format fake articles fool both models | Known ceiling |
| Short input weakness | Headlines < 50 words → intercept bias toward Fake in LR | Mitigated by BERT routing |
| Static training | Model doesn't update with new disinformation patterns | Version 2 roadmap |

### What This System IS and IS NOT

✅ **IS:** A first-pass triage classifier that handles high-confidence cases automatically and flags uncertain ones for review.

❌ **IS NOT:** A fact-checker. Does not verify claims against external knowledge bases.

---

## Roadmap

### Version 2 (In Planning)

- **Live Data Integration** — NewsAPI + GDELT for real-time article fetching
- **Claim Verification** — Support/Refute/Neutral verdicts using NLI (cross-encoder/nli-distilroberta-base) + Gemini reasoning
- **RAG Pipeline** — ChromaDB vector store for evidence retrieval against user claims
- **Weak Supervision** — Snorkel labeling functions for automatic labeling of fetched articles
- **Auto-Retraining** — Scheduled pipeline: fetch → label → validate → retrain
- **Multimodal Input** — OCR (Tesseract) for screenshot/image inputs, newspaper3k for URL extraction
- **Diverse Dataset** — MIND dataset (160k, 18 categories) to fix domain bias

---

## Technical Stack

| Component | Technology |
|---|---|
| Package manager | uv |
| ML framework | scikit-learn 1.8+ |
| Deep learning | transformers, torch |
| Text vectorization | TF-IDF (scikit-learn) |
| Model serialization | joblib |
| NLP preprocessing | NLTK (stopwords, WordNetLemmatizer) |
| UI framework | Streamlit 1.52+ |
| Model hosting | HuggingFace Hub |
| Training environment | Google Colab (T4 GPU) |

---

## Acknowledgements

- **WELFake Dataset** — Verma, P.K. et al., IEEE Transactions on Computational Social Systems
- **HuggingFace Transformers** — for DistilBERT pretrained weights and fine-tuning infrastructure
- **Streamlit** — for the UI framework