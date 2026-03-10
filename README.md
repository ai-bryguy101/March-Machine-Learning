# 🏀 March Machine Learning Mania 2026

Neural network approach to predicting NCAA March Madness tournament outcomes.

**Competition:** [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)  
**Prize Pool:** $50,000  
**Evaluation:** Log Loss (lower is better)

---

## 🎯 Our Approach: MarchNet

Instead of traditional ML (XGBoost, logistic regression), we built a custom neural network that:

1. **Learns sequential context** — Processes games chronologically through the season using a GRU, so the model understands a team's journey (early struggles, late-season run, etc.)

2. **Uses matchup-specific attention** — For Duke vs Kansas, the model attends over Duke's game history asking "which of Duke's games are most relevant for playing against Kansas?" (Maybe their game vs Baylor, a similar team)

3. **Aggressively calibrates** — March Madness is chaotic. We shrink all predictions toward 50% to avoid catastrophic log loss penalties from confident wrong predictions.

---

## 📁 Project Structure

```
march-madness-2026/
├── configs/                 # Hyperparameters (YAML)
│   └── default.yaml
├── data/
│   ├── raw/                 # Kaggle CSVs (download separately)
│   ├── processed/           # Preprocessed tensors
│   └── submissions/         # Generated submission CSVs
├── docs/
│   ├── ARCHITECTURE.md      # Neural network specification
│   ├── STRATEGY.md          # Competition strategy
│   └── PROMPTS.md           # AI prompts used during development
├── notebooks/               # Jupyter exploration notebooks
├── src/
│   ├── data/
│   │   └── preprocessing.py # CSV → model-ready data
│   ├── models/
│   │   ├── team_encoder.py      # Component 1: stats → embeddings
│   │   ├── game_processor.py    # Component 2: GRU sequential updater
│   │   ├── attention_matchup.py # Component 3: multi-head attention
│   │   ├── prediction_head.py   # Component 4: → win probability
│   │   └── marchnet.py          # Full combined model
│   ├── training/            # Training scripts
│   └── utils/               # Helper functions
├── tests/                   # Unit tests
├── PROJECT_LOG.md           # Session-by-session progress
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/ai-bryguy101/March-Machine-Learning.git
cd March-Machine-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download the competition data from Kaggle and place all CSVs in `data/raw/`:

```bash
# Using Kaggle CLI (install with: pip install kaggle)
kaggle competitions download -c march-machine-learning-mania-2026
unzip march-machine-learning-mania-2026.zip -d data/raw/
```

### 3. Preprocess Data

```bash
python -m src.data.preprocessing data/raw/
```

### 4. Train Model

```bash
# Coming soon - training scripts
python -m src.training.pretrain --config configs/default.yaml
python -m src.training.finetune --config configs/default.yaml
python -m src.training.calibrate --config configs/default.yaml
```

### 5. Generate Submission

```bash
# Coming soon
python -m src.predict.generate_submission --checkpoint best_model.pt
```

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         MarchNet                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Team A's Season                    Team B's Season              │
│  ┌─────────────┐                    ┌─────────────┐              │
│  │ Game Stats  │ ×30                │ Game Stats  │ ×30          │
│  └──────┬──────┘                    └──────┬──────┘              │
│         ▼                                  ▼                     │
│  ┌─────────────┐                    ┌─────────────┐              │
│  │Team Encoder │ (shared weights)   │Team Encoder │              │
│  └──────┬──────┘                    └──────┬──────┘              │
│         ▼                                  ▼                     │
│  ┌─────────────┐                    ┌─────────────┐              │
│  │     GRU     │ Sequential         │     GRU     │              │
│  │  Processor  │ Processing         │  Processor  │              │
│  └──────┬──────┘                    └──────┬──────┘              │
│         │                                  │                     │
│         ▼                                  ▼                     │
│     [Embedding]◄───────────────────►[Embedding]                  │
│     + History      Attention         + History                   │
│         │              │                  │                      │
│         └──────────────┼──────────────────┘                      │
│                        ▼                                         │
│              ┌─────────────────┐                                 │
│              │ Prediction Head │                                 │
│              │   + Shrinkage   │                                 │
│              └────────┬────────┘                                 │
│                       ▼                                          │
│                 P(A wins) ∈ [0.35, 0.65]                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding dim | 512 | High-dimensional for nuance |
| GRU layers | 1 | Simple, avoid overfitting |
| Attention heads | 8 | Standard transformer default |
| Dropout | 0.3 | Moderate regularization |
| Shrinkage | 0.5 | Aggressive pull toward 50% |

---

## 📈 Progress

See `PROJECT_LOG.md` for detailed session-by-session notes.

**Current Status:**
- ✅ Architecture designed
- ✅ Core components implemented (Encoder, GRU, Attention, Head)
- ✅ Preprocessing pipeline
- 🔲 Training loops
- 🔲 Validation on Stage 1
- 🔲 Generate Stage 2 submission

---

## 🛠 Tools Used

- **PyTorch** — Neural network framework
- **Claude AI** — Architecture design, code generation, debugging
- **Pandas/NumPy** — Data processing

This project uses AI assistance for development. All tools meet the competition's Reasonableness Standard.

---

## 📄 License

Competition data is CC-BY 4.0. Code is MIT licensed.
