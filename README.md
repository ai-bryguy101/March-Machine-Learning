# March Machine Learning Mania 2026

Kaggle competition to predict NCAA March Madness tournament outcomes for both men's and women's brackets.

**Competition:** [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)  
**Sponsor:** Google LLC  
**Prize Pool:** $50,000 (8 winners)  
**Data License:** CC-BY 4.0  

## Goal

Build two separate models (men's + women's) that predict the probability of Team A beating Team B for every possible tournament matchup. Submissions are evaluated on log loss.

## Project Structure

```
march-madness-2026/
├── data/
│   ├── raw/              # Original Kaggle CSVs (not tracked in git)
│   ├── processed/        # Cleaned & feature-engineered datasets
│   └── submissions/      # Generated submission CSVs
├── notebooks/            # Jupyter notebooks for exploration & modeling
├── src/
│   ├── features/         # Feature engineering scripts
│   ├── models/           # Model training & prediction scripts
│   └── utils/            # Helper functions (data loading, etc.)
├── docs/                 # Project documentation
├── tests/                # Unit tests
├── PROJECT_LOG.md        # Session-by-session progress tracker
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── .gitignore            # Files to exclude from git
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/march-madness-2026.git
cd march-madness-2026

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt

# Download data from Kaggle and place CSVs in data/raw/

# Run notebooks in order (once created)
```

## Approach Overview

See `PROJECT_LOG.md` for detailed session notes. High-level plan:

1. **Explore** — Understand the data, find patterns, visualize trends
2. **Engineer Features** — Build team-level stats (offensive/defensive ratings, strength of schedule, etc.)
3. **Train Models** — Start simple (logistic regression), iterate toward ensemble methods
4. **Validate** — Use 2022-2025 tournament results (Stage 1) to test predictions
5. **Submit** — Generate Stage 2 predictions for the 2026 tournament

## Tools & AI Usage

This project uses AI assistance (Claude, Cursor) for code generation, data analysis, and learning ML concepts. All tools used are publicly available and meet the competition's Reasonableness Standard for external tools.
