# NEURAL NETWORK ARCHITECTURE — MarchNet

## Overview

MarchNet is a custom neural network that learns team representations by sequentially
processing games through a season, then uses attention-based matchup prediction to
forecast tournament outcomes.

**Framework:** PyTorch
**Philosophy:** Let the network find patterns humans can't — don't over-engineer features.

---

## Architecture Components

### Component 1: Team Encoder
**Purpose:** Project raw team stats into a high-dimensional embedding space.

```
Input:  Raw game stats (28 features per team per game)
        - FG%, 3PT%, FT%, offensive rebounds, defensive rebounds,
          assists, turnovers, steals, blocks, personal fouls,
          points scored, points allowed, tempo estimate
        
Output: 512-dimensional team embedding

Layers: Linear(num_features → 256) → ReLU → Dropout(0.3)
        Linear(256 → 512) → ReLU → Dropout(0.3)
        Linear(512 → 512) → LayerNorm
        
Shared: Same encoder weights process ALL teams (forces the network
        to learn a universal "team quality" language)
```

### Component 2: Game Processor (GRU-based Sequential Updater)
**Purpose:** Update team embeddings after each game, building context through the season.

```
How it works:
  - Before each game, we have Team A's current embedding and Team B's current embedding
  - We combine: [TeamA_embedding | TeamB_embedding | game_stats | game_outcome]
  - Feed through a GRU cell
  - GRU outputs an UPDATED embedding for each team
  - A learned "game importance" gate controls how much each game affects the embedding
  
Why GRU over LSTM:
  - Simpler (fewer parameters, less overfitting risk with our data size)
  - Faster to train
  - Performs comparably on sequences of this length (~30 games per season)
  
GRU specs:
  - Input size: 512 (projected from combined context)
  - Hidden size: 512
  - Layers: 1 (keep it simple)
```

### Component 3: Attention-Based Matchup Layer
**Purpose:** For a specific matchup (A vs B), figure out which of A's past games
are most relevant for predicting how A will do against B, and vice versa.

```
Mechanism: Scaled dot-product attention (same math as transformers)

For predicting Team A vs Team B:

  Query:   Team B's final embedding (what kind of team is B?)
  Keys:    Team A's game history embeddings (each past game A played)
  Values:  Team A's game history embeddings

  Attention weights = softmax(Q · K^T / sqrt(d_k))
  A_matchup_repr = Attention weights · V

  (Then reverse: use A as query into B's history)
  
Multi-head attention:
  - 8 attention heads
  - Each head can learn different "reasons" a past game is relevant
  - Concatenate heads → Linear projection back to 512 dims
```

### Component 4: Prediction Head
**Purpose:** Take the two matchup-specific representations and output a win probability.

```
Input:  [A_matchup_repr | B_matchup_repr | A_matchup_repr - B_matchup_repr | seed_diff]
        (512 + 512 + 512 + 1 = 1537 dimensions)

Layers: Linear(1537 → 512) → ReLU → Dropout(0.3)
        Linear(512 → 256) → ReLU → Dropout(0.3)
        Linear(256 → 1) → Sigmoid

Output: Raw win probability (0 to 1)
        → Apply shrinkage → Final prediction (~0.35 to ~0.65)
```

---

## Data Flow Example

```
Season 2024, predicting: Duke (TeamID 1181) vs Kansas (TeamID 1242)

1. ENCODE: Duke's per-game stats → encoder → initial 512-dim embedding
           Kansas's per-game stats → encoder → initial 512-dim embedding
           (same encoder, shared weights)

2. PROCESS SEASON SEQUENTIALLY:
   Game 1:  Duke vs Team X → update Duke embedding, store in Duke's history
   Game 2:  Duke vs Team Y → update Duke embedding, store in Duke's history
   ...
   Game 30: Duke vs Team Z → update Duke embedding, store in Duke's history
   
   (Same for Kansas, processing their ~30 games)

3. ATTEND:
   "Which of Duke's 30 games are most informative about Duke-vs-Kansas?"
   → Maybe Game 7 (vs Baylor, similar style) gets weight 0.25
   → Maybe Game 22 (vs Iowa State, similar defense) gets weight 0.18
   → Maybe Game 3 (vs weak team) gets weight 0.01
   → Weighted combination = Duke's "Kansas-specific" representation
   
   (Reverse for Kansas attending over their history with Duke as query)

4. PREDICT:
   Combine both matchup-specific representations → prediction head → 0.62
   Apply shrinkage → 0.56
   
   Interpretation: Duke has ~56% chance of beating Kansas
```

---

## Training Strategy

### Phase 1: Pre-training on Regular Season Games
```
Data:    ALL regular season games, 2003-2025 (~100,000+ games)
Task:    Predict winner of each game
Process: Step through each season chronologically
         At each game, predict outcome using current embeddings
         Update embeddings after seeing result
         Backprop through the whole sequence

Epochs:  Start with 10-20, monitor validation loss
LR:      1e-3 with cosine annealing
```

### Phase 2: Fine-tuning on Tournament Games
```
Data:    NCAA tournament games, 2015-2025 (~700 games)
Task:    Predict tournament game outcomes
Process: Use pre-trained encoder + game processor
         Fine-tune attention layer and prediction head
         Lower learning rate (1e-4)
```

### Phase 3: Calibration
```
Data:    2022-2025 tournaments (Stage 1 validation)
Task:    Tune shrinkage factor to minimize log loss
Process: Grid search over shrinkage values [0.3, 0.35, 0.4, ..., 0.7]
         calibrated = shrinkage * raw_pred + (1 - shrinkage) * 0.5
         Pick shrinkage that gives lowest log loss on Stage 1
```

---

## Hyperparameters (Starting Points)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding dim | 512 | High-dimensional; room for nuance |
| GRU hidden size | 512 | Match embedding dim |
| Attention heads | 8 | Standard transformer default |
| Dropout | 0.3 | Moderate regularization |
| Learning rate | 1e-3 (pretrain), 1e-4 (finetune) | Standard starting points |
| Batch size | 64 games | Balance speed and gradient stability |
| Weight decay | 1e-5 | Light L2 regularization |
| Gradient clipping | 1.0 | Prevent exploding gradients in GRU |

---

## File Structure

```
src/
├── models/
│   ├── team_encoder.py          # Component 1: stats → 512-dim embedding
│   ├── game_processor.py        # Component 2: GRU sequential updater
│   ├── attention_matchup.py     # Component 3: multi-head attention layer
│   ├── prediction_head.py       # Component 4: embedding → win probability
│   └── marchnet.py              # Full model combining all components
├── data/
│   ├── dataset.py               # PyTorch Dataset for game sequences
│   └── preprocessing.py         # Raw CSV → model-ready tensors
├── training/
│   ├── pretrain.py              # Phase 1: regular season training
│   ├── finetune.py              # Phase 2: tournament fine-tuning
│   └── calibrate.py             # Phase 3: shrinkage tuning
└── predict/
    └── generate_submission.py   # Produce final Kaggle submission CSV
```

---

## Women's Model Adaptation

Same architecture, trained separately. Key differences:
- No Massey Ordinals features available
- Less training data (2010+ vs 2003+) — may need stronger regularization
- Different game dynamics — model learns this naturally
- Consider slightly lower embedding dim (256) if overfitting
