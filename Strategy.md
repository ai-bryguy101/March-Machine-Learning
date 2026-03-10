# STRATEGY & FEATURE PLAN

## Core Philosophy

**"Respect the chaos."**

March Madness is notoriously unpredictable. Our strategy optimizes for log loss by:
1. Weighting current season data most heavily
2. Giving extra emphasis to late-season form (last ~4 weeks)
3. Aggressively shrinking predictions toward 50% (target range: ~35%-65%)
4. Using historical data only for structural calibration, not team-specific prediction

---

## How Log Loss Works (And Why Our Strategy Fits)

Log loss formula per game: `-[y*log(p) + (1-y)*log(1-p)]`

Where `y` = actual outcome (1 if lower-ID team won, 0 if not), `p` = our predicted probability.

| Your Prediction | Actual Winner | Log Loss (penalty) |
|----------------|---------------|-------------------|
| 0.95 confident | Correct | 0.05 (tiny) |
| 0.95 confident | **Wrong** | **3.00 (massive)** |
| 0.65 confident | Correct | 0.43 (moderate) |
| 0.65 confident | Wrong | 1.05 (moderate) |
| 0.50 (coin flip) | Either | 0.69 (safe floor) |

**Key insight:** One badly wrong confident prediction can wipe out 20+ correct confident predictions. Our aggressive shrinkage toward 50% protects against this. We sacrifice a little reward on "obvious" games to massively reduce risk on upsets.

---

## The Three Layers

### Layer 1: Current Season Team Profile (HEAVIEST WEIGHT)

These features describe how each team is playing RIGHT NOW in the 2026 season.

#### Offensive Features
| Feature | Description | Source File |
|---------|-------------|-------------|
| `pts_per_game` | Average points scored per game | CompactResults |
| `fg_pct` | Field goal percentage (FGM/FGA) | DetailedResults |
| `fg3_pct` | 3-point percentage (FGM3/FGA3) | DetailedResults |
| `ft_pct` | Free throw percentage (FTM/FTA) | DetailedResults |
| `off_rebounds_per_game` | Offensive rebounds per game | DetailedResults |
| `assists_per_game` | Assists per game | DetailedResults |
| `turnovers_per_game` | Turnovers committed per game (lower = better) | DetailedResults |

#### Defensive Features
| Feature | Description | Source File |
|---------|-------------|-------------|
| `pts_allowed_per_game` | Average points allowed | CompactResults |
| `opp_fg_pct` | Opponent field goal percentage | DetailedResults |
| `opp_fg3_pct` | Opponent 3-point percentage | DetailedResults |
| `def_rebounds_per_game` | Defensive rebounds per game | DetailedResults |
| `steals_per_game` | Steals per game | DetailedResults |
| `blocks_per_game` | Blocks per game | DetailedResults |

#### Composite Features
| Feature | Description | How to Calculate |
|---------|-------------|-----------------|
| `scoring_margin` | Average win/loss margin | pts_scored - pts_allowed per game |
| `off_efficiency` | Points per possession (approx) | Score / estimated_possessions |
| `def_efficiency` | Points allowed per possession | Opp_score / estimated_possessions |
| `net_efficiency` | Offensive - Defensive efficiency | off_efficiency - def_efficiency |
| `tempo` | Estimated possessions per game | FGA - OR + TO + 0.475*FTA |
| `win_pct` | Season win percentage | Wins / total games |

#### Late-Season Form (EXTRA WEIGHT — last ~4 weeks before tournament)
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `recent_win_pct` | Win % in last ~4 weeks (DayNum 100-132) | Teams peak or slump heading into March |
| `recent_scoring_margin` | Avg margin in last ~4 weeks | Captures momentum / decline |
| `recent_off_efficiency` | Offensive efficiency last ~4 weeks | Are they heating up? |
| `recent_def_efficiency` | Defensive efficiency last ~4 weeks | Is their defense tightening? |

**Implementation:** We'll compute all features twice — full season AND last-4-weeks — and include both. The model can learn which timeframe matters more for different stats.

---

### Layer 2: Historical / Structural Calibration (LIGHTER WEIGHT)

These features come from patterns that repeat across many years of tournaments.

| Feature | Description | Source |
|---------|-------------|--------|
| `seed` | Tournament seed (1-16) | NCAATourneySeeds |
| `seed_diff` | Difference in seeds between matchup teams | Calculated |
| `historical_seed_win_rate` | How often this seed beats that seed historically | TourneyCompactResults + Seeds |
| `massey_ordinal_avg` | Average ranking across multiple systems | MMasseyOrdinals (men only) |
| `massey_ordinal_best` | Best (lowest) ranking across systems | MMasseyOrdinals (men only) |
| `conf_strength` | Average performance of conference teams | Calculated from results |
| `tourney_experience` | # of times team made tournament in last 5 years | TourneySeeds historical |

---

### Layer 3: Randomness / Calibration Layer

After the model outputs a raw probability, we apply shrinkage:

```
calibrated_prob = shrinkage_factor * raw_prob + (1 - shrinkage_factor) * 0.5
```

With `shrinkage_factor` around 0.5 to 0.6, this maps:
- Raw 0.90 → Calibrated ~0.70-0.74 → After aggressive shrink ~0.60-0.62
- Raw 0.75 → Calibrated ~0.62-0.65 → After aggressive shrink ~0.56-0.58
- Raw 0.50 → Stays at 0.50

We'll tune the exact shrinkage factor by testing against Stage 1 (2022-2025) results.

**Additional randomness approaches to test:**
- **Platt scaling:** Fit a logistic regression on top of model outputs to calibrate
- **Temperature scaling:** Divide model logits by a temperature parameter > 1
- **Seed-aware shrinkage:** Shrink more for close seed matchups (7v10), less for extreme ones (1v16)

---

## Matchup Feature Engineering

For each possible matchup (Team A vs Team B), we create features as DIFFERENCES:

```
matchup_scoring_margin = TeamA_scoring_margin - TeamB_scoring_margin
matchup_off_efficiency = TeamA_off_efficiency - TeamB_off_efficiency
matchup_seed_diff = TeamA_seed - TeamB_seed
... (same pattern for all features)
```

By convention, Team A is always the lower TeamID (matching the submission format).

---

## Model Pipeline

```
┌─────────────────────────────────────────────────┐
│  1. LOAD & CLEAN DATA                           │
│     - Load all CSVs                             │
│     - Handle missing values                     │
│     - Align men's and women's data formats      │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  2. BUILD TEAM PROFILES (per season)            │
│     - Full season stats                         │
│     - Late-season stats (last 4 weeks)          │
│     - Rankings (men's Massey Ordinals)          │
│     - Seed info                                 │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  3. CREATE MATCHUP FEATURES                     │
│     - For each possible A-vs-B pairing          │
│     - Compute feature differences               │
│     - Add historical seed matchup rates         │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  4. TRAIN MODEL                                 │
│     - Train on recent tournament matchups       │
│     - Start: Logistic Regression (baseline)     │
│     - Then: XGBoost / LightGBM                  │
│     - Separate models for men's + women's       │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  5. CALIBRATE & SHRINK                          │
│     - Apply aggressive shrinkage toward 0.50    │
│     - Target prediction range: ~0.35 to ~0.65   │
│     - Tune shrinkage factor on Stage 1 data     │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  6. GENERATE SUBMISSION                         │
│     - Stage 1: 2022-2025 matchups (validate)    │
│     - Stage 2: 2026 matchups (competition)      │
└─────────────────────────────────────────────────┘
```

---

## Women's Model Considerations

The women's model follows the same pipeline but with key differences:
- **No Massey Ordinals** — we lose the ranking features, so team stats carry more weight
- **Less historical data** — detailed results only go back to 2010 (vs 2003 for men)
- **Different game dynamics** — women's game has different tempo, scoring patterns
- **~1.5% missing detailed data** in 2010-2012 seasons — need to handle gracefully
- **Tournament scheduling varies** — can't rely on DayNum for round detection like men's

---

## Training Data Strategy

| What | Years | Purpose |
|------|-------|---------|
| Feature engineering | 2003-2026 (men), 2010-2026 (women) | Build team profiles per season |
| Model training | 2015-2025 tournaments | Learn matchup outcome patterns |
| Validation (Stage 1) | 2022-2025 tournaments | Test predictions against known results |
| Final predictions (Stage 2) | 2026 tournament | The actual competition submission |

**Why start training at 2015?** College basketball evolves — the 3-point revolution, pace of play changes, rule changes. Recent tournaments are more representative of 2026 than the 2003-2010 era. But we'll test this assumption by comparing models trained on different year ranges.

---

## Success Metrics

| Milestone | Target | How We Know |
|-----------|--------|-------------|
| Baseline (seed-only) | Log loss ~0.55-0.60 | First submission |
| + Team stats | Log loss ~0.50-0.55 | Should beat seed-only |
| + Rankings + calibration | Log loss ~0.48-0.52 | Competitive range |
| + Shrinkage tuning | Log loss < 0.50 | Top-tier if achieved |

(These are rough estimates based on historical competition scores.)

---

## Open Questions

- [ ] Exact shrinkage factor — need to tune on Stage 1
- [ ] Which Massey Ordinal systems are most predictive? (Test top 5-10 vs. using all)
- [ ] How much does late-season weighting actually help? (A/B test full-season vs. recent)
- [ ] Should we use any geography features? (Home court proximity in early rounds?)
- [ ] Coach tournament experience — does it add signal or just noise?
