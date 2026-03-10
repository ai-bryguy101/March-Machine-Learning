# PROJECT LOG — March Machine Learning Mania 2026

This document tracks progress session-by-session. Upload this to Claude at the start of each session to pick up where you left off.

---

## Competition Quick Reference

- **What:** Predict win probabilities for every possible NCAA tournament matchup
- **Models needed:** 2 (one for men's, one for women's)
- **Submission format:** CSV with `ID` (SSSS_XXXX_YYYY) and `Pred` (probability lower-ID team wins)
- **Evaluation metric:** Log loss (lower is better)
- **Stage 1:** Predict 2022-2025 matchups (for model development — scored against known results)
- **Stage 2:** Predict 2026 matchups (the real competition — scored after tournament ends)
- **Key dates:** Selection Sunday is March 15, 2026 (DayNum=132). Tournament starts ~DayNum=134.

---

## Data Inventory

### Files We Have (35 CSVs, ~180MB total)

**Section 1 — The Basics (1985-present for men, 1998-present for women)**
| File | Description | Key Columns |
|------|-------------|-------------|
| MTeams / WTeams | Team IDs and names | TeamID, TeamName |
| MSeasons / WSeasons | Season info, DayZero dates, region names | Season, DayZero, RegionW/X/Y/Z |
| MNCAATourneySeeds / WNCAATourneySeeds | Tournament seeds by year | Season, Seed, TeamID |
| MRegularSeasonCompactResults / W... | Game scores (all regular season) | Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT |
| MNCAATourneyCompactResults / W... | Game scores (tournament only) | Same as above |
| SampleSubmissionStage1 / Stage2 | Submission format templates | ID, Pred |

**Section 2 — Detailed Box Scores (2003+ men, 2010+ women)**
| File | Description | Extra Columns |
|------|-------------|---------------|
| MRegularSeasonDetailedResults / W... | Per-game team stats | FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF (for W and L teams) |
| MNCAATourneyDetailedResults / W... | Same but for tournament games | Same as above |

**Section 3 — Geography (2010+)**
| File | Description |
|------|-------------|
| Cities | City names and states |
| MGameCities / WGameCities | Which city each game was played in |

**Section 4 — Rankings (Men only, 2003+)**
| File | Description |
|------|-------------|
| MMasseyOrdinals | Weekly rankings from dozens of systems (Pomeroy, Sagarin, RPI, ESPN, etc.) |

**Section 5 — Supplements**
| File | Description |
|------|-------------|
| MTeamCoaches | Head coach per team per season |
| Conferences | Conference names and abbreviations |
| MTeamConferences / WTeamConferences | Which conference each team was in each year |
| MConferenceTourneyGames / W... | Conference tournament game identifiers |
| MSecondaryTourneyTeams / W... | NIT and other post-season participants |
| MSecondaryTourneyCompactResults / W... | NIT and other post-season scores |
| MTeamSpellings / WTeamSpellings | Alternative team name mappings |
| MNCAATourneySlots / WNCAATourneySlots | Bracket structure (how seeds pair up) |
| MNCAATourneySeedRoundSlots | Men's bracket DayNum mapping per seed/round |

---

## Key Concepts for a Beginner

### What is Log Loss?
Log loss measures how confident AND correct your predictions are. If you predict 90% chance Team A wins and they do win, that's great (low loss). But if they lose, you get heavily penalized. Predicting 50% for everything is safe but won't win. The goal is to be both confident and right.

### What Makes a Good March Madness Model?
- **Seed** is the single strongest predictor (a 1-seed almost always beats a 16-seed)
- **Team strength metrics** (offensive/defensive efficiency, tempo, etc.) add signal
- **Recent form** matters more than early-season performance
- **Conference strength** helps contextualize win/loss records
- **Historical upset patterns** (certain seed matchups produce more upsets)

### Our Modeling Plan (Simple → Complex)
1. **Baseline:** Seed-based model (just use historical seed-vs-seed win rates)
2. **Level 2:** Add team stats (scoring margin, shooting %, turnover rate, etc.)
3. **Level 3:** Use ranking systems (Massey Ordinals) as features
4. **Level 4:** Ensemble multiple models together for final predictions

---

## Session Log

### Session 1 — [DATE: 2026-03-10]
**What we did:**
- Reviewed competition rules — confirmed AI/LLM tools are allowed
- Reviewed full dataset description (35 CSVs across 5 sections)
- Set up project structure and documentation
- Decision: Build separate men's and women's models
- Decision: Use GitHub for version control

**Key decisions:**
- Start with men's model first (more data available, has Massey Ordinals)
- Women's model will follow same pipeline but with adapted features
- Approach: Start simple (seed-based baseline), iterate toward complexity

**Next session — TODO:**
- [ ] Upload Kaggle data files to `data/raw/`
- [ ] Initial data exploration in a Jupyter notebook
  - Load each CSV, check shapes, dtypes, missing values
  - Understand the DayNum system and how seasons work
  - Look at seed vs. outcome distributions
- [ ] Build a seed-based baseline model
- [ ] Generate first Stage 1 submission to establish a benchmark score

**Questions to resolve:**
- How far back should we train? (All data since 1985, or only recent years with detailed stats?)
- Which Massey Ordinal systems are most predictive?
- How to handle the women's model lacking Massey Ordinals data?

---

*Add new sessions below this line. Copy the session template:*

### Session N — [DATE: YYYY-MM-DD]
**What we did:**
- 

**Key decisions:**
- 

**Results:**
- 

**Next session — TODO:**
- [ ] 

---
