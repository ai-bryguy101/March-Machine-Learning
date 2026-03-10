"""
preprocessing.py — Transform raw Kaggle CSVs into model-ready data structures.

This module handles:
1. Loading all CSV files
2. Computing per-game feature vectors for each team
3. Organizing games chronologically per season
4. Preparing data for the sequential Game Processor

Key concept: We don't compute season averages. Instead, we preserve the
game-by-game sequence so the GRU can process games in order and build
team embeddings incrementally.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# These are the raw stats we extract from each game for each team.
# The Team Encoder will project these into 512-dimensional embeddings.

# From Detailed Results files (available 2003+ men, 2010+ women)
DETAILED_STATS = [
    'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
    'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF'
]

# Computed features we derive from raw stats
COMPUTED_FEATURES = [
    'FGPct',        # Field goal percentage (FGM / FGA)
    'FG3Pct',       # 3-point percentage (FGM3 / FGA3)
    'FTPct',        # Free throw percentage (FTM / FTA)
    'Possessions',  # Estimated possessions (FGA - OR + TO + 0.475 * FTA)
    'OffEff',       # Offensive efficiency (Score / Possessions * 100)
    'DefEff',       # Defensive efficiency (OppScore / Possessions * 100)
    'NetEff',       # Net efficiency (OffEff - DefEff)
    'Tempo',        # Pace of play (Possessions per game)
    'TORatio',      # Turnover ratio (TO / Possessions)
    'AstRatio',     # Assist ratio (Ast / FGM)
    'RebMargin',    # Rebound margin (OR + DR - OppOR - OppDR)
    'ScoreMargin',  # Point differential (Score - OppScore)
]

# Total features per team per game
# 13 raw stats + 12 computed + some context features (home/away, DayNum, etc.)
TOTAL_FEATURES_PER_GAME = len(DETAILED_STATS) + len(COMPUTED_FEATURES) + 3  # +3 for context


def load_all_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the raw data directory into a dictionary.
    
    Args:
        data_dir: Path to folder containing all Kaggle CSVs
        
    Returns:
        Dictionary mapping filename (without .csv) to DataFrame
        
    Example:
        data = load_all_data('data/raw/')
        men_teams = data['MTeams']
        women_results = data['WRegularSeasonDetailedResults']
    """
    all_data = {}
    
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.csv'):
            key = filename.replace('.csv', '')
            filepath = os.path.join(data_dir, filename)
            all_data[key] = pd.read_csv(filepath)
            print(f"  Loaded {key}: {all_data[key].shape}")
    
    print(f"\nTotal files loaded: {len(all_data)}")
    return all_data


def compute_game_features(row: pd.Series, team_perspective: str) -> np.ndarray:
    """
    Compute the feature vector for ONE team in ONE game.
    
    This is called twice per game — once from the winner's perspective,
    once from the loser's perspective. The features are the same but
    the "team" vs "opponent" columns swap.
    
    Args:
        row: A single row from a DetailedResults DataFrame
        team_perspective: Either 'W' (winner) or 'L' (loser)
        
    Returns:
        numpy array of features for this team in this game
    """
    # Determine which prefix is "us" vs "opponent"
    if team_perspective == 'W':
        us, them = 'W', 'L'
    else:
        us, them = 'L', 'W'
    
    # --- Raw stats (13 features) ---
    raw = []
    for stat in DETAILED_STATS:
        raw.append(row[f'{us}{stat}'])
    
    # --- Computed features (12 features) ---
    fgm = row[f'{us}FGM']
    fga = row[f'{us}FGA']
    fgm3 = row[f'{us}FGM3']
    fga3 = row[f'{us}FGA3']
    ftm = row[f'{us}FTM']
    fta = row[f'{us}FTA']
    oreb = row[f'{us}OR']
    dreb = row[f'{us}DR']
    ast = row[f'{us}Ast']
    to = row[f'{us}TO']
    score = row[f'{us}Score']
    
    opp_score = row[f'{them}Score']
    opp_or = row[f'{them}OR']
    opp_dr = row[f'{them}DR']
    opp_fga = row[f'{them}FGA']
    opp_to = row[f'{them}TO']
    opp_fta = row[f'{them}FTA']
    
    # Percentages (handle division by zero)
    fg_pct = fgm / fga if fga > 0 else 0.0
    fg3_pct = fgm3 / fga3 if fga3 > 0 else 0.0
    ft_pct = ftm / fta if fta > 0 else 0.0
    
    # Possessions estimate (standard formula)
    possessions = fga - oreb + to + 0.475 * fta
    opp_possessions = opp_fga - opp_or + opp_to + 0.475 * opp_fta
    avg_possessions = (possessions + opp_possessions) / 2
    
    # Efficiency ratings (points per 100 possessions)
    off_eff = (score / avg_possessions * 100) if avg_possessions > 0 else 0.0
    def_eff = (opp_score / avg_possessions * 100) if avg_possessions > 0 else 0.0
    net_eff = off_eff - def_eff
    
    # Tempo
    tempo = avg_possessions
    
    # Ratios
    to_ratio = to / avg_possessions if avg_possessions > 0 else 0.0
    ast_ratio = ast / fgm if fgm > 0 else 0.0
    
    # Margins
    reb_margin = (oreb + dreb) - (opp_or + opp_dr)
    score_margin = score - opp_score
    
    computed = [
        fg_pct, fg3_pct, ft_pct,
        avg_possessions, off_eff, def_eff, net_eff, tempo,
        to_ratio, ast_ratio, reb_margin, score_margin
    ]
    
    # --- Context features (3 features) ---
    if team_perspective == 'W':
        loc = row['WLoc']
        home_val = 1.0 if loc == 'H' else (-1.0 if loc == 'A' else 0.0)
    else:
        loc = row['WLoc']
        home_val = -1.0 if loc == 'H' else (1.0 if loc == 'A' else 0.0)
    
    day_num = row['DayNum'] / 154.0  # Normalize to [0, 1] range
    num_ot = row['NumOT'] / 4.0      # Normalize (4 OT is extremely rare)
    
    context = [home_val, day_num, num_ot]
    
    return np.array(raw + computed + context, dtype=np.float32)


def build_season_game_sequences(
    detailed_results: pd.DataFrame,
    season: int
) -> Dict[int, List[dict]]:
    """
    For a given season, build a chronological list of games for every team.
    
    Each team gets a list of games in the order they were played, with
    the feature vector from that team's perspective and info about their opponent.
    
    Args:
        detailed_results: DataFrame from DetailedResults CSV
        season: Which season to process (e.g., 2024)
        
    Returns:
        Dictionary mapping TeamID → list of game dicts, sorted by DayNum
    """
    season_data = detailed_results[detailed_results['Season'] == season].copy()
    season_data = season_data.sort_values('DayNum')
    
    team_sequences = {}
    
    for _, row in season_data.iterrows():
        w_team = row['WTeamID']
        l_team = row['LTeamID']
        
        # Winner's perspective
        w_features = compute_game_features(row, 'W')
        if w_team not in team_sequences:
            team_sequences[w_team] = []
        team_sequences[w_team].append({
            'day_num': int(row['DayNum']),
            'features': w_features,
            'opponent_id': int(l_team),
            'won': True,
            'score_margin': int(row['WScore'] - row['LScore']),
        })
        
        # Loser's perspective (same game, different viewpoint)
        l_features = compute_game_features(row, 'L')
        if l_team not in team_sequences:
            team_sequences[l_team] = []
        team_sequences[l_team].append({
            'day_num': int(row['DayNum']),
            'features': l_features,
            'opponent_id': int(w_team),
            'won': False,
            'score_margin': int(row['LScore'] - row['WScore']),
        })
    
    # Sort each team's games chronologically
    for team_id in team_sequences:
        team_sequences[team_id].sort(key=lambda g: g['day_num'])
    
    return team_sequences


def get_tournament_matchups(
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
    season: int
) -> List[dict]:
    """
    Get all tournament matchups for a given season with their outcomes.
    Standardized so lower TeamID = team_a (matching submission format).
    """
    season_games = tourney_results[tourney_results['Season'] == season]
    season_seeds = seeds[seeds['Season'] == season].set_index('TeamID')
    
    matchups = []
    for _, row in season_games.iterrows():
        w_team = row['WTeamID']
        l_team = row['LTeamID']
        
        team_a = min(w_team, l_team)
        team_b = max(w_team, l_team)
        team_a_won = (team_a == w_team)
        
        seed_a = int(season_seeds.loc[team_a, 'Seed'][1:3]) if team_a in season_seeds.index else 16
        seed_b = int(season_seeds.loc[team_b, 'Seed'][1:3]) if team_b in season_seeds.index else 16
        
        matchups.append({
            'season': season,
            'team_a': team_a,
            'team_b': team_b,
            'team_a_won': team_a_won,
            'seed_a': seed_a,
            'seed_b': seed_b,
        })
    
    return matchups


def normalize_features(
    all_game_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and std across all games, then normalize.
    Neural networks train much better when inputs are centered around 0.
    """
    means = np.mean(all_game_features, axis=0)
    stds = np.std(all_game_features, axis=0)
    stds[stds == 0] = 1.0
    normalized = (all_game_features - means) / stds
    return normalized, means, stds


def preprocess_all(data_dir: str, gender: str = 'M') -> dict:
    """
    Run the full preprocessing pipeline for men's or women's data.
    
    Args:
        data_dir: Path to raw data CSVs
        gender: 'M' for men's, 'W' for women's
        
    Returns:
        Dictionary with game_sequences, tournament_matchups, normalization stats, etc.
    """
    prefix = gender
    print(f"\n{'='*60}")
    print(f"  Preprocessing {'Mens' if gender == 'M' else 'Womens'} Data")
    print(f"{'='*60}\n")
    
    print("Loading CSV files...")
    all_data = load_all_data(data_dir)
    
    teams = all_data[f'{prefix}Teams']
    seeds = all_data[f'{prefix}NCAATourneySeeds']
    
    detailed_key = f'{prefix}RegularSeasonDetailedResults'
    tourney_key = f'{prefix}NCAATourneyCompactResults'
    
    detailed_results = all_data.get(detailed_key)
    tourney_results = all_data.get(tourney_key)
    
    if detailed_results is not None:
        detailed_seasons = sorted(detailed_results['Season'].unique())
        print(f"\nDetailed results available for seasons: {detailed_seasons[0]}-{detailed_seasons[-1]}")
    else:
        detailed_seasons = []
    
    # Build game sequences for each season
    print("\nBuilding game sequences...")
    game_sequences = {}
    all_features_for_normalization = []
    
    for season in detailed_seasons:
        game_sequences[season] = build_season_game_sequences(detailed_results, season)
        
        for team_id, games in game_sequences[season].items():
            for game in games:
                all_features_for_normalization.append(game['features'])
        
        num_teams = len(game_sequences[season])
        num_games = sum(len(g) for g in game_sequences[season].values()) // 2
        print(f"  Season {season}: {num_teams} teams, {num_games} games")
    
    # Normalize features
    print("\nNormalizing features...")
    all_features_array = np.stack(all_features_for_normalization)
    _, means, stds = normalize_features(all_features_array)
    
    for season in game_sequences:
        for team_id in game_sequences[season]:
            for game in game_sequences[season][team_id]:
                game['features'] = (game['features'] - means) / stds
    
    # Build tournament matchup labels
    print("\nBuilding tournament matchups...")
    tournament_matchups = {}
    if tourney_results is not None:
        for season in sorted(tourney_results['Season'].unique()):
            if season in game_sequences:
                matchups = get_tournament_matchups(tourney_results, seeds, season)
                tournament_matchups[season] = matchups
                print(f"  Season {season}: {len(matchups)} tournament games")
    
    print(f"\n{'='*60}")
    print(f"  Preprocessing complete!")
    print(f"  Seasons with game data: {len(game_sequences)}")
    print(f"  Seasons with tournament labels: {len(tournament_matchups)}")
    print(f"  Features per game: {TOTAL_FEATURES_PER_GAME}")
    print(f"{'='*60}\n")
    
    return {
        'game_sequences': game_sequences,
        'tournament_matchups': tournament_matchups,
        'feature_means': means,
        'feature_stds': stds,
        'teams': teams,
        'seeds': seeds,
        'num_features': TOTAL_FEATURES_PER_GAME,
    }


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    men_data = preprocess_all(data_dir, gender='M')
    women_data = preprocess_all(data_dir, gender='W')
