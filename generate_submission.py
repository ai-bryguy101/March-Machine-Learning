"""
generate_submission.py — Produce final Kaggle submission CSV

Takes a trained + calibrated MarchNet and generates predictions for all
possible tournament matchups in the target season(s).

Submission format (from Kaggle):
    ID,Pred
    2026_1101_1102,0.55
    2026_1101_1103,0.48
    ...

Where ID = Season_LowerTeamID_HigherTeamID and Pred = P(lower team wins).

Usage:
    # Stage 1: Predictions for 2022-2025 (validation)
    python generate_submission.py --stage 1 --data-dir data/raw

    # Stage 2: Predictions for 2026 (competition)
    python generate_submission.py --stage 2 --data-dir data/raw
"""

import argparse
import os
from itertools import combinations

import torch
import numpy as np
import pandas as pd

from marchnet import MarchNet
from preprocessing import preprocess_all
from calibrate import apply_seed_aware_shrinkage, get_seed_bucket


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_season_context(data: dict, season: int):
    """Build the game list needed for GRU season processing."""
    game_sequences = data['game_sequences']
    if season not in game_sequences:
        return None, None

    season_seqs = game_sequences[season]
    team_ids = list(season_seqs.keys())

    # Deduplicate and sort games chronologically
    seen_games = set()
    season_games = []

    for team_id, team_games in season_seqs.items():
        for game in team_games:
            opp_id = game['opponent_id']
            game_key = (game['day_num'], min(team_id, opp_id), max(team_id, opp_id))
            if game_key not in seen_games:
                seen_games.add(game_key)
                a_id = min(team_id, opp_id)
                b_id = max(team_id, opp_id)

                if team_id == a_id:
                    # Find opponent's matching game record
                    opp_features = game['features']  # fallback
                    for g in season_seqs.get(opp_id, []):
                        if g['day_num'] == game['day_num'] and g['opponent_id'] == team_id:
                            opp_features = g['features']
                            break

                    season_games.append({
                        'team_a': a_id,
                        'team_b': b_id,
                        'features_a': torch.tensor(game['features'], dtype=torch.float32),
                        'features_b': torch.tensor(opp_features, dtype=torch.float32),
                        'a_won': game['won'],
                        'margin_a': game['score_margin'],
                        'day_num': game['day_num'],
                    })
                else:
                    opp_features = game['features']
                    for g in season_seqs.get(opp_id, []):
                        if g['day_num'] == game['day_num'] and g['opponent_id'] == team_id:
                            opp_features = g['features']
                            break

                    season_games.append({
                        'team_a': a_id,
                        'team_b': b_id,
                        'features_a': torch.tensor(opp_features, dtype=torch.float32),
                        'features_b': torch.tensor(game['features'], dtype=torch.float32),
                        'a_won': not game['won'],
                        'margin_a': -game['score_margin'],
                        'day_num': game['day_num'],
                    })

    season_games.sort(key=lambda g: g['day_num'])
    return team_ids, season_games


def generate_predictions(
    model: MarchNet,
    data: dict,
    season: int,
    tournament_teams: list,
    bucket_shrinkage: dict,
    device: torch.device,
) -> list:
    """
    Generate predictions for all possible pairings of tournament teams.

    Returns list of (ID_string, probability) tuples.
    """
    model.eval()

    # Get seeds for this season
    seeds_df = data['seeds']
    season_seeds = seeds_df[seeds_df['Season'] == season].set_index('TeamID')

    def get_seed(team_id):
        if team_id in season_seeds.index:
            seed_str = season_seeds.loc[team_id, 'Seed']
            if isinstance(seed_str, pd.Series):
                seed_str = seed_str.iloc[0]
            return int(seed_str[1:3])
        return 16  # default

    # Process the full season
    team_ids, season_games = build_season_context(data, season)
    if team_ids is None:
        print(f"  WARNING: No game data for season {season}")
        return []

    with torch.no_grad():
        final_embs, histories, momentum = model.process_season(
            team_ids, season_games, device
        )

    # Set model to raw output for calibration
    old_shrinkage = model.prediction_head.shrinkage
    model.prediction_head.set_shrinkage(1.0)

    predictions = []

    # Generate all possible matchups (lower ID = team_a)
    for team_a, team_b in combinations(sorted(tournament_teams), 2):
        if team_a not in final_embs or team_b not in final_embs:
            # Team not in our data — predict 0.5 (safe default)
            game_id = f"{season}_{team_a}_{team_b}"
            predictions.append((game_id, 0.5))
            continue

        seed_a = get_seed(team_a)
        seed_b = get_seed(team_b)
        seed_diff = torch.tensor(seed_a - seed_b, dtype=torch.float32).to(device)

        mom_a = momentum.get(team_a, 0.0)
        mom_b = momentum.get(team_b, 0.0)
        momentum_diff = torch.tensor(mom_a - mom_b, dtype=torch.float32).to(device)

        with torch.no_grad():
            raw_prob = model.predict_matchup(
                final_embs[team_a], histories[team_a],
                final_embs[team_b], histories[team_b],
                seed_diff, momentum_diff,
            ).squeeze().item()

        # Apply seed-aware calibration
        calibrated = apply_seed_aware_shrinkage(
            raw_prob, seed_a, seed_b, bucket_shrinkage
        )

        game_id = f"{season}_{team_a}_{team_b}"
        predictions.append((game_id, calibrated))

    model.prediction_head.set_shrinkage(old_shrinkage)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission CSV")
    parser.add_argument('--data-dir', type=str, default='data/raw')
    parser.add_argument('--gender', type=str, default='M', choices=['M', 'W'])
    parser.add_argument('--model-checkpoint', type=str, default=None,
                        help='Path to fine-tuned model')
    parser.add_argument('--cal-checkpoint', type=str, default=None,
                        help='Path to calibration config')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                        help='Stage 1 = validation (2022-2025), Stage 2 = competition (2026)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: submission_stage{N}_{gender}.csv)')

    args = parser.parse_args()
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  MarchNet Submission Generator")
    print(f"{'='*60}")
    print(f"  Stage: {args.stage}")
    print(f"  Gender: {'Mens' if args.gender == 'M' else 'Womens'}")

    # Load data
    data = preprocess_all(args.data_dir, gender=args.gender)

    # Load model
    model = MarchNet(
        num_features=data['num_features'],
        embedding_dim=512,
        num_attention_heads=8,
        dropout=0.0,  # No dropout for inference
        shrinkage=1.0,  # Applied manually via calibration
    ).to(device)

    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"  Loaded model: {args.model_checkpoint}")
    else:
        print(f"  WARNING: No model checkpoint — using random weights (predictions will be noise)")

    # Load calibration
    bucket_shrinkage = {
        'blowout': 0.8, 'heavy_favorite': 0.65,
        'moderate': 0.5, 'competitive': 0.4, 'tossup': 0.3,
    }
    if args.cal_checkpoint and os.path.exists(args.cal_checkpoint):
        cal = torch.load(args.cal_checkpoint, map_location='cpu')
        bucket_shrinkage = cal['bucket_shrinkage']
        print(f"  Loaded calibration: {args.cal_checkpoint}")
        print(f"    Calibrated log loss: {cal['calibrated_log_loss']:.6f}")
    else:
        print(f"  Using default calibration (no checkpoint)")

    print(f"\n  Calibration config:")
    for bucket, s in sorted(bucket_shrinkage.items()):
        print(f"    {bucket:20s}: {s:.3f}")

    # Determine target seasons and tournament teams
    seeds_df = data['seeds']

    if args.stage == 1:
        target_seasons = [2022, 2023, 2024, 2025]
    else:
        target_seasons = [2026]

    # Generate predictions
    all_predictions = []
    for season in target_seasons:
        season_seeds = seeds_df[seeds_df['Season'] == season]
        if season_seeds.empty:
            print(f"\n  No seeds found for {season} — skipping")
            continue

        tournament_teams = sorted(season_seeds['TeamID'].unique().tolist())
        print(f"\n  Season {season}: {len(tournament_teams)} tournament teams")
        n_matchups = len(tournament_teams) * (len(tournament_teams) - 1) // 2
        print(f"  Generating {n_matchups} matchup predictions...")

        preds = generate_predictions(
            model, data, season, tournament_teams, bucket_shrinkage, device
        )
        all_predictions.extend(preds)
        print(f"  Done: {len(preds)} predictions")

    # Build submission DataFrame
    df = pd.DataFrame(all_predictions, columns=['ID', 'Pred'])

    # Sanity checks
    print(f"\n  Submission stats:")
    print(f"    Total predictions: {len(df)}")
    print(f"    Pred range: [{df['Pred'].min():.4f}, {df['Pred'].max():.4f}]")
    print(f"    Pred mean: {df['Pred'].mean():.4f}")
    print(f"    Pred std: {df['Pred'].std():.4f}")

    # Save
    output_path = args.output or f"submission_stage{args.stage}_{args.gender}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Saved submission to: {output_path}")

    print(f"\n{'='*60}")
    print(f"  Submission generated!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
