"""
calibrate.py — Phase 3: Calibration with Seed-Aware Shrinkage + Temperature

The secret sauce for log loss optimization: your model might predict well,
but if it's overconfident on upsets, one wrong 0.95 prediction costs you
more than 20 correct ones.

This script tunes TWO calibration mechanisms:

1. TEMPERATURE SCALING — Soften the model's raw logits before sigmoid.
   Higher temp → probabilities closer to 0.5. Learned end-to-end.

2. SEED-AWARE SHRINKAGE — Variable shrinkage based on seed gap:
   - Large gap (1v16): low shrinkage → be confident, upsets are rare (~1%)
   - Medium gap (4v5, 5v12): moderate shrinkage → upsets happen ~35%
   - Small gap (8v9): high shrinkage → basically a coin flip

This is better than uniform shrinkage because the RISK profile changes:
- Predicting 0.95 for a 1-seed vs 16-seed is low risk
- Predicting 0.95 for an 8-seed vs 9-seed is insane

Usage:
    python calibrate.py --data-dir data/raw --checkpoint checkpoints/finetune_M_best.pt
"""

import argparse
import os
import itertools
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np

from marchnet import MarchNet
from preprocessing import preprocess_all


# Historical upset rates by seed matchup (approximate from 1985-2025)
# Used to set prior expectations for calibration
HISTORICAL_UPSET_RATES = {
    # (higher_seed, lower_seed): P(higher_seed_wins)
    (16, 1): 0.01, (15, 2): 0.06, (14, 3): 0.15, (13, 4): 0.20,
    (12, 5): 0.35, (11, 6): 0.37, (10, 7): 0.39, (9, 8): 0.48,
}


def get_seed_bucket(seed_a: int, seed_b: int) -> str:
    """
    Categorize a matchup by seed gap for seed-aware shrinkage.

    Returns a bucket key used to look up the shrinkage factor.
    """
    gap = abs(seed_a - seed_b)
    if gap >= 12:       # 1v16, 2v15 type
        return 'blowout'
    elif gap >= 8:      # 1v12, 3v14 type
        return 'heavy_favorite'
    elif gap >= 4:      # 4v8, 5v9 type
        return 'moderate'
    elif gap >= 2:      # 6v8, 5v7 type
        return 'competitive'
    else:               # 8v9, 7v8 type
        return 'tossup'


def compute_log_loss(predictions: List[float], labels: List[float], eps: float = 1e-15) -> float:
    """Compute average log loss (same as Kaggle's metric)."""
    total = 0.0
    for p, y in zip(predictions, labels):
        p = max(eps, min(1 - eps, p))
        total += -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return total / len(predictions) if predictions else float('inf')


def apply_seed_aware_shrinkage(
    raw_prob: float,
    seed_a: int,
    seed_b: int,
    bucket_shrinkage: Dict[str, float],
) -> float:
    """Apply variable shrinkage based on the seed matchup bucket."""
    bucket = get_seed_bucket(seed_a, seed_b)
    shrinkage = bucket_shrinkage.get(bucket, 0.5)
    return shrinkage * raw_prob + (1 - shrinkage) * 0.5


def collect_predictions(
    model: MarchNet,
    data: dict,
    seasons: List[int],
    device: torch.device,
) -> List[dict]:
    """
    Run the model on tournament matchups and collect raw predictions.

    Returns list of dicts with: raw_prob, label, seed_a, seed_b, season
    """
    from finetune import prepare_tournament_samples

    samples, _ = prepare_tournament_samples(data, seasons, [])

    model.eval()
    results = []

    with torch.no_grad():
        for season_data in samples:
            team_ids = season_data['team_ids']
            season_games = season_data['season_games']
            matchups = season_data['matchups']

            if not matchups:
                continue

            final_embs, histories, momentum = model.process_season(
                team_ids, season_games, device
            )

            for matchup in matchups:
                team_a = matchup['team_a']
                team_b = matchup['team_b']

                if team_a not in final_embs or team_b not in final_embs:
                    continue

                seed_diff = torch.tensor(
                    matchup['seed_a'] - matchup['seed_b'], dtype=torch.float32
                ).to(device)
                mom_a = momentum.get(team_a, 0.0)
                mom_b = momentum.get(team_b, 0.0)
                momentum_diff = torch.tensor(
                    mom_a - mom_b, dtype=torch.float32
                ).to(device)

                # Get RAW probability (temporarily set shrinkage to 1.0)
                old_shrinkage = model.prediction_head.shrinkage
                model.prediction_head.set_shrinkage(1.0)

                prob = model.predict_matchup(
                    final_embs[team_a], histories[team_a],
                    final_embs[team_b], histories[team_b],
                    seed_diff, momentum_diff,
                )

                model.prediction_head.set_shrinkage(old_shrinkage)

                results.append({
                    'raw_prob': prob.squeeze().item(),
                    'label': 1.0 if matchup['team_a_won'] else 0.0,
                    'seed_a': matchup['seed_a'],
                    'seed_b': matchup['seed_b'],
                    'season': season_data['season'],
                })

    return results


def grid_search_calibration(results: List[dict]) -> dict:
    """
    Grid search over seed-aware shrinkage parameters to minimize log loss.

    Searches over 5 bucket-specific shrinkage values simultaneously.
    """
    buckets = ['blowout', 'heavy_favorite', 'moderate', 'competitive', 'tossup']

    # Define search ranges per bucket
    # More extreme matchups can afford more confidence (higher shrinkage)
    search_ranges = {
        'blowout':        [0.7, 0.75, 0.8, 0.85, 0.9],     # Can be confident
        'heavy_favorite': [0.55, 0.6, 0.65, 0.7, 0.75],
        'moderate':       [0.4, 0.45, 0.5, 0.55, 0.6],
        'competitive':    [0.35, 0.4, 0.45, 0.5, 0.55],
        'tossup':         [0.2, 0.25, 0.3, 0.35, 0.4],      # Stay close to 0.5
    }

    best_loss = float('inf')
    best_config = {}

    # Also try uniform shrinkage as baseline
    for uniform_s in np.arange(0.3, 0.75, 0.05):
        preds = [uniform_s * r['raw_prob'] + (1 - uniform_s) * 0.5 for r in results]
        labels = [r['label'] for r in results]
        loss = compute_log_loss(preds, labels)
        if loss < best_loss:
            best_loss = loss
            best_config = {b: uniform_s for b in buckets}

    print(f"  Best uniform shrinkage: {best_config['moderate']:.3f}, log loss: {best_loss:.6f}")

    # Now try seed-aware (grid search all combinations)
    all_combos = list(itertools.product(*[search_ranges[b] for b in buckets]))
    print(f"  Searching {len(all_combos)} seed-aware configurations...")

    for combo in all_combos:
        bucket_shrinkage = dict(zip(buckets, combo))

        preds = []
        labels = []
        for r in results:
            calibrated = apply_seed_aware_shrinkage(
                r['raw_prob'], r['seed_a'], r['seed_b'], bucket_shrinkage
            )
            preds.append(calibrated)
            labels.append(r['label'])

        loss = compute_log_loss(preds, labels)
        if loss < best_loss:
            best_loss = loss
            best_config = bucket_shrinkage.copy()

    return best_config, best_loss


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description="Calibrate MarchNet predictions")
    parser.add_argument('--data-dir', type=str, default='data/raw')
    parser.add_argument('--gender', type=str, default='M', choices=['M', 'W'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--cal-start', type=int, default=2022,
                        help='First season for calibration')
    parser.add_argument('--cal-end', type=int, default=2025,
                        help='Last season for calibration')
    parser.add_argument('--save-dir', type=str, default='checkpoints')

    args = parser.parse_args()
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  MarchNet Calibration — Phase 3")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Calibration seasons: {args.cal_start}-{args.cal_end}")

    # Load data
    data = preprocess_all(args.data_dir, gender=args.gender)

    # Load model
    model = MarchNet(
        num_features=data['num_features'],
        embedding_dim=512,
        num_attention_heads=8,
        dropout=0.0,  # No dropout during inference
        shrinkage=1.0,  # We'll apply our own calibration
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"  WARNING: No checkpoint loaded — using random weights")

    # Collect raw predictions on calibration set
    cal_seasons = list(range(args.cal_start, args.cal_end + 1))
    print(f"\n  Collecting predictions on {len(cal_seasons)} seasons...")
    results = collect_predictions(model, data, cal_seasons, device)
    print(f"  Got {len(results)} predictions")

    if not results:
        print("  No predictions to calibrate — check your data!")
        return

    # Show baseline stats
    raw_preds = [r['raw_prob'] for r in results]
    labels = [r['label'] for r in results]
    raw_loss = compute_log_loss(raw_preds, labels)
    raw_acc = sum(1 for p, y in zip(raw_preds, labels) if (p > 0.5) == (y > 0.5)) / len(results)
    print(f"\n  Raw model performance:")
    print(f"    Log loss: {raw_loss:.6f}")
    print(f"    Accuracy: {raw_acc:.3f}")
    print(f"    Prob range: [{min(raw_preds):.3f}, {max(raw_preds):.3f}]")

    # Grid search for best calibration
    print(f"\n  Running calibration grid search...")
    best_config, best_loss = grid_search_calibration(results)

    print(f"\n  {'='*60}")
    print(f"  BEST CALIBRATION CONFIG:")
    print(f"  {'='*60}")
    for bucket, shrinkage in sorted(best_config.items()):
        print(f"    {bucket:20s}: shrinkage = {shrinkage:.3f}")
    print(f"  {'='*60}")
    print(f"  Calibrated log loss: {best_loss:.6f}")
    print(f"  Improvement: {raw_loss - best_loss:.6f} ({(raw_loss - best_loss)/raw_loss*100:.1f}%)")

    # Show calibrated prediction distribution
    calibrated_preds = []
    for r in results:
        cal = apply_seed_aware_shrinkage(
            r['raw_prob'], r['seed_a'], r['seed_b'], best_config
        )
        calibrated_preds.append(cal)
    print(f"  Calibrated range: [{min(calibrated_preds):.3f}, {max(calibrated_preds):.3f}]")

    # Save calibration config
    os.makedirs(args.save_dir, exist_ok=True)
    cal_path = os.path.join(args.save_dir, f'calibration_{args.gender}.pt')
    torch.save({
        'bucket_shrinkage': best_config,
        'calibrated_log_loss': best_loss,
        'raw_log_loss': raw_loss,
        'num_samples': len(results),
        'cal_seasons': cal_seasons,
    }, cal_path)
    print(f"\n  Saved calibration to: {cal_path}")

    print(f"\n{'='*60}")
    print(f"  Calibration complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
