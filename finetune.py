"""
finetune.py — Phase 2: Fine-tuning on Tournament Games

After Phase 1 pre-training teaches the model general basketball patterns,
this script fine-tunes specifically on tournament games where:
- Stakes are higher (single elimination)
- Seed matchups matter
- Pressure/experience factors come into play
- Upsets happen at known rates

Key differences from pre-training:
- Uses the FULL pipeline: encoder → GRU season processing → attention → prediction
- Includes seed information
- Includes momentum signals
- Lower learning rate (1e-4) to not destroy pre-trained knowledge
- Trains on ~700 tournament games (2015-2025)

Usage:
    python finetune.py --data-dir data/raw --checkpoint checkpoints/pretrain_M_best.pt
"""

import argparse
import os
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from marchnet import MarchNet, count_parameters
from preprocessing import preprocess_all, NUM_FEATURES


def prepare_tournament_samples(
    data: dict,
    train_seasons: List[int],
    val_seasons: List[int],
) -> Tuple[List[dict], List[dict]]:
    """
    Prepare tournament matchup samples with full season context.

    For each tournament game, we need:
    - Both teams' season game sequences (for GRU processing)
    - Seed information
    - The outcome label

    Returns:
        train_samples, val_samples: Lists of season-grouped matchup data
    """
    game_sequences = data['game_sequences']
    tournament_matchups = data['tournament_matchups']

    def build_samples(seasons):
        samples = []
        for season in seasons:
            if season not in tournament_matchups or season not in game_sequences:
                continue

            season_seqs = game_sequences[season]
            matchups = tournament_matchups[season]

            # Build chronological pairwise game list for GRU processing
            # Collect all games across all teams, deduplicate, sort by day
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
                            season_games.append({
                                'team_a': a_id,
                                'team_b': b_id,
                                'features_a': torch.tensor(game['features'], dtype=torch.float32),
                                'features_b': torch.tensor(
                                    # Find opponent's features for this game
                                    next((g['features'] for g in season_seqs.get(opp_id, [])
                                          if g['day_num'] == game['day_num'] and g['opponent_id'] == team_id),
                                         game['features']),
                                    dtype=torch.float32
                                ),
                                'a_won': game['won'],
                                'margin_a': game['score_margin'],
                                'day_num': game['day_num'],
                            })
                        else:
                            season_games.append({
                                'team_a': a_id,
                                'team_b': b_id,
                                'features_a': torch.tensor(
                                    next((g['features'] for g in season_seqs.get(opp_id, [])
                                          if g['day_num'] == game['day_num'] and g['opponent_id'] == team_id),
                                         game['features']),
                                    dtype=torch.float32
                                ),
                                'features_b': torch.tensor(game['features'], dtype=torch.float32),
                                'a_won': not game['won'],
                                'margin_a': -game['score_margin'],
                                'day_num': game['day_num'],
                            })

            season_games.sort(key=lambda g: g['day_num'])

            samples.append({
                'season': season,
                'team_ids': list(season_seqs.keys()),
                'season_games': season_games,
                'matchups': matchups,
            })

        return samples

    train = build_samples(train_seasons)
    val = build_samples(val_seasons)
    return train, val


def train_on_season(
    model: MarchNet,
    season_data: dict,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> Tuple[float, float, int]:
    """
    Process one season through GRU, then predict all tournament matchups.

    Returns: (total_loss, num_correct, num_games)
    """
    model.train()

    team_ids = season_data['team_ids']
    season_games = season_data['season_games']
    matchups = season_data['matchups']

    if not matchups:
        return 0.0, 0, 0

    # Process the full season through GRU to get final embeddings + histories
    final_embs, histories, momentum = model.process_season(
        team_ids, season_games, device
    )

    total_loss = 0.0
    correct = 0

    for matchup in matchups:
        team_a = matchup['team_a']
        team_b = matchup['team_b']

        if team_a not in final_embs or team_b not in final_embs:
            continue

        emb_a = final_embs[team_a]
        emb_b = final_embs[team_b]
        hist_a = histories[team_a]
        hist_b = histories[team_b]

        seed_diff = torch.tensor(
            matchup['seed_a'] - matchup['seed_b'], dtype=torch.float32
        ).to(device)

        mom_a = momentum.get(team_a, 0.0)
        mom_b = momentum.get(team_b, 0.0)
        momentum_diff = torch.tensor(mom_a - mom_b, dtype=torch.float32).to(device)

        label = torch.tensor(
            1.0 if matchup['team_a_won'] else 0.0, dtype=torch.float32
        ).to(device)

        optimizer.zero_grad()

        prob = model.predict_matchup(
            emb_a, hist_a, emb_b, hist_b, seed_diff, momentum_diff
        )

        loss = criterion(prob.squeeze(), label)
        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        predicted = (prob.squeeze() > 0.5).float()
        correct += (predicted == label).item()

    return total_loss, correct, len(matchups)


def evaluate_season(
    model: MarchNet,
    season_data: dict,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, int]:
    """Evaluate on one season's tournament matchups without updating weights."""
    model.eval()

    team_ids = season_data['team_ids']
    season_games = season_data['season_games']
    matchups = season_data['matchups']

    if not matchups:
        return 0.0, 0, 0

    with torch.no_grad():
        final_embs, histories, momentum = model.process_season(
            team_ids, season_games, device
        )

        total_loss = 0.0
        correct = 0

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
            momentum_diff = torch.tensor(mom_a - mom_b, dtype=torch.float32).to(device)

            label = torch.tensor(
                1.0 if matchup['team_a_won'] else 0.0, dtype=torch.float32
            ).to(device)

            prob = model.predict_matchup(
                final_embs[team_a], histories[team_a],
                final_embs[team_b], histories[team_b],
                seed_diff, momentum_diff,
            )

            loss = criterion(prob.squeeze(), label)
            total_loss += loss.item()
            predicted = (prob.squeeze() > 0.5).float()
            correct += (predicted == label).item()

    return total_loss, correct, len(matchups)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MarchNet on tournament games")
    parser.add_argument('--data-dir', type=str, default='data/raw')
    parser.add_argument('--gender', type=str, default='M', choices=['M', 'W'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to Phase 1 pretrained checkpoint')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Lower LR to preserve pretrained knowledge')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-start', type=int, default=2015)
    parser.add_argument('--train-end', type=int, default=2021)
    parser.add_argument('--val-start', type=int, default=2022)
    parser.add_argument('--val-end', type=int, default=2025)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"\n{'='*60}")
    print(f"  MarchNet Fine-tuning — Phase 2")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Gender: {'Mens' if args.gender == 'M' else 'Womens'}")
    print(f"  Train seasons: {args.train_start}-{args.train_end}")
    print(f"  Val seasons: {args.val_start}-{args.val_end}")

    # Load data
    print(f"\n  Loading data...")
    try:
        data = preprocess_all(args.data_dir, gender=args.gender)
    except Exception as e:
        print(f"\n  Error loading data: {e}")
        print(f"  Make sure Kaggle CSVs are in: {args.data_dir}")
        return

    # Prepare season-level samples
    train_seasons = list(range(args.train_start, args.train_end + 1))
    val_seasons = list(range(args.val_start, args.val_end + 1))

    print(f"\n  Preparing tournament samples...")
    train_samples, val_samples = prepare_tournament_samples(
        data, train_seasons, val_seasons
    )

    train_games = sum(len(s['matchups']) for s in train_samples)
    val_games = sum(len(s['matchups']) for s in val_samples)
    print(f"  Train: {len(train_samples)} seasons, {train_games} tournament games")
    print(f"  Val: {len(val_samples)} seasons, {val_games} tournament games")

    # Create model
    model = MarchNet(
        num_features=data['num_features'],
        embedding_dim=512,
        num_attention_heads=8,
        dropout=0.3,
        shrinkage=0.6,  # Start slightly more confident for tournament
    ).to(device)

    # Load pre-trained weights if available
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n  Loading pretrained weights from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Load with strict=False in case prediction head shape changed
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    else:
        print(f"\n  No pretrained checkpoint — training from scratch")

    print(f"  Total parameters: {count_parameters(model):,}")

    # Optimizer — lower LR for fine-tuning
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.BCELoss()

    # Training loop
    print(f"\n{'='*60}")
    print(f"  Fine-tuning...")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train on each season
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # Shuffle season order each epoch
        season_order = np.random.permutation(len(train_samples))
        for idx in season_order:
            loss, correct, total = train_on_season(
                model, train_samples[idx], optimizer, criterion, device,
                args.gradient_clip
            )
            epoch_loss += loss
            epoch_correct += correct
            epoch_total += total

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        # Validate
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        for sample in val_samples:
            loss, correct, total = evaluate_season(
                model, sample, criterion, device
            )
            val_loss_total += loss
            val_correct += correct
            val_total += total

        val_loss = val_loss_total / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step()
        elapsed = time.time() - start_time

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                args.save_dir, f'finetune_{args.gender}_best.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'feature_means': data['feature_means'],
                'feature_stds': data['feature_stds'],
            }, checkpoint_path)
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")

    # Save final
    final_path = os.path.join(args.save_dir, f'finetune_{args.gender}_final.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'feature_means': data['feature_means'],
        'feature_stds': data['feature_stds'],
    }, final_path)

    print(f"\n{'='*60}")
    print(f"  Fine-tuning complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Models saved to: {args.save_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
