"""
pretrain.py — Phase 1: Pre-training on Regular Season Games

This script trains MarchNet on ALL regular season games (100k+) to learn:
- How to encode team statistics
- How to update team representations after each game
- General patterns of which teams beat which

After this phase, the model understands basketball but hasn't specialized
for tournament prediction yet. That comes in Phase 2 (finetune.py).

Usage:
    python -m src.training.pretrain --config configs/default.yaml
    python -m src.training.pretrain --data-dir data/raw --epochs 20
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models import MarchNet, count_parameters
from src.data import preprocess_all, NUM_FEATURES


class SeasonGameDataset(Dataset):
    """
    Dataset for pre-training: predicts the outcome of each game.
    
    Each sample is a single game with features for both teams.
    The model predicts P(team_a wins).
    """
    
    def __init__(
        self,
        all_games: Dict[int, List[dict]],
        seasons: Optional[List[int]] = None,
    ):
        """
        Args:
            all_games: {season: [game_dicts]} from preprocessing
            seasons: Which seasons to include (None = all)
        """
        self.samples = []
        
        seasons_to_use = seasons if seasons else list(all_games.keys())
        
        for season in seasons_to_use:
            if season not in all_games:
                continue
            for game in all_games[season]:
                self.samples.append({
                    'features_a': torch.tensor(game['features_a'], dtype=torch.float32),
                    'features_b': torch.tensor(game['features_b'], dtype=torch.float32),
                    'label': torch.tensor(1.0 if game['a_won'] else 0.0, dtype=torch.float32),
                    'seed_diff': torch.tensor(0.0, dtype=torch.float32),  # No seeds in regular season
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_epoch(
    model: MarchNet,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        avg_loss: Average loss over the epoch
        accuracy: Prediction accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        features_a = batch['features_a'].to(device)
        features_b = batch['features_b'].to(device)
        labels = batch['label'].to(device)
        seed_diff = batch['seed_diff'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (simple mode without attention for pre-training)
        probs = model(features_a, features_b, seed_diff)
        
        # Loss
        loss = criterion(probs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Stats
        total_loss += loss.item() * len(labels)
        predictions = (probs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += len(labels)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: MarchNet,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            features_a = batch['features_a'].to(device)
            features_b = batch['features_b'].to(device)
            labels = batch['label'].to(device)
            seed_diff = batch['seed_diff'].to(device)
            
            probs = model(features_a, features_b, seed_diff)
            loss = criterion(probs, labels)
            
            total_loss += loss.item() * len(labels)
            predictions = (probs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Pre-train MarchNet on regular season games")
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Path to raw CSV data')
    parser.add_argument('--gender', type=str, default='M', choices=['M', 'W'], help="Men's or Women's")
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--gradient-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  MarchNet Pre-training — Phase 1")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Gender: {'Mens' if args.gender == 'M' else 'Womens'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    
    # Load and preprocess data
    print(f"\n{'='*60}")
    print("  Loading data...")
    
    try:
        data = preprocess_all(args.data_dir, gender=args.gender)
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        print(f"   Make sure Kaggle CSVs are in: {args.data_dir}")
        return
    
    all_games = data['all_games']
    num_features = data['num_features']
    
    # Create dataset
    print("  Creating dataset...")
    full_dataset = SeasonGameDataset(all_games)
    print(f"  Total games: {len(full_dataset):,}")
    
    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    print(f"  Train: {train_size:,}, Val: {val_size:,}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for now
        pin_memory=True if device.type == 'cuda' else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )
    
    # Create model
    print(f"\n{'='*60}")
    print("  Creating model...")
    model = MarchNet(
        num_features=num_features,
        embedding_dim=512,
        num_attention_heads=8,
        dropout=0.3,
        shrinkage=0.5,
    ).to(device)
    
    print(f"  Total parameters: {count_parameters(model):,}")
    
    # Optimizer and loss
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
    print("  Training...")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.gradient_clip
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {elapsed:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                args.save_dir, f'pretrain_{args.gender}_best.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    # Save final model
    final_path = os.path.join(args.save_dir, f'pretrain_{args.gender}_final.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, final_path)
    
    print(f"\n{'='*60}")
    print(f"  Pre-training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Models saved to: {args.save_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
