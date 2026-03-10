"""
marchnet.py — The Complete MarchNet Model

Combines all four components into a unified model:
1. TeamEncoder: Raw stats → 512-dim embeddings
2. GameProcessor: Sequential GRU that evolves embeddings through the season
3. MatchupAttention: Finds relevant past games for a specific opponent
4. PredictionHead: Outputs calibrated win probabilities

Data Flow:
    Season data → Encoder → GRU → Final embeddings + Game histories
                                            ↓
    Matchup (A vs B) → Attention → Matchup-specific representations
                                            ↓
                       Prediction Head → P(A wins) ∈ [~0.35, ~0.65]
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np

from .team_encoder import TeamEncoder
from .game_processor import GameProcessor
from .attention_matchup import BidirectionalMatchupAttention
from .prediction_head import PredictionHead


class MarchNet(nn.Module):
    """
    Complete MarchNet architecture for NCAA tournament prediction.
    
    This model:
    1. Encodes each team's game stats into embeddings
    2. Processes games sequentially to build season-long team representations
    3. Uses attention to find matchup-relevant game history
    4. Predicts win probabilities with calibration shrinkage
    """
    
    def __init__(
        self,
        num_features: int = 28,
        embedding_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.3,
        shrinkage: float = 0.5,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # Component 1: Team Encoder
        self.encoder = TeamEncoder(
            num_features=num_features,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        
        # Component 2: Game Processor (GRU)
        self.game_processor = GameProcessor(
            embedding_dim=embedding_dim,
            num_game_features=num_features,
            dropout=dropout,
        )
        
        # Component 3: Matchup Attention
        self.matchup_attention = BidirectionalMatchupAttention(
            embedding_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # Component 4: Prediction Head
        self.prediction_head = PredictionHead(
            embedding_dim=embedding_dim,
            dropout=dropout,
            shrinkage=shrinkage,
        )
        
        # Learnable initial embedding (used when a team has no prior games)
        self.initial_embedding = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.normal_(self.initial_embedding, std=0.02)
    
    def encode_games(
        self,
        game_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode raw game stats into embeddings.
        
        Args:
            game_features: (batch, num_features) or (num_features,)
            
        Returns:
            embeddings: Same shape with embedding_dim instead of num_features
        """
        return self.encoder(game_features)
    
    def process_season(
        self,
        team_ids: List[int],
        season_games: List[dict],
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Process an entire season to get final team embeddings and histories.
        
        Args:
            team_ids: List of all team IDs in the season
            season_games: Chronological list of games from preprocessing
            device: Device to use
            
        Returns:
            final_embeddings: {team_id: embedding}
            game_histories: {team_id: stacked_history_tensor}
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Initialize all teams with the learnable initial embedding
        initial_embeddings = {
            tid: self.initial_embedding.clone()
            for tid in team_ids
        }
        
        # Process season
        final_embs, histories = self.game_processor.process_season(
            initial_embeddings, season_games, device
        )
        
        # Stack histories into tensors
        stacked_histories = {}
        for tid, hist_list in histories.items():
            if hist_list:
                stacked_histories[tid] = torch.stack(hist_list)
            else:
                # Team with no games gets a single zero-context
                stacked_histories[tid] = self.initial_embedding.unsqueeze(0)
        
        return final_embs, stacked_histories
    
    def predict_matchup(
        self,
        team_a_embedding: torch.Tensor,
        team_a_history: torch.Tensor,
        team_b_embedding: torch.Tensor,
        team_b_history: torch.Tensor,
        seed_diff: torch.Tensor,
        mask_a: Optional[torch.Tensor] = None,
        mask_b: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Predict P(Team A wins) for a specific matchup.
        
        Args:
            team_a_embedding: A's final season embedding
            team_a_history: A's game history (seq_a, emb_dim)
            team_b_embedding: B's final season embedding
            team_b_history: B's game history (seq_b, emb_dim)
            seed_diff: seed_a - seed_b
            mask_a, mask_b: Optional padding masks
            return_attention: Whether to return attention weights
            
        Returns:
            prob: P(A wins), or tuple (prob, attn_a, attn_b) if return_attention
        """
        # Get matchup-specific representations via attention
        repr_a, repr_b, attn_a, attn_b = self.matchup_attention(
            team_a_embedding.unsqueeze(0) if team_a_embedding.dim() == 1 else team_a_embedding,
            team_a_history.unsqueeze(0) if team_a_history.dim() == 2 else team_a_history,
            team_b_embedding.unsqueeze(0) if team_b_embedding.dim() == 1 else team_b_embedding,
            team_b_history.unsqueeze(0) if team_b_history.dim() == 2 else team_b_history,
            mask_a, mask_b,
        )
        
        # Predict
        prob = self.prediction_head(repr_a, repr_b, seed_diff)
        
        if return_attention:
            return prob, attn_a, attn_b
        return prob
    
    def forward(
        self,
        team_a_features: torch.Tensor,
        team_b_features: torch.Tensor,
        seed_diff: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simple forward pass for single-game prediction (without season context).
        
        This is useful for quick predictions when you already have team features
        aggregated, but doesn't use the full sequential/attention capabilities.
        
        Args:
            team_a_features: (batch, num_features) aggregated stats for A
            team_b_features: (batch, num_features) aggregated stats for B
            seed_diff: (batch,) seed differences
            
        Returns:
            prob: (batch,) P(A wins)
        """
        # Encode
        emb_a = self.encoder(team_a_features)
        emb_b = self.encoder(team_b_features)
        
        # Skip attention (use embeddings directly as matchup repr)
        prob = self.prediction_head(emb_a, emb_b, seed_diff)
        
        return prob
    
    def set_shrinkage(self, shrinkage: float):
        """Update the calibration shrinkage factor."""
        self.prediction_head.set_shrinkage(shrinkage)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("Testing MarchNet...")
    print("=" * 60)
    
    model = MarchNet(
        num_features=28,
        embedding_dim=512,
        num_attention_heads=8,
        dropout=0.3,
        shrinkage=0.5,
    )
    
    print(f"\nModel Architecture:")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Encoder params: {count_parameters(model.encoder):,}")
    print(f"  Game Processor params: {count_parameters(model.game_processor):,}")
    print(f"  Attention params: {count_parameters(model.matchup_attention):,}")
    print(f"  Prediction Head params: {count_parameters(model.prediction_head):,}")
    
    # Test simple forward pass
    print("\n" + "-" * 40)
    print("Testing simple forward pass...")
    
    batch_size = 16
    team_a_feats = torch.randn(batch_size, 28)
    team_b_feats = torch.randn(batch_size, 28)
    seeds = torch.randint(-15, 16, (batch_size,))
    
    probs = model(team_a_feats, team_b_feats, seeds)
    print(f"  Input: ({batch_size}, 28) features each")
    print(f"  Output: {probs.shape}")
    print(f"  Prob range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
    
    # Test full matchup prediction with history
    print("\n" + "-" * 40)
    print("Testing full matchup prediction...")
    
    emb_a = torch.randn(512)
    emb_b = torch.randn(512)
    hist_a = torch.randn(28, 512)  # 28 games
    hist_b = torch.randn(32, 512)  # 32 games
    seed_diff = torch.tensor(-3)
    
    prob, attn_a, attn_b = model.predict_matchup(
        emb_a, hist_a, emb_b, hist_b, seed_diff, return_attention=True
    )
    
    print(f"  Team A history: {hist_a.shape}")
    print(f"  Team B history: {hist_b.shape}")
    print(f"  P(A wins): {prob.item():.4f}")
    print(f"  Attention A shape: {attn_a.shape}")
    print(f"  Attention B shape: {attn_b.shape}")
    
    # Show which games got most attention
    avg_attn_a = attn_a.mean(dim=1).squeeze()  # Average across heads
    top_games = avg_attn_a.topk(5)
    print(f"\n  Top 5 attended games from A's history:")
    for i, (idx, weight) in enumerate(zip(top_games.indices, top_games.values)):
        print(f"    Game {idx.item()}: {weight.item():.3f} attention")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
