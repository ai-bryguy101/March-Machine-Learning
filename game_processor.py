"""
game_processor.py — Component 2 of MarchNet

Processes games sequentially through a season using a GRU (Gated Recurrent Unit).
After each game, both teams' embeddings are UPDATED to reflect what happened.

Think of it like this:
  Before Season: Duke's embedding = "generic good team"
  After Game 1 (beat UNC by 12): embedding = "good team that dominates rivals"
  After Game 2 (lost to unranked): embedding = "good but inconsistent team"
  After Game 30: rich summary of their entire season journey

The GRU decides how much of new game info to incorporate and how much of the
old understanding to keep. Same concept as language models processing text
word-by-word, but here we process basketball games game-by-game.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict


class GameProcessor(nn.Module):
    """
    Sequential game processor that evolves team embeddings through a season.
    
    For each game:
      1. Takes both teams' current embeddings
      2. Combines them with the game's feature vector
      3. Feeds through a GRU cell
      4. Outputs updated embeddings for both teams
      5. Stores game context for attention layer later
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_game_features: int = 28,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # GRU input: team embedding + opponent embedding + game features + win/loss
        gru_input_size = embedding_dim + embedding_dim + num_game_features + 1
        
        # Project combined input to embedding dimension
        self.input_projection = nn.Sequential(
            nn.Linear(gru_input_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # The GRU cell — the "memory" component
        # Reset gate: "How much of old embedding to forget?"
        # Update gate: "How much should this new game change understanding?"
        # The GRU LEARNS these gating decisions from data.
        self.gru = nn.GRUCell(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Learned "game importance" — some games matter more than others
        # A conference tournament game > a November cupcake game
        self.game_importance = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def process_game(
        self,
        team_embedding: torch.Tensor,
        opponent_embedding: torch.Tensor,
        game_features: torch.Tensor,
        won: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single game and return updated team embedding.
        
        Args:
            team_embedding: Current embedding (embedding_dim,)
            opponent_embedding: Opponent's current embedding (embedding_dim,)
            game_features: This game's feature vector (num_features,)
            won: 1.0 for win, 0.0 for loss
            
        Returns:
            updated_embedding: New team embedding after this game
            game_context: Game representation (stored for attention later)
        """
        combined = torch.cat([
            team_embedding,
            opponent_embedding,
            game_features,
            won.unsqueeze(-1) if won.dim() == 0 else won,
        ], dim=-1)
        
        projected = self.input_projection(combined)
        importance = self.game_importance(projected)
        
        # GRU update: blend old embedding with new information
        new_embedding = self.gru(projected, team_embedding)
        
        # Importance weighting — cupcake games barely change embedding
        updated_embedding = team_embedding + importance * (new_embedding - team_embedding)
        updated_embedding = self.layer_norm(updated_embedding)
        
        game_context = projected.detach().clone()
        
        return updated_embedding, game_context
    
    def process_season(
        self,
        initial_embeddings: Dict[int, torch.Tensor],
        season_games: List[dict],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[torch.Tensor]], Dict[int, float]]:
        """
        Process an entire season of games chronologically.

        Args:
            initial_embeddings: {team_id: initial_embedding_tensor}
            season_games: List of game dicts sorted by DayNum, each with:
                - 'team_a', 'team_b': int team IDs
                - 'features_a', 'features_b': game feature tensors
                - 'a_won': bool
                - 'margin_a': score margin from A's perspective (optional)

        Returns:
            final_embeddings: {team_id: embedding} after full season
            game_histories: {team_id: [game_context_tensors]} for attention
            momentum: {team_id: float} exponential moving avg of recent performance
        """
        current_embeddings = {
            tid: emb.clone() for tid, emb in initial_embeddings.items()
        }
        game_histories = {tid: [] for tid in initial_embeddings.keys()}

        # Momentum: exponential moving average of normalized score margins
        # Alpha=0.15 means ~last 6-7 games dominate the signal
        momentum = {tid: 0.0 for tid in initial_embeddings.keys()}
        alpha = 0.15

        for game in season_games:
            team_a = game['team_a']
            team_b = game['team_b']

            if team_a not in current_embeddings or team_b not in current_embeddings:
                continue

            emb_a = current_embeddings[team_a]
            emb_b = current_embeddings[team_b]
            a_won = torch.tensor(1.0 if game['a_won'] else 0.0)
            b_won = torch.tensor(1.0 - a_won)

            new_emb_a, context_a = self.process_game(
                emb_a, emb_b, game['features_a'], a_won
            )
            new_emb_b, context_b = self.process_game(
                emb_b, emb_a, game['features_b'], b_won
            )

            current_embeddings[team_a] = new_emb_a
            current_embeddings[team_b] = new_emb_b
            game_histories[team_a].append(context_a)
            game_histories[team_b].append(context_b)

            # Update momentum with score margin (normalized to [-1, 1] range)
            # If margin not available, use +0.3 for win, -0.3 for loss
            margin_a = game.get('margin_a', (0.3 if game['a_won'] else -0.3))
            margin_b = -margin_a
            # Normalize margin: divide by 20 and clamp (20-pt win → ~1.0)
            norm_margin_a = max(-1.0, min(1.0, margin_a / 20.0))
            norm_margin_b = max(-1.0, min(1.0, margin_b / 20.0))
            momentum[team_a] = (1 - alpha) * momentum[team_a] + alpha * norm_margin_a
            momentum[team_b] = (1 - alpha) * momentum[team_b] + alpha * norm_margin_b

        return current_embeddings, game_histories, momentum


if __name__ == '__main__':
    print("Testing GameProcessor...")
    processor = GameProcessor(embedding_dim=512, num_game_features=28)
    total_params = sum(p.numel() for p in processor.parameters())
    print(f"Total parameters: {total_params:,}")
    
    team_emb = torch.randn(512)
    opp_emb = torch.randn(512)
    features = torch.randn(28)
    updated, ctx = processor.process_game(team_emb, opp_emb, features, torch.tensor(1.0))
    print(f"Embedding: {team_emb.shape} → Updated: {updated.shape}")
    print(f"Game context: {ctx.shape}")
    print("Test passed!")
