"""
attention_matchup.py — Component 3 of MarchNet

The key innovation: For a specific matchup (A vs B), figure out which of A's 
past games are most relevant for predicting how A will perform against B.

Example: Duke vs Kansas
  - Duke's game against Baylor (similar style to Kansas) → HIGH attention weight
  - Duke's game against a weak non-conference team → LOW attention weight
  
This is the same scaled dot-product attention used in transformers, but applied
to basketball game sequences instead of text tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MatchupAttention(nn.Module):
    """
    Multi-head attention for matchup-specific team representations.
    
    Mechanism:
        Query: The opponent's final embedding (what kind of team are they?)
        Keys:  The team's game history (each past game they played)
        Values: Same as keys
        
        Attention weights = softmax(Q · K^T / sqrt(d_k))
        Output = weighted combination of values
        
    This lets the model learn: "Given that we're playing Kansas, which of 
    Duke's past games are most informative about how Duke will perform?"
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert embedding_dim % num_heads == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection (combines multi-head outputs)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization."""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value_sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matchup-specific representation using attention.
        
        Args:
            query: The opponent's embedding (batch_size, embedding_dim)
                   "What kind of team is the opponent?"
            key_value_sequence: The team's game history 
                   (batch_size, seq_len, embedding_dim)
                   "What games did this team play?"
            mask: Optional boolean mask (batch_size, seq_len)
                   True = valid game, False = padding
                   
        Returns:
            output: Matchup-specific representation (batch_size, embedding_dim)
            attention_weights: (batch_size, num_heads, seq_len) for visualization
        """
        batch_size = query.size(0)
        seq_len = key_value_sequence.size(1)
        
        # Handle unbatched inputs
        if query.dim() == 1:
            query = query.unsqueeze(0)
        if key_value_sequence.dim() == 2:
            key_value_sequence = key_value_sequence.unsqueeze(0)
        
        batch_size = query.size(0)
        seq_len = key_value_sequence.size(1)
        
        # Project to Q, K, V
        Q = self.query_proj(query)  # (batch, embedding_dim)
        K = self.key_proj(key_value_sequence)  # (batch, seq_len, embedding_dim)
        V = self.value_proj(key_value_sequence)  # (batch, seq_len, embedding_dim)
        
        # Reshape for multi-head attention
        # Q: (batch, 1, num_heads, head_dim) → we attend from one query
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V: (batch, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch, num_heads, 1, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # → (batch, num_heads, 1, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # mask: (batch, seq_len) → (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # (batch, num_heads, 1, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # → (batch, num_heads, 1, head_dim)
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back: (batch, num_heads, 1, head_dim) → (batch, embedding_dim)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.embedding_dim)
        
        # Output projection
        output = self.output_proj(attended)
        output = self.dropout(output)
        
        # Residual connection with query (the opponent embedding)
        # This preserves information about the opponent while adding matchup context
        output = self.layer_norm(output + query)
        
        # Return attention weights for visualization (squeeze the query dim)
        attention_weights = attention_weights.squeeze(2)  # (batch, num_heads, seq_len)
        
        return output, attention_weights


class BidirectionalMatchupAttention(nn.Module):
    """
    Applies attention in BOTH directions for a matchup:
    1. Team A attending over their history with B as query
    2. Team B attending over their history with A as query
    
    This produces matchup-specific representations for both teams.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Shared attention mechanism (same learned weights for both directions)
        self.attention = MatchupAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    
    def forward(
        self,
        team_a_embedding: torch.Tensor,
        team_a_history: torch.Tensor,
        team_b_embedding: torch.Tensor,
        team_b_history: torch.Tensor,
        mask_a: Optional[torch.Tensor] = None,
        mask_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute bidirectional matchup attention.
        
        Args:
            team_a_embedding: A's final season embedding (batch, emb_dim)
            team_a_history: A's game history (batch, seq_a, emb_dim)
            team_b_embedding: B's final season embedding (batch, emb_dim)
            team_b_history: B's game history (batch, seq_b, emb_dim)
            mask_a, mask_b: Optional padding masks
            
        Returns:
            a_matchup_repr: A's matchup-specific representation
            b_matchup_repr: B's matchup-specific representation
            a_attention: A's attention weights (for visualization)
            b_attention: B's attention weights
        """
        # A attending over A's history, using B as query
        # "Which of A's games are relevant for playing against B?"
        a_matchup_repr, a_attention = self.attention(
            query=team_b_embedding,
            key_value_sequence=team_a_history,
            mask=mask_a,
        )
        
        # B attending over B's history, using A as query
        # "Which of B's games are relevant for playing against A?"
        b_matchup_repr, b_attention = self.attention(
            query=team_a_embedding,
            key_value_sequence=team_b_history,
            mask=mask_b,
        )
        
        return a_matchup_repr, b_matchup_repr, a_attention, b_attention


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("Testing MatchupAttention...")
    print("-" * 40)
    
    attention = MatchupAttention(embedding_dim=512, num_heads=8)
    
    total_params = sum(p.numel() for p in attention.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test single sample
    query = torch.randn(512)  # Opponent embedding
    history = torch.randn(30, 512)  # 30 games of history
    
    output, weights = attention(query, history)
    print(f"\nSingle sample:")
    print(f"  Query (opponent): {query.shape}")
    print(f"  History (games): {history.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {weights.shape}")
    
    # Test batch
    batch_size = 16
    query_batch = torch.randn(batch_size, 512)
    history_batch = torch.randn(batch_size, 30, 512)
    
    output_batch, weights_batch = attention(query_batch, history_batch)
    print(f"\nBatch:")
    print(f"  Query batch: {query_batch.shape}")
    print(f"  History batch: {history_batch.shape}")
    print(f"  Output batch: {output_batch.shape}")
    print(f"  Attention weights batch: {weights_batch.shape}")
    
    # Test with masking (variable sequence lengths)
    mask = torch.ones(batch_size, 30, dtype=torch.bool)
    mask[:, 25:] = False  # Last 5 positions are padding
    
    output_masked, weights_masked = attention(query_batch, history_batch, mask=mask)
    print(f"\nWith masking:")
    print(f"  Attention on padded positions: {weights_masked[0, 0, 25:].sum().item():.6f} (should be ~0)")
    
    print("\n" + "="*40)
    print("Testing BidirectionalMatchupAttention...")
    print("-" * 40)
    
    bidir = BidirectionalMatchupAttention(embedding_dim=512, num_heads=8)
    
    emb_a = torch.randn(batch_size, 512)
    emb_b = torch.randn(batch_size, 512)
    hist_a = torch.randn(batch_size, 28, 512)  # Team A played 28 games
    hist_b = torch.randn(batch_size, 32, 512)  # Team B played 32 games
    
    repr_a, repr_b, attn_a, attn_b = bidir(emb_a, hist_a, emb_b, hist_b)
    print(f"Team A matchup repr: {repr_a.shape}")
    print(f"Team B matchup repr: {repr_b.shape}")
    
    print("\n✓ All tests passed!")
