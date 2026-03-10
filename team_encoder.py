"""
team_encoder.py — Component 1 of MarchNet

Projects raw per-game stats into a high-dimensional embedding space.
Uses shared weights so every team is encoded the same way — the network
learns a universal "language" for describing team quality.

Think of it like this:
  Raw stats = "Duke shot 48% from the field and had 12 turnovers"
  Embedding = A 512-number fingerprint that captures what kind of team Duke is
"""

import torch
import torch.nn as nn


class TeamEncoder(nn.Module):
    """
    Encodes raw game statistics into a high-dimensional team embedding.
    
    Architecture:
        Input (28 features) → 256 → ReLU → Dropout
                             → 512 → ReLU → Dropout
                             → 512 (output embedding) → LayerNorm
    
    The same encoder is used for ALL teams (shared weights). This forces
    the network to learn general basketball concepts rather than memorizing
    specific teams.
    """
    
    def __init__(
        self,
        num_features: int = 28,
        embedding_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # The encoding layers
        # Linear: multiplies input by learned weight matrix + bias (the "learning" part)
        # ReLU:   max(0, x) — adds nonlinearity so network can learn complex patterns
        # Dropout: randomly zeros neurons during training (prevents overfitting)
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # Keeps embedding values in a reasonable range for the GRU
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Xavier initialization for stable training
        self._init_weights()
    
    def _init_weights(self):
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, game_stats: torch.Tensor) -> torch.Tensor:
        """
        Encode raw game stats into a team embedding.
        
        Args:
            game_stats: shape (batch_size, num_features) or (num_features,)
        Returns:
            embedding: shape (batch_size, embedding_dim) or (embedding_dim,)
        """
        embedding = self.encoder(game_stats)
        embedding = self.layer_norm(embedding)
        return embedding


if __name__ == '__main__':
    print("Testing TeamEncoder...")
    encoder = TeamEncoder(num_features=28, embedding_dim=512)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
    
    x = torch.randn(32, 28)
    out = encoder(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    print("Test passed!")
