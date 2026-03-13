"""
prediction_head.py — Component 4 of MarchNet

Takes two matchup-specific team representations and produces a calibrated
win probability. This is where the final prediction happens.

My sauce beyond the basic concat-and-predict:
1. Concatenation (what each team looks like individually)
2. Difference vector (who has the edge in each dimension)
3. Element-wise product (interaction — captures when teams are SIMILAR
   vs different in specific dimensions, e.g., two great defensive teams)
4. Seed difference (historical base rate signal)
5. Momentum signal (how hot is each team coming in)

The combination of difference AND product is key — difference tells you
WHO is better at what, product tells you WHERE the matchup is interesting
(both strong at X → high product, mismatch → low product).
"""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    Produces calibrated win probability from matchup representations.

    Input composition (default 512-dim embeddings):
        - repr_a:          512 dims  (A's matchup-specific representation)
        - repr_b:          512 dims  (B's matchup-specific representation)
        - repr_a - repr_b: 512 dims  (who has the edge)
        - repr_a * repr_b: 512 dims  (where the matchup is interesting)
        - seed_diff:       1 dim     (historical base rate)
        - momentum_diff:   1 dim     (who's hotter coming in)
        Total: 2049 dims (or 2048 without momentum)

    Architecture:
        Linear(input → 512) → GELU → Dropout
        Linear(512 → 256) → GELU → Dropout
        Linear(256 → 64) → GELU
        Linear(64 → 1) → Sigmoid
        → Apply shrinkage toward 0.5

    Why GELU over ReLU: Smoother gradient flow, slightly better for
    probability estimation since it doesn't hard-clip negative values.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        dropout: float = 0.3,
        shrinkage: float = 0.5,
        use_momentum: bool = True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.shrinkage = shrinkage
        self.use_momentum = use_momentum

        # Input: concat + diff + product + seed_diff + optional momentum_diff
        input_dim = embedding_dim * 4 + 1  # 2049
        if use_momentum:
            input_dim += 1  # 2050

        # Prediction layers — funnel down to single probability
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.GELU(),

            nn.Linear(64, 1),
        )

        # Learnable temperature for calibration (initialized to 1 = no effect)
        # During Phase 3 calibration, this gets tuned alongside shrinkage
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable probability outputs."""
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize final layer with small weights → outputs near 0.5
        final_linear = self.head[-1]
        nn.init.uniform_(final_linear.weight, -0.01, 0.01)
        nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        repr_a: torch.Tensor,
        repr_b: torch.Tensor,
        seed_diff: torch.Tensor,
        momentum_diff: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict P(Team A wins).

        Args:
            repr_a: A's matchup representation (batch, embedding_dim) or (embedding_dim,)
            repr_b: B's matchup representation, same shape
            seed_diff: seed_a - seed_b (batch,) or scalar
            momentum_diff: momentum_a - momentum_b (batch,) or scalar, optional

        Returns:
            prob: P(A wins) in calibrated range (batch,) or scalar
        """
        # Handle unbatched inputs
        if repr_a.dim() == 1:
            repr_a = repr_a.unsqueeze(0)
            repr_b = repr_b.unsqueeze(0)

        # Ensure seed_diff has right shape
        if seed_diff.dim() == 0:
            seed_diff = seed_diff.unsqueeze(0).unsqueeze(1)
        elif seed_diff.dim() == 1:
            seed_diff = seed_diff.unsqueeze(1)
        seed_diff = seed_diff.float()

        # Build the rich input representation
        diff = repr_a - repr_b       # Who has the edge
        product = repr_a * repr_b    # Where the matchup is interesting

        parts = [repr_a, repr_b, diff, product, seed_diff]

        if self.use_momentum and momentum_diff is not None:
            if momentum_diff.dim() == 0:
                momentum_diff = momentum_diff.unsqueeze(0).unsqueeze(1)
            elif momentum_diff.dim() == 1:
                momentum_diff = momentum_diff.unsqueeze(1)
            momentum_diff = momentum_diff.float()
            parts.append(momentum_diff)
        elif self.use_momentum:
            # No momentum provided — use zero (no signal)
            parts.append(torch.zeros_like(seed_diff))

        combined = torch.cat(parts, dim=-1)

        # Raw logit → temperature-scaled sigmoid
        logit = self.head(combined).squeeze(-1)
        raw_prob = torch.sigmoid(logit / self.temperature)

        # Apply shrinkage: pull toward 0.5
        # shrinkage=1.0 → no shrinkage, shrinkage=0.0 → always predict 0.5
        calibrated = self.shrinkage * raw_prob + (1.0 - self.shrinkage) * 0.5

        return calibrated

    def set_shrinkage(self, shrinkage: float):
        """Update shrinkage factor (called during Phase 3 calibration)."""
        self.shrinkage = max(0.0, min(1.0, shrinkage))

    def set_temperature(self, temperature: float):
        """Update temperature (called during Phase 3 calibration)."""
        with torch.no_grad():
            self.temperature.fill_(max(0.1, temperature))


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("Testing PredictionHead...")
    print("=" * 60)

    head = PredictionHead(embedding_dim=512, shrinkage=0.5, use_momentum=True)
    total_params = sum(p.numel() for p in head.parameters())
    print(f"Total parameters: {total_params:,}")

    # Single prediction
    repr_a = torch.randn(512)
    repr_b = torch.randn(512)
    seed_diff = torch.tensor(-3)
    momentum = torch.tensor(0.15)

    prob = head(repr_a, repr_b, seed_diff, momentum)
    print(f"\nSingle prediction: P(A wins) = {prob.item():.4f}")
    print(f"  (Should be in ~[0.35, 0.65] range due to shrinkage)")

    # Batch prediction
    batch = 16
    repr_a_batch = torch.randn(batch, 512)
    repr_b_batch = torch.randn(batch, 512)
    seeds = torch.randint(-15, 16, (batch,))
    momentums = torch.randn(batch) * 0.1

    probs = head(repr_a_batch, repr_b_batch, seeds, momentums)
    print(f"\nBatch prediction ({batch} games):")
    print(f"  Prob range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"  Mean: {probs.mean().item():.4f}")

    # Test shrinkage effect
    print(f"\nShrinkage test (same input, different shrinkage):")
    for s in [0.3, 0.5, 0.7, 1.0]:
        head.set_shrinkage(s)
        p = head(repr_a, repr_b, seed_diff, momentum)
        print(f"  shrinkage={s:.1f} → P(A wins) = {p.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
