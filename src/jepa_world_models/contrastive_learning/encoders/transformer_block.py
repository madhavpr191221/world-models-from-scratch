"""
ViT encoder — transformer block.

One full transformer block, repeated N times to form the encoder:

    x' = x + MultiHeadAttention(LayerNorm(x))       # attention sublayer
    y  = x' + MLP(LayerNorm(x'))                    # MLP sublayer

Pre-norm convention (LayerNorm before each sublayer, not after) -- more
stable than post-norm for deep networks, and the modern standard in ViT
and most transformer implementations since 2020.

Residual connections (the + in both lines above) carry the original signal
through unchanged, so each sublayer only needs to learn a *correction* to
its input rather than the full transformation from scratch. This is the
same residual principle as ResNets, transplanted into attention-based
architectures.
"""

import torch
from torch import Tensor

from jepa_world_models.contrastive_learning.encoders.attention import MultiHeadAttention


class MLP(torch.nn.Module):
    """
    Position-wise feedforward sublayer.

    (B, L, d) -> (B, L, d)

    Linear(d -> 4d) -> GELU -> Linear(4d -> d)

    Applied identically to every position (same weights across all L
    tokens, no interaction between positions -- that's attention's job).
    The 4x expansion is conventional, dating to the original Transformer
    paper, giving the network capacity to compute richer per-token
    transformations before projecting back down.
    """

    def __init__(self, embed_dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.fc1 = torch.nn.Linear(embed_dim, hidden_dim)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))
    

class TransformerBlock(torch.nn.Module):
    """
    One full transformer block.

    (B, L, d) -> (B, L, d)

    Pre-norm residual structure:
        x' = x + MultiHeadAttention(LayerNorm(x))
        y  = x' + MLP(LayerNorm(x'))

    Two separate LayerNorm instances -- one for each sublayer -- each
    with their own learned gamma and beta parameters, since the
    distributions before attention and before the MLP differ.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        # attention sublayer: pre-norm + residual
        x = x + self.attn(self.norm1(x))
        # MLP sublayer: pre-norm + residual
        x = x + self.mlp(self.norm2(x))
        return x