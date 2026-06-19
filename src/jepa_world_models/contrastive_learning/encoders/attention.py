"""
Multi-head self-attention.

Block 1: attention scores
---------------------------
S = Q K^T / sqrt(d_k)

Given Q, K in R^{L x d_k} (L patches, each a d_k-dimensional query/key
vector), S_ij = <q_i, k_j> / sqrt(d_k) -- the compatibility score between
patch i (as a query) and patch j (as a key), for every pair simultaneously.

Scaling by 1/sqrt(d_k): if q, k ~ N(0, I), then q^T k ~ N(0, d_k) -- the
dot product's variance grows with the dimension, since it's a sum of d_k
independent products. Dividing by sqrt(d_k) restores unit variance
regardless of d_k, preventing the softmax (applied next, not in this
function) from saturating into a near-one-hot distribution purely as an
artifact of dimensionality, rather than genuine relevance.
"""

import math

import torch
from torch import Tensor


def attention_scores(Q: Tensor, K: Tensor) -> Tensor:
    """
    Q: (..., L, d_k), K: (..., L, d_k) -> S: (..., L, L)

    Leading dimensions (...) are batch/head dimensions and are carried
    through unchanged -- this function doesn't care whether it's called
    on a single (L, d_k) pair or a batched (B, h, L, d_k) tensor.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)  # (..., L, d_k) @ (..., d_k, L) -> (..., L, L)
    return scores / math.sqrt(d_k)


class SingleHeadAttention(torch.nn.Module):
    """
    Block 2: single-head self-attention.

    X: (L, d) -> output: (L, d_k)

    Computes Q = X Wq, K = X Wk, V = X Wv (three independent learned
    projections of the same input), scores S = attention_scores(Q, K),
    attention weights A = softmax(S) applied row-wise (each patch's
    weights over all patches sum to 1), and output = A V -- each patch's
    output is a weighted average of every patch's value vector, weighted
    by how relevant that patch was deemed to be.
    """

    def __init__(self, embed_dim: int, head_dim: int) -> None:
        super().__init__()
        self.W_q = torch.nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = torch.nn.Linear(embed_dim, head_dim, bias=False)
        self.W_v = torch.nn.Linear(embed_dim, head_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., L, embed_dim)
        Q = self.W_q(x)  # (..., L, head_dim)
        K = self.W_k(x)  # (..., L, head_dim)
        V = self.W_v(x)  # (..., L, head_dim)

        scores = attention_scores(Q, K)         # (..., L, L)
        attn_weights = torch.softmax(scores, dim=-1)  # row-wise: each row sums to 1
        output = attn_weights @ V                # (..., L, L) @ (..., L, head_dim) -> (..., L, head_dim)
        return output
    


class MultiHeadAttention(torch.nn.Module):
    """
    Block 3: multi-head self-attention.

    X: (B, L, d) -> output: (B, L, d)

    Runs h independent attention heads in parallel, each operating in a
    d_h = d / h dimensional subspace, then concatenates and mixes outputs.

    Implementation uses the efficient single-projection approach:
    rather than h separate W_Q, W_K, W_V matrices (each d x d_h), we use
    one large W_Q, W_K, W_V (each d x d), compute all heads' projections
    in one matrix multiply, then expose the head axis via reshape + permute:

        (B, L, d) -> (B, L, h, d_h) -> (B, h, L, d_h)

    This is mathematically identical to h separate projections but uses
    one large GPU-efficient matrix multiply instead of h small ones.

    After attention: reverse the permute/reshape to get (B, L, d), then
    pass through W_O (d x d) to mix head contributions together.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # d_h = d / h

        # One large projection per Q, K, V -- all heads packed together
        self.W_q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        # Output mixing: combines the h heads' contributions
        self.W_o = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        (B, L, d) -> (B, h, L, d_h)

        Exposes the head axis by splitting the last dimension into
        (num_heads, head_dim) then permuting so heads sit next to batch.
        Pure reshape + permute: no computation, just relabeling.
        """
        B, L, d = x.shape
        x = x.reshape(B, L, self.num_heads, self.head_dim)  # (B, L, h, d_h)
        return x.permute(0, 2, 1, 3)                         # (B, h, L, d_h)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        (B, h, L, d_h) -> (B, L, d)

        Reverses _split_heads: permute back, then merge head and head_dim
        axes into a single d-dimensional axis.
        """
        B, h, L, d_h = x.shape
        x = x.permute(0, 2, 1, 3)                   # (B, L, h, d_h)
        return x.reshape(B, L, h * d_h)              # (B, L, d)

    def forward(self, x: Tensor) -> Tensor:
        # Project all heads at once: (B, L, d) -> (B, L, d)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Expose head axis: (B, L, d) -> (B, h, L, d_h)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # attention_scores already handles (..., L, d_h) leading dims
        scores = attention_scores(Q, K)                    # (B, h, L, L)
        attn_weights = torch.softmax(scores, dim=-1)       # row-wise
        out = attn_weights @ V                             # (B, h, L, d_h)

        # Merge heads back: (B, h, L, d_h) -> (B, L, d)
        out = self._merge_heads(out)

        # Mix head contributions
        return self.W_o(out)                               # (B, L, d)