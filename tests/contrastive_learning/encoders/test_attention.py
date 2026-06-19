"""
Tests for jepa_world_models.contrastive_learning.encoders.attention
"""

import math

import torch
import pytest 
from jepa_world_models.contrastive_learning.encoders.attention import (
    SingleHeadAttention,
    attention_scores,
    MultiHeadAttention
)


class TestAttentionScores:
    def test_output_shape(self) -> None:
        Q = torch.randn(3, 2)
        K = torch.randn(3, 2)
        S = attention_scores(Q, K)
        print(f"\nQ shape: {Q.shape}, K shape: {K.shape} -> S shape: {S.shape}")
        assert S.shape == (3, 3)

    def test_matches_manual_dot_product_with_scaling(self) -> None:
        torch.manual_seed(0)
        Q = torch.randn(4, 8)
        K = torch.randn(4, 8)
        d_k = 8

        S = attention_scores(Q, K)
        print(f"\nFull score matrix S (4x4):\n{S}")

        for i in range(4):
            for j in range(4):
                expected = torch.dot(Q[i], K[j]) / math.sqrt(d_k)
                torch.testing.assert_close(S[i, j], expected, atol=1e-5, rtol=1e-5)
        print("every entry S[i,j] matches manual <q_i, k_j> / sqrt(d_k)")

    def test_handles_batched_multihead_shape(self) -> None:
        Q = torch.randn(2, 4, 144, 32)
        K = torch.randn(2, 4, 144, 32)
        S = attention_scores(Q, K)
        print(f"\nbatched/multi-head: Q {Q.shape}, K {K.shape} -> S {S.shape}")
        assert S.shape == (2, 4, 144, 144)

    def test_scaling_reduces_variance_as_dimension_grows(self) -> None:
        torch.manual_seed(0)
        L = 50

        print()
        for d_k in [8, 64, 256]:
            Q = torch.randn(L, d_k)
            K = torch.randn(L, d_k)

            unscaled = Q @ K.transpose(-2, -1)
            scaled = attention_scores(Q, K)

            print(
                f"d_k={d_k:>4}:  unscaled var = {unscaled.var().item():>8.2f}   "
                f"scaled var = {scaled.var().item():>6.3f}"
            )
            assert 0.5 < scaled.var().item() < 2.0, (
                f"scaled variance for d_k={d_k} was {scaled.var().item():.3f}, "
                f"expected it to stay near 1 regardless of d_k"
            )


class TestSingleHeadAttention:
    def test_output_shape(self) -> None:
        attn = SingleHeadAttention(embed_dim=192, head_dim=32)
        x = torch.randn(4, 144, 192)
        out = attn(x)
        print(f"\ninput {x.shape} -> output {out.shape}")
        assert out.shape == (4, 144, 32)

    def test_output_shape_unbatched(self) -> None:
        attn = SingleHeadAttention(embed_dim=192, head_dim=32)
        x = torch.randn(144, 192)
        out = attn(x)
        print(f"\nunbatched: input {x.shape} -> output {out.shape}")
        assert out.shape == (144, 32)

    def test_attention_weights_sum_to_one_per_row(self) -> None:
        torch.manual_seed(0)
        attn = SingleHeadAttention(embed_dim=16, head_dim=4)
        x = torch.randn(5, 16)

        Q = attn.W_q(x)
        K = attn.W_k(x)
        scores = attention_scores(Q, K)
        weights = torch.softmax(scores, dim=-1)

        print(f"\nattention weights (5x5, each row is one patch's distribution):\n{weights}")
        row_sums = weights.sum(dim=-1)
        print(f"row sums: {row_sums}")
        torch.testing.assert_close(row_sums, torch.ones(5), atol=1e-6, rtol=1e-6)

    def test_no_bias_on_projections(self) -> None:
        attn = SingleHeadAttention(embed_dim=16, head_dim=4)
        print(f"\nW_q.bias = {attn.W_q.bias}, W_k.bias = {attn.W_k.bias}, W_v.bias = {attn.W_v.bias}")
        assert attn.W_q.bias is None
        assert attn.W_k.bias is None
        assert attn.W_v.bias is None

    def test_gradients_flow_to_input_and_weights(self) -> None:
        attn = SingleHeadAttention(embed_dim=192, head_dim=32)
        x = torch.randn(2, 144, 192, requires_grad=True)

        out = attn(x)
        out.sum().backward()

        print(f"\ngrad norm wrt input x: {x.grad.norm().item():.4f}")
        print(f"grad norm wrt W_q: {attn.W_q.weight.grad.norm().item():.4f}")
        print(f"grad norm wrt W_k: {attn.W_k.weight.grad.norm().item():.4f}")
        print(f"grad norm wrt W_v: {attn.W_v.weight.grad.norm().item():.4f}")

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert attn.W_q.weight.grad is not None
        assert attn.W_k.weight.grad is not None
        assert attn.W_v.weight.grad is not None

    def test_identical_input_patches_get_identical_output(self) -> None:
        torch.manual_seed(0)
        attn = SingleHeadAttention(embed_dim=16, head_dim=4)
        attn.eval()

        patch_content = torch.randn(16)
        x = torch.zeros(6, 16)
        x[0] = patch_content
        x[3] = patch_content

        with torch.no_grad():
            Q = attn.W_q(x)

        print(f"\nQ[0] (position 0): {Q[0]}")
        print(f"Q[3] (position 3, identical input): {Q[3]}")
        torch.testing.assert_close(Q[0], Q[3])



class TestMultiHeadAttention:
    def test_output_shape_matches_input_shape(self) -> None:
        """Multi-head attention is a sequence-to-sequence map that preserves
        shape -- output (B, L, d) must equal input (B, L, d), since this
        is what allows N blocks to be stacked without dimension changes."""
        mha = MultiHeadAttention(embed_dim=192, num_heads=6)
        x = torch.randn(4, 144, 192)
        out = mha(x)
        print(f"\ninput {x.shape} -> output {out.shape}")
        assert out.shape == x.shape

    def test_parameter_count(self) -> None:
        """4 square matrices (Wq, Wk, Wv, Wo), each d x d, no bias:
        4 * d^2 total parameters."""
        d = 192
        mha = MultiHeadAttention(embed_dim=d, num_heads=6)
        n_params = sum(p.numel() for p in mha.parameters())
        expected = 4 * d * d
        print(f"\nparameters: {n_params}, expected: {expected}")
        assert n_params == expected

    def test_raises_when_embed_dim_not_divisible_by_num_heads(self) -> None:
        """head_dim = embed_dim / num_heads must be an integer -- the
        assert catches this at construction time rather than letting a
        cryptic reshape error surface later."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(embed_dim=192, num_heads=7)

    def test_head_dim_is_correct(self) -> None:
        mha = MultiHeadAttention(embed_dim=192, num_heads=6)
        print(f"\nhead_dim: {mha.head_dim}, expected: {192 // 6}")
        assert mha.head_dim == 192 // 6  # 32

    def test_split_heads_shape(self) -> None:
        """_split_heads must produce (B, h, L, d_h) from (B, L, d) --
        the head axis sits next to batch so attention_scores can treat
        (B, h) as independent leading dimensions."""
        mha = MultiHeadAttention(embed_dim=192, num_heads=6)
        x = torch.randn(4, 144, 192)
        split = mha._split_heads(x)
        print(f"\n_split_heads: {x.shape} -> {split.shape}")
        assert split.shape == (4, 6, 144, 32)

    def test_merge_heads_is_inverse_of_split_heads(self) -> None:
        """_merge_heads must perfectly undo _split_heads -- the round trip
        must recover the original tensor exactly, since both operations
        are pure reshape/permute with no computation."""
        torch.manual_seed(0)
        mha = MultiHeadAttention(embed_dim=192, num_heads=6)
        x = torch.randn(4, 144, 192)
        roundtrip = mha._merge_heads(mha._split_heads(x))
        print(f"\nround-trip max diff: {(roundtrip - x).abs().max().item():.2e}")
        torch.testing.assert_close(roundtrip, x)

    def test_each_head_sees_different_subspace(self) -> None:
        """After _split_heads, head i should contain exactly columns
        [i*d_h : (i+1)*d_h] of the pre-split projection output -- confirming
        the reshape/permute correctly partitions the d-dimensional space
        into h non-overlapping d_h-dimensional subspaces."""
        torch.manual_seed(0)
        mha = MultiHeadAttention(embed_dim=192, num_heads=6)
        x = torch.randn(2, 144, 192)

        Q_flat = mha.W_q(x)
        Q_heads = mha._split_heads(Q_flat)

        d_h = mha.head_dim
        for h in range(mha.num_heads):
            expected = Q_flat[:, :, h * d_h:(h + 1) * d_h]
            print(f"head {h}: cols {h*d_h}:{(h+1)*d_h} match: "
                  f"{torch.allclose(Q_heads[:, h], expected)}")
            torch.testing.assert_close(Q_heads[:, h], expected)

    def test_attention_weights_sum_to_one_per_row_per_head(self) -> None:
        """Each head's attention distribution must sum to 1 per row --
        same property as single-head, now verified for every head."""
        torch.manual_seed(0)
        mha = MultiHeadAttention(embed_dim=192, num_heads=6)
        x = torch.randn(2, 10, 192)

        Q = mha._split_heads(mha.W_q(x))
        K = mha._split_heads(mha.W_k(x))
        scores = attention_scores(Q, K)
        weights = torch.softmax(scores, dim=-1)

        row_sums = weights.sum(dim=-1)
        print(f"\nattention weight row sums (min, max): "
              f"{row_sums.min().item():.6f}, {row_sums.max().item():.6f}")
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6
        )

    def test_gradients_flow_to_all_weight_matrices(self) -> None:
        mha = MultiHeadAttention(embed_dim=192, num_heads=6)
        x = torch.randn(2, 144, 192, requires_grad=True)

        out = mha(x)
        out.sum().backward()

        for name, param in mha.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
            print(f"grad norm {name}: {param.grad.norm().item():.4f}")

    def test_no_bias_on_any_projection(self) -> None:
        mha = MultiHeadAttention(embed_dim=192, num_heads=6)
        assert mha.W_q.bias is None
        assert mha.W_k.bias is None
        assert mha.W_v.bias is None
        assert mha.W_o.bias is None
        print("\nall projection matrices have no bias -- confirmed")