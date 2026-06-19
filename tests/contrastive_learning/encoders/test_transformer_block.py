"""
Tests for jepa_world_models.contrastive_learning.encoders.transformer_block
"""

import pytest
import torch

from jepa_world_models.contrastive_learning.encoders.transformer_block import (
    MLP,
    TransformerBlock,
)


class TestMLP:
    def test_output_shape(self) -> None:
        mlp = MLP(embed_dim=192)
        x = torch.randn(4, 144, 192)
        out = mlp(x)
        print(f"\ninput {x.shape} -> output {out.shape}")
        assert out.shape == (4, 144, 192)

    def test_parameter_count(self) -> None:
        """fc1: (d -> 4d), fc2: (4d -> d), both with bias.
        Total: (d * 4d + 4d) + (4d * d + d) = 8d^2 + 5d."""
        d = 192
        mlp = MLP(embed_dim=d)
        n_params = sum(p.numel() for p in mlp.parameters())
        expected = (d * (4*d) + (4*d)) + ((4*d) * d + d)
        print(f"\nparameters: {n_params}, expected: {expected}")
        assert n_params == expected

    def test_expansion_to_4d(self) -> None:
        """Hidden layer should be 4 * embed_dim wide -- confirm by
        inspecting fc1's weight shape directly."""
        d = 192
        mlp = MLP(embed_dim=d, mlp_ratio=4)
        print(f"\nfc1 weight shape: {mlp.fc1.weight.shape}, expected: ({4*d}, {d})")
        assert mlp.fc1.weight.shape == (4 * d, d)
        assert mlp.fc2.weight.shape == (d, 4 * d)

    def test_custom_mlp_ratio(self) -> None:
        """mlp_ratio is a free design choice -- confirm it's respected."""
        mlp = MLP(embed_dim=64, mlp_ratio=2)
        assert mlp.fc1.weight.shape == (128, 64)
        x = torch.randn(2, 10, 64)
        out = mlp(x)
        assert out.shape == (2, 10, 64)

    def test_gradients_flow(self) -> None:
        mlp = MLP(embed_dim=192)
        x = torch.randn(2, 144, 192, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        print(f"\ngrad norm wrt input: {x.grad.norm().item():.4f}")


class TestTransformerBlock:
    def test_output_shape_matches_input(self) -> None:
        """TransformerBlock must be shape-preserving: (B, L, d) -> (B, L, d).
        This is what makes N blocks stackable without dimension changes."""
        block = TransformerBlock(embed_dim=192, num_heads=6)
        x = torch.randn(4, 144, 192)
        out = block(x)
        print(f"\ninput {x.shape} -> output {out.shape}")
        assert out.shape == x.shape

    def test_residual_connection_active(self) -> None:
        """The residual connection means the block's output is NOT a
        full rewrite of the input -- the skip signal passes through
        unchanged. At random init, the correction (out - x) should be
        meaningfully smaller than x itself (ratio well below 1)."""
        torch.manual_seed(0)
        block = TransformerBlock(embed_dim=192, num_heads=6)
        x = torch.randn(4, 144, 192)
        out = block(x)

        diff_norm = (out - x).norm().item()
        input_norm = x.norm().item()
        ratio = diff_norm / input_norm
        print(f"\ninput norm: {input_norm:.3f}, diff norm: {diff_norm:.3f}, "
              f"ratio: {ratio:.3f}")
        assert ratio < 1.0

    def test_six_blocks_stackable(self) -> None:
        """Shape must be preserved through N=6 stacked blocks -- the
        fundamental requirement for building a deep ViT."""
        x = torch.randn(4, 144, 192)
        for i in range(6):
            block = TransformerBlock(embed_dim=192, num_heads=6)
            x = block(x)
        print(f"\nafter 6 stacked blocks: {x.shape}")
        assert x.shape == (4, 144, 192)

    def test_pre_norm_order(self) -> None:
        """Pre-norm: LayerNorm is applied BEFORE attention and BEFORE MLP,
        not after. Verify by confirming norm1 normalizes its input to
        near-zero mean, near-unit std."""
        torch.manual_seed(0)
        block = TransformerBlock(embed_dim=192, num_heads=6)
        x = torch.randn(4, 144, 192) * 5.0  # deliberately large values

        normed = block.norm1(x)
        mean = normed.mean(dim=-1).abs().mean().item()
        std = normed.std(dim=-1).mean().item()
        print(f"\nnorm1 output: mean~{mean:.4f} (should be ~0), "
              f"std~{std:.4f} (should be ~1)")
        assert mean < 0.01
        assert 0.9 < std < 1.1

    def test_layernorm_has_learned_parameters(self) -> None:
        """Each LayerNorm has its own gamma (weight) and beta (bias),
        so the network can rescale/shift after normalization if needed."""
        block = TransformerBlock(embed_dim=192, num_heads=6)
        assert block.norm1.weight.shape == (192,)
        assert block.norm1.bias.shape == (192,)
        assert block.norm2.weight.shape == (192,)
        assert block.norm2.bias.shape == (192,)
        print(f"\nnorm1 weight shape: {block.norm1.weight.shape}, "
              f"norm2 weight shape: {block.norm2.weight.shape}")

    def test_gradients_flow_through_both_sublayers(self) -> None:
        """Gradients must reach: the input tensor, all attention weights,
        all MLP weights, and both LayerNorm parameters."""
        block = TransformerBlock(embed_dim=192, num_heads=6)
        x = torch.randn(2, 144, 192, requires_grad=True)

        block(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        print(f"\ngrad norm wrt input: {x.grad.norm().item():.4f}")

        for name, param in block.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert torch.isfinite(param.grad).all(), f"NaN grad for {name}"
            print(f"grad norm {name}: {param.grad.norm().item():.4f}")

    def test_output_differs_from_input(self) -> None:
        """The block should not be an identity map -- its output should
        differ from its input."""
        torch.manual_seed(0)
        block = TransformerBlock(embed_dim=192, num_heads=6)
        x = torch.randn(2, 144, 192)
        out = block(x)
        assert not torch.allclose(out, x)
        print(f"\nmax diff between output and input: {(out - x).abs().max().item():.4f}")