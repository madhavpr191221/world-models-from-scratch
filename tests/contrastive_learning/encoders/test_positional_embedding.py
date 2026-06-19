"""
Tests for jepa_world_models.contrastive_learning.encoders.positional_embedding
"""

import torch

from jepa_world_models.contrastive_learning.encoders.positional_embedding import (
    PositionalEmbedding,
)


class TestPositionalEmbedding:
    def test_output_shape(self) -> None:
        pos_embed = PositionalEmbedding(num_patches=144, embed_dim=192)
        x = torch.randn(4, 144, 192)
        out = pos_embed(x)
        assert out.shape == (4, 144, 192)

    def test_has_learnable_parameters(self) -> None:
        """144 positions x 192 dimensions, one learnable table, no bias."""
        pos_embed = PositionalEmbedding(num_patches=144, embed_dim=192)
        n_params = sum(p.numel() for p in pos_embed.parameters())
        assert n_params == 144 * 192

    def test_same_table_added_to_every_batch_element(self) -> None:
        """The positional table must be IDENTICAL across the batch -- image
        0's patch at position 5 and image 1's patch at position 5 should
        receive the exact same positional vector added, since position 5
        always means 'this spot in the grid' regardless of which image."""
        torch.manual_seed(0)
        pos_embed = PositionalEmbedding(num_patches=144, embed_dim=192)
        x = torch.randn(4, 144, 192)

        out = pos_embed(x)

        added_to_image_0 = out[0] - x[0]
        added_to_image_1 = out[1] - x[1]
        torch.testing.assert_close(added_to_image_0, added_to_image_1)

    def test_different_positions_get_different_embeddings(self) -> None:
        """Sanity check against a degenerate case: if every row of the
        positional table were identical, positional embedding would add
        no positional information at all. With random initialization,
        different positions should (almost certainly) get different
        vectors."""
        torch.manual_seed(0)
        pos_embed = PositionalEmbedding(num_patches=144, embed_dim=192)

        position_0 = pos_embed.pos_embed[0]
        position_1 = pos_embed.pos_embed[1]

        assert not torch.allclose(position_0, position_1)

    def test_gradients_flow(self) -> None:
        """Confirms the positional table is actually trainable."""
        pos_embed = PositionalEmbedding(num_patches=144, embed_dim=192)
        x = torch.randn(2, 144, 192, requires_grad=True)

        out = pos_embed(x)
        out.sum().backward()

        assert pos_embed.pos_embed.grad is not None
        assert torch.isfinite(pos_embed.pos_embed.grad).all()