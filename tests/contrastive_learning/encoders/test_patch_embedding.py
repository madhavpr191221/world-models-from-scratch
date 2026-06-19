import pytest
import torch

from jepa_world_models.contrastive_learning.encoders.patch_embedding import (
    patchify,
    PatchEmbedding,
)


class TestPatchify:
    def test_output_shape_for_stl10_dimensions(self) -> None:
        images = torch.randn(4, 3, 96, 96)
        out = patchify(images, patch_size=8)
        assert out.shape == (4, 144, 192)

    def test_patch_content_matches_original_pixels(self) -> None:
        torch.manual_seed(0)
        images = torch.randn(2, 3, 16, 16)
        out = patchify(images, patch_size=4)
        expected_patch_0 = images[0, :, 0:4, 0:4].flatten()
        torch.testing.assert_close(out[0, 0], expected_patch_0)
        expected_patch_1 = images[0, :, 0:4, 4:8].flatten()
        torch.testing.assert_close(out[0, 1], expected_patch_1)

    def test_does_not_mix_pixels_across_patches(self) -> None:
        images = torch.zeros(1, 1, 8, 8)
        images[0, 0, 0:4, 0:4] = 1.0
        images[0, 0, 0:4, 4:8] = 2.0
        images[0, 0, 4:8, 0:4] = 3.0
        images[0, 0, 4:8, 4:8] = 4.0
        out = patchify(images, patch_size=4)
        assert torch.all(out[0, 0] == 1.0)
        assert torch.all(out[0, 1] == 2.0)
        assert torch.all(out[0, 2] == 3.0)
        assert torch.all(out[0, 3] == 4.0)

    def test_raises_or_misbehaves_clearly_on_non_divisible_patch_size(self) -> None:
        images = torch.randn(1, 3, 10, 10)
        with pytest.raises(RuntimeError):
            patchify(images, patch_size=4)


class TestPatchEmbedding:
    def test_output_shape(self) -> None:
        patches = torch.randn(4, 144, 192)
        embed = PatchEmbedding(patch_dim=192, embed_dim=192)
        out = embed(patches)
        assert out.shape == (4, 144, 192)

    def test_output_shape_with_different_embed_dim(self) -> None:
        patches = torch.randn(4, 144, 192)
        embed = PatchEmbedding(patch_dim=192, embed_dim=256)
        out = embed(patches)
        assert out.shape == (4, 144, 256)

    def test_has_learnable_parameters(self) -> None:
        embed = PatchEmbedding(patch_dim=192, embed_dim=192)
        n_params = sum(p.numel() for p in embed.parameters())
        assert n_params == 192 * 192 + 192

    def test_same_weights_applied_to_every_patch(self) -> None:
        torch.manual_seed(0)
        embed = PatchEmbedding(patch_dim=192, embed_dim=192)
        embed.eval()

        patch_content = torch.randn(192)
        patches = torch.zeros(1, 144, 192)
        patches[0, 0] = patch_content
        patches[0, 100] = patch_content

        with torch.no_grad():
            out = embed(patches)

        torch.testing.assert_close(out[0, 0], out[0, 100])

    def test_gradients_flow(self) -> None:
        patches = torch.randn(2, 144, 192, requires_grad=True)
        embed = PatchEmbedding(patch_dim=192, embed_dim=192)

        out = embed(patches)
        out.sum().backward()

        assert embed.proj.weight.grad is not None
        assert embed.proj.bias.grad is not None
        assert torch.isfinite(embed.proj.weight.grad).all()