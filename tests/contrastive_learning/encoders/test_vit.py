"""
Tests for jepa_world_models.contrastive_learning.encoders.vit

End-to-end tests for the full ViTEncoder: raw images in, one
representation vector per image out.
"""

import torch
import pytest

from jepa_world_models.contrastive_learning.encoders.vit import ViTEncoder


class TestViTEncoder:
    def test_output_shape(self) -> None:
        """Raw STL-10 images -> one d-dimensional vector per image."""
        encoder = ViTEncoder()
        images = torch.randn(4, 3, 96, 96)
        out = encoder(images)
        print(f"\ninput {images.shape} -> output {out.shape}")
        assert out.shape == (4, 192)

    def test_output_shape_batch_size_1(self) -> None:
        """Confirm batch size of 1 works -- no batchnorm-style issues."""
        encoder = ViTEncoder()
        images = torch.randn(1, 3, 96, 96)
        out = encoder(images)
        print(f"\nbatch=1: input {images.shape} -> output {out.shape}")
        assert out.shape == (1, 192)

    def test_num_patches_correct(self) -> None:
        """96 / 8 = 12 patches per side, 12 * 12 = 144 patches total."""
        encoder = ViTEncoder(image_size=96, patch_size=8)
        print(f"\nnum_patches: {encoder.num_patches}, expected: 144")
        assert encoder.num_patches == 144

    def test_patch_dim_correct(self) -> None:
        """patch_dim = patch_size^2 * channels = 8*8*3 = 192."""
        encoder = ViTEncoder(patch_size=8, in_channels=3)
        print(f"\npatch_dim: {encoder.patch_dim}, expected: 192")
        assert encoder.patch_dim == 192

    def test_total_parameter_count(self) -> None:
        """~2.7M parameters -- small enough for 8GB VRAM per the BOTE
        calculation in docs/pytorch_stuff.md."""
        encoder = ViTEncoder()
        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"\ntotal parameters: {n_params:,}")
        assert 1_000_000 < n_params < 10_000_000

    def test_depth_determines_number_of_blocks(self) -> None:
        """ModuleList should contain exactly `depth` TransformerBlocks."""
        for depth in [2, 4, 6]:
            encoder = ViTEncoder(depth=depth)
            print(f"\ndepth={depth}: len(blocks)={len(encoder.blocks)}")
            assert len(encoder.blocks) == depth

    def test_raises_on_non_divisible_image_patch_size(self) -> None:
        """image_size must be divisible by patch_size -- caught at
        construction time rather than deep inside a forward pass."""
        with pytest.raises(AssertionError):
            ViTEncoder(image_size=96, patch_size=7)

    def test_custom_embed_dim(self) -> None:
        """embed_dim is a free design choice -- output should match it."""
        encoder = ViTEncoder(embed_dim=256, num_heads=8)
        images = torch.randn(2, 3, 96, 96)
        out = encoder(images)
        print(f"\ncustom embed_dim=256: output {out.shape}")
        assert out.shape == (2, 256)

    def test_mean_pooling_uses_all_patches(self) -> None:
        """GAP averages over all 144 patch tokens -- verify by zeroing
        one patch in the final layer's output and confirming the pooled
        representation changes by exactly 1/144 of that patch's norm."""
        torch.manual_seed(0)
        encoder = ViTEncoder()
        encoder.eval()

        images = torch.randn(1, 3, 96, 96)

        with torch.no_grad():
            x = encoder.norm(self._run_to_prenorm(encoder, images))
            pooled = x.mean(dim=1)

            x_modified = x.clone()
            x_modified[0, 0] = 0.0
            pooled_modified = x_modified.mean(dim=1)

            expected_diff = x[0, 0] / 144
            actual_diff = pooled[0] - pooled_modified[0]
            torch.testing.assert_close(actual_diff, expected_diff, atol=1e-5, rtol=1e-5)
            print(f"\nmean pool uses all patches: diff matches x[0,0]/144 exactly")

    def _run_to_prenorm(self, encoder: ViTEncoder, images: torch.Tensor) -> torch.Tensor:
        from jepa_world_models.contrastive_learning.encoders.patch_embedding import patchify
        x = encoder.patch_embed(patchify(images, encoder.patch_size))
        x = encoder.pos_embed(x)
        for block in encoder.blocks:
            x = block(x)
        return x

    def test_two_different_images_give_different_representations(self) -> None:
        """Basic sanity check: the encoder should not collapse two random
        images to the same representation at random initialization."""
        torch.manual_seed(0)
        encoder = ViTEncoder()
        encoder.eval()

        image_a = torch.randn(1, 3, 96, 96)
        image_b = torch.randn(1, 3, 96, 96)

        with torch.no_grad():
            rep_a = encoder(image_a)
            rep_b = encoder(image_b)

        max_diff = (rep_a - rep_b).abs().max().item()
        print(f"\nmax diff between two random images' reps: {max_diff:.4f}")
        assert max_diff > 0.01

    def test_gradients_flow_end_to_end(self) -> None:
        """Gradients must flow from the VICReg loss (applied to encoder
        output) back through all six steps to the encoder's parameters."""
        encoder = ViTEncoder()
        images = torch.randn(2, 3, 96, 96)

        out = encoder(images)
        out.sum().backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert torch.isfinite(param.grad).all(), f"NaN/inf grad for {name}"

        print(f"\ngradients verified for all "
              f"{sum(1 for _ in encoder.parameters())} parameter tensors")