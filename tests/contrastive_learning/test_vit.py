"""
Tests for jepa_world_models.contrastive_learning.encoders.vit

Step 1: patchify
"""

import pytest
import torch

from jepa_world_models.contrastive_learning.encoders.vit import patchify


class TestPatchify:
    def test_output_shape_for_stl10_dimensions(self) -> None:
        """For STL-10 (96x96x3) with patch_size=8, expect 144 patches of
        dimension 192 each (12x12 spatial grid, 8*8*3 flattened patch)."""
        images = torch.randn(4, 3, 96, 96)
        out = patchify(images, patch_size=8)
        print(f'out.shape: {out.shape}')
        assert out.shape == (4, 144, 192)

    def test_patch_content_matches_original_pixels(self) -> None:
        """Verify patchify doesn't scramble pixels: the top-left patch of
        image 0 should contain exactly the pixels from images[0, :, 0:8, 0:8],
        in the same relative order (channel, then row, then column)."""
        torch.manual_seed(0)
        images = torch.randn(2, 3, 16, 16)  # small image, patch_size=4 -> 4x4=16 patches
        out = patchify(images, patch_size=4)

        # patch index 0 should be the top-left 4x4 block: rows 0:4, cols 0:4
        expected_patch_0 = images[0, :, 0:4, 0:4].flatten()  # (C*P*P,) = (48,)
        torch.testing.assert_close(out[0, 0], expected_patch_0)

        # patch index 1 should be the next block over (rows 0:4, cols 4:8),
        # since patches are ordered row-major: (k_h=0,k_w=0), (k_h=0,k_w=1), ...
        expected_patch_1 = images[0, :, 0:4, 4:8].flatten()
        torch.testing.assert_close(out[0, 1], expected_patch_1)

    def test_does_not_mix_pixels_across_patches(self) -> None:
        """Regression check against the 'naive reshape gives you rows, not
        patches' failure mode: a constant-valued image should patchify into
        patches that are each internally constant (since nothing should leak
        in from a different patch)."""
        images = torch.zeros(1, 1, 8, 8)
        images[0, 0, 0:4, 0:4] = 1.0   # top-left patch -> all 1s
        images[0, 0, 0:4, 4:8] = 2.0   # top-right patch -> all 2s
        images[0, 0, 4:8, 0:4] = 3.0   # bottom-left patch -> all 3s
        images[0, 0, 4:8, 4:8] = 4.0   # bottom-right patch -> all 4s

        out = patchify(images, patch_size=4)  # (1, 4, 16)

        assert torch.all(out[0, 0] == 1.0)
        assert torch.all(out[0, 1] == 2.0)
        assert torch.all(out[0, 2] == 3.0)
        assert torch.all(out[0, 3] == 4.0)

    def test_raises_or_misbehaves_clearly_on_non_divisible_patch_size(self) -> None:
        """patch_size must evenly divide H and W. Document current behavior:
        reshape will raise a RuntimeError if the dimensions don't divide
        evenly, rather than silently truncating or padding."""
        images = torch.randn(1, 3, 10, 10)  # 10 is not divisible by patch_size=4
        with pytest.raises(RuntimeError):
            patchify(images, patch_size=4)