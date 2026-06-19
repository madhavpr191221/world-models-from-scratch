"""
Tests for jepa_world_models.contrastive_learning.augmentations
"""

import torch
import pytest
from PIL import Image
import numpy as np

from jepa_world_models.contrastive_learning.augmentations import (
    vicreg_augmentation,
    vicreg_augmentation_unnormalized,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def make_pil_image(size: int = 96) -> Image.Image:
    """Create a random RGB PIL image of the given size."""
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestVICRegAugmentation:
    def test_output_shape(self) -> None:
        """Output must be (3, image_size, image_size) regardless of
        which random augmentations fired."""
        torch.manual_seed(0)
        transform = vicreg_augmentation(image_size=96)
        img = make_pil_image(96)
        out = transform(img)
        print(f"\noutput shape: {out.shape}")
        assert out.shape == (3, 96, 96)

    def test_output_is_float_tensor(self) -> None:
        transform = vicreg_augmentation(image_size=96)
        img = make_pil_image(96)
        out = transform(img)
        print(f"\ndtype: {out.dtype}")
        assert out.dtype == torch.float32

    def test_normalized_output_range(self) -> None:
        """After ImageNet normalization, values can go below 0 and above
        1 -- that's expected and correct. Verify the range is plausible
        (not wildly outside what normalization should produce)."""
        torch.manual_seed(0)
        transform = vicreg_augmentation(image_size=96)
        img = make_pil_image(96)
        out = transform(img)
        print(f"\nnormalized range: [{out.min().item():.3f}, {out.max().item():.3f}]")
        # normalized values for natural images should stay roughly in (-3, 3)
        assert out.min().item() > -5.0
        assert out.max().item() < 5.0

    def test_two_views_of_same_image_differ(self) -> None:
        """Calling the transform twice on the same PIL image must produce
        different tensors -- the random augmentations should fire
        differently each time. This is the entire point of the two-view
        design: same image, different augmentation outcomes."""
        torch.manual_seed(0)
        transform = vicreg_augmentation(image_size=96)

        # run over several images to make statistical failure unlikely
        found_different = False
        for _ in range(10):
            img = make_pil_image(96)
            view_a = transform(img)
            view_b = transform(img)
            if not torch.allclose(view_a, view_b):
                found_different = True
                break

        print(f"\ntwo views differ: {found_different}")
        assert found_different, (
            "All pairs of views were identical -- random augmentations "
            "may not be applying correctly"
        )

    def test_custom_image_size(self) -> None:
        """image_size is configurable -- the crop and all subsequent
        transforms should respect it."""
        for size in [64, 96, 128]:
            transform = vicreg_augmentation(image_size=size)
            img = make_pil_image(size)
            out = transform(img)
            print(f"\nimage_size={size}: output {out.shape}")
            assert out.shape == (3, size, size)

    def test_imagenet_constants_are_correct_length(self) -> None:
        """IMAGENET_MEAN and IMAGENET_STD must each have exactly 3
        values -- one per channel."""
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        print(f"\nIMAGENET_MEAN: {IMAGENET_MEAN}")
        print(f"IMAGENET_STD:  {IMAGENET_STD}")


class TestVICRegAugmentationUnnormalized:
    def test_output_shape(self) -> None:
        transform = vicreg_augmentation_unnormalized(image_size=96)
        img = make_pil_image(96)
        out = transform(img)
        print(f"\nunnormalized output shape: {out.shape}")
        assert out.shape == (3, 96, 96)

    def test_values_in_zero_one_range(self) -> None:
        """Without normalization, ToTensor() maps uint8 [0,255] to
        float [0,1] -- verify this holds so saved PNGs look correct."""
        torch.manual_seed(0)
        transform = vicreg_augmentation_unnormalized(image_size=96)

        for _ in range(5):
            img = make_pil_image(96)
            out = transform(img)
            print(f"\npixel range: [{out.min().item():.4f}, {out.max().item():.4f}]")
            assert out.min().item() >= 0.0, "unnormalized values below 0"
            assert out.max().item() <= 1.0, "unnormalized values above 1"

    def test_unnormalized_differs_from_normalized(self) -> None:
        """The normalized and unnormalized transforms should produce
        different outputs for the same input -- confirms the Normalize
        step is actually present in one and absent in the other."""
        torch.manual_seed(0)
        img = make_pil_image(96)

        torch.manual_seed(42)
        normalized = vicreg_augmentation(image_size=96)(img)
        torch.manual_seed(42)
        unnormalized = vicreg_augmentation_unnormalized(image_size=96)(img)

        max_diff = (normalized - unnormalized).abs().max().item()
        print(f"\nmax diff between normalized and unnormalized: {max_diff:.4f}")
        assert max_diff > 0.01, (
            "Normalized and unnormalized outputs are too similar -- "
            "Normalize step may be missing from vicreg_augmentation"
        )

    def test_two_views_differ(self) -> None:
        """Same property as the normalized version -- two independent
        calls produce different results."""
        torch.manual_seed(0)
        transform = vicreg_augmentation_unnormalized(image_size=96)

        found_different = False
        for _ in range(10):
            img = make_pil_image(96)
            view_a = transform(img)
            view_b = transform(img)
            if not torch.allclose(view_a, view_b):
                found_different = True
                break

        print(f"\ntwo views differ: {found_different}")
        assert found_different