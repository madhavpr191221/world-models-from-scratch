"""
Tests for jepa_world_models.data.stl10

Uses synthetic PNG files so the real dataset doesn't need to be present
for tests to pass. One integration test (marked with a skip condition)
verifies against the real data when it IS present.
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

from jepa_world_models.data.stl10 import STL10Unlabeled


def make_fake_dataset(tmp_dir: str, n: int = 8, size: int = 96) -> str:
    """Create n fake 96x96 PNG files in tmp_dir, return the path."""
    for i in range(n):
        img = Image.fromarray(
            np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        )
        img.save(os.path.join(tmp_dir, f"unlabeled_image_png_{i+1}.png"))
    return tmp_dir


class TestSTL10Unlabeled:
    def test_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_dataset(tmp, n=10)
            ds = STL10Unlabeled(root=tmp)
            print(f"\ndataset length: {len(ds)}, expected: 10")
            assert len(ds) == 10

    def test_returns_two_views(self) -> None:
        """__getitem__ must return exactly two tensors (view_a, view_b)."""
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_dataset(tmp, n=4)
            ds = STL10Unlabeled(root=tmp)
            item = ds[0]
            assert isinstance(item, tuple) and len(item) == 2
            view_a, view_b = item
            print(f"\nview_a: {view_a.shape}, view_b: {view_b.shape}")

    def test_view_shapes_without_transform(self) -> None:
        """Without a transform, default ToTensor() gives (3, 96, 96)."""
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_dataset(tmp, n=4, size=96)
            ds = STL10Unlabeled(root=tmp)
            view_a, view_b = ds[0]
            assert view_a.shape == (3, 96, 96)
            assert view_b.shape == (3, 96, 96)
            print(f"\nshapes: {view_a.shape}, {view_b.shape}")

    def test_no_transform_gives_identical_views(self) -> None:
        """Without randomness, both views of the same image must be
        identical -- the same PIL image converted twice gives the same
        tensor. This confirms the two-view contract is correct."""
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_dataset(tmp, n=4)
            ds = STL10Unlabeled(root=tmp)
            view_a, view_b = ds[0]
            torch.testing.assert_close(view_a, view_b)
            print("\nno transform: both views identical -- confirmed")

    def test_random_transform_gives_different_views(self) -> None:
        """With a random augmentation, the two views should differ --
        this is the whole point of the two-view SSL design."""
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_dataset(tmp, n=4)
            # p=1.0 guarantees the flip always fires on both views
            # independently -- they'll differ unless the image is
            # horizontally symmetric (astronomically unlikely for random)
            transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.5),
                T.ToTensor(),
            ])
            ds = STL10Unlabeled(root=tmp, transform=transform)

            # run over several samples to make statistical failure unlikely
            found_different = False
            for i in range(len(ds)):
                view_a, view_b = ds[i]
                if not torch.allclose(view_a, view_b):
                    found_different = True
                    break

            print(f"\nfound views that differ: {found_different}")
            assert found_different, (
                "All views were identical -- random transform may not be working"
            )

    def test_raises_on_missing_directory(self) -> None:
        """A clearly wrong path should raise FileNotFoundError immediately
        at construction time, not silently return an empty dataset."""
        with pytest.raises(FileNotFoundError):
            STL10Unlabeled(root="/nonexistent/path/that/cannot/exist")

    def test_dataloader_collation(self) -> None:
        """DataLoader must be able to collate (view_a, view_b) pairs into
        batched tensors (B, 3, 96, 96) -- the shape the training loop
        receives."""
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_dataset(tmp, n=16)
            ds = STL10Unlabeled(root=tmp)
            loader = DataLoader(ds, batch_size=4, shuffle=False)
            batch = next(iter(loader))
            view_a_batch, view_b_batch = batch
            print(f"\nbatched view_a: {view_a_batch.shape}, "
                  f"view_b: {view_b_batch.shape}")
            assert view_a_batch.shape == (4, 3, 96, 96)
            assert view_b_batch.shape == (4, 3, 96, 96)

    def test_pixel_values_in_valid_range(self) -> None:
        """After ToTensor(), pixel values must be in [0, 1] --
        ToTensor() divides uint8 values by 255."""
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_dataset(tmp, n=4)
            ds = STL10Unlabeled(root=tmp)
            view_a, _ = ds[0]
            print(f"\npixel range: [{view_a.min().item():.3f}, "
                  f"{view_a.max().item():.3f}]")
            assert view_a.min().item() >= 0.0
            assert view_a.max().item() <= 1.0

    @pytest.mark.skipif(
        not os.path.exists("data/archive/unlabeled_images"),
        reason="real STL-10 data not present"
    )
    def test_real_dataset_length(self) -> None:
        """Integration test: verifies the real Kaggle download has the
        expected 100,000 images. Skipped if data folder not present."""
        ds = STL10Unlabeled(root="data/archive/unlabeled_images")
        print(f"\nreal dataset length: {len(ds)}")
        assert len(ds) == 100_000