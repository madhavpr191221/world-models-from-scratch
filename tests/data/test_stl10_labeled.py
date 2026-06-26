"""
Tests for the labeled STL-10 wrapper used by probing and retrieval.
"""

from pathlib import Path

import pytest

from jepa_world_models.data.stl10 import STL10Labeled


def _raw_data_root() -> Path:
    return Path("data_raw")


@pytest.mark.skipif(
    not (_raw_data_root() / "stl10_binary").exists(),
    reason="raw STL-10 binary data not present",
)
class TestSTL10Labeled:
    def test_train_split_length(self) -> None:
        ds = STL10Labeled(root=str(_raw_data_root()), split="train")
        assert len(ds) == 5000

    def test_labels_are_zero_indexed(self) -> None:
        ds = STL10Labeled(root=str(_raw_data_root()), split="train")
        _, label = ds[0]
        assert 0 <= int(label) <= 9

    def test_return_index_contract(self) -> None:
        ds = STL10Labeled(root=str(_raw_data_root()), split="train", return_index=True)
        item = ds[0]
        assert isinstance(item, tuple) and len(item) == 3
        _, label, idx = item
        assert isinstance(idx, int)
        assert 0 <= int(label) <= 9

