"""
Tests for jepa_world_models.vic_reg_loss.config
"""

import pytest
from jepa_world_models.vic_reg_loss.config import VICRegConfig


class TestVICRegConfig:
    def test_default_construction(self) -> None:
        cfg = VICRegConfig()
        print(f"\n{cfg}")
        assert cfg.embed_dim == 192
        assert cfg.batch_size == 256
        assert cfg.epochs == 100

    def test_derived_num_patches(self) -> None:
        cfg = VICRegConfig(image_size=96, patch_size=8)
        print(f"\nnum_patches: {cfg.num_patches}")
        assert cfg.num_patches == 144

    def test_derived_patch_dim(self) -> None:
        cfg = VICRegConfig(patch_size=8, in_channels=3)
        print(f"\npatch_dim: {cfg.patch_dim}")
        assert cfg.patch_dim == 192

    def test_override_single_value(self) -> None:
        cfg = VICRegConfig(batch_size=64)
        assert cfg.batch_size == 64
        assert cfg.embed_dim == 192  # others unchanged

    def test_raises_on_incompatible_embed_dim_and_heads(self) -> None:
        with pytest.raises(AssertionError):
            VICRegConfig(embed_dim=192, num_heads=7)

    def test_raises_on_incompatible_image_and_patch_size(self) -> None:
        with pytest.raises(AssertionError):
            VICRegConfig(image_size=96, patch_size=7)

    def test_raises_on_batch_size_1(self) -> None:
        """BatchNorm1d in the projector requires B >= 2."""
        with pytest.raises(AssertionError):
            VICRegConfig(batch_size=1)

    def test_vicreg_loss_defaults_match_paper(self) -> None:
        """lambda=25, mu=25, nu=1, gamma=1 are the original paper values."""
        cfg = VICRegConfig()
        assert cfg.lambda_ == 25.0
        assert cfg.mu == 25.0
        assert cfg.nu == 1.0
        assert cfg.gamma == 1.0
        print(f"\nloss weights: lambda={cfg.lambda_}, mu={cfg.mu}, "
              f"nu={cfg.nu}, gamma={cfg.gamma}")