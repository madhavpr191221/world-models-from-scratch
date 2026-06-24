"""
Regression tests for the VideoMAE reconstruction helpers.
"""

import torch

from jepa_world_models.analysis.video_reconstruction import (
    TubeletDecoderHead,
    batch_patchify_clips,
    masked_reconstruction_metrics,
    patchify_clip,
    unpatchify_tubelets,
)


def test_patchify_and_unpatchify_round_trip() -> None:
    torch.manual_seed(0)
    clip = torch.randn(16, 3, 32, 32)
    tubelets, (t_blocks, h_blocks, w_blocks) = patchify_clip(clip, patch_size=16, tubelet_size=2)
    rebuilt = unpatchify_tubelets(tubelets, t_blocks, h_blocks, w_blocks, patch_size=16, tubelet_size=2)
    assert torch.allclose(clip[: rebuilt.shape[0], :, : rebuilt.shape[2], : rebuilt.shape[3]], rebuilt)


def test_batch_patchify_returns_expected_shape() -> None:
    torch.manual_seed(0)
    clips = torch.randn(2, 16, 3, 32, 32)
    tubelets, shape = batch_patchify_clips(clips, patch_size=16, tubelet_size=2)
    assert tubelets.shape == (2, 32, 2, 3, 16, 16)
    assert shape == (8, 2, 2)


def test_tubelet_decoder_head_shape() -> None:
    head = TubeletDecoderHead(embed_dim=192, tubelet_dim=1536, hidden_dim=256, num_blocks=2)
    x = torch.randn(2, 5, 192)
    y = head(x)
    assert y.shape == (2, 5, 1536)


def test_masked_reconstruction_metrics_are_mask_local() -> None:
    target = torch.zeros(2, 2, 3, 4, 4)
    predicted = target.clone()
    predicted[1] = 1.0
    first_only = masked_reconstruction_metrics(predicted, target, torch.tensor([True, False]))
    second_only = masked_reconstruction_metrics(predicted, target, torch.tensor([False, True]))
    assert first_only["reconstruction_loss"] == 0.0
    assert first_only["psnr_db"] == float("inf")
    assert second_only["reconstruction_loss"] > 0.0
    assert second_only["masked_pixel_mse"] > 0.0
