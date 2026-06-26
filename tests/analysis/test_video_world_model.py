"""
Regression tests for the latent world-model helpers.
"""

import math

import torch

from jepa_world_models.analysis.video_world_model import (
    TemporalLatentPredictor,
    _latent_metrics,
    _split_indices,
)


def test_temporal_latent_predictor_returns_future_sequence() -> None:
    torch.manual_seed(0)
    model = TemporalLatentPredictor(
        latent_dim=192,
        context_steps=6,
        future_steps=2,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )
    x = torch.randn(3, 6, 192)
    y = model(x)
    assert y.shape == (3, 2, 192)


def test_latent_metrics_are_perfect_for_identical_tensors() -> None:
    x = torch.randn(2, 2, 8)
    metrics = _latent_metrics(x, x)
    assert math.isclose(metrics["latent_mse"], 0.0, abs_tol=1e-8)
    assert math.isclose(metrics["normalized_latent_mse"], 0.0, abs_tol=1e-8)
    assert math.isclose(metrics["cosine_similarity"], 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_split_indices_cover_every_sample_once() -> None:
    train, val, test = _split_indices(20, seed=0)
    combined = train + val + test
    assert sorted(combined) == list(range(20))
    assert len(set(combined)) == 20
