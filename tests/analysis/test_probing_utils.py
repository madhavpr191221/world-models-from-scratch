"""
Unit tests for probing helpers on synthetic data.
"""

import torch

from jepa_world_models.analysis.probing import (
    balanced_subset_indices,
    cosine_knn_predict,
    train_linear_probe,
)


def test_balanced_subset_indices_is_class_balanced() -> None:
    labels = torch.tensor([i for i in range(10) for _ in range(50)])
    indices = balanced_subset_indices(labels, fraction=0.1, seed=0)
    sampled = labels[indices]
    counts = torch.bincount(sampled, minlength=10)
    assert counts.min().item() == counts.max().item()
    assert counts.sum().item() == 50


def test_linear_probe_fits_simple_separable_data() -> None:
    torch.manual_seed(0)
    n_per_class = 64
    class0 = torch.randn(n_per_class, 8) * 0.2 - 2.0
    class1 = torch.randn(n_per_class, 8) * 0.2 + 2.0
    train_features = torch.cat([class0, class1], dim=0)
    train_labels = torch.tensor([0] * n_per_class + [1] * n_per_class)
    test_features = train_features.clone()
    test_labels = train_labels.clone()

    result, _ = train_linear_probe(
        train_features,
        train_labels,
        test_features,
        test_labels,
        num_classes=2,
        device="cpu",
        epochs=50,
        batch_size=32,
        lr=1e-2,
    )

    assert result.test_accuracy > 0.98


def test_cosine_knn_predict_returns_expected_class() -> None:
    train_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    train_labels = torch.tensor([0, 1])
    test_features = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    preds = cosine_knn_predict(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        k=1,
    )
    assert preds.tolist() == [0, 1]

