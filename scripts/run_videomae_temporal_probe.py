from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm

from jepa_world_models.analysis.videomae_pipeline import (
    SomethingSomethingVideoDataset,
    extract_videomae_features,
)


@dataclass
class ProbeResult:
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    report_path: str
    model_path: str
    feature_shape: tuple[int, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a forward-vs-reverse probe on VideoMAE features.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--source-split", type=str, default="train")
    parser.add_argument("--subset-size", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="logs/videomae/probe")
    parser.add_argument("--cache-dir", type=str, default="logs/videomae/probe/cache")
    return parser


def _split_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(n)
    train_end = max(1, int(n * 0.7))
    val_end = max(train_end + 1, int(n * 0.9))
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading VideoMAE temporal probe dataset...")
    dataset = SomethingSomethingVideoDataset(
        data_root=args.data_root,
        split=args.source_split,
        image_size=args.image_size,
        num_frames=args.num_frames,
        limit=args.subset_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    print(f"Loaded {len(dataset)} source clips")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    feature_batches = []
    label_batches = []
    print("Encoding forward and reversed clips...")
    for batch in tqdm(loader, desc="Encoding clips", unit="batch"):
        clip = batch["clip"]
        forward = clip
        reverse = torch.flip(clip, dims=[1])
        both = torch.cat([forward, reverse], dim=0)
        feats = extract_videomae_features(args.checkpoint, both)
        feature_batches.append(feats)
        label_batches.append(np.zeros(len(clip), dtype=np.int64))
        label_batches.append(np.ones(len(clip), dtype=np.int64))

    print("Stacking encoded features...")
    features = torch.cat(feature_batches, dim=0)
    y = np.concatenate(label_batches, axis=0)
    feature_shape = tuple(features.shape)
    train_idx, val_idx, test_idx = _split_indices(len(y))

    print("Training logistic regression probe...")
    flat = features.reshape(features.shape[0], -1).numpy()
    x_train = flat[train_idx]
    x_val = flat[val_idx]
    x_test = flat[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)

    train_pred = clf.predict(x_train)
    val_pred = clf.predict(x_val)
    test_pred = clf.predict(x_test)
    train_acc = float(accuracy_score(y_train, train_pred))
    val_acc = float(accuracy_score(y_val, val_pred))
    test_acc = float(accuracy_score(y_test, test_pred))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "videomae_temporal_probe.pkl"
    report_path = out_dir / "result.json"
    torch.save({"coef": clf.coef_, "intercept": clf.intercept_}, model_path)
    report_path.write_text(
        json.dumps(
            {
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
                "feature_shape": feature_shape,
                "model_path": str(model_path),
                "classification_report": classification_report(y_test, test_pred, output_dict=True),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Probe complete.")
    print(f"Wrote probe report to {report_path}")
    print(f"train_accuracy={train_acc:.4f}")
    print(f"val_accuracy={val_acc:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"feature_shape={feature_shape}")
    print(f"model_path={model_path}")


if __name__ == "__main__":
    main()
