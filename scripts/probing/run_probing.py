"""
Run the STL-10 probing suite on a trained VICReg checkpoint.

Outputs:
    logs/probing/probe_summary.json
    logs/probing/probe_results.csv
    logs/probing/knn_results.csv
    logs/probing/layerwise_results.csv
    logs/probing/retrieval_index_encoder.pt
    logs/probing/retrieval_index_projector.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch

from jepa_world_models.analysis.probing import run_probe_suite
from jepa_world_models.analysis.retrieval import RetrievalIndex, save_retrieval_index
from jepa_world_models.analysis.common import l2_normalize


def _parse_fractions(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STL-10 probing suite")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/vicreg/best.pt",
        help="Path to a VICReg checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data_raw",
        help="Root containing stl10_binary",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu; defaults to auto-detect",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--fractions",
        type=str,
        default="0.01,0.1,1.0",
        help="Comma-separated train fractions for low-shot probes",
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/probing",
        help="Directory for probe reports and retrieval caches",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="logs/probing/feature_banks",
        help="Directory for cached train/test feature banks",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached embeddings and recompute them",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_probe_suite(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_epochs=args.train_epochs,
        probe_lr=args.probe_lr,
        probe_weight_decay=args.probe_weight_decay,
        fractions=_parse_fractions(args.fractions),
        k=args.k,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )

    probe_rows = [asdict(row) for row in results["probe_results"]]
    knn_rows = [asdict(row) for row in results["knn_results"]]
    layerwise_rows = [asdict(row) for row in results["layerwise_results"]]

    _write_csv(output_dir / "probe_results.csv", probe_rows)
    _write_csv(output_dir / "knn_results.csv", knn_rows)
    _write_csv(output_dir / "layerwise_results.csv", layerwise_rows)

    summary = {
        "checkpoint": results["checkpoint"],
        "device": results["device"],
        "cache_dir": args.cache_dir,
        "class_names": list(results["class_names"]),
        "probe_results": probe_rows,
        "knn_results": knn_rows,
        "layerwise_results": layerwise_rows,
    }
    _save_summary(output_dir / "probe_summary.json", summary)

    train_banks = results["train_features"]
    for feature_space, bank in train_banks.items():
        index = RetrievalIndex(
            feature_space=feature_space,
            embeddings=l2_normalize(bank.features).cpu(),
            labels=bank.labels.cpu(),
            indices=bank.indices.cpu(),
            class_names=tuple(results["class_names"]),
            checkpoint_path=results["checkpoint"],
        )
        save_retrieval_index(index, output_dir / f"retrieval_index_{feature_space}.pt")

    print(f"Wrote probing artifacts to {output_dir}")
    print(f"Feature cache dir: {args.cache_dir}")
    for row in probe_rows:
        print(
            f"{row['feature_space']:>9} | fraction={row['train_fraction']:<5} "
            f"| train_acc={row['train_accuracy']:.4f} | test_acc={row['test_accuracy']:.4f}"
        )
    for row in knn_rows:
        print(f"{row['feature_space']:>9} | k={row['k']} | knn_acc={row['accuracy']:.4f}")


if __name__ == "__main__":
    main()
