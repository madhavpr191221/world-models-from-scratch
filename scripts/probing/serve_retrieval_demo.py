from __future__ import annotations

import argparse
import cgi
import base64
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from torchvision.transforms import functional as TF

from jepa_world_models.analysis.common import build_labeled_splits
from jepa_world_models.analysis.retrieval import build_retrieval_index, load_retrieval_index, query_neighbors
from jepa_world_models.data.test_images import DownloadedImageDataset
from jepa_world_models.data.stl10 import stl10_eval_transform
from jepa_world_models.vic_reg_loss.train import load_model_from_checkpoint


STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the retrieval demo.")
    parser.add_argument("--checkpoint", required=True, help="Path to a VICReg checkpoint.")
    parser.add_argument(
        "--index-path",
        default="logs/retrieval/retrieval_index_test_images.pt",
        help="Path to the cached retrieval index.",
    )
    parser.add_argument(
        "--corpus-root",
        default="data/test_images",
        help="Root directory for the retrieval image corpus.",
    )
    parser.add_argument(
        "--manifest",
        default="data/test_images_manifest.csv",
        help="CSV manifest for the retrieval image corpus.",
    )
    parser.add_argument(
        "--frontend-dir",
        default="frontend",
        help="Directory containing the static frontend files.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to return.",
    )
    parser.add_argument(
        "--feature-space",
        choices=["encoder", "projector"],
        default="encoder",
        help="Which feature space to use for the query embedding.",
    )
    return parser.parse_args()


def _load_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(file_bytes))
        return image.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Uploaded file is not a valid image.") from exc


def _lookup_label(index: Any, dataset_index: int, raw_label: Any = None) -> str:
    if raw_label is not None:
        if isinstance(raw_label, str):
            return raw_label
        try:
            label_index = int(raw_label)
            if 0 <= label_index < len(STL10_CLASSES):
                return STL10_CLASSES[label_index]
            return str(label_index)
        except Exception:  # noqa: BLE001
            return str(raw_label)

    for attr in ("labels", "targets", "y", "classes", "class_names"):
        values = getattr(index, attr, None)
        if values is None:
            continue

        try:
            if attr in {"classes", "class_names"}:
                if isinstance(values, (list, tuple)) and values:
                    return str(values[dataset_index])
            else:
                value = values[dataset_index]
                if isinstance(value, str):
                    return value
                label_index = int(value)
                if 0 <= label_index < len(STL10_CLASSES):
                    return STL10_CLASSES[label_index]
                return str(label_index)
        except Exception:  # noqa: BLE001
            continue

    return f"sample {dataset_index}"


def _normalize_neighbor(index: Any, item: Any, rank: int) -> dict[str, Any]:
    if isinstance(item, dict):
        dataset_index = item.get("index", item.get("dataset_index", item.get("idx")))
        score = item.get("score", item.get("similarity", item.get("sim")))
        label = item.get("label")
        filename = item.get("filename", item.get("file_name"))
    elif hasattr(item, "__dict__"):
        payload = vars(item)
        dataset_index = payload.get("index", payload.get("dataset_index", payload.get("idx")))
        score = payload.get("score", payload.get("similarity", payload.get("sim")))
        label = payload.get("label")
        filename = payload.get("filename", payload.get("file_name"))
    elif isinstance(item, (list, tuple)):
        dataset_index = item[0] if item else None
        score = item[1] if len(item) > 1 else None
        label = item[2] if len(item) > 2 else None
        filename = item[3] if len(item) > 3 else None
    else:
        dataset_index = item
        score = None
        label = None
        filename = None

    dataset_index = int(dataset_index) if dataset_index is not None else rank
    label_text = _lookup_label(index, dataset_index, label)
    score_text = None if score is None else float(score)

    return {
        "rank": rank,
        "index": dataset_index,
        "filename": filename,
        "label": label_text,
        "score": score_text,
    }


def _normalize_neighbors(index: Any, raw_neighbors: Any) -> list[dict[str, Any]]:
    if raw_neighbors is None:
        return []

    if isinstance(raw_neighbors, dict):
        for key in ("neighbors", "items", "results", "topk"):
            if key in raw_neighbors:
                raw_neighbors = raw_neighbors[key]
                break

    if isinstance(raw_neighbors, tuple) and len(raw_neighbors) == 2:
        left, right = raw_neighbors
        if len(left) == len(right):
            raw_neighbors = list(zip(left, right))

    if isinstance(raw_neighbors, torch.Tensor):
        raw_neighbors = raw_neighbors.tolist()

    if not isinstance(raw_neighbors, (list, tuple)):
        raw_neighbors = [raw_neighbors]

    neighbors: list[dict[str, Any]] = []
    for rank, item in enumerate(raw_neighbors, start=1):
        neighbors.append(_normalize_neighbor(index, item, rank))
    return neighbors


def _tensor_to_data_uri(image_tensor: torch.Tensor) -> str:
    tensor = image_tensor.detach().cpu()
    if tensor.ndim != 3:
      raise ValueError("Expected a CHW tensor.")

    if tensor.shape[0] == 3:
        mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        tensor = tensor * std + mean

    tensor = tensor.clamp(0.0, 1.0)
    pil_image = TF.to_pil_image(tensor)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _attach_thumbnails(
    neighbors: list[dict[str, Any]],
    index: Any,
    dataset,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for neighbor in neighbors:
        dataset_index = int(neighbor["index"])
        try:
            sample = dataset[dataset_index]
            image_tensor = sample[0]
            neighbor["thumbnail"] = _tensor_to_data_uri(image_tensor)
        except Exception:  # noqa: BLE001
            neighbor["thumbnail"] = None
        enriched.append(neighbor)
    return enriched


def _lookup_filename(dataset: Any, dataset_index: int) -> str:
    getter = getattr(dataset, "get_filename", None)
    if callable(getter):
        try:
            return str(getter(dataset_index))
        except Exception:  # noqa: BLE001
            pass

    filenames = getattr(dataset, "filenames", None)
    if filenames is not None:
        try:
            return str(filenames[dataset_index])
        except Exception:  # noqa: BLE001
            pass

    return f"sample_{dataset_index:04d}.png"


def _run_query(
    encoder: torch.nn.Module,
    projector: torch.nn.Module,
    retrieval_index: Any,
    retrieval_dataset: Any,
    image: Image.Image,
    query_filename: str | None,
    feature_space: str,
    top_k: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    transform = stl10_eval_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        query_tensor = image_tensor.squeeze(0)
        raw_scores, raw_neighbor_idx = query_neighbors(
            query_tensor,
            retrieval_index,
            encoder,
            projector,
            device=device,
            top_k=min(top_k + 1, int(retrieval_index.embeddings.shape[0])),
        )

    raw_neighbors = []
    for rank, (score, row_idx) in enumerate(
        zip(raw_scores.tolist(), raw_neighbor_idx.tolist()),
        start=1,
    ):
        dataset_idx = int(retrieval_index.indices[row_idx].item())
        filename = _lookup_filename(retrieval_dataset, dataset_idx)
        if query_filename and filename == query_filename:
            continue
        raw_neighbors.append(
            {
                "rank": rank,
                "index": dataset_idx,
                "filename": filename,
                "score": float(score),
                "label": int(retrieval_index.labels[row_idx].item()),
            }
        )

    return _normalize_neighbors(retrieval_index, raw_neighbors)


class RetrievalDemoHandler(SimpleHTTPRequestHandler):
    model: torch.nn.Module
    encoder: torch.nn.Module
    projector: torch.nn.Module
    retrieval_index: Any
    retrieval_dataset: Any
    device: torch.device
    feature_space: str
    top_k: int

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/retrieve":
            self.send_error(404, "Unknown endpoint")
            return

        content_type = self.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart form data")
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": self.headers.get("content-length", "0"),
                },
                keep_blank_values=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.send_error(400, f"Could not parse upload: {exc}")
            return

        file_item = form["image"] if "image" in form else None
        if file_item is None or not getattr(file_item, "file", None):
            self.send_error(400, "Missing image upload")
            return

        try:
            file_bytes = file_item.file.read()
            image = _load_image(file_bytes)
            neighbors = _run_query(
                self.encoder,
                self.projector,
                self.retrieval_index,
                self.retrieval_dataset,
                image,
                file_item.filename,
                self.feature_space,
                self.top_k,
                self.device,
            )
            neighbors = _attach_thumbnails(
                neighbors,
                self.retrieval_index,
                self.retrieval_dataset,
            )
        except ValueError as exc:
            self.send_error(400, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self.send_error(500, f"Retrieval failed: {exc}")
            return

        payload = {
            "query": {
                "filename": file_item.filename,
                "content_type": file_item.type,
                "width": image.width,
                "height": image.height,
                "feature_space": self.feature_space,
            },
            "neighbors": neighbors,
        }

        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print(format % args)


def main() -> None:
    args = parse_args()
    default_index_paths = {
        "encoder": "logs/probing/retrieval_index_encoder.pt",
        "projector": "logs/probing/retrieval_index_projector.pt",
    }
    if args.index_path == default_index_paths["encoder"] and args.feature_space == "projector":
        args.index_path = default_index_paths["projector"]

    frontend_dir = Path(args.frontend_dir).resolve()
    if not frontend_dir.exists():
        raise SystemExit(f"Frontend directory not found: {frontend_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {args.checkpoint}")
    cfg, encoder, projector, _ = load_model_from_checkpoint(args.checkpoint)
    encoder = encoder.to(device)
    projector = projector.to(device)
    encoder.eval()
    projector.eval()

    index_path = Path(args.index_path).resolve()
    corpus_root = Path(args.corpus_root).resolve()
    manifest_path = Path(args.manifest).resolve()

    if corpus_root.exists() and manifest_path.exists():
        retrieval_dataset = DownloadedImageDataset(
            root=corpus_root,
            manifest=manifest_path,
            image_size=cfg.image_size,
            return_index=True,
        )
        if index_path.exists():
            retrieval_index = load_retrieval_index(index_path)
        else:
            retrieval_index = build_retrieval_index(
                args.checkpoint,
                device=device,
                batch_size=256,
                num_workers=0,
                feature_space=args.feature_space,
                cfg=cfg,
                encoder=encoder,
                projector=projector,
                dataset=retrieval_dataset,
            )
            index_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(retrieval_index, index_path)
    else:
        if not index_path.exists():
            raise SystemExit(
                f"Retrieval index not found: {index_path}. "
                "Run the probing pipeline first to build the cached index."
            )
        retrieval_index = load_retrieval_index(index_path)
        splits = build_labeled_splits(data_root="data_raw", image_size=96)
        retrieval_dataset = splits.train

    handler = type(
        "ConfiguredRetrievalDemoHandler",
        (RetrievalDemoHandler,),
        {},
    )
    handler.encoder = encoder
    handler.projector = projector
    handler.retrieval_index = retrieval_index
    handler.retrieval_dataset = retrieval_dataset
    handler.device = device
    handler.feature_space = args.feature_space
    handler.top_k = args.top_k

    server = ThreadingHTTPServer(
        (args.host, args.port),
        partial(handler, directory=str(frontend_dir)),
    )

    print(f"Serving frontend from {frontend_dir}")
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
