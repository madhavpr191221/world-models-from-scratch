from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from jepa_world_models.analysis.common import (
    FeatureBank,
    build_eval_loader,
    build_labeled_splits,
    extract_feature_bank,
    l2_normalize,
    load_checkpointed_models,
    resolve_device,
)


@dataclass(slots=True)
class RetrievalIndex:
    feature_space: str
    embeddings: Tensor
    labels: Tensor
    indices: Tensor
    class_names: tuple[str, ...]
    checkpoint_path: str


def build_retrieval_index(
    checkpoint_path: str | Path,
    *,
    data_root: str | Path = "data_raw",
    device: str | torch.device | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
    feature_space: str = "encoder",
    cfg=None,
    encoder=None,
    projector=None,
    dataset=None,
) -> RetrievalIndex:
    device_obj = resolve_device(device)
    if encoder is None or projector is None:
        cfg, encoder, projector, _ = load_checkpointed_models(checkpoint_path, device=device_obj)
    if cfg is None:
        from jepa_world_models.vic_reg_loss.config import VICRegConfig

        cfg = VICRegConfig()
    if dataset is None:
        splits = build_labeled_splits(data_root=data_root, image_size=cfg.image_size)
        dataset = splits.train
        class_names = splits.class_names
    else:
        class_names = tuple(getattr(dataset, "classes", ()))

    train_loader = build_eval_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    bank = extract_feature_bank(
        encoder,
        projector,
        train_loader,
        device=device_obj,
        feature_space=feature_space,
    )
    return RetrievalIndex(
        feature_space=feature_space,
        embeddings=l2_normalize(bank.features).cpu(),
        labels=bank.labels.cpu(),
        indices=bank.indices.cpu(),
        class_names=class_names,
        checkpoint_path=str(checkpoint_path),
    )


def save_retrieval_index(index: RetrievalIndex, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, path)
    return str(path)


def load_retrieval_index(path: str | Path) -> RetrievalIndex:
    return torch.load(path, map_location="cpu", weights_only=False)


def query_neighbors(
    query_image: Tensor,
    index: RetrievalIndex,
    encoder,
    projector,
    device: str | torch.device | None = None,
    top_k: int = 5,
) -> tuple[Tensor, Tensor]:
    device_obj = resolve_device(device)
    encoder.eval()
    projector.eval()
    with torch.inference_mode():
        query = query_image.unsqueeze(0).to(device_obj)
        if index.feature_space == "encoder":
            query_embedding = encoder(query)
        elif index.feature_space == "projector":
            query_embedding = projector(encoder(query))
        else:
            raise ValueError(f"Unsupported feature_space: {index.feature_space}")
        query_embedding = l2_normalize(query_embedding).cpu()
        sims = query_embedding @ index.embeddings.T
        scores, neighbor_idx = sims.topk(k=min(top_k, index.embeddings.shape[0]), dim=1)
    return scores.squeeze(0), neighbor_idx.squeeze(0)


def neighbor_payload(
    query_image: Tensor,
    index: RetrievalIndex,
    encoder,
    projector,
    dataset,
    device: str | torch.device | None = None,
    top_k: int = 5,
):
    scores, neighbor_idx = query_neighbors(
        query_image=query_image,
        index=index,
        encoder=encoder,
        projector=projector,
        device=device,
        top_k=top_k,
    )
    payload = []
    for score, row_idx in zip(scores.tolist(), neighbor_idx.tolist()):
        dataset_idx = int(index.indices[row_idx].item())
        image, label = dataset[dataset_idx][:2]
        class_name = index.class_names[int(label)]
        payload.append((image, f"{class_name} | sim={score:.3f}"))
    return payload


def build_gradio_app(
    checkpoint_path: str | Path,
    *,
    data_root: str | Path = "data_raw",
    device: str | torch.device | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
    feature_space: str = "encoder",
    index_cache_path: str | Path | None = None,
):
    import gradio as gr
    from torchvision import transforms as T

    device_obj = resolve_device(device)
    cfg, encoder, projector, _ = load_checkpointed_models(checkpoint_path, device=device_obj)
    splits = build_labeled_splits(data_root=data_root, image_size=cfg.image_size)
    index = None
    if index_cache_path is not None and Path(index_cache_path).exists():
        index = load_retrieval_index(index_cache_path)
    else:
        index = build_retrieval_index(
            checkpoint_path,
            data_root=data_root,
            device=device_obj,
            batch_size=batch_size,
            num_workers=num_workers,
            feature_space=feature_space,
            cfg=cfg,
            encoder=encoder,
            projector=projector,
        )
        if index_cache_path is not None:
            save_retrieval_index(index, index_cache_path)

    upload_transform = T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    def search(image, k=5):
        if image is None:
            return []
        image_tensor = upload_transform(image)
        return neighbor_payload(
            image_tensor,
            index,
            encoder,
            projector,
            splits.train.dataset,
            device=device_obj,
            top_k=int(k),
        )

    with gr.Blocks(title="STL-10 Similarity Search") as demo:
        gr.Markdown(
            "# STL-10 Similarity Search\n"
            "Upload an image and retrieve the nearest neighbors in the learned embedding space."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload an image")
                k_input = gr.Slider(3, 10, value=5, step=1, label="Neighbors")
                search_button = gr.Button("Find neighbors")
            with gr.Column():
                gallery = gr.Gallery(
                    label="Nearest neighbors",
                    columns=2,
                    height=420,
                    object_fit="contain",
                )
        search_button.click(search, inputs=[image_input, k_input], outputs=gallery)
        image_input.change(search, inputs=[image_input, k_input], outputs=gallery)

    return demo
