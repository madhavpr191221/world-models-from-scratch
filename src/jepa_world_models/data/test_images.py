from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from torch.utils.data import Dataset

from jepa_world_models.data.stl10 import stl10_eval_transform


STL10_CLASSES = (
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
)


@dataclass(slots=True)
class TestImageRecord:
    class_name: str
    saved_name: str
    path: Path


class DownloadedImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "data/test_images",
        manifest: str | Path = "data/test_images_manifest.csv",
        *,
        image_size: int = 96,
        return_index: bool = True,
    ) -> None:
        self.root = Path(root)
        self.manifest = Path(manifest)
        self.return_index = return_index
        self.classes = tuple(STL10_CLASSES)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.transform = stl10_eval_transform(image_size=image_size)

        if not self.root.exists():
            raise FileNotFoundError(f"Image corpus root not found: {self.root}")
        if not self.manifest.exists():
            raise FileNotFoundError(f"Image manifest not found: {self.manifest}")

        self.records = self._load_records()
        self.filenames = tuple(record.saved_name for record in self.records)

    def _load_records(self) -> list[TestImageRecord]:
        records: list[TestImageRecord] = []
        with self.manifest.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                class_name = str(row["class_name"]).strip()
                saved_name = str(row["saved_name"]).strip()
                if class_name not in self.class_to_idx:
                    continue
                path = self.root / class_name / saved_name
                if not path.exists():
                    continue
                records.append(
                    TestImageRecord(
                        class_name=class_name,
                        saved_name=saved_name,
                        path=path,
                    )
                )
        if not records:
            raise RuntimeError(
                "No downloadable test images were found. "
                "Check data/test_images and data/test_images_manifest.csv."
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = self._load_image(record.path)
        image_tensor = self.transform(image)
        label = self.class_to_idx[record.class_name]
        if self.return_index:
            return image_tensor, label, index
        return image_tensor, label

    def _load_image(self, path: Path) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            tensor = read_image(str(path))
            if tensor.shape[0] == 4:
                tensor = tensor[:3]
            if tensor.dtype != torch.uint8:
                tensor = tensor.to(torch.uint8)
            return TF.to_pil_image(tensor)

    def get_filename(self, index: int) -> str:
        return self.records[index].saved_name

    def get_path(self, index: int) -> Path:
        return self.records[index].path
