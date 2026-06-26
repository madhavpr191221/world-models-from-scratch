"""
STL-10 unlabeled dataset — PNG folder format.

Reads directly from the Kaggle PNG distribution of STL-10:
    data/archive/unlabeled_images/unlabeled_image_png_N.png

Why not torchvision.datasets.STL10
------------------------------------
torchvision.datasets.STL10 expects the original Stanford binary format
(unlabeled_X.bin packed as uint8 arrays). The Kaggle distribution ships
pre-extracted PNGs instead -- same images, different packaging. Rather
than converting formats, we read the PNGs directly with PIL, which is
what torchvision does internally anyway.

Two-view design
----------------
__getitem__ applies the transform TWICE independently to the same PIL
image, returning (view_a, view_b) -- two differently-augmented versions
of the same underlying image. This is the input contract for VICReg:
    z_a = projector(encoder(view_a))
    z_b = projector(encoder(view_b))
    loss = VICRegLoss(z_a, z_b)

The augmentation transform itself lives in
contrastive_learning/augmentations.py -- this class is intentionally
decoupled from any specific augmentation recipe. Pass any callable
that accepts a PIL Image and returns a Tensor.
"""

from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import datasets as D

from jepa_world_models.contrastive_learning.augmentations import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class STL10Unlabeled(Dataset):
    """
    STL-10 unlabeled split, PNG folder format.

    Returns two independently augmented views (view_a, view_b) of the
    same image per __getitem__ call -- the two-view contract for VICReg.

    Args:
        root:      path to the folder containing unlabeled PNG files.
                   Default: 'data/archive/unlabeled_images' relative to
                   the project root (where the Kaggle download lives).
        transform: callable applied independently twice per image to
                   produce the two views. Should accept a PIL Image and
                   return a Tensor of shape (3, 96, 96). If None, a
                   plain ToTensor() is applied (no augmentation --
                   useful for sanity checking, not for training).
    """

    DEFAULT_ROOT = "data/archive/unlabeled_images"

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.png"))

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No PNG files found in {self.root}. "
                f"Expected the Kaggle STL-10 unlabeled_images folder at this path."
            )

        self.transform = transform
        self._default_transform = T.ToTensor()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.files[idx]).convert("RGB")

        t = self.transform if self.transform is not None else self._default_transform

        # Apply independently: same PIL image, two different random
        # augmentation outcomes (when transform contains randomness).
        view_a = t(img)
        view_b = t(img)

        return view_a, view_b


def stl10_eval_transform(image_size: int = 96) -> T.Compose:
    """Deterministic eval transform for labeled STL-10 images and uploads."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class STL10Labeled(Dataset):
    """
    STL-10 labeled split backed by the official binary files in data_raw.

    torchvision.datasets.STL10 expects the binary format already unpacked
    under `root/stl10_binary`, which matches the raw data checked into this
    repo. Labels are remapped from 1..10 to 0..9 for standard classification.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_index: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.return_index = return_index
        self.transform = transform if transform is not None else stl10_eval_transform()
        self.dataset = D.STL10(
            root=str(self.root),
            split=split,
            download=False,
            transform=self.transform,
        )
        self.classes = tuple(self.dataset.classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        if self.return_index:
            return image, label, idx
        return image, label
