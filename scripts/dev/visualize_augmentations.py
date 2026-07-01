"""
Augmentation visualization script.

Takes 5 random images from the STL-10 unlabeled set, applies each
augmentation individually plus the full pipeline, and saves the results
to disk for visual inspection.

Usage (from project root):
    uv run python scripts/dev/visualize_augmentations.py

Output:
    augmentation_samples/
        image_<N>_original.png
        image_<N>_crop.png
        image_<N>_hflip.png
        image_<N>_colorjitter.png
        image_<N>_grayscale.png
        image_<N>_blur.png
        image_<N>_full_view_a.png
        image_<N>_full_view_b.png

Where N is the image index (1-5). The full_view_a and full_view_b files
show what the actual VICReg training loop sees for each image -- two
independently augmented views from the complete pipeline.
"""

import random
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image

# --- config ---
DATA_ROOT = "data/archive/unlabeled_images"
OUTPUT_DIR = "augmentation_samples"
N_IMAGES = 5
IMAGE_SIZE = 96
SEED = 42


def save_tensor_as_png(tensor: torch.Tensor, path: Path) -> None:
    """Convert a (3, H, W) float tensor in [0,1] to a PNG file."""
    tensor = tensor.clamp(0.0, 1.0)
    img = T.ToPILImage()(tensor)
    img.save(path)


def individual_augmentation(name: str, image_size: int = IMAGE_SIZE) -> T.Compose:
    """Returns a transform that applies ONLY the named augmentation,
    so we can see each one in isolation."""
    base = [T.ToTensor()]  # always at the end

    transforms = {
        "original": [T.Resize((image_size, image_size))],
        "crop": [T.RandomResizedCrop(
            size=image_size,
            scale=(0.08, 1.0),
            ratio=(3/4, 4/3),
            interpolation=InterpolationMode.BICUBIC,
        )],
        "hflip": [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=1.0),  # force flip
        ],
        "colorjitter": [
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        ],
        "grayscale": [
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),
        ],
        "blur": [
            T.Resize((image_size, image_size)),
            T.GaussianBlur(kernel_size=9, sigma=2.0),  # strong blur, always applied
        ],
    }

    return T.Compose(transforms[name] + base)


def full_pipeline_unnormalized(image_size: int = IMAGE_SIZE) -> T.Compose:
    """Full VICReg pipeline WITHOUT normalize -- keeps [0,1] for saving."""
    return T.Compose([
        T.RandomResizedCrop(
            size=image_size,
            scale=(0.08, 1.0),
            ratio=(3/4, 4/3),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(
            [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8,
        ),
        T.RandomGrayscale(p=0.2),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
            p=0.5,
        ),
        T.ToTensor(),
    ])


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    data_root = Path(DATA_ROOT)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # pick N_IMAGES random files
    all_files = sorted(data_root.glob("*.png"))
    if not all_files:
        raise FileNotFoundError(
            f"No PNG files found in {data_root}. "
            f"Run from the project root: uv run python scripts/dev/visualize_augmentations.py"
        )

    chosen = random.sample(all_files, N_IMAGES)
    full_pipeline = full_pipeline_unnormalized()

    augmentation_names = ["original", "crop", "hflip", "colorjitter", "grayscale", "blur"]

    print(f"Saving augmentation samples to {output_dir}/")
    print(f"{'File':<45} {'Saved'}")
    print("-" * 60)

    for i, filepath in enumerate(chosen, start=1):
        img = Image.open(filepath).convert("RGB")

        # individual augmentations
        for aug_name in augmentation_names:
            transform = individual_augmentation(aug_name)
            tensor = transform(img)
            out_path = output_dir / f"image_{i}_{aug_name}.png"
            save_tensor_as_png(tensor, out_path)
            print(f"  image_{i}_{aug_name}.png")

        # full pipeline, two independent views
        view_a = full_pipeline(img)
        view_b = full_pipeline(img)
        out_a = output_dir / f"image_{i}_full_view_a.png"
        out_b = output_dir / f"image_{i}_full_view_b.png"
        save_tensor_as_png(view_a, out_a)
        save_tensor_as_png(view_b, out_b)
        print(f"  image_{i}_full_view_a.png")
        print(f"  image_{i}_full_view_b.png")
        print()

    total = N_IMAGES * (len(augmentation_names) + 2)
    print(f"Done. {total} images saved to {output_dir}/")
    print(f"\nFor each image, compare:")
    print(f"  image_N_original.png  -- the raw STL-10 image (resized to 96x96)")
    print(f"  image_N_crop.png      -- random crop (notice scale and position change)")
    print(f"  image_N_hflip.png     -- horizontal flip")
    print(f"  image_N_colorjitter.png -- color/brightness/contrast shift")
    print(f"  image_N_grayscale.png -- grayscale conversion")
    print(f"  image_N_blur.png      -- Gaussian blur (strong, always applied here)")
    print(f"  image_N_full_view_a.png -- full VICReg pipeline, view a")
    print(f"  image_N_full_view_b.png -- full VICReg pipeline, view b (different!)")


if __name__ == "__main__":
    main()
