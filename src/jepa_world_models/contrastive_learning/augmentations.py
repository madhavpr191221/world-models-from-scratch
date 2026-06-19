"""
SSL two-view augmentation pipeline for VICReg on STL-10.

Design goal
-----------
The augmentation pipeline defines the invariances the encoder is forced
to learn via the VICReg invariance term:

    L_inv = (1/N) sum_i || z_i' - z_i'' ||^2

For this to teach the encoder something useful, the two views must:
  - differ in ways irrelevant to semantic content (augmentations handle this)
  - be the same in ways that ARE semantically relevant (the encoder learns this)

If both views were identical, collapse is trivially satisfied and nothing is
learned. If they differed too much, the encoder would map unrelated content
to the same point. The augmentation recipe defines exactly which variations
are "irrelevant" -- the encoder will become invariant to all of them.

The five transforms, in order
------------------------------
1. RandomResizedCrop -- THE most important SSL augmentation. Forces
   scale and position invariance: two crops of the same image at different
   zoom levels and positions must share a representation.

2. RandomHorizontalFlip (p=0.5) -- left-right flip. Semantically neutral
   for almost all natural image content (dogs, cars, animals face both
   ways). NOT vertical flip -- natural images are not vertically symmetric.

3. ColorJitter (p=0.8) -- random brightness, contrast, saturation, hue
   perturbation. Forces invariance to lighting and color shift. Applied
   with p=0.8, not 1.0, so the encoder occasionally sees unaugmented
   color and doesn't over-adapt to color distortion.

4. RandomGrayscale (p=0.2) -- converts to grayscale 20% of the time.
   Forces the encoder to rely on shape/texture, not color alone. Without
   this, color becomes a dominant cue and representations are less
   transferable to grayscale or color-shifted domains.

5. GaussianBlur (p=0.5) -- random Gaussian blur. Forces invariance to
   high-frequency detail and fine-grained sharpness. The encoder should
   care about semantic structure, not pixel-level crispness.

Then: ToTensor() and Normalize() with ImageNet per-channel statistics.

Order constraint (important)
-----------------------------
ColorJitter, RandomGrayscale, GaussianBlur must come BEFORE ToTensor --
they operate on PIL Images. Normalize must come AFTER ToTensor -- it
operates on tensors. Applying PIL transforms to tensors raises a cryptic
error; this ordering avoids it.

Hyperparameters
---------------
Taken directly from the original VICReg paper (Bardes et al. 2022),
which in turn follows the SimCLR augmentation recipe (Chen et al. 2020).
"""

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


# ImageNet per-channel statistics -- standard normalization for models
# pretrained on or evaluated against ImageNet-like data. STL-10 images
# come from ImageNet, so these statistics are appropriate.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def vicreg_augmentation(image_size: int = 96) -> T.Compose:
    """
    Two-view SSL augmentation pipeline for VICReg.

    Returns a callable that accepts a PIL Image and returns a normalized
    Tensor of shape (3, image_size, image_size). Call it TWICE on the
    same PIL image to get two independently augmented views:

        transform = vicreg_augmentation(image_size=96)
        view_a = transform(pil_image)
        view_b = transform(pil_image)

    This is exactly what STL10Unlabeled does in __getitem__ when this
    transform is passed as the `transform` argument.

    Args:
        image_size: output spatial size. Default 96 for STL-10.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        # 1. Random crop + resize: scale invariance and position invariance.
        #    scale=(0.08, 1.0): crops between 8% and 100% of image area.
        #    ratio=(0.75, 1.333): aspect ratio range (3:4 to 4:3).
        #    This is the most impactful single augmentation in SSL.
        T.RandomResizedCrop(
            size=image_size,
            scale=(0.08, 1.0),
            ratio=(3/4, 4/3),
            interpolation=InterpolationMode.BICUBIC,
        ),

        # 2. Horizontal flip: left-right symmetry invariance.
        T.RandomHorizontalFlip(p=0.5),

        # 3. Color jitter (applied with p=0.8, not always):
        #    brightness, contrast, saturation each perturbed up to 0.4x,
        #    hue perturbed up to 0.1x (subtle -- large hue shifts are
        #    semantically significant, e.g. green apple vs red apple).
        T.RandomApply(
            [T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            )],
            p=0.8,
        ),

        # 4. Grayscale (p=0.2): forces shape/texture reliance over color.
        #    output_channels=3 keeps the tensor 3-channel (required for
        #    the encoder which expects 3-channel input) -- the three
        #    channels are just identical copies of the grayscale value.
        T.RandomGrayscale(p=0.2),

        # 5. Gaussian blur (p=0.5): fine-detail invariance.
        #    kernel_size must be odd; 9 is ~10% of 96px image width.
        #    sigma range (0.1, 2.0) from the SimCLR / VICReg recipe.
        T.RandomApply(
            [T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
            p=0.5,
        ),

        # Convert PIL Image to float Tensor in [0, 1], shape (3, H, W).
        T.ToTensor(),

        # Normalize to ImageNet statistics (zero mean, unit std per channel).
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def vicreg_augmentation_unnormalized(image_size: int = 96) -> T.Compose:
    """
    Same as vicreg_augmentation but WITHOUT the final Normalize step.

    Useful for visualization -- normalized tensors have values outside
    [0, 1] and look wrong when saved as images. Use this when you want
    to inspect what the augmented images actually look like on disk.
    """
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
        # No Normalize -- keeps values in [0, 1] for saving as PNG
    ])