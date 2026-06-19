"""
ViT encoder — patch embedding.

Step 1: patchify
-----------------
Turns an image into a sequence of flattened patches, since self-attention
operates on sequences of vectors, not 2D pixel grids.

(B, C, H, W) -> (B, num_patches, patch_dim)

For STL-10 (96x96x3) with patch_size=8: (B, 3, 96, 96) -> (B, 144, 192).

A plain reshape on the raw image tensor would NOT correctly extract
spatial patches -- it respects raw memory order (row by row), not spatial
adjacency. The fix used here: reshape the H and W axes into
(num_blocks, block_size) pairs first (which IS something a plain reshape
can do correctly, since it's splitting one axis into two nested axes, not
regrouping non-adjacent memory), then permute so the two "which patch"
indices sit together and the "patch content" indices sit together, then
merge each group into a single axis.

Step 2: patch embedding
------------------------
(B, num_patches, patch_dim) -> (B, num_patches, embed_dim)

A single learned linear layer, applied identically to every patch (the
same weight matrix shared across all 144 positions -- not 144 independent
layers). patch_dim and embed_dim are deliberately decoupled: patch_dim is
fixed by patch_size * channels (an arithmetic accident of how the image
was cut up), while embed_dim is a free design choice -- the dimension the
rest of the network actually reasons in. Even when patch_dim == embed_dim
(no dimension change), the layer is not a no-op: gradient descent,
driven by the loss at the end of the whole network, reshapes what each
output axis *means* over training, starting from axes that originally
meant nothing more than "pixel value at this exact spot."
"""

import torch
from torch import Tensor


def patchify(images: Tensor, patch_size: int) -> Tensor:
    """
    (B, C, H, W) -> (B, num_patches, patch_dim)

    Splits each image into non-overlapping P x P patches and flattens each
    patch into a single vector, turning the image into a sequence of tokens
    (the format self-attention expects).
    """
    B, C, H, W = images.shape
    P = patch_size
    Hp, Wp = H // P, W // P  # patches along each spatial axis (12, 12 for 96/8)

    # (B, C, H, W) -> (B, C, Hp, P, Wp, P)
    # Splits H into (Hp blocks, P pixels each) and W likewise -- this works
    # because H = Hp * P exactly, so reshape can split the axis cleanly
    # without needing unfold's sliding-window logic.
    x = images.reshape(B, C, Hp, P, Wp, P)

    # -> (B, Hp, Wp, C, P, P)
    # Reorders axes so the two "which patch" indices (Hp, Wp) sit together,
    # and the "patch content" indices (C, P, P) sit together.
    x = x.permute(0, 2, 4, 1, 3, 5)

    # -> (B, Hp * Wp, C * P * P) = (B, num_patches, patch_dim)
    # Merges Hp,Wp into a single sequence-length axis (144), and merges
    # C,P,P into a single feature axis per patch (192).
    x = x.reshape(B, Hp * Wp, C * P * P)
    return x


class PatchEmbedding(torch.nn.Module):
    """
    Step 2: patch embedding.

    (B, num_patches, patch_dim) -> (B, num_patches, embed_dim)

    A single linear layer applied identically to every patch:
        E = X W_e + b_e   (matching the row-stacked convention: X has one
                           patch per row, W_e in R^{patch_dim x embed_dim})

    This decouples the model's working dimension (embed_dim) from the
    accidental arithmetic of patch_size * channels (patch_dim), and gives
    gradient descent its first opportunity to reshape what each axis of
    the representation means -- raw pixel intensities going in, a learned
    coordinate system coming out.
    """

    def __init__(self, patch_dim: int = 192, embed_dim: int = 192) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(patch_dim, embed_dim)

    def forward(self, patches: Tensor) -> Tensor:
        # patches: (B, num_patches, patch_dim) -> (B, num_patches, embed_dim)
        return self.proj(patches)