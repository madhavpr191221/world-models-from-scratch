"""
ViT encoder — final assembly.

Wires together all six steps into one ViTEncoder class:

    1. patchify:            (B, 3, 96, 96)  -> (B, 144, 192)
    2. patch embedding:     (B, 144, 192)   -> (B, 144, d)
    3. positional embedding:(B, 144, d)     -> (B, 144, d)
    4. N transformer blocks:(B, 144, d)     -> (B, 144, d)
    5. final LayerNorm:     (B, 144, d)     -> (B, 144, d)
    6. mean pool:           (B, 144, d)     -> (B, d)

Output: one d-dimensional representation per image, suitable for VICReg.

Pooling choice: global average pooling (GAP) over the 144 patch tokens,
rather than a CLS token. The original ViT paper (Dosovitskiy et al. 2021,
Appendix D.3, Figure 9) showed CLS and GAP perform identically once
learning rates are tuned appropriately -- the earlier "GAP performed very
poorly" result was a learning rate mismatch artifact, not an architectural
difference. GAP is the simpler choice and is used by the original VICReg
paper when pairing with a ViT backbone.
"""

import torch
from torch import Tensor

from jepa_world_models.contrastive_learning.encoders.patch_embedding import (
    PatchEmbedding,
    patchify,
)
from jepa_world_models.contrastive_learning.encoders.positional_embedding import (
    PositionalEmbedding,
)
from jepa_world_models.contrastive_learning.encoders.transformer_block import (
    TransformerBlock,
)


class ViTEncoder(torch.nn.Module):
    """
    Small ViT encoder for SSL pretraining on STL-10.

    Default config (ViT-Tiny scale, fits comfortably within 8GB VRAM
    per the BOTE calculation in docs/pytorch_stuff.md):
        patch_size  = 8    -> 12x12 = 144 patches per image
        embed_dim   = 192  -> d = 192 throughout
        depth       = 6    -> 6 stacked TransformerBlocks
        num_heads   = 6    -> 6 attention heads, head_dim = 32
        mlp_ratio   = 4    -> MLP hidden dim = 4 * 192 = 768

    Input:  (B, 3, H, W) images, H = W = 96 for STL-10
    Output: (B, embed_dim) one representation vector per image
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()

        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 144 for 96/8
        self.patch_dim = patch_size * patch_size * in_channels  # 192 for 8*8*3
        self.embed_dim = embed_dim

        # Step 2: patch embedding (step 1, patchify, is a pure function called in forward)
        self.patch_embed = PatchEmbedding(
            patch_dim=self.patch_dim,
            embed_dim=embed_dim,
        )

        # Step 3: positional embedding
        self.pos_embed = PositionalEmbedding(
            num_patches=self.num_patches,
            embed_dim=embed_dim,
        )

        # Step 4: N stacked transformer blocks
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        # Step 5: final LayerNorm (applied before pooling)
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, images: Tensor) -> Tensor:
        # Step 1: patchify -- (B, 3, H, W) -> (B, num_patches, patch_dim)
        x = patchify(images, self.patch_size)

        # Step 2: patch embedding -- (B, num_patches, patch_dim) -> (B, num_patches, d)
        x = self.patch_embed(x)

        # Step 3: add positional embedding -- (B, num_patches, d) -> (B, num_patches, d)
        x = self.pos_embed(x)

        # Step 4: N transformer blocks -- shape preserved throughout
        for block in self.blocks:
            x = block(x)

        # Step 5: final LayerNorm
        x = self.norm(x)

        # Step 6: mean pool over patch dimension -- (B, num_patches, d) -> (B, d)
        x = x.mean(dim=1)

        return x