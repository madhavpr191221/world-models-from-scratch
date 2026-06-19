import torch 
from torch import Tensor 

class PositionalEmbedding(torch.nn.Module):
    """
    Step 3: positional embedding.

    (B, num_patches, embed_dim) -> (B, num_patches, embed_dim), same shape

    A learned table, one row per patch position, added elementwise to the
    patch embeddings. Necessary because patchify + PatchEmbedding treat
    every patch identically regardless of where it sits in the spatial
    grid -- nothing so far tells the model "this was the top-left patch"
    vs "this was the bottom-right patch".

    Sequence length here is fixed (always 144, since STL-10 images are
    always 96x96 with patch_size=8) -- there is no requirement to
    generalize to unseen sequence lengths, so a learned table is strictly
    more expressive than a fixed formula (e.g. the sinusoidal encoding
    from the original Transformer paper, which exists specifically to
    extrapolate beyond training-time sequence lengths -- a property we
    don't need).
    """

    def __init__(self, num_patches: int = 144, embed_dim: int = 192) -> None:
        super().__init__()
        # One learnable d-dimensional vector per patch position.
        # Shape (num_patches, embed_dim), broadcasts over the batch dim
        # when added to (B, num_patches, embed_dim).
        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, embed_dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, num_patches, embed_dim) -> same shape, position added in
        return x + self.pos_embed