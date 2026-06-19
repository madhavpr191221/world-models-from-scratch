"""
VICReg projector MLP.

Why the projector exists
-------------------------
The VICReg loss operates on projector outputs z', z'' -- NOT directly on
the encoder's representation. This separation is deliberate:

    encoder f_theta: image -> r in R^d        (the representation we care about)
    projector g_phi: r -> z in R^d_proj       (what the loss actually sees)

The VICReg loss imposes specific geometric constraints on z (variance pinned
to gamma, off-diagonal covariance penalized). Applying those constraints
directly to the encoder output would "use up" representational capacity
enforcing loss-specific geometry rather than general visual structure.

The projector absorbs those constraints. At evaluation time it is discarded
entirely -- the encoder is frozen and a linear classifier is trained on top
of r directly. This is what makes SSL representations transfer well.

Architecture
-------------
3-layer MLP matching the original VICReg paper (Bardes et al. 2022):

    Linear(d, d_proj, bias=False) -> BatchNorm1d -> ReLU
    Linear(d_proj, d_proj, bias=False) -> BatchNorm1d -> ReLU
    Linear(d_proj, d_proj, bias=True)

BatchNorm here, not LayerNorm: the projector sees one vector per image
(not a sequence), so batch statistics are meaningful and stable. The paper
obtained its results with BatchNorm specifically.

bias=False on the first two layers because BatchNorm already applies a
learned shift (beta) after normalization -- a bias before BN would be
immediately cancelled out and is a wasted parameter.

Scale choice
-------------
Original paper: d_proj = 8192. Our choice: d_proj = 512.
BOTE calculation (see docs/pytorch_stuff.md): at d_proj=512, the projector
adds ~13 MiB at B=128 against a 7179 MiB headroom -- essentially free.
Parameter count is 625K (~23% of the 2.7M encoder), keeping the projector
smaller than the encoder feeding it, which is the right relationship.
"""

import torch
import torch.nn as nn
from torch import Tensor


class Projector(nn.Module):
    """
    VICReg projector MLP.

    (B, in_dim) -> (B, proj_dim)

    in_dim:   must match the encoder's output dimension (embed_dim=192 for
              our default ViTEncoder).
    proj_dim: dimension of the space the VICReg loss operates in. Default
              512, chosen by BOTE analysis for our model scale on STL-10.
    """

    def __init__(self, in_dim: int = 192, proj_dim: int = 512) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim

        self.net = nn.Sequential(
            # Layer 1: expand from encoder dim to projector dim
            nn.Linear(in_dim, proj_dim, bias=False),   # bias=False: BN absorbs it
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),

            # Layer 2: project within projector dim
            nn.Linear(proj_dim, proj_dim, bias=False),  # bias=False: BN absorbs it
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),

            # Layer 3: final projection, no BN after (VICReg loss sees raw output)
            nn.Linear(proj_dim, proj_dim, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, in_dim) -> (B, proj_dim)
        return self.net(x)