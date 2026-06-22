"""
VICReg training configuration.

All hyperparameters in one place. Every value here is either taken
directly from the original VICReg paper (Bardes et al. 2022) or was
chosen explicitly during this curriculum with documented reasoning.

Usage:
    from jepa_world_models.vic_reg_loss.config import VICRegConfig
    cfg = VICRegConfig()              # defaults
    cfg = VICRegConfig(batch_size=64) # override one value

The dataclass is frozen=False so you can override values after
construction if needed during experimentation.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VICRegConfig:
    """
    Full configuration for a VICReg pretraining run on STL-10.

    Sections:
        Data        -- dataset path and loading
        Encoder     -- ViTEncoder architecture
        Projector   -- MLP projector head
        Loss        -- VICReg loss hyperparameters
        Optimizer   -- Adam + cosine schedule
        Training    -- epochs, batch size, device
        Logging     -- checkpoint and log paths
    """

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    # Path to the Kaggle STL-10 unlabeled PNG folder, relative to
    # project root. Matches the path confirmed during setup.
    data_root: str = "data/archive/unlabeled_images"

    # Number of DataLoader worker processes. 0 = load on main process
    # (safe on Windows, which has issues with multiprocessing + CUDA).
    # Increase to 4 on Linux for faster data loading.
    num_workers: int = 4

    # Pin memory for faster CPU->GPU transfers. Only useful with
    # num_workers > 0; set False when num_workers=0.
    pin_memory: bool = True

    # -------------------------------------------------------------------------
    # Encoder (ViTEncoder)
    # -------------------------------------------------------------------------

    image_size: int = 96      # STL-10 image size, fixed
    patch_size: int = 8       # -> 12x12 = 144 patches per image
    in_channels: int = 3      # RGB

    # Embedding dimension -- chosen by BOTE analysis: d=192 comfortably
    # fits within 8GB VRAM even at large batch sizes. Matches patch_dim
    # (8*8*3=192) by coincidence of our patch size choice, but they are
    # deliberately decoupled via the PatchEmbedding layer.
    embed_dim: int = 192

    # Depth and heads -- ViT-Tiny scale, 6 blocks x 6 heads x head_dim=32.
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: int = 4        # MLP hidden dim = 4 * embed_dim = 768

    # -------------------------------------------------------------------------
    # Projector
    # -------------------------------------------------------------------------

    # proj_dim chosen by BOTE: 512 adds ~13 MiB at B=128 (essentially free)
    # and keeps the projector (~625K params) smaller than the encoder (2.7M).
    # The original paper uses 8192; we scale down proportionally to our
    # smaller encoder and dataset.
    proj_dim: int = 512

    # -------------------------------------------------------------------------
    # VICReg loss hyperparameters
    # -------------------------------------------------------------------------

    # lambda, mu, nu: weights for invariance, variance, covariance terms.
    # Original paper: lambda=25, mu=25, nu=1.
    # The high weight on invariance and variance relative to covariance
    # reflects that collapse prevention (var) and view alignment (inv)
    # are the primary objectives; covariance regularization is secondary.
    lambda_: float = 25.0
    mu: float = 25.0
    nu: float = 1.0

    # gamma: target standard deviation for each dimension (variance term
    # penalizes std below gamma). Original paper: gamma=1.0.
    gamma: float = 1.0

    # epsilon: numerical stability inside the sqrt in the variance term.
    # Prevents NaN gradients when a dimension's variance -> 0.
    eps: float = 1e-4

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    # Adam with default betas -- standard for transformer pretraining.
    # The original VICReg paper uses LARS optimizer for large batches
    # (4096); Adam is appropriate for our smaller batch sizes.
    lr: float = 1e-4           # learning rate
    weight_decay: float = 1e-6 # L2 regularization, light for SSL
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Warmup: linearly ramp lr from 0 to lr over the first warmup_epochs.
    # Important for transformer training -- jumping straight to the full
    # learning rate causes instability in the early epochs.
    warmup_epochs: int = 10

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    # Batch size: B=256 chosen based on BOTE analysis showing ~1.9 GiB
    # total at d=192, B=256, N=6, h=6 -- well within 8GB VRAM even after
    # doubling for VICReg's two views.
    batch_size: int = 256

    # Number of pretraining epochs. SSL methods typically need 100-400
    # epochs to converge; 100 is a reasonable starting point for STL-10
    # at our model scale.
    epochs: int = 100

    # Device: "cuda" if available, else "cpu". Set explicitly here so
    # train.py doesn't silently fall back to CPU without warning.
    device: str = "cuda"

    # -------------------------------------------------------------------------
    # Logging and checkpointing
    # -------------------------------------------------------------------------

    # Log every N batches (not every batch -- avoids I/O overhead).
    log_every_n_steps: int = 50

    # Save a checkpoint every N epochs.
    checkpoint_every_n_epochs: int = 5

    # Where to save checkpoints and logs.
    checkpoint_dir: str = "checkpoints/vicreg"
    log_dir: str = "logs/vicreg"

    def __post_init__(self) -> None:
        """Validate that the config is internally consistent."""
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        assert self.image_size % self.patch_size == 0, (
            f"image_size ({self.image_size}) must be divisible by "
            f"patch_size ({self.patch_size})"
        )
        assert self.batch_size >= 2, (
            f"batch_size must be >= 2 for BatchNorm1d in the projector "
            f"(got {self.batch_size})"
        )

    @property
    def num_patches(self) -> int:
        """Number of patches per image, derived from image_size / patch_size."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        """Flattened patch dimension: patch_size^2 * channels."""
        return self.patch_size ** 2 * self.in_channels

    def __str__(self) -> str:
        """Pretty-print for logging at the start of a training run."""
        lines = ["VICRegConfig:"]
        for key, val in self.__dict__.items():
            lines.append(f"  {key:<30} {val}")
        lines.append(f"  {'num_patches':<30} {self.num_patches}")
        lines.append(f"  {'patch_dim':<30} {self.patch_dim}")
        return "\n".join(lines)
