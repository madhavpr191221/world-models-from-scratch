"""
contrastive_learning public API.

External code (train.py, analysis scripts) should import from here,
not from the internal sub-modules directly. This keeps the public
interface stable even if internal file structure changes.

Usage:
    from jepa_world_models.contrastive_learning import ViTEncoder, Projector
"""

from jepa_world_models.contrastive_learning.encoders.vit import ViTEncoder
from jepa_world_models.contrastive_learning.projector import Projector

__all__ = ["ViTEncoder", "Projector"]