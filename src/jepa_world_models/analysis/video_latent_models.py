from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import video as tv_video


def _ensure_5d(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected video tensor with shape (B, T, C, H, W), got {tuple(video.shape)}")
    return video


def _split_tubelets(video: torch.Tensor, tubelet_size: int) -> torch.Tensor:
    video = _ensure_5d(video)
    if video.shape[1] % tubelet_size != 0:
        raise ValueError(f"num_frames={video.shape[1]} must be divisible by tubelet_size={tubelet_size}.")
    return video.reshape(video.shape[0], video.shape[1] // tubelet_size, tubelet_size, *video.shape[2:])


class BaseTubeletEncoder(nn.Module):
    def __init__(self, latent_dim: int, tubelet_size: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.tubelet_size = tubelet_size

    def encode_tubelet(self, tubelet: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def forward_sequence(self, video: torch.Tensor) -> torch.Tensor:
        tubelets = _split_tubelets(video, self.tubelet_size)
        outputs = [self.encode_tubelet(tubelets[:, i]).unsqueeze(1) for i in range(tubelets.shape[1])]
        return torch.cat(outputs, dim=1)


class _BackboneHeadlessEncoder(BaseTubeletEncoder):
    def __init__(self, latent_dim: int, tubelet_size: int = 2, pretrained: bool = False) -> None:
        super().__init__(latent_dim=latent_dim, tubelet_size=tubelet_size)
        self.pretrained = pretrained
        self.backbone: nn.Module
        self.feature_dim: int
        self.proj: nn.Module | None = None

    def _finalize(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        if feature_dim != self.latent_dim:
            self.proj = nn.Linear(feature_dim, self.latent_dim)
        else:
            self.proj = nn.Identity()

    def encode_tubelet(self, tubelet: torch.Tensor) -> torch.Tensor:
        features = self.backbone(tubelet.permute(0, 2, 1, 3, 4))
        if features.ndim > 2:
            features = features.flatten(1)
        assert self.proj is not None
        return self.proj(features)


def _headless_torchvision_model(builder: Callable[..., nn.Module], weights=None) -> tuple[nn.Module, int]:
    model = builder(weights=weights)
    feature_dim: int | None = None
    if hasattr(model, "head"):
        head = getattr(model, "head")
        if isinstance(head, nn.Module) and hasattr(head, "in_features"):
            feature_dim = int(head.in_features)  # type: ignore[attr-defined]
        model.head = nn.Identity()
    if hasattr(model, "fc"):
        fc = getattr(model, "fc")
        if feature_dim is None and isinstance(fc, nn.Module) and hasattr(fc, "in_features"):
            feature_dim = int(fc.in_features)  # type: ignore[attr-defined]
        model.fc = nn.Identity()
    if hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if feature_dim is None and isinstance(classifier, nn.Module) and hasattr(classifier, "in_features"):
            feature_dim = int(classifier.in_features)  # type: ignore[attr-defined]
        model.classifier = nn.Identity()
    if feature_dim is None:
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 2, 224, 224)
            out = model(dummy)
            feature_dim = int(out.shape[-1])
    return model, feature_dim


class Swin3DTubeletEncoder(_BackboneHeadlessEncoder):
    def __init__(self, latent_dim: int, tubelet_size: int = 2, pretrained: bool = False, variant: str = "t") -> None:
        super().__init__(latent_dim=latent_dim, tubelet_size=tubelet_size, pretrained=pretrained)
        builder_map = {
            "t": tv_video.swin3d_t,
            "s": tv_video.swin3d_s,
        }
        builder = builder_map.get(variant, tv_video.swin3d_t)
        weights = None
        if pretrained:
            weight_enum = getattr(tv_video, f"Swin3D_{variant.upper()}_Weights", None)
            weights = getattr(weight_enum, "DEFAULT", None) if weight_enum is not None else None
        self.backbone, feature_dim = _headless_torchvision_model(builder, weights=weights)
        self._finalize(feature_dim)


class _TubeletTransformerEncoder(BaseTubeletEncoder):
    def __init__(
        self,
        latent_dim: int,
        tubelet_size: int = 2,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(latent_dim=latent_dim, tubelet_size=tubelet_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.spatial_tokens = (224 // patch_size) ** 2
        self.proj = nn.Conv3d(3, embed_dim, kernel_size=(tubelet_size, patch_size, patch_size), stride=(tubelet_size, patch_size, patch_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, latent_dim) if embed_dim != latent_dim else nn.Identity()
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _encode_patch_tokens(self, tubelet: torch.Tensor) -> torch.Tensor:
        x = tubelet.permute(0, 2, 1, 3, 4)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

    def encode_tubelet(self, tubelet: torch.Tensor) -> torch.Tensor:
        tokens = self._encode_patch_tokens(tubelet)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)


class TimeSformerLikeEncoder(BaseTubeletEncoder):
    def __init__(
        self,
        latent_dim: int,
        tubelet_size: int = 2,
        patch_size: int = 16,
        embed_dim: int = 192,
        spatial_depth: int = 2,
        temporal_depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(latent_dim=latent_dim, tubelet_size=tubelet_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        spatial_tokens = (224 // patch_size) ** 2
        self.spatial_pos = nn.Parameter(torch.zeros(1, spatial_tokens, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, tubelet_size, embed_dim))
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_blocks = nn.TransformerEncoder(spatial_layer, num_layers=spatial_depth)
        self.temporal_blocks = nn.TransformerEncoder(temporal_layer, num_layers=temporal_depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, latent_dim) if embed_dim != latent_dim else nn.Identity()
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def encode_tubelet(self, tubelet: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = tubelet.shape
        flat_frames = tubelet.reshape(batch * frames, channels, height, width)
        tokens = self.patch_embed(flat_frames).flatten(2).transpose(1, 2)
        tokens = tokens + self.spatial_pos[:, : tokens.shape[1]]
        tokens = self.spatial_blocks(tokens)
        frame_tokens = tokens.mean(dim=1).reshape(batch, frames, self.embed_dim)
        frame_tokens = frame_tokens + self.temporal_pos[:, :frames]
        frame_tokens = self.temporal_blocks(frame_tokens)
        pooled = self.norm(frame_tokens).mean(dim=1)
        return self.head(pooled)


class Hybrid3DCNNTransformerEncoder(BaseTubeletEncoder):
    def __init__(
        self,
        latent_dim: int,
        tubelet_size: int = 2,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(latent_dim=latent_dim, tubelet_size=tubelet_size)
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.Conv3d(128, embed_dim, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, latent_dim) if embed_dim != latent_dim else nn.Identity()
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode_tubelet(self, tubelet: torch.Tensor) -> torch.Tensor:
        x = tubelet.permute(0, 2, 1, 3, 4)
        x = self.stem(x)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        tokens = self.blocks(tokens)
        pooled = self.norm(tokens).mean(dim=1)
        return self.head(pooled)


class CausalTransformerPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_steps: int,
        future_steps: int,
        hidden_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.context_steps = context_steps
        self.future_steps = future_steps
        self.input_norm = nn.LayerNorm(latent_dim)
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.context_pos = nn.Parameter(torch.zeros(1, context_steps, hidden_dim))
        self.future_query = nn.Parameter(torch.zeros(1, future_steps, hidden_dim))
        self.future_pos = nn.Parameter(torch.zeros(1, future_steps, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, latent_dim)
        nn.init.trunc_normal_(self.context_pos, std=0.02)
        nn.init.trunc_normal_(self.future_query, std=0.02)
        nn.init.trunc_normal_(self.future_pos, std=0.02)

    def forward(self, context_latents: torch.Tensor) -> torch.Tensor:
        context = self.input_proj(self.input_norm(context_latents)) + self.context_pos
        future = self.future_query.expand(context.shape[0], -1, -1) + self.future_pos
        tokens = torch.cat([context, future], dim=1)
        total = tokens.shape[1]
        mask = torch.triu(torch.full((total, total), float("-inf"), device=tokens.device), diagonal=1)
        encoded = self.blocks(tokens, mask=mask)
        future_encoded = encoded[:, -self.future_steps :, :]
        return self.output_head(self.out_norm(future_encoded))


class CrossAttentionPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_steps: int,
        future_steps: int,
        hidden_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.context_steps = context_steps
        self.future_steps = future_steps
        self.input_norm = nn.LayerNorm(latent_dim)
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.context_pos = nn.Parameter(torch.zeros(1, context_steps, hidden_dim))
        self.future_query = nn.Parameter(torch.zeros(1, future_steps, hidden_dim))
        self.future_pos = nn.Parameter(torch.zeros(1, future_steps, hidden_dim))
        self.self_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.cross_attn = nn.ModuleList([nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, latent_dim)
        nn.init.trunc_normal_(self.context_pos, std=0.02)
        nn.init.trunc_normal_(self.future_query, std=0.02)
        nn.init.trunc_normal_(self.future_pos, std=0.02)

    def forward(self, context_latents: torch.Tensor) -> torch.Tensor:
        memory = self.input_proj(self.input_norm(context_latents)) + self.context_pos
        query = self.future_query.expand(context_latents.shape[0], -1, -1) + self.future_pos
        for self_block, cross_attn, ffn in zip(self.self_blocks, self.cross_attn, self.ffn):
            query = self_block(query)
            attn_out, _ = cross_attn(query, memory, memory, need_weights=False)
            query = query + attn_out
            query = query + ffn(query)
        return self.output_head(self.out_norm(query))


class MambaStylePredictor(nn.Module):
    """
    Lightweight state-space style predictor.

    This is intentionally dependency-free. It is not the external mamba-ssm package,
    but it follows the same modeling intent: a causal latent dynamics block with a
    compact hidden state and recurrent updates.
    """

    def __init__(
        self,
        latent_dim: int,
        context_steps: int,
        future_steps: int,
        hidden_dim: int = 192,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del num_heads
        self.latent_dim = latent_dim
        self.context_steps = context_steps
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim
        self.input_norm = nn.LayerNorm(latent_dim)
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.step_pos = nn.Parameter(torch.zeros(1, context_steps + future_steps, hidden_dim))
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.state_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, latent_dim)
        nn.init.trunc_normal_(self.step_pos, std=0.02)

    def _encode_context(self, context_latents: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(self.input_norm(context_latents)) + self.step_pos[:, : context_latents.shape[1]]
        for layer in self.layers:
            x = x + layer(x)
        state = x[:, -1, :]
        return state

    def forward(self, context_latents: torch.Tensor) -> torch.Tensor:
        state = self._encode_context(context_latents)
        outputs: list[torch.Tensor] = []
        prev = state
        for step in range(self.future_steps):
            step_token = self.step_pos[:, context_latents.shape[1] + step, :].expand(context_latents.shape[0], -1)
            state = self.state_cell(step_token, prev)
            prev = state
            outputs.append(self.output_head(state).unsqueeze(1))
        return torch.cat(outputs, dim=1)


class OneLagMLPPredictor(nn.Module):
    """A simple one-lag baseline that maps the latest latent to future latents."""

    def __init__(
        self,
        latent_dim: int,
        context_steps: int,
        future_steps: int,
        hidden_dim: int = 192,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del context_steps, num_heads
        self.latent_dim = latent_dim
        self.future_steps = future_steps
        layers: list[nn.Module] = [nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.GELU()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, future_steps * latent_dim)

    def forward(self, context_latents: torch.Tensor) -> torch.Tensor:
        if context_latents.ndim != 3:
            raise ValueError(f"Expected context_latents with shape (B, C, D), got {tuple(context_latents.shape)}")
        x = context_latents[:, -1, :]
        x = self.backbone(x)
        future = self.output_head(x)
        return future.view(context_latents.shape[0], self.future_steps, self.latent_dim)


class TemporalConvPredictor(nn.Module):
    """A causal temporal convolution baseline over the latent context window."""

    def __init__(
        self,
        latent_dim: int,
        context_steps: int,
        future_steps: int,
        hidden_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del num_heads
        self.latent_dim = latent_dim
        self.context_steps = context_steps
        self.future_steps = future_steps
        self.input_norm = nn.LayerNorm(latent_dim)
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        dilation = 1
        for _ in range(max(1, num_layers)):
            self.layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            dilation *= 2
        self.output_head = nn.Linear(hidden_dim, future_steps * latent_dim)

    def forward(self, context_latents: torch.Tensor) -> torch.Tensor:
        if context_latents.ndim != 3:
            raise ValueError(f"Expected context_latents with shape (B, C, D), got {tuple(context_latents.shape)}")
        x = self.input_proj(self.input_norm(context_latents)).transpose(1, 2)
        for conv, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            padding = (conv.kernel_size[0] - 1) * conv.dilation[0]
            x_pad = F.pad(x, (padding, 0))
            x = conv(x_pad)
            x = F.gelu(x)
            x = dropout(x.transpose(1, 2)).transpose(1, 2)
            x = norm(x.transpose(1, 2)).transpose(1, 2)
        state = x[:, :, -1]
        future = self.output_head(state)
        return future.view(context_latents.shape[0], self.future_steps, self.latent_dim)


class GRUPredictor(nn.Module):
    """A recurrent baseline for temporal latent forecasting."""

    def __init__(
        self,
        latent_dim: int,
        context_steps: int,
        future_steps: int,
        hidden_dim: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del num_heads
        self.latent_dim = latent_dim
        self.context_steps = context_steps
        self.future_steps = future_steps
        self.input_norm = nn.LayerNorm(latent_dim)
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.context_dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=max(1, num_layers), batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.step_embed = nn.Parameter(torch.zeros(1, future_steps, hidden_dim))
        self.decoder_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, latent_dim)
        nn.init.trunc_normal_(self.step_embed, std=0.02)

    def forward(self, context_latents: torch.Tensor) -> torch.Tensor:
        if context_latents.ndim != 3:
            raise ValueError(f"Expected context_latents with shape (B, C, D), got {tuple(context_latents.shape)}")
        x = self.context_dropout(self.input_proj(self.input_norm(context_latents)))
        _, state = self.encoder(x)
        hidden = state[-1]
        outputs: list[torch.Tensor] = []
        for step in range(self.future_steps):
            step_token = self.step_embed[:, step, :].expand(context_latents.shape[0], -1)
            hidden = self.decoder_cell(step_token, hidden)
            outputs.append(self.output_head(hidden).unsqueeze(1))
        return torch.cat(outputs, dim=1)


@dataclass(slots=True)
class EncoderSpec:
    name: str
    latent_dim: int = 192
    tubelet_size: int = 2
    pretrained: bool = False
    variant: str = "t"


@dataclass(slots=True)
class PredictorSpec:
    name: str
    latent_dim: int = 192
    context_steps: int = 8
    future_steps: int = 4
    hidden_dim: int = 192
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1


ENCODER_REGISTRY = {
    "swin": Swin3DTubeletEncoder,
    "timesformer": TimeSformerLikeEncoder,
    "vivit": _TubeletTransformerEncoder,
    "hybrid": Hybrid3DCNNTransformerEncoder,
}

PREDICTOR_REGISTRY = {
    "one_lag_mlp": OneLagMLPPredictor,
    "causal_transformer": CausalTransformerPredictor,
    "cross_attention": CrossAttentionPredictor,
    "gru": GRUPredictor,
    "tcn": TemporalConvPredictor,
    "mamba": MambaStylePredictor,
}


def build_video_encoder(spec: EncoderSpec) -> BaseTubeletEncoder:
    key = spec.name.strip().lower()
    if key not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{spec.name}'. Available: {sorted(ENCODER_REGISTRY)}")
    if key == "swin":
        return ENCODER_REGISTRY[key](latent_dim=spec.latent_dim, tubelet_size=spec.tubelet_size, pretrained=spec.pretrained, variant=spec.variant)
    if key == "timesformer":
        return ENCODER_REGISTRY[key](latent_dim=spec.latent_dim, tubelet_size=spec.tubelet_size)
    if key == "vivit":
        return ENCODER_REGISTRY[key](latent_dim=spec.latent_dim, tubelet_size=spec.tubelet_size)
    if key == "hybrid":
        return ENCODER_REGISTRY[key](latent_dim=spec.latent_dim, tubelet_size=spec.tubelet_size)
    raise AssertionError("unreachable")


def build_temporal_predictor(spec: PredictorSpec) -> nn.Module:
    key = spec.name.strip().lower()
    if key not in PREDICTOR_REGISTRY:
        raise ValueError(f"Unknown predictor '{spec.name}'. Available: {sorted(PREDICTOR_REGISTRY)}")
    return PREDICTOR_REGISTRY[key](
        latent_dim=spec.latent_dim,
        context_steps=spec.context_steps,
        future_steps=spec.future_steps,
        hidden_dim=spec.hidden_dim,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        dropout=spec.dropout,
    )
