"""
Tests for jepa_world_models.contrastive_learning.projector
"""

import torch
import pytest

from jepa_world_models.contrastive_learning.projector import Projector


class TestProjector:
    def test_output_shape(self) -> None:
        """(B, in_dim) -> (B, proj_dim)."""
        proj = Projector(in_dim=192, proj_dim=512)
        x = torch.randn(4, 192)
        out = proj(x)
        print(f"\ninput {x.shape} -> output {out.shape}")
        assert out.shape == (4, 512)

    def test_parameter_count(self) -> None:
        """Verify parameter count matches BOTE expectation (~625K).
        Layer 1: in_dim * proj_dim (no bias)
        BN 1:    proj_dim * 2 (gamma + beta)
        Layer 2: proj_dim^2 (no bias)
        BN 2:    proj_dim * 2
        Layer 3: proj_dim^2 + proj_dim (with bias)"""
        d, p = 192, 512
        proj = Projector(in_dim=d, proj_dim=p)
        n_params = sum(param.numel() for param in proj.parameters())
        expected = (
            d * p +          # layer 1 weight (no bias)
            p * 2 +          # BN 1 gamma + beta
            p * p +          # layer 2 weight (no bias)
            p * 2 +          # BN 2 gamma + beta
            p * p + p        # layer 3 weight + bias
        )
        print(f"\nparameters: {n_params:,}, expected: {expected:,}")
        assert n_params == expected

    def test_first_two_layers_have_no_bias(self) -> None:
        """bias=False on layers 1 and 2 because BatchNorm's learned shift
        (beta) already plays that role -- a bias before BN is a wasted
        parameter that gets immediately cancelled out."""
        proj = Projector(in_dim=192, proj_dim=512)
        layer1 = proj.net[0]  # Linear
        layer2 = proj.net[3]  # Linear
        layer3 = proj.net[6]  # Linear
        print(f"\nlayer1.bias: {layer1.bias}, layer2.bias: {layer2.bias}, "
              f"layer3.bias shape: {layer3.bias.shape}")
        assert layer1.bias is None
        assert layer2.bias is None
        assert layer3.bias is not None

    def test_batchnorm_not_layernorm(self) -> None:
        """Projector uses BatchNorm1d specifically -- the projector sees
        one vector per image (not a sequence), so batch statistics are
        meaningful. Verify the correct norm type is in the network."""
        proj = Projector(in_dim=192, proj_dim=512)
        assert isinstance(proj.net[1], torch.nn.BatchNorm1d)
        assert isinstance(proj.net[4], torch.nn.BatchNorm1d)
        print(f"\nnorm type: {type(proj.net[1]).__name__} -- confirmed BatchNorm1d")

    def test_output_dimension_matches_proj_dim(self) -> None:
        """The VICReg loss operates in proj_dim space -- confirm the
        output dimension matches proj_dim exactly."""
        for proj_dim in [256, 512, 1024]:
            proj = Projector(in_dim=192, proj_dim=proj_dim)
            out = proj(torch.randn(2, 192))
            print(f"\nproj_dim={proj_dim}: output shape {out.shape}")
            assert out.shape == (2, proj_dim)

    def test_gradients_flow_to_all_parameters(self) -> None:
        """All parameters must receive gradients -- the projector is
        trained jointly with the encoder."""
        proj = Projector(in_dim=192, proj_dim=512)
        x = torch.randn(4, 192, requires_grad=True)

        out = proj(x)
        out.sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        print(f"\ngrad norm wrt input: {x.grad.norm().item():.4f}")

        for name, param in proj.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert torch.isfinite(param.grad).all(), f"NaN grad for {name}"
            print(f"grad norm {name}: {param.grad.norm().item():.4f}")

    def test_train_vs_eval_mode(self) -> None:
        """BatchNorm behaves differently in train vs eval mode -- in train
        mode it uses batch statistics; in eval mode it uses running stats
        accumulated during training. This means output will differ between
        modes for the same input (expected behavior, not a bug)."""
        torch.manual_seed(0)
        proj = Projector(in_dim=192, proj_dim=512)
        x = torch.randn(8, 192)

        proj.train()
        with torch.no_grad():
            out_train = proj(x)

        proj.eval()
        with torch.no_grad():
            out_eval = proj(x)

        max_diff = (out_train - out_eval).abs().max().item()
        print(f"\ntrain vs eval max diff: {max_diff:.4f} (expected nonzero)")
        assert max_diff > 0.0

    def test_batch_size_1_fails_in_train_mode(self) -> None:
        """BatchNorm1d requires B >= 2 in train mode to compute batch
        statistics -- B=1 raises a ValueError. This is a known BatchNorm
        limitation; use eval mode or larger batches during training."""
        proj = Projector(in_dim=192, proj_dim=512)
        proj.train()
        x = torch.randn(1, 192)
        with pytest.raises(ValueError):
            proj(x)
        print("\nB=1 correctly raises ValueError in train mode")

    def test_encoder_projector_pipeline(self) -> None:
        """End-to-end: ViTEncoder -> Projector, the actual pipeline used
        during VICReg training. Input images -> loss-ready representations."""
        from jepa_world_models.contrastive_learning.encoders.vit import ViTEncoder

        encoder = ViTEncoder()
        proj = Projector(in_dim=192, proj_dim=512)

        images = torch.randn(4, 3, 96, 96)
        representations = encoder(images)    # (4, 192)
        projections = proj(representations)  # (4, 512)

        print(f"\nimages {images.shape} -> encoder {representations.shape} "
              f"-> projector {projections.shape}")
        assert representations.shape == (4, 192)
        assert projections.shape == (4, 512)