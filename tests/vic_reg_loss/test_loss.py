"""
Tests for jepa_world_models.vic_reg_loss.loss

These tests verify that the implementation matches the theoretical
predictions from the derivation (see loss.py's module docstring and the
theory thread, Week 5-6):

1. The covariance estimator is unbiased (matches torch.cov, which uses N-1).
2. Collapse (all representations equal to a constant vector) produces a
   STRICTLY POSITIVE loss, not zero -- this is the central claim of the
   theory thread: collapse is a saddle point of the full VICReg objective,
   even though it is the global minimum of the invariance term alone.
3. A well-spread, decorrelated representation scores lower than a
   collapsed one (sanity check that the loss actually prefers "good"
   representations).
4. Gradients are finite and nonzero (no silently broken backward pass).
5. invariance_loss matches the OFFICIAL VICReg paper's released pseudocode
   (Algorithm 1, Appendix A): sim_loss = mse_loss(z_a, z_b), averaged over
   ALL N*d elements -- NOT (1/N) sum_i ||z_i - z_i'||^2 (summed over
   features, then averaged over batch), which is what the paper's PROSE
   equation (Eq. 5) literally states. The prose and the pseudocode
   disagree by a factor of d (the embedding dimension). We match the
   pseudocode because that is what was actually run to produce the
   paper's results and what lambda=25, mu=25, nu=1 were calibrated
   against. Using the prose formula instead causes L_inv's gradient to be
   ~d times larger than intended relative to L_var and L_cov, which
   empirically causes the optimizer to collapse the representation rather
   than learn genuine invariances (confirmed by isolating this exact
   change against a known-working reference implementation -- see
   docs/pytorch_stuff.md training notes for the full debugging trace).
"""

import pytest
import torch

from jepa_world_models.vic_reg_loss.loss import (
    VICRegLoss,
    batch_covariance,
    covariance_loss,
    invariance_loss,
    variance_loss,
)


@pytest.fixture
def loss_fn() -> VICRegLoss:
    return VICRegLoss(lambda_=25.0, mu=25.0, nu=1.0, gamma=1.0, eps=1e-4)


@pytest.fixture
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


class TestBatchCovariance:
    def test_matches_torch_cov_unbiased(self, rng: torch.Generator) -> None:
        z = torch.randn(64, 32, generator=rng)
        s_ours = batch_covariance(z)
        s_torch = torch.cov(z.T)
        torch.testing.assert_close(s_ours, s_torch, atol=1e-5, rtol=1e-5)

    def test_raises_on_insufficient_samples(self) -> None:
        z = torch.randn(1, 8)
        with pytest.raises(ValueError):
            batch_covariance(z)

    def test_diagonal_is_per_dimension_variance(self, rng: torch.Generator) -> None:
        z = torch.randn(100, 5, generator=rng)
        s = batch_covariance(z)
        for j in range(5):
            expected_var = z[:, j].var(unbiased=True)
            torch.testing.assert_close(s[j, j], expected_var, atol=1e-5, rtol=1e-5)


class TestVarianceLoss:
    def test_zero_when_std_at_or_above_gamma(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(200, 16)
        z = z / z.std(dim=0, unbiased=True, keepdim=True)
        loss = variance_loss(z, gamma=1.0, eps=0.0)
        assert loss.item() < 1e-3

    def test_dead_dimension_gives_loss_near_gamma(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(50, 4)
        z[:, 0] = 7.0
        loss = variance_loss(z, gamma=1.0, eps=1e-4)
        assert loss.item() > 0.2


class TestCovarianceLoss:
    def test_zero_for_perfectly_decorrelated_dimensions(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(5000, 8)
        loss = covariance_loss(z)
        assert loss.item() < 0.05

    def test_positive_for_correlated_dimensions(self) -> None:
        torch.manual_seed(0)
        base = torch.randn(200, 1)
        z = torch.cat([base, base, torch.randn(200, 2)], dim=1)
        loss = covariance_loss(z)
        assert loss.item() > 0.3


class TestInvarianceLoss:
    def test_zero_for_identical_views(self, rng: torch.Generator) -> None:
        z = torch.randn(32, 16, generator=rng)
        assert invariance_loss(z, z.clone()).item() == pytest.approx(0.0, abs=1e-7)

    def test_matches_official_pseudocode_mse_convention(
        self, rng: torch.Generator
    ) -> None:
        """L_inv must equal mse_loss(z_a, z_b) -- mean squared error
        averaged over ALL N*d elements, matching the official VICReg
        paper's released pseudocode (Algorithm 1, Appendix A), NOT the
        paper's prose equation (Eq. 5), which sums over features then
        averages over the batch (an extra factor of d larger)."""
        z_a = torch.randn(10, 7, generator=rng)
        z_b = torch.randn(10, 7, generator=rng)

        expected = torch.nn.functional.mse_loss(z_a, z_b)
        actual = invariance_loss(z_a, z_b)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

        prose_convention = ((z_a - z_b) ** 2).sum(dim=1).mean()
        d = z_a.shape[1]
        torch.testing.assert_close(
            prose_convention, actual * d, atol=1e-5, rtol=1e-5
        )

    def test_raises_on_shape_mismatch(self) -> None:
        z_a = torch.randn(10, 7)
        z_b = torch.randn(10, 8)
        with pytest.raises(ValueError):
            invariance_loss(z_a, z_b)


class TestVICRegLossCollapseIsNotAMinimum:
    def test_collapse_gives_zero_invariance_but_positive_total(
        self, loss_fn: VICRegLoss
    ) -> None:
        n, d = 64, 32
        c = torch.randn(1, d).expand(n, d).clone()
        out = loss_fn(c, c)
        assert out.inv.item() == pytest.approx(0.0, abs=1e-6)
        assert out.total.item() > 20.0

    def test_collapse_loss_matches_predicted_mu_gamma(
        self, loss_fn: VICRegLoss
    ) -> None:
        n, d = 64, 32
        c = torch.randn(1, d).expand(n, d).clone()
        out = loss_fn(c, c)
        predicted = loss_fn.mu * loss_fn.gamma
        assert abs(out.total.item() - predicted) < 1.0

    def test_well_spread_representation_beats_collapse(
        self, loss_fn: VICRegLoss, rng: torch.Generator
    ) -> None:
        n, d = 64, 32
        collapsed = torch.randn(1, d, generator=rng).expand(n, d).clone()
        out_collapsed = loss_fn(collapsed, collapsed)
        well_spread = torch.randn(n, d, generator=rng) * loss_fn.gamma
        out_good = loss_fn(well_spread, well_spread.clone())
        assert out_good.total.item() < out_collapsed.total.item()


class TestVICRegLossGradients:
    def test_gradients_are_finite_and_nonzero(
        self, loss_fn: VICRegLoss, rng: torch.Generator
    ) -> None:
        z_a = torch.randn(64, 32, generator=rng, requires_grad=True)
        z_b = torch.randn(64, 32, generator=rng, requires_grad=True)
        out = loss_fn(z_a, z_b)
        out.total.backward()
        assert z_a.grad is not None and z_b.grad is not None
        assert torch.isfinite(z_a.grad).all()
        assert torch.isfinite(z_b.grad).all()
        assert z_a.grad.norm().item() > 0.0
        assert z_b.grad.norm().item() > 0.0

    def test_gradient_vanishes_at_exact_collapse_for_invariance_term_only(self) -> None:
        d = 16
        c = torch.randn(1, d).expand(8, d).clone()
        z_a = c.clone().requires_grad_(True)
        z_b = c.clone().requires_grad_(True)
        inv = invariance_loss(z_a, z_b)
        inv.backward()
        torch.testing.assert_close(
            z_a.grad, torch.zeros_like(z_a.grad), atol=1e-6, rtol=1e-6
        )


class TestVICRegLossOutputShape:
    def test_returns_scalars(self, loss_fn: VICRegLoss, rng: torch.Generator) -> None:
        z_a = torch.randn(32, 16, generator=rng)
        z_b = torch.randn(32, 16, generator=rng)
        out = loss_fn(z_a, z_b)
        for field in (out.total, out.inv, out.var, out.cov):
            assert field.ndim == 0

    def test_total_is_correctly_weighted_sum(
        self, loss_fn: VICRegLoss, rng: torch.Generator
    ) -> None:
        z_a = torch.randn(32, 16, generator=rng)
        z_b = torch.randn(32, 16, generator=rng)
        out = loss_fn(z_a, z_b)
        expected_total = (
            loss_fn.lambda_ * out.inv + loss_fn.mu * out.var + loss_fn.nu * out.cov
        )
        torch.testing.assert_close(out.total, expected_total, atol=1e-5, rtol=1e-5)