"""
VICReg loss — derived from first principles.

================================================================================
DERIVATION (mirrors the theory thread, Week 6)
================================================================================

Setup
-----
For each input x_i, two augmented views are sampled:

    x_i' ~ T(x_i),   x_i'' ~ T(x_i)

and passed through an encoder + projector f_theta : R^n -> R^d:

    z_i' = f_theta(x_i'),   z_i'' = f_theta(x_i'')

Stacking N samples in a batch gives representation matrices

    Z', Z'' in R^{N x d}

This module operates directly on (Z', Z''), i.e. it does not know about the
encoder, the augmentations, or the projector. It is a pure function of two
batches of representation vectors.

Why three terms are needed
---------------------------
Recall the collapse theorem: with only an invariance objective
    L_inv = (1/N) sum_i || z_i' - z_i'' ||^2
the global minimum is f_theta(x) = c for all x (a constant vector). This is a
THEOREM about the loss landscape, not an empirical failure mode -- L_inv = 0
is achieved trivially and is the lowest possible value. VICReg prevents this
by explicitly regularizing the marginal statistics of Z' and Z'' so that
collapse is no longer the global minimum of the full objective.

--------------------------------------------------------------------------
1. The covariance matrix (computed from the batch, matching the theory
   thread's entry-by-entry derivation)
--------------------------------------------------------------------------
Let Z in R^{N x d} be a batch of representations (we'll do this for Z' and
Z'' separately). Sample mean (per dimension j):

    z_bar_j = (1/N) sum_i z_{ij}

Center the batch:

    Z_tilde = Z - 1 z_bar^T        (Z_tilde in R^{N x d}, rows are z_i - z_bar)

Sample covariance, derived entry-by-entry from
Cov(Z_j, Z_k) = E[(Z_j - mu_j)(Z_k - mu_k)], with Bessel's correction since
mu is itself estimated from the same N samples (UNBIASED estimator):

    S_{jk} = (1 / (N - 1)) * sum_i (z_{ij} - z_bar_j)(z_{ik} - z_bar_k)

In matrix form:

    S = (1 / (N - 1)) * Z_tilde^T Z_tilde            (S in R^{d x d})

--------------------------------------------------------------------------
2. The dead dimension chain (why we monitor the diagonal of S)
--------------------------------------------------------------------------
    sigma_j = 0  =>  v_j in null(Z_tilde)  =>  Z_tilde v_j = 0
             =>  <f_theta(x_i) - z_bar, v_j> = 0  for all i
             =>  Var(C_j) = lambda_j = 0  =>  C_j = E[C_j] a.s.
             =>  v_j is an uninformative direction

where C_j = <Z, v_j> is the projection of the representation onto the j-th
eigenvector of S, and lambda_j = v_j^T Sigma v_j is the population variance
along that direction.

The eigenbasis {v_j} changes every gradient step (it depends on theta), so
tracking variance in that basis would require an SVD at every step -- too
expensive, and the basis itself isn't stationary. Instead we monitor
variance in the STANDARD basis {e_j}: S_{jj} = Var(Z_j) directly, read off
the diagonal of S with no decomposition needed. This is what the variance
term penalizes.

--------------------------------------------------------------------------
3. Variance term -- no dead dimensions
--------------------------------------------------------------------------
Penalize the standard deviation along each dimension falling below a target
gamma (default gamma = 1):

    L_var(Z) = (1/d) sum_{j=1}^d max(0, gamma - sqrt(S_{jj} + eps))

epsilon is added INSIDE the square root (not after) purely for numerical
stability -- it prevents NaN gradients when S_{jj} -> 0 (sqrt is not
differentiable at exactly 0). Applied symmetrically to both views:

    L_var(Z', Z'') = (1/2) [ L_var(Z') + L_var(Z'') ]

--------------------------------------------------------------------------
4. Covariance term -- no redundant dimensions
--------------------------------------------------------------------------
Penalize off-diagonal covariance entries (encourages decorrelation across
dimensions, which is what spreads information across all d dimensions
instead of letting many dimensions encode the same thing):

    L_cov(Z) = (1/d) sum_{j != k} S_{jk}^2

Implemented as ||S||_F^2 - sum_j S_{jj}^2, divided by d.

Note (kept for documentation, matches the theory thread): the
scale-invariant correlation penalty (1/d) sum_{j!=k} S_{jk}^2 / (S_{jj}
S_{kk}) is theoretically cleaner but numerically unstable as S_{jj} -> 0.
Since L_var pins S_{jj} >= gamma^2 in the trained regime, the two penalties
become approximately proportional, so the simpler (and numerically stable)
raw-covariance form is used here, matching the original paper.

Applied symmetrically to both views:

    L_cov(Z', Z'') = L_cov(Z') + L_cov(Z'')

--------------------------------------------------------------------------
5. Invariance term -- pull views of the same input together
--------------------------------------------------------------------------
    L_inv(Z', Z'') = (1/N) sum_i || z_i' - z_i'' ||^2

This is the only term that uses BOTH Z' and Z'' jointly (rather than each
view's own statistics); it is exactly the objective that collapse trivially
minimizes, which is why it cannot be used alone.

--------------------------------------------------------------------------
6. Full VICReg loss
--------------------------------------------------------------------------
    L_VICReg = lambda * L_inv(Z', Z'') + mu * L_var(Z', Z'') + nu * L_cov(Z', Z'')

Original paper hyperparameters: lambda = 25, mu = 25, nu = 1, gamma = 1,
eps = 1e-4.

At collapse (Z' = Z'' = c for all rows): every S_{jj} = 0, so
L_var = mu * gamma > 0. The collapsed point is NOT a minimum of the full
objective -- both L_var and L_cov jointly push S towards gamma^2 * I, while
L_inv alone would have been satisfied (= 0) at that same collapsed point.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def centered(z: Tensor) -> Tensor:
    """
    Center a batch of representations along the batch dimension.

    z: (N, d) -> z_tilde: (N, d), with z_tilde_i = z_i - z_bar

    z_bar = (1/N) sum_i z_i is the sample mean, computed along dim=0 (the
    batch dimension), matching the theory thread's z_bar definition.
    """
    return z - z.mean(dim=0, keepdim=True)


def batch_covariance(z: Tensor) -> Tensor:
    """
    Unbiased sample covariance matrix of a batch of representations.

    z: (N, d) representation matrix (one view, one batch).
    Returns S: (d, d), S = (1 / (N - 1)) * z_tilde^T z_tilde.

    Uses Bessel's correction (N - 1, not N) because the mean used for
    centering is itself estimated from the same N samples -- this is the
    UNBIASED estimator of the population covariance, matching the
    derivation above exactly (S_{jk} = Cov(Z_j, Z_k) derived entry-by-entry
    from E[(Z_j - mu_j)(Z_k - mu_k)]).
    """
    n = z.shape[0]
    if n < 2:
        raise ValueError(
            f"batch_covariance requires N >= 2 samples for the unbiased "
            f"(N - 1) estimator, got N = {n}."
        )
    z_tilde = centered(z)
    return (z_tilde.T @ z_tilde) / (n - 1)


def variance_loss(z: Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    """
    L_var(Z) = (1/d) sum_j max(0, gamma - sqrt(S_jj + eps))

    Penalizes any dimension whose standard deviation falls below gamma.
    Reads S_jj directly off the diagonal of the covariance matrix -- no SVD,
    no eigendecomposition, per the "monitor the standard basis" argument in
    the dead dimension chain above.

    z: (N, d). Returns a scalar.
    """
    s = batch_covariance(z)
    variances = torch.diagonal(s)  # S_jj = Var(Z_j), shape (d,)
    std = torch.sqrt(variances + eps)
    return torch.clamp(gamma - std, min=0.0).mean()


def covariance_loss(z: Tensor) -> Tensor:
    """
    L_cov(Z) = (1/d) sum_{j != k} S_jk^2

    Penalizes off-diagonal covariance entries, encouraging different
    dimensions of the representation to be decorrelated (spreading
    information across dimensions rather than duplicating it).

    z: (N, d). Returns a scalar.
    """
    d = z.shape[1]
    s = batch_covariance(z)
    off_diagonal_sq_sum = (s ** 2).sum() - torch.diagonal(s).pow(2).sum()
    return off_diagonal_sq_sum / d


def invariance_loss(z_a: Tensor, z_b: Tensor) -> Tensor:
    """
    L_inv(Z', Z'') = (1/N) sum_i || z_i' - z_i'' ||^2

    Pulls the two augmented views of the same input together. This is the
    only term that requires a correspondence between z_a and z_b (same
    underlying image, two different augmentations) -- the variance and
    covariance terms only ever see one view's statistics at a time.

    z_a, z_b: (N, d), must contain corresponding pairs (z_a[i] and z_b[i]
    are the two views of the same input). Returns a scalar.
    """
    if z_a.shape != z_b.shape:
        raise ValueError(
            f"invariance_loss requires matching shapes, got {z_a.shape} "
            f"and {z_b.shape}."
        )
    # NOTE: torch.nn.functional.mse_loss with default reduction averages
    # over ALL elements (N * d), i.e. it would compute (1/(N*d)) sum_i sum_j
    # (.)^2 -- off by a factor of d from the derivation, which sums the
    # squared norm per-sample (sum over j) and only then averages over the
    # batch (mean over i):
    #     L_inv = (1/N) sum_i || z_i' - z_i'' ||^2
    #           = (1/N) sum_i [ sum_j (z_ij' - z_ij'')^2 ]
    # So: sum over the feature dimension first, then mean over the batch.
    squared_norms = ((z_a - z_b) ** 2).sum(dim=1)  # (N,), one ||.||^2 per sample
    return squared_norms.mean()


@dataclass
class VICRegLossOutput:
    """Container for the total loss plus its three components, for logging."""

    total: Tensor
    inv: Tensor
    var: Tensor
    cov: Tensor


class VICRegLoss(torch.nn.Module):
    """
    Full VICReg loss:

        L_VICReg = lambda * L_inv(Z', Z'')
                 + mu     * L_var(Z', Z'')
                 + nu     * L_cov(Z', Z'')

    where:
        L_var(Z', Z'') = (1/2) [ L_var(Z') + L_var(Z'') ]
        L_cov(Z', Z'') =        L_cov(Z') + L_cov(Z'')

    Default hyperparameters (lambda=25, mu=25, nu=1, gamma=1, eps=1e-4) match
    the original paper.
    """

    def __init__(
        self,
        lambda_: float = 25.0,
        mu: float = 25.0,
        nu: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu
        self.gamma = gamma
        self.eps = eps

    def forward(self, z_a: Tensor, z_b: Tensor) -> VICRegLossOutput:
        """
        z_a, z_b: (N, d) projector outputs for the two augmented views.
        Both must come from the SAME batch of N inputs, i.e. z_a[i] and
        z_b[i] are the two views of input i (the invariance term depends on
        this correspondence; the variance and covariance terms do not).
        """
        inv = invariance_loss(z_a, z_b)

        var_a = variance_loss(z_a, gamma=self.gamma, eps=self.eps)
        var_b = variance_loss(z_b, gamma=self.gamma, eps=self.eps)
        var = 0.5 * (var_a + var_b)

        cov_a = covariance_loss(z_a)
        cov_b = covariance_loss(z_b)
        cov = cov_a + cov_b

        total = self.lambda_ * inv + self.mu * var + self.nu * cov

        return VICRegLossOutput(total=total, inv=inv, var=var, cov=cov)