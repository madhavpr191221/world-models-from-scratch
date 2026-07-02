# Requirements: Longer Rollout Horizons

## 1. Goal

The goal is to measure and improve latent-rollout stability as the prediction horizon grows.

This feature must answer:

1. How does teacher-forced error compare to free-rollout error as `r` increases?
2. How quickly does drift grow with horizon?
3. Does multi-scale supervision reduce long-horizon collapse?
4. Does a rollout-aware penalty improve the relationship between theory and practice?
5. Which horizon range is the model actually reliable over?

## 2. Non-Goals

This feature does not primarily target:

- changing the video encoder architecture,
- replacing the latent representation pipeline,
- pixel-space decoding quality,
- browser UI changes,
- broad architecture sweeps unrelated to horizon behavior.

## 3. Operating Assumptions

- The encoder remains pluggable and can be frozen for the main experiments.
- The predictor remains pluggable.
- The latent trajectory is the main object of study.
- Baseline metrics and rollout diagnostics must be saved for every run.
- The feature must support both evaluation-only sweeps and training runs.

## 4. Mathematical Contract

Let a video clip be encoded into a latent trajectory:

$$
\mathbf{z}_{1:T} = (\mathbf{z}_1,\mathbf{z}_2,\ldots,\mathbf{z}_T), \qquad \mathbf{z}_t \in \mathbb{R}^d.
$$

Let a context window of length `C` be given by:

$$
\mathbf{z}_{t-C+1:t} = (\mathbf{z}_{t-C+1}, \ldots, \mathbf{z}_t).
$$

The predictor maps context latents to future latents:

$$
P_\theta(\mathbf{z}_{t-C+1:t}) \mapsto \hat{\mathbf{z}}_{t+1:t+R}.
$$

Here `R` is the rollout horizon under study.

### 4.1 Teacher-forced and rollout predictions

Teacher-forced prediction uses the ground-truth latent history at each step:

$$
\hat{\mathbf{z}}_{t+r}^{\mathrm{TF}} = P_\theta(\mathbf{z}_{t-C+1:t+r-1}).
$$

Free rollout feeds predictions back into the predictor:

$$
\hat{\mathbf{z}}_{t+r}^{\mathrm{RO}} = P_\theta(\mathbf{z}_{t-C+1:t}, \hat{\mathbf{z}}_{t+1}^{\mathrm{RO}}, \ldots, \hat{\mathbf{z}}_{t+r-1}^{\mathrm{RO}}).
$$

Define per-horizon errors:

$$
\boldsymbol{\varepsilon}_r^{\mathrm{TF}} = \mathbf{z}_{t+r} - \hat{\mathbf{z}}_{t+r}^{\mathrm{TF}},
\qquad
\boldsymbol{\varepsilon}_r^{\mathrm{RO}} = \mathbf{z}_{t+r} - \hat{\mathbf{z}}_{t+r}^{\mathrm{RO}}.
$$

The rollout drift is:

$$
\mathbf{d}_r = \hat{\mathbf{z}}_{t+r}^{\mathrm{TF}} - \hat{\mathbf{z}}_{t+r}^{\mathrm{RO}}.
$$

The exact decomposition is:

$$
\boldsymbol{\varepsilon}_r^{\mathrm{RO}} = \boldsymbol{\varepsilon}_r^{\mathrm{TF}} + \mathbf{d}_r.
$$

### 4.2 Core losses

For a predicted latent $\hat{\mathbf{z}}$ and target latent $\mathbf{z}$:

$$
\ell_{\mathrm{mse}}(\hat{\mathbf{z}}, \mathbf{z}) = \frac{1}{d}\sum_{i=1}^{d}(\hat{z}_i - z_i)^2.
$$

Normalized MSE:

$$
\tilde{\mathbf{z}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2 + \epsilon},
\qquad
\tilde{\hat{\mathbf{z}}} = \frac{\hat{\mathbf{z}}}{\|\hat{\mathbf{z}}\|_2 + \epsilon},
$$

$$
\ell_{\mathrm{norm}}(\hat{\mathbf{z}}, \mathbf{z}) = \frac{1}{d}\sum_{i=1}^{d}(\tilde{\hat{z}}_i - \tilde{z}_i)^2.
$$

Cosine loss:

$$
\ell_{\cos}(\hat{\mathbf{z}}, \mathbf{z}) = 1 - \frac{\hat{\mathbf{z}}^\top \mathbf{z}}{\|\hat{\mathbf{z}}\|_2 \|\mathbf{z}\|_2 + \epsilon}.
$$

### 4.3 Horizon-weighted multi-scale supervision

Let the horizon set be:

$$
\mathcal{H} = \{r_1, r_2, \ldots, r_k\}
$$

such as `\{1, 2, 4, 8, 16\}`.

The generic multi-scale loss is:

$$
\mathcal{L}_{\mathrm{multi}} = \sum_{r \in \mathcal{H}} \lambda_r \, \ell(\hat{\mathbf{z}}_{t+r}, \mathbf{z}_{t+r}),
$$

where `\ell` may be one of:

- `mse`,
- `normalized_mse`,
- `cosine`,
- a balanced mixture of the above.

If rollout weighting is enabled, then:

$$
\lambda_r = \frac{\gamma^{r-1}}{\sum_{j \in \mathcal{H}} \gamma^{j-1}}.
$$

The key idea is:

- short horizons protect local fit,
- long horizons force stability farther out,
- larger `r` should not be ignored by the objective.

### 4.4 Drift-aware regularization

If enabled, the training objective may also penalize drift directly:

$$
\mathcal{L}_{\mathrm{drift}} = \sum_{r \in \mathcal{H}} \|\mathbf{d}_r\|_2^2.
$$

An optional alignment penalty can be added using the cosine between teacher-forced error and drift:

$$
\mathcal{L}_{\mathrm{align}} = \sum_{r \in \mathcal{H}} \left(1 - \cos\angle(\boldsymbol{\varepsilon}_r^{\mathrm{TF}}, \mathbf{d}_r)\right).
$$

These are not mandatory for the first diagnostic sweep, but they are part of the planned training extension.

### 4.5 Total objective

The default training objective for this feature is a configurable mixture:

$$
\mathcal{L}
=
\alpha \mathcal{L}_{\mathrm{multi}}
+
\beta \mathcal{L}_{\mathrm{drift}}
+
\delta \mathcal{L}_{\mathrm{align}},
$$

with all weights configurable and defaulting to a conservative setup.

If a delta prediction mode is used, the same formulas apply to residuals:

$$
\Delta \mathbf{z}_{t+r} = \mathbf{z}_{t+r} - \mathbf{z}_t,
\qquad
\Delta \hat{\mathbf{z}}_{t+r} = \hat{\mathbf{z}}_{t+r} - \mathbf{z}_t.
$$

This allows the model to focus on motion rather than raw latent offsets.

## 5. Expected Behavior

The feature should expose whether:

- rollout error grows smoothly or sharply,
- teacher-forced fit is misleadingly optimistic,
- multi-scale supervision widens or narrows the gap over a fixed baseline,
- drift-aware terms reduce long-horizon divergence,
- the latent predictor remains useful at higher `r`.

## 6. Artifact Contract

Each run should save:

- `result.json`
- `rollout_validation.json`
- `metrics.json`
- `predictions.csv`
- `plots/`
- optional `profile/`
- epoch checkpoints

The report should include per-horizon curves for:

- teacher-forced MSE,
- rollout MSE,
- drift norm,
- alignment cosine,
- local amplification proxy,
- gradient norm contribution if available.

## 7. Success Criteria

This feature is successful if:

1. horizons can be increased without breaking the pipeline,
2. rollout diagnostics are generated for each horizon,
3. multi-scale loss can be toggled on and off,
4. drift-aware terms can be measured and compared,
5. the results explain where the model becomes unreliable.

