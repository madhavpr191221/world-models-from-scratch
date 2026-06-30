# Latent Video Dynamics Pipeline Spec

## 1. Purpose

Build a single video-learning pipeline with three explicit stages:

1. A video encoder that converts short tubelet sequences into latent vectors `z_t`.
2. A temporal dynamics model that predicts future latent trajectories.
3. An error-analysis and visualization layer that measures where rollout fails and how error grows over time.

The application area is video prediction, but the research target is broader: understand whether latent dynamics can become a usable world-model substrate.

## 2. Core Claim

We are not trying to copy pixels directly.

We are trying to learn:

- a representation space `z_t` that is compact, stable, and semantically meaningful
- a dynamics model that can forecast `z_{t+1}, z_{t+2}, ..., z_{t+K}`
- diagnostics that reveal whether rollout error is caused by representation collapse, dynamics drift, or long-horizon compounding error

## 3. Notation

Let a video clip be

`x = {x_1, x_2, ..., x_T}`

where each `x_t` is a frame or sampled frame at time index `t`.

Let the encoder operate on tubelets, not single frames. A tubelet is a spatiotemporal patch of shape:

`(tau, p, p)`

where:

- `tau` is the number of frames per tubelet
- `p` is the spatial patch size

The encoder outputs a latent sequence:

`z = {z_1, z_2, ..., z_N}`, with `z_t in R^d`

where `d` is the latent dimension and `N` is the number of latent tokens after tubeletification and temporal pooling/aggregation.

## 4. Video Encoder

### 4.1 Goal

The encoder maps a short clip into latent vectors that summarize motion and appearance.

Formally:

`E_theta: x_{1:T} -> z_{1:N}`

Each latent should preserve information useful for predicting future latent states.

### 4.2 Desired properties

The encoder should:

- accept configurable clip lengths in seconds
- support configurable sampling FPS
- process tubelets so the representation is explicitly spatiotemporal
- produce a latent space suitable for linear projection, nearest-neighbor inspection, and temporal prediction

### 4.3 Why tubelets

Tubelets make the encoder encode motion locally in time and space instead of learning only framewise appearance.

That matters because the downstream temporal model should predict dynamics in a space already aligned with short-horizon motion.

## 5. Temporal Dynamics Model

### 5.1 Goal

Given observed context latents, predict future latent states.

Let:

`z_{1:C}` be the context latents
`z_{C+1:C+K}` be the target future latents

The temporal model is:

`F_phi: z_{1:C} -> \hat z_{C+1:C+K}`

### 5.2 Rollout

At inference time, the model is rolled forward autoregressively or via multi-step prediction:

`hat z_{C+1} = F_phi(z_{1:C})`

`hat z_{C+2} = F_phi(z_{2:C}, hat z_{C+1})`

and so on, depending on the implementation.

The important point is that the model must survive its own prediction errors.

## 6. Training Objective

We want the temporal model to minimize a weighted sum of rollout losses.

### 6.1 Latent MSE

For one step:

`L_mse = (1/d) * ||hat z_t - z_t||_2^2`

Over a horizon:

`L_roll = (1/K) * sum_{k=1..K} (1/d) * ||hat z_{C+k} - z_{C+k}||_2^2`

### 6.2 Normalized latent MSE

To reduce the effect of scale differences across samples:

`z_t^ = z_t / (||z_t||_2 + eps)`

`hat z_t^ = hat z_t / (||hat z_t||_2 + eps)`

`L_norm = (1/K) * sum_{k=1..K} ||hat z_{C+k}^ - z_{C+k}^||_2^2`

### 6.3 Cosine alignment

Cosine similarity measures directional agreement:

`cos(a, b) = (a^T b) / (||a||_2 ||b||_2 + eps)`

Loss form:

`L_cos = (1/K) * sum_{k=1..K} (1 - cos(hat z_{C+k}, z_{C+k}))`

### 6.4 Multi-step rollout weighting

Later steps should matter, but not dominate the signal.

One reasonable weighted objective is:

`L_total = sum_{k=1..K} w_k * [alpha * L_mse(k) + beta * L_norm(k) + gamma * L_cos(k)]`

where:

- `w_k` can increase with horizon if we want to penalize long-range drift
- `alpha`, `beta`, `gamma` are tunable coefficients

### 6.5 Baseline comparison

Always compare the learned dynamics against:

- repeat-last: `hat z_{C+k} = z_C`
- mean-context: `hat z_{C+k} = mean(z_{1:C})`

If the learned model does not beat these baselines on the same metric, the temporal model is not yet doing useful work.

## 7. Error Analysis

The key research object is not just the prediction itself, but the rollout error curve.

For each horizon step `k`, compute:

`e_k = ||hat z_{C+k} - z_{C+k}||_2^2`

Then inspect:

- mean error vs horizon
- variance of error vs horizon
- cosine similarity vs horizon
- failure cases by clip category or action class
- trajectory plots in 2D after projection

Useful diagnostics:

- short-horizon accuracy
- long-horizon drift
- collapse to the mean
- oversmoothing
- mode switching errors

## 8. Projection and Visualization

The latent space should be inspectable.

We want:

- PCA projection for stable global structure
- t-SNE for local neighborhood structure
- 2D animated trajectories for context, true future, and predicted future

These visualizations are diagnostic tools, not training objectives.

## 9. Data Strategy

Use full video data once the pipeline is stable.

Primary datasets:

- Kinetics-400
- Something-Something V2

Initial experiments may use subsets for sanity checks, but the target is full-scale latent dynamics training.

## 10. Model Requirements

The implementation should satisfy:

- configurable context seconds
- configurable future seconds
- configurable sampling FPS
- configurable tubelet encoder
- configurable latent dimension `d`
- configurable temporal horizon `K`
- training-time profiling support
- checkpoint saving
- reproducible evaluation summaries

## 11. Acceptance Criteria

The system is acceptable when all of the following are true:

1. The encoder produces stable `z_t` vectors for arbitrary clip lengths within supported limits.
2. The temporal model predicts future latent trajectories from observed context.
3. The learned model beats repeat-last or mean-context on at least one meaningful rollout metric.
4. The repository can generate an error-analysis report and a 2D latent trajectory visualization.
5. The training script can run with profiling enabled when needed.

## 12. Implementation Stages

### Stage A: Encoder and latent export

- finalize tubelet encoder interface
- expose latent dimensionality and sequence length
- cache latent sequences for faster temporal training

### Stage B: Temporal forecasting

- train a dynamics model on latent sequences
- support autoregressive rollout
- compare against repeat-last and mean-context baselines

### Stage C: Error analysis

- compute horizon-wise metrics
- export plots and CSV summaries
- identify failure regimes

### Stage D: Browser visualization

- load a clip
- project the latent trajectory into 2D
- animate observed context and forecasted future
- let the user compare true vs predicted rollout

## 13. Research Direction

This repo should become a compact but serious latent world-model workspace:

- representation learning from video
- latent dynamics forecasting
- rollout error analysis
- interactive inspection in the browser

That combination is niche enough to be interesting and concrete enough to demonstrate real engineering and ML depth.