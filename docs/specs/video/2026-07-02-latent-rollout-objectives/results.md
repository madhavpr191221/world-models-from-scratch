# Results: Latent Rollout Objectives

This report records the completed latent rollout objective run on the current feature branch.

## 1. Run Summary

- Encoder checkpoint: `logs/encoder_checkpoints/encoder_swin_pretrained.pt`
- Encoder family: `swin`
- Predictor family: `causal_transformer`
- Objective: `rollout_balanced`
- Rollout decay: `0.95`
- Context duration: `4.0 s`
- Future duration: `2.0 s`
- Sample FPS: `4.0`
- Derived context frames: `16`
- Derived future frames: `8`
- Total frames per clip: `24`
- Context lag steps: `8`
- Dataset split layout: `train` / `validation` / `test`
- Train subset size: `50000`
- Validation subset size: `50000` sampled from the validation split
- Test subset size: `5000` sampled from the test split
- Encoder latent dimension: `192`
- Model context latent window: `8`
- Run output directory: `logs/video_world_model_swin_mamba_50k_rollout`

Important implementation note:

- The final completed run used the default predictor name `causal_transformer` because `--predictor-name` was not overridden on the launch command.
- The objective and diagnostics were still rollout-aware, and the run completed successfully.

## 2. What Was Trained

The training pipeline learns a temporal predictor over frozen latent sequences.

At a high level:

1. video clips are converted into latent sequences,
2. the predictor sees a context window of latent states,
3. it predicts the future latent trajectory,
4. the objective penalizes direct fit and rollout drift,
5. evaluation compares the learned predictor against trivial baselines.

The latent shape recorded in the artifact is:

$$
\text{latent shape} = (C, F, d) = (8, 4, 192),
$$

where:

- `C = 8` context latent steps,
- `F = 4` future latent steps in the internal reporting path,
- `d = 192` latent width.

## 3. Objective Used

The training objective was `rollout_balanced`, meaning the model was trained with a horizon-weighted multi-step rollout loss.

In compact form:

$$
\mathcal{L}_{\mathrm{rollout}}
=
\sum_{r=1}^{F}
w_r
\Big(
\ell_{\mathrm{norm}}(\hat{\mathbf{z}}_{r}, \mathbf{z}_{r})
+ 0.1\,\ell_{\mathrm{mse}}(\hat{\mathbf{z}}_{r}, \mathbf{z}_{r})
+ 0.1\,\ell_{\cos}(\hat{\mathbf{z}}_{r}, \mathbf{z}_{r})
\Big),
\qquad
w_r \propto \gamma^{r-1},
$$

with `\gamma = 0.95`.

Interpretation:

- the later rollout steps are slightly discounted,
- the predictor is not trained to just copy the most recent latent,
- the loss still measures scale, direction, and normalized fit.

## 4. Final Metrics

### 4.1 Learned model

- Train loss: `0.277516`
- Validation loss: `0.271434`
- Test loss: `1.081907`

### 4.2 Latent-space metrics

Train:

- `latent_mse = 0.037641`
- `normalized_latent_mse = 0.002722`
- `cosine_similarity = 0.738699`

Validation:

- `latent_mse = 0.039227`
- `normalized_latent_mse = 0.002845`
- `cosine_similarity = 0.726835`

Test:

- `latent_mse = 0.039200`
- `normalized_latent_mse = 0.002850`
- `cosine_similarity = 0.726431`

## 5. Baseline Comparison

Two trivial baselines were used:

1. `repeat_last`
2. `mean_context`

Definitions:

- `repeat_last` predicts every future latent as the last observed context latent.
- `mean_context` predicts every future latent as the mean of the context latents.

### 5.1 Train split

| Metric | Model | Best baseline | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| latent MSE | 0.037641 | 0.040061 | 0.002419 | 6.04% |
| normalized latent MSE | 0.002722 | 0.002859 | 0.000137 | 4.80% |
| cosine similarity | 0.738699 | 0.725512 | 0.013187 | 1.82% |

### 5.2 Validation split

| Metric | Model | Best baseline | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| latent MSE | 0.039227 | 0.040219 | 0.000992 | 2.47% |
| normalized latent MSE | 0.002845 | 0.002870 | 0.000025 | 0.86% |
| cosine similarity | 0.726835 | 0.724464 | 0.002371 | 0.33% |

### 5.3 Test split

| Metric | Model | Best baseline | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| latent MSE | 0.039200 | 0.039955 | 0.000755 | 1.89% |
| normalized latent MSE | 0.002850 | 0.002855 | 0.000005 | 0.18% |
| cosine similarity | 0.726431 | 0.725948 | 0.000483 | 0.07% |

### 5.4 Verdict

The model beats the trivial baselines on all three splits, but the margin over `mean_context` is modest on validation and test.

Interpretation:

- the predictor learned more than "repeat the last latent",
- but the gain over a simple averaging baseline is still small,
- so the latent motion prior is present but not yet strong.

## 6. Rollout Diagnostics

The rollout validation artifact was saved to:

- `logs/video_world_model_swin_mamba_50k_rollout/rollout_validation.json`

Key qualitative observations from the rollout report:

- teacher-forced error stays lower than free-rollout error,
- drift grows with horizon,
- alignment is positive but not perfect,
- local rollout amplification remains below 1 on average for some horizons, which indicates partial damping rather than explosive divergence,
- decomposition checks are numerically consistent.

The validation logic confirms:

$$
\varepsilon_r^{\mathrm{RO}} = \varepsilon_r^{\mathrm{TF}} + d_r
$$

up to floating-point tolerance.

That means the rollout decomposition is internally consistent and the error growth analysis is meaningful.

## 7. Plots

### 7.1 Training steps

![Training steps](../../../../logs/video_world_model_swin_mamba_50k_rollout/plots/training_steps.png)

This shows the per-batch objective trajectory during training.

### 7.2 Training history

![Training history](../../../../logs/video_world_model_swin_mamba_50k_rollout/plots/training_history.png)

This shows epoch-level train/validation loss progression.

### 7.3 Training components

![Training components](../../../../logs/video_world_model_swin_mamba_50k_rollout/plots/training_components.png)

This separates the combined loss into objective, MSE, normalized MSE, cosine, rollout, and delta components.

### 7.4 Metric comparison

![Metric comparison](../../../../logs/video_world_model_swin_mamba_50k_rollout/plots/metric_comparison.png)

This compares the learned predictor against `repeat_last` and `mean_context` on train/val/test.

### 7.5 Rollout validation

![Rollout validation](../../../../logs/video_world_model_swin_mamba_50k_rollout/plots/rollout_validation.png)

This summarizes horizon-wise teacher-forced error, rollout error, drift, alignment, and gradient norm.

### 7.6 Rollout spectrum

![Rollout spectrum](../../../../logs/video_world_model_swin_mamba_50k_rollout/plots/rollout_spectrum.png)

This records the singular-spectrum / directional diagnostics used in the theory note.

## 8. Artifact Paths

- Metrics JSON: `logs/video_world_model_swin_mamba_50k_rollout/result.json`
- Rollout JSON: `logs/video_world_model_swin_mamba_50k_rollout/rollout_validation.json`
- Predictions CSV: `logs/video_world_model_swin_mamba_50k_rollout/predictions.csv`
- Checkpoint: `logs/video_world_model_swin_mamba_50k_rollout/decoder_causal_transformer.pt`
- Plots directory: `logs/video_world_model_swin_mamba_50k_rollout/plots`

## 9. Interpretation

The run is useful because it validates the feature end-to-end:

1. the objective is configurable,
2. the predictor is trainable on a substantial dataset,
3. the run beats trivial baselines,
4. rollout diagnostics and plots are generated automatically,
5. the saved artifacts are sufficient to reproduce the analysis.

The main limitation is not a failure of the pipeline, but the size of the improvement over the stronger baseline:

- the model is better than `repeat_last`,
- it is only slightly better than `mean_context` on held-out data,
- so there is still room to improve the predictor family, rollout objective, or context-length regime.

That is a decent result for a first full-scale rollout-aware latent dynamics run, but not a finished world model.
