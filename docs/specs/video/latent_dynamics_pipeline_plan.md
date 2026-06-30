# Latent Video Dynamics Pipeline Implementation Plan

## Summary

Build the latent video world-model stack as one pipeline:

1. a tubelet-based video encoder that produces latent vectors `z_t`
2. a temporal model that predicts future latent dynamics from observed context
3. an error-analysis and visualization layer that makes rollout failure visible

This plan follows the canonical spec at `docs/specs/video/latent_dynamics_pipeline_spec.md` and keeps the browser projection view as the main inspection tool.

## Implementation Changes

### 1. Encoder and latent export

- Keep the encoder tubelet-based so video is represented as a sequence of latent states rather than a pixel reconstruction target.
- Make clip length, sampling FPS, and latent cache shape explicit in the training and serving paths.
- Standardize the latent export contract so every clip yields:
  - context latent sequence
  - future latent sequence
  - derived total frame count
  - latent dimensionality and sequence length metadata
- Preserve cached latent banks so temporal training does not re-encode clips every epoch.

### 2. Temporal forecasting

- Train the latent dynamics model on cached latent sequences.
- Support configurable rollout horizon in seconds, converted to frame counts from `sample_fps`.
- Keep the predictor architecture small enough to train iteratively but expressive enough to model sequence dynamics.
- Compare learned predictions against repeat-last and mean-context baselines on the same metrics.

### 3. Rollout loss and diagnostics

- Report horizon-wise latent MSE, normalized latent MSE, and cosine alignment.
- Track train, validation, and test metrics separately.
- Add rollout curves so long-horizon drift is visible instead of hidden inside one aggregate loss.
- Record checkpoint paths, prediction CSVs, and result JSON in the run output directory.

### 4. Browser inspection

- Keep the latent projection browser wired to the same latent bank and forecasting outputs.
- Allow PCA and t-SNE projection methods.
- Make the user flow explicit:
  - load clip
  - choose projection method
  - inspect observed context
  - compare true future vs predicted future in latent space
- Preserve uploaded-video support and dataset-backed clip selection.

### 5. Training-time profiling

- Preserve the `--profile` path on training runs.
- Export a profiler trace and a JSON summary that records the main timing and memory information.
- Ensure profiling remains optional and does not change model behavior.

## Test Plan

- Run a small sanity training job on a small subset and verify:
  - latent caches are created
  - training completes end-to-end
  - checkpoint, metrics JSON, and predictions CSV are written
- Verify the learned model is evaluated against repeat-last and mean-context.
- Run with `--profile` and confirm the trace and profile summary are generated.
- Launch the latent projection server and verify the browser can:
  - load a dataset example
  - load an uploaded local clip
  - switch between PCA and t-SNE
  - render context, true future, and predicted future trajectories
- Check that the rollout metrics are stable across the same subset when rerun with the same seed.

## Assumptions

- Tubelet encoding is the correct representation boundary for the video encoder.
- Pixel reconstruction is out of scope for this workstream.
- The primary datasets remain Something-Something V2 and Kinetics-400.
- The current implementation already has the right entrypoints; this plan is about organizing and extending them rather than inventing a new pipeline.
- A good run is defined by both rollout quality and interpretability, not just low loss.
