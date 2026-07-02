# Roadmap

## Phase 0: Constitution

- Mission
- Tech stack
- Roadmap
- SpecDD workflow rules

## Phase 1: Stable Latent Encoding

- Finalize the encoder interface.
- Confirm tubelet extraction and latent cache format.
- Verify that cached latents are reproducible and well documented.

## Phase 2: Baseline Temporal Prediction

- Implement and compare simple predictors.
- Establish repeat-last and mean-context baselines.
- Keep the single-lag ablation available as a sanity check.

## Phase 3: Strong Temporal Predictors

- Compare stronger sequence models.
- Sweep context length.
- Sweep forecast horizon.
- Evaluate the predictor/encoder cross product.

## Phase 4: Rollout Analysis

- Teacher-forced versus rollout error decomposition.
- Alignment analysis.
- Gradient-flow inspection.
- Singular-spectrum diagnostics.

## Phase 5: Reporting and Serving

- Generate plots automatically during training.
- Save metrics, traces, and reports in the run folder.
- Provide browser-based inspection or serving scripts for visual review.

## Phase 6: Replanning

- Update the constitution when the project changes direction.
- Update feature specs when the roadmap changes.
- Keep the workflow stable even when the implementation changes.

## Phase 7: Rollout Objectives

- Compare direct latent regression against delta prediction.
- Add rollout-aware objectives and horizon weighting.
- Preserve the pluggable encoder and predictor interfaces.
- Treat saved results, plots, and validation artifacts as part of the feature.
- Sweep longer rollout horizons so drift, alignment, and error growth can be studied as a function of `r`.
- Keep multi-scale supervision available as the baseline long-horizon objective.

