# Video Latent Dynamics Docs

This folder is the active source of truth for the video latent-dynamics work.

## Canonical Docs

- [Latent Video Dynamics Spec](./latent_dynamics_pipeline_spec.md)
- [Latent Video Dynamics Implementation Plan](./latent_dynamics_pipeline_plan.md)

## Canonical Runtime Contract

The training pipeline is centered around:

- `scripts/run_video_world_model.py`
- `scripts/video/run_video_world_model_validation.py`

Primary outputs for a run:

- `result.json`
- `metrics.json`
- `predictions.csv`
- `rollout_validation.json`
- `plots/training_steps.png`
- `plots/training_history.png`
- `plots/training_components.png`
- `plots/metric_comparison.png`
- `plots/rollout_validation.png`

If a file or script is not part of this flow, it should be treated as legacy until explicitly promoted into the spec.

