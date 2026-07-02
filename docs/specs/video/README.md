# Video Latent Dynamics Docs

This folder is the active source of truth for the video latent-dynamics work.

The project-level constitution lives in:

- `docs/specs/constitution/README.md`
- `docs/specs/constitution/mission.md`
- `docs/specs/constitution/tech-stack.md`
- `docs/specs/constitution/roadmap.md`

## Active Feature

- [Latent Rollout Objectives](./2026-07-02-latent-rollout-objectives/README.md)
- [Feature Spec](./2026-07-02-latent-rollout-objectives/requirements.md)
- [Feature Plan](./2026-07-02-latent-rollout-objectives/plan.md)
- [Validation](./2026-07-02-latent-rollout-objectives/validation.md)

## Previous Feature

- [Improving Temporal Predictor](./2026-07-01-improving-temporal-predictor/README.md)
- [Results](./2026-07-01-improving-temporal-predictor/results.md)

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
