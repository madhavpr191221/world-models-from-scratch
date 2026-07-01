# Scripts

The top-level entrypoints stay small:

- `run_training.py`
- `run_video_world_model.py`
- `video/run_video_latent_cache.py`

Everything else is grouped by purpose:

- `scripts/dev/` for local debugging and profiling helpers
- `scripts/data/` for data download and collection helpers
- `scripts/video/` for video pretraining, reconstruction, demos, and probes
- `scripts/probing/` for retrieval and probing entrypoints

## Video World Model Outputs

`run_video_world_model.py` writes its artifacts under `--output-dir`.
Typical outputs are:

- `result.json`: full training summary and scalar metrics
- `metrics.json`: canonical metrics report used by downstream tooling
- `predictions.csv`: per-sample evaluation rows for the held-out split
- `rollout_validation.json`: rollout decomposition, alignment, and gradient checks
- `latent_sequence_bank_<fingerprint>.json`: cache provenance written next to the latent bank
- `plots/training_steps.png`: per-batch loss components
- `plots/training_history.png`: epoch-level combined loss curves
- `plots/training_components.png`: epoch-level MSE, normalized MSE, and cosine loss
- `plots/metric_comparison.png`: learned model versus repeat-last and mean-context
- `plots/rollout_validation.png`: horizon-wise rollout validation curves
- `plots/rollout_spectrum.png`: singular-spectrum diagnostics by horizon
- `profile/profile_summary.json`: profiler summary when `--profile` is enabled
- `profile/video_world_model_trace.json`: Chrome trace when `--profile` is enabled

Use `--skip-plots` or `--skip-rollout-validation` to suppress the corresponding post-processing artifacts.
Use `--predictor-mode one-lag` to run the single-previous-latent ablation.
