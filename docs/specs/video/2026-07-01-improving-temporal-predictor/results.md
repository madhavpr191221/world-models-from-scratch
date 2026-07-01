# Results

## Current Status

This feature is focused on learning latent video dynamics from tubelet-encoded clips, predicting future latent vectors, and validating the rollout theory with error decomposition, drift, alignment, and spectral analysis.

Current conclusion:
- the pipeline is working end-to-end
- the predictor is learning nontrivial temporal structure
- the learned models still do **not** beat the trivial baselines
- the main bottleneck is still the temporal predictor/objective, not the encoder

## Experiment Setup

### Encoder / Predictor Contract

The feature is built around a pluggable interface:
- encoder: maps video clips to a latent sequence `z_{1:T}`
- temporal predictor: maps a context window of latent vectors to future latent vectors

The current implementation supports:
- encoders: `swin`, `timesformer`, `vivit`, `hybrid`
- predictors: `one_lag_mlp`, `causal_transformer`, `gru`, `tcn`, `mamba`, `cross_attention`

The configurable lag/window contract is:

`hat{z}_{t+1:t+F} = f_theta(z_{t-L+1:t})`

where:
- `L` is `context_lag_steps`
- `F` is the future horizon in latent steps
- `f_theta` is the temporal predictor

### Dataset

For the sweep summarized here:
- dataset root: `data`
- physical split layout:
  - `data/data_videos/train`
  - `data/data_videos/validation`
  - `data/data_videos/test`
- source split used for training: `train`
- validation split: `validation`
- test split: `test`
- subset size per run: `256` videos

Clip geometry:
- `context_seconds = 4.0`
- `future_seconds = 2.0`
- `sample_fps = 4.0`
- derived context frames = `16`
- derived future frames = `8`
- total frames per clip = `24`

### Training Configuration

Common settings:
- epochs: `20`
- batch size: `16`
- latent extraction feature batch size: `1`
- num workers: `2`
- learning rate: default run setting from the CLI
- frozen encoder: yes
- GPU used: yes
- latent dimension: `192`
- predictor hidden width: `192`
- predictor depth: `4`
- attention heads: `4`
- dropout: `0.1`

The training loop saves:
- per-epoch checkpoints in `checkpoints/`
- final checkpoint as `decoder_<predictor>.pt`
- metrics as `result.json`
- rollout validation as `rollout_validation.json`
- plots under `plots/`

## Completed Runs

All completed runs used the `swin` encoder and a 256-video subset.

| Run | Lag `L` | Train Loss | Val Loss | Test Loss | Test Verdict |
|---|---:|---:|---:|---:|---|
| `swin-one_lag_mlp-lag1` | 1 | 0.007084 | 0.011930 | 0.011771 | did_not_beat_baselines |
| `swin-one_lag_mlp-lag2` | 2 | 0.006874 | 0.011894 | 0.011667 | did_not_beat_baselines |
| `swin-one_lag_mlp-lag4` | 4 | 0.006874 | 0.011894 | 0.011667 | did_not_beat_baselines |
| `swin-causal_transformer-lag1` | 1 | 0.008204 | 0.011743 | 0.011750 | did_not_beat_baselines |
| `swin-causal_transformer-lag2` | 2 | 0.006122 | 0.011444 | 0.011351 | did_not_beat_baselines |
| `swin-causal_transformer-lag4` | 4 | 0.005735 | 0.011140 | 0.011005 | did_not_beat_baselines |
| `swin-mamba-lag1` | 1 | 0.003461 | 0.011127 | 0.011048 | did_not_beat_baselines |

## Metric Interpretation

### What the losses mean

The training objective combines:
- latent MSE
- normalized latent MSE
- cosine loss

In the logged summaries:
- `train_loss`, `val_loss`, `test_loss` are the combined objective values
- `latent_mse` is the raw squared error in latent space
- `normalized_latent_mse` rescales error by latent magnitude
- `cosine_similarity` measures directional agreement between predicted and target latents

For the reported runs:
- the models improve steadily during training
- validation is usually close to training
- there is no obvious classic overfitting signal
- the models are still weaker than the trivial baselines

### Baseline verdict

Each completed run currently reports:
- `did_not_beat_baselines`

And on the test split:
- `latent_mse_better = false`
- `normalized_latent_mse_better = false`
- `cosine_similarity_better = false`

So the learned predictors have **not yet beaten**:
- `repeat_last`
- `mean_context`

## Rollout Theory Diagnostics

The rollout validation pass confirms the decomposition and norm identities numerically, which means the analytical bookkeeping is correct.

Representative observations from the validation reports:
- decomposition error is near floating-point tolerance, around `1e-8`
- norm identity error is near floating-point tolerance, around `1e-6`
- horizon-wise drift increases with rollout horizon
- alignment cosine changes sign across horizons depending on the predictor
- singular spectrum energy shifts upward under rollout

Example snapshot from `swin-causal_transformer-lag4`:

| Horizon | Teacher-forced MSE | Rollout MSE | Drift Norm | Mean Alignment Cosine | Local Lipschitz Ratio Mean |
|---|---:|---:|---:|---:|---:|
| 1 | 0.057520 | 0.057520 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 0.059078 | 0.067618 | 0.970758 | 0.125130 | 0.291993 |
| 3 | 0.054220 | 0.072804 | 1.601028 | 0.084538 | 0.327011 |
| 4 | 0.054009 | 0.079539 | 2.062140 | 0.046956 | 0.334181 |

Interpretation:
- teacher forcing is stable at short horizons
- free rollout degrades as horizon grows
- the drift term is nontrivial and grows with horizon
- the rollout path is not fully aligned with the teacher-forced direction, so error propagates instead of self-correcting

That means:
- the rollout math is consistent
- the model dynamics are still unstable / misaligned over longer horizons

## Artifact Locations

For each completed run:
- final checkpoint: `logs/video_latent_dynamics_sweep_2026_07_01_laggrid_small/<run>/decoder_<predictor>.pt`
- per-epoch checkpoints: `logs/video_latent_dynamics_sweep_2026_07_01_laggrid_small/<run>/checkpoints/`
- metrics: `logs/video_latent_dynamics_sweep_2026_07_01_laggrid_small/<run>/result.json`
- rollout validation: `logs/video_latent_dynamics_sweep_2026_07_01_laggrid_small/<run>/rollout_validation.json`
- plots:
  - `training_steps.png`
  - `training_history.png`
  - `training_components.png`
  - `metric_comparison.png`
  - `rollout_validation.png`
  - `rollout_spectrum.png`

## What This Means

The current evidence says:
- the encoder is not the blocker
- the temporal predictor is still the blocker
- small lag changes help a little, but not enough
- a stronger rollout-aware objective is likely needed
- the current best loss values still do not beat the trivial baselines

The next useful experiments are:
1. delta prediction in latent space
2. multi-horizon rollout loss
3. longer context sweeps
4. stronger predictor families with the same encoder
5. compare `lag=1,2,4,8` once the current sweep finishes

## Reproducibility

The sweep was launched from the CLI with:

```powershell
uv run python scripts/video/run_video_latent_dynamics_sweep.py `
  --data-root data `
  --source-split train `
  --validation-source-split validation `
  --test-source-split test `
  --subset-size 256 `
  --context-seconds 4.0 `
  --future-seconds 2.0 `
  --sample-fps 4.0 `
  --feature-batch-size 1 `
  --num-workers 2 `
  --batch-size 16 `
  --epochs 20 `
  --output-root logs/video_latent_dynamics_sweep_2026_07_01_laggrid_small `
  --cache-root logs/video_latent_dynamics_sweep_2026_07_01_cache `
  --encoders swin `
  --predictors one_lag_mlp causal_transformer mamba cross_attention `
  --context-lag-grid 1 2 4 `
  --encoder-pretrained
```
