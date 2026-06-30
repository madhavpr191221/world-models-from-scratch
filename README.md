# JEPA World Models

This repository is a spec-driven research workspace for latent video world models.

The core question is simple:

- given a short observed video context, can we predict future latent dynamics
- and can we understand how that prediction fails when we roll it forward?

That makes the repo about three things at once:

- video prediction in latent space
- rollout error analysis and failure analysis
- browser-based inspection of the learned dynamics

The longer-term goal is a JEPA-style world model that is useful for future prediction, not just reconstruction.

## Current Thesis

The project is centered on the idea that a useful world model should:

- encode video into a latent trajectory
- predict the future in that latent space
- remain stable when rolled out autoregressively
- beat trivial baselines such as repeat-last and mean-context
- expose its behavior through a visual frontend and explicit error analysis

The emphasis is on understanding the model, not just training it.

## What Lives Here

### Latent video modeling

- frozen video encoder paths
- temporal predictors over latent sequences
- multi-step future prediction
- rollout-based evaluation
- latent-space baselines and metric reports

### Error analysis

- teacher-forced vs rollout error
- input sensitivity and compounding error analysis
- per-horizon rollout error curves
- mathematical notes on stability and sensitivity

### Browser inspection

- a local latent projection browser
- dataset clip loading
- local clip upload support
- PCA and t-SNE projection views
- context, true future, and predicted future trajectories

### Spec-driven development

- canonical specs under `docs/specs/`
- detailed execution plans for major experiments
- explicit acceptance criteria and test plans
- theory notes that connect the math to the implementation

## Repo Structure

- `src/jepa_world_models/analysis/`: latent analysis, video world model logic, rollout inspection
- `scripts/video/`: training, serving, and analysis entrypoints for video experiments
- `frontend/`: local browser UI for latent projection and inspection
- `docs/specs/`: canonical specs and experiment plans
- `docs/video/`: video-oriented notes, plans, and theory documents
- `logs/`: checkpoints, metrics, predictions, profiler traces, and browser artifacts

## What The Repository Is For

This repo is not trying to be a generic ML sandbox.
It is meant to show a disciplined research workflow for video world models:

1. write a spec
2. implement the experiment
3. train on video data
4. analyze teacher-forced and rollout behavior
5. inspect the latent dynamics in the browser
6. iterate based on failure modes

## Main Datasets

The current latent-dynamics work is aimed at large-scale video corpora such as:

- Something-Something V2
- Kinetics-400

Those datasets give two useful signals:

- motion-heavy, action-conditioned dynamics
- broad semantic and scene diversity

## Key Documents

- [Latent dynamics pipeline spec](docs/specs/video/latent_dynamics_pipeline_spec.md)
- [Latent dynamics pipeline plan](docs/specs/video/latent_dynamics_pipeline_plan.md)
- [Spec guide](docs/specs/README.md)
- [Spec template](docs/specs/template.md)
- [Rollout error analysis note](docs/video/rollout_error_analysis_latent_dynamics_markdown_edited.md)

## Useful Commands

### Small sanity retrain

```powershell
uv run python scripts/run_video_world_model.py `
  --checkpoint logs/videomae_large/best_videomae.pt `
  --data-root data `
  --source-split train `
  --subset-size 256 `
  --context-seconds 4.0 `
  --future-seconds 2.0 `
  --sample-fps 4.0 `
  --feature-batch-size 1 `
  --batch-size 8 `
  --epochs 3 `
  --output-dir logs/video_world_model_sanity `
  --cache-dir logs/video_world_model/cache `
  --profile
```

### Medium retrain

```powershell
uv run python scripts/run_video_world_model.py `
  --checkpoint logs/videomae_large/best_videomae.pt `
  --data-root data `
  --source-split train `
  --subset-size 2000 `
  --context-seconds 4.0 `
  --future-seconds 2.0 `
  --sample-fps 4.0 `
  --feature-batch-size 1 `
  --batch-size 8 `
  --epochs 10 `
  --output-dir logs/video_world_model_medium `
  --cache-dir logs/video_world_model/cache `
  --profile
```

### Latent projection browser

```powershell
uv run python scripts/video/serve_video_latent_projection.py `
  --world-model-checkpoint logs/video_world_model_medium/latent_world_model_best_videomae_1400170f03_train_2000_224_24_16_8.pt `
  --data-root data `
  --source-split train `
  --subset-size 2000 `
  --context-seconds 4.0 `
  --future-seconds 2.0 `
  --sample-fps 4.0 `
  --feature-batch-size 1 `
  --projection-method pca `
  --host 127.0.0.1 `
  --port 8002
```

## How To Read The Results

When a run finishes, look at:

- `train_loss`, `val_loss`, and `test_loss`
- latent MSE and normalized latent MSE
- cosine similarity
- repeat-last and mean-context baselines
- rollout curves and per-horizon behavior
- the browser visualization of context vs true future vs predicted future

A good run is not just low loss.
It should also show that the model is learning more than a trivial continuation of the past.

## Why This Repo Is Niche

The niche is not just video prediction.
It is:

- latent video world models
- explicit rollout error analysis
- spec-driven experimentation
- browser-hosted inspection
- mathematically grounded failure analysis

That combination makes the project more useful than a pile of checkpoints or notebooks.

## Current Direction

The immediate work is to improve latent dynamics prediction under rollout.
The research path is:

- train a stronger latent predictor
- analyze teacher-forced vs rollout error
- scale the training data
- keep the browser as the inspection layer
- use the results to guide the next model change

That is the project story.



