# Mission

## Why this project exists

This repository is a latent video dynamics lab.

The goal is to study and improve future-state prediction for video in latent space, with enough rigor to support rollout analysis, error decomposition, and spectral diagnostics.

## What we care about

- Encoding video into tubelet-level latent vectors.
- Predicting future latent vectors with a pluggable temporal model.
- Measuring teacher-forced error and free-rollout error.
- Studying how error grows with horizon.
- Explaining failure modes with geometry, alignment, gradient flow, and singular-spectrum analysis.
- Keeping the encoder, predictor, metrics, plots, and reports swappable and reproducible.

## What we are not optimizing for first

- Pixel-perfect video synthesis.
- Classification accuracy as the primary goal.
- Aesthetic demos without diagnostics.
- One-off experiments that cannot be reproduced or compared.

## Success Criteria

The project is successful when each run can answer:

1. What latent dynamics were learned?
2. How does rollout error change with horizon?
3. Which predictor and context settings help?
4. Does the model beat simple baselines?
5. Are the plots, metrics, and artifacts sufficient to explain the result?

