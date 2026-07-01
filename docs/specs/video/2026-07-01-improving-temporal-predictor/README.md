# Improving Temporal Predictor

This feature spec defines the next step after the latent-dynamics baseline.

## Scope

- Keep the encoder frozen for this feature.
- Improve the temporal predictor.
- Start with the one-lag ablation, then sweep `lag = 2, 4, 8, ...` style context windows.
- Then expand to stronger predictor families and rollout-aware training.
- Keep rollout analysis, baselines, plots, and reports as part of the feature deliverable.

## Files

- [Requirements](./requirements.md)
- [Plan](./plan.md)
- [Validation](./validation.md)

## Branch Rule

Implement this feature on its own branch and do not merge until validation passes.
