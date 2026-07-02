# Longer Rollout Horizons

This feature studies what happens when the rollout horizon `r` grows.

The intent is to keep the current encoder/predictor interface stable and answer a simple question:

> If we let the latent predictor run farther and farther into the future, how quickly do the errors grow, and which losses actually help?

## Focus

- Reuse the existing encoder and predictor as the baseline setup.
- Sweep rollout horizons to observe drift, alignment, and error amplification.
- Compare teacher-forced evaluation against free rollout.
- Add multi-scale and rollout-aware objectives as optional training targets.
- Keep the latent cache, plots, metrics, and validation artifacts in the same run layout.

## Files

- [Requirements](./requirements.md)
- [Plan](./plan.md)
- [Validation](./validation.md)

## Relationship to prior work

This feature builds directly on the rollout-error analysis and the earlier latent-rollout objective work. It does not replace that work; it extends it by pushing the horizon farther and by making the horizon itself a first-class experimental variable.

