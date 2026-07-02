# Validation: Longer Rollout Horizons

## 1. Validation Goal

Validate that increasing the rollout horizon exposes the expected drift behavior and that the training objectives can be compared on the same latent trajectory.

## 2. Validation Steps

### 2.1 Baseline sweep

Run a frozen encoder + frozen predictor sweep over horizons:

$$
R \in \{4, 8, 12, 16\}
$$

with a fixed context length `C`.

Check that:

- teacher-forced and rollout predictions are both produced,
- rollout error increases with horizon,
- drift norms are non-decreasing in the aggregate,
- the decomposition residual stays near zero.

### 2.2 Multi-scale training check

Train with a multi-scale horizon set:

$$
\mathcal{H} = \{1, 2, 4, 8, 16\}
$$

or a similar set chosen in the run config.

Check that:

- the run logs loss per horizon,
- later horizons receive nonzero weight,
- training history and rollout validation plots are generated,
- the model does not collapse to a trivial constant predictor.

### 2.3 Drift-aware ablation

Enable the drift penalty on a small run.

Check that:

- `\mathcal{L}_{\mathrm{drift}}` decreases,
- the free-rollout curve improves or at least does not degrade catastrophically,
- the objective still respects the exact decomposition identities.

### 2.4 Sanity checks

Verify:

- `r = 1` rollout and teacher-forced predictions match,
- the first-horizon drift is approximately zero,
- normalization does not produce NaNs,
- cosine values remain finite,
- checkpointing and artifact writing succeed for every epoch.

## 3. Pass Criteria

The feature passes validation if:

1. the horizon sweep runs successfully,
2. the metrics show how error evolves with `r`,
3. the multi-scale objective can be toggled and compared,
4. the plots and JSON outputs exist in the run folder,
5. the report can be read without needing hidden context.

## 4. Failure Modes

Common failures include:

- horizon bookkeeping mismatches,
- wrong indexing between context and future steps,
- drift computation using mismatched rollouts,
- teacher-forced and rollout paths accidentally sharing the same inputs,
- objectives that collapse to a trivial constant or repeat-last solution.

## 5. Evidence to Save

Save the following for each validation run:

- `result.json`
- `rollout_validation.json`
- `metrics.json`
- `predictions.csv`
- training plots
- rollout plots
- checkpoint files

This evidence is sufficient to judge whether longer horizons are helping or only making the task harder.

