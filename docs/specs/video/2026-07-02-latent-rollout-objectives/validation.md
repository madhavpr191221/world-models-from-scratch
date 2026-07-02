# Validation: Latent Rollout Objectives

## 1. What must be verified

The feature is valid only if the saved artifacts show:

1. the predictor trains with the selected objective,
2. one-lag and multi-context modes both work,
3. delta prediction works if enabled,
4. teacher-forced and rollout predictions are both computed,
5. rollout identities hold numerically,
6. plots and metrics are emitted automatically,
7. the result can be compared against trivial baselines.

## 2. Mathematical checks

### 2.1 Decomposition identity

For every horizon `r`, verify:

$$
\varepsilon_r^{\mathrm{RO}} \approx \varepsilon_r^{\mathrm{TF}} + d_r
$$

to floating-point tolerance.

### 2.2 Norm identity

Verify:

$$
\|\varepsilon_r^{\mathrm{RO}}\|_2^2
\approx
\|\varepsilon_r^{\mathrm{TF}}\|_2^2
+ \|d_r\|_2^2
+ 2\langle \varepsilon_r^{\mathrm{TF}}, d_r \rangle.
$$

### 2.3 Horizon-wise reporting

Confirm the run artifacts include:

- teacher-forced error by horizon,
- rollout error by horizon,
- drift by horizon,
- alignment by horizon,
- singular spectrum by horizon,
- gradient norms if computed.

## 3. Operational checks

- Check that the configured objective is recorded in the run artifact.
- Check that the configured lag length is recorded in the run artifact.
- Check that checkpoint files exist for each epoch if enabled.
- Check that plots exist in the run folder.
- Check that the profiler outputs exist when profiling is enabled.
- Check that the run folder contains a readable summary.

## 4. Manual commands

The feature should support a manual training command and a separate validation command.

The exact commands may evolve, but the interface must remain documented in the feature README and scripts README.

## 5. Acceptance Criteria

The feature passes validation if:

1. the objective selector works,
2. the one-lag and multi-context paths both run,
3. the rollout algebra checks pass,
4. the saved outputs are complete,
5. the next experiment can reuse the same interface.

