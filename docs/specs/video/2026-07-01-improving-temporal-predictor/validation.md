# Validation: Improving Temporal Predictor

## 1. What must be verified

The feature is valid only if the following are visible in saved artifacts:

1. the predictor trains without interface mismatches,
2. the one-lag ablation works,
3. the context-window predictor works for configurable lag lengths,
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

### 2.3 Alignment

Compute:

$$
\cos \theta_r
=
\frac{\langle \varepsilon_r^{\mathrm{TF}}, d_r \rangle}
{\|\varepsilon_r^{\mathrm{TF}}\|_2 \, \|d_r\|_2}.
$$

### 2.4 Baseline comparison

Compare at minimum against:

- repeat-last,
- mean-context.

The learned predictor should at least be competitive with the trivial control settings before more complex model changes are justified.

## 3. Operational checks

- Check that the cached latent shapes match the configured context/future settings.
- Check that the configured lag length is recorded in the run artifact.
- Check that training checkpoints are written.
- Check that plots exist in the run folder.
- Check that the profiler outputs exist when profiling is enabled.
- Check that the run folder contains a readable summary.

## 4. Manual commands

The feature should support a manual training command and a separate validation command.

The exact commands may evolve, but the interface must remain documented in the feature README and scripts README.

## 5. Acceptance Criteria

The feature passes validation if:

1. the one-lag and multi-context paths both run,
2. the rollout algebra checks pass,
3. the saved outputs are complete,
4. the feature can be explained from the artifacts without reading code,
5. the next experiment can reuse the same interface.
