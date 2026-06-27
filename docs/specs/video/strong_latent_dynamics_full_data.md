# Strong Latent Dynamics v1

## Status

`draft`

## Objective

Build a materially stronger video world model that predicts future latent dynamics on full-scale video data.

This is the next serious step beyond the current inspection prototype.
The goal is not just to make the latent forecast look reasonable in 2D.
The goal is to learn a temporal model that captures real future structure in video.

## Context

The current latent projection browser and temporal predictor are useful for inspection, but the behavior is still limited.
The model is learning dynamics in a frozen latent space, yet the results are not strong enough to be the final system.

The repository has access to large video corpora, including:

- Kinetics-400
- Something-Something V2

That gives us enough scale to train a substantially stronger model than the current small prototype.

## Scope

In scope:

- full-data or near-full-data training
- stronger latent temporal prediction
- longer context windows
- longer future horizons
- multi-step rollout evaluation
- latent-space inspection in the browser
- saved checkpoints and reports

Out of scope:

- pixel-space diffusion video generation
- browser-based training control
- classification-only training
- a final production UI

## Proposed Modeling Direction

The model should predict future latent dynamics, not just repeat the last latent or regress to the mean.

Recommended direction:

- frozen video encoder
- temporal predictor over latent sequences
- multi-step rollout objective
- multi-scale future supervision
- optional stochastic future head if deterministic forecasts collapse

Let the encoder produce latent tokens

$$
Z = [\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_T],
$$

with future target sequence

$$
Y = [\mathbf{z}_{t+1}, \mathbf{z}_{t+2}, \dots, \mathbf{z}_{t+K}].
$$

The predictor outputs

$$
\hat{Y} = [\hat{\mathbf{z}}_{t+1}, \hat{\mathbf{z}}_{t+2}, \dots, \hat{\mathbf{z}}_{t+K}].
$$

The core loss should include:

$$
\mathcal{L}_{\text{mse}} = \frac{1}{K}\sum_{k=1}^{K}\|\hat{\mathbf{z}}_{t+k} - \mathbf{z}_{t+k}\|_2^2
$$

and a normalized directional term:

$$
\mathcal{L}_{\text{cos}} = \frac{1}{K}\sum_{k=1}^{K}
\left(1 - \left\langle
\frac{\hat{\mathbf{z}}_{t+k}}{\|\hat{\mathbf{z}}_{t+k}\|},
\frac{\mathbf{z}_{t+k}}{\|\mathbf{z}_{t+k}\|}
\right\rangle\right).
$$

If rollout quality is weak, add multi-step autoregressive rollout loss:

$$
\mathcal{L}_{\text{rollout}} = \sum_{r=1}^{R} \lambda_r \, \mathcal{L}_{\text{future}}^{(r)}.
$$

## Data

Primary training sources:

- `Kinetics-400`
- `Something-Something V2`

Rationale:

- Kinetics-400 provides broad action diversity and scene variation.
- Something-Something V2 provides strong motion-dependent temporal structure.

The combined dataset should help the model learn both:

- object/action semantics
- motion-conditioned future evolution

## Data Strategy

Recommended training order:

1. validate on the current small-scale pipeline
2. train on a larger subset of Something-Something V2
3. train on Kinetics-400
4. train on a mixed dataset
5. compare mixed vs single-dataset behavior

If compute is limited, start with Something-Something V2 because the motion signal is more direct.
Use Kinetics-400 to improve diversity and generalization.

## Temporal Setup

Start with a longer horizon than the prototype.

Suggested initial configuration:

- context: 4 to 8 seconds
- future: 2 to 4 seconds
- sampling rate: 4 fps for baseline experiments

If memory allows, test higher frame rates and longer windows later.

The key is to preserve a long enough context that the model can infer motion trends, not just the last frame state.

## Architecture

Baseline architecture:

- frozen encoder
- temporal transformer or state-space predictor
- hidden latent projector only if needed for stability
- multi-step output head

Possible extensions:

- hierarchical temporal predictor
- stochastic latent head
- separate short-horizon and long-horizon branches
- auxiliary contrastive objective against negative futures

## Evaluation

We should not judge the model by a single loss value.

Required evaluation:

- latent MSE
- normalized latent MSE
- cosine similarity
- repeat-last baseline
- mean-context baseline
- rollout stability across multiple steps
- qualitative inspection in the browser

If possible, track:

- per-horizon error curves
- error as a function of rollout length
- dataset-specific performance

## Acceptance Criteria

- the model trains on substantially more data than the current prototype
- the predictor outperforms repeat-last on at least some meaningful validation slices
- rollout quality degrades gracefully instead of collapsing immediately
- the browser can inspect full-scale samples with the same latent projection tooling
- checkpoints and reports are saved consistently

## Risks

- deterministic latent prediction may still average out uncertain futures
- more data may not help if the temporal objective is too weak
- longer horizons will increase memory and compute pressure
- the frozen encoder may cap the quality ceiling

## Test Plan

- run a small sanity training job first
- run a medium-scale job on one dataset
- compare against repeat-last and mean-context baselines
- inspect browser trajectories for multiple clips
- then scale up to the full mixed-data run

## Open Questions

- should Kinetics-400 and Something-Something V2 be mixed from the start or staged separately?
- should the predictor be deterministic or stochastic?
- should the next step include a decoder for pixel-space visualization?
- should we keep the frozen encoder or fine-tune it later?


## Detailed Execution Plan

This section is the implementation plan for the spec above.
It is intentionally detailed so that each experiment can be run, compared, and revised without guessing.

### Phase 0: Stabilize the current baseline

Goal:

- confirm the existing prototype still runs end to end
- preserve the current latent browser workflow
- establish a reliable baseline for comparisons

Tasks:

- run the current latent dynamics pipeline on a small subset of Something-Something V2
- verify the same checkpoint can be loaded by the browser demo
- save training, validation, and test metrics
- inspect a few clips in the latent projection page

Outputs:

- baseline checkpoint
- `result.json`
- prediction artifacts
- browser verification notes

Exit criteria:

- no crashes during training or evaluation
- browser inspection works for dataset clips and uploads
- the metrics and artifacts are written to disk consistently

### Phase 1: One-dataset scale-up

Goal:

- improve temporal prediction quality on a single dataset before mixing corpora

Recommended dataset order:

1. Something-Something V2 first
2. Kinetics-400 second

Reasoning:

- Something-Something V2 is more motion-heavy and directly exercises temporal prediction
- Kinetics-400 adds diversity and helps generalization

Tasks:

- increase the number of clips sampled into the latent bank
- use a longer context window
- use a longer future horizon
- train for more epochs than the baseline
- compare against repeat-last and mean-context baselines

Outputs:

- improved single-dataset checkpoint
- per-horizon loss curves
- latent forecast examples for multiple clips

Exit criteria:

- validation improves over the baseline run
- repeat-last is no longer the best-performing trivial strategy on meaningful slices
- rollout remains stable for several steps

### Phase 2: Multi-scale prediction

Goal:

- force the model to represent both short-term motion and longer-term evolution

Tasks:

- add multiple future heads or multi-scale target branches
- supervise several horizons at once
- keep the rollout objective active
- compare short, medium, and long horizon errors separately

Suggested horizon set:

- short: around 0.5 s
- medium: around 1 s to 2 s
- long: around 4 s

Outputs:

- multi-scale checkpoint
- error table by horizon
- rollout plots by horizon

Exit criteria:

- short-horizon quality remains good
- medium and long horizons improve relative to the single-head version
- rollout curves degrade smoothly rather than collapsing

### Phase 3: Full-scale training

Goal:

- train on a substantially larger corpus and check whether the model scales with data

Tasks:

- move from subsets to large-scale or full-data runs
- save checkpoints at regular intervals
- evaluate on held-out splits
- run browser inspection against representative examples

Outputs:

- large-scale checkpoint series
- reproducible logs
- evaluation summaries
- browser-ready artifacts for representative clips

Exit criteria:

- training remains numerically stable
- validation loss does not diverge
- the model shows a consistent improvement trend with scale

### Phase 4: Mixed-dataset training

Goal:

- combine the strengths of Something-Something V2 and Kinetics-400

Tasks:

- create a mixed sampler or alternating training schedule
- compare mixed training against single-dataset training
- track dataset-specific performance separately

Outputs:

- mixed-data checkpoint
- per-dataset validation breakdowns
- comparison plots across data sources

Exit criteria:

- mixed training is not worse than the best single-dataset run on every slice
- the mixed model generalizes better on diverse clips

## Architecture Plan

### Baseline choice

Keep the encoder frozen initially.
That keeps the comparison clean and lets us focus on temporal prediction quality.

### Predictor options

Use the strongest temporal model that fits the GPU budget comfortably:

- deeper transformer
- causal transformer with rollout
- state-space model
- hybrid temporal stack

### Practical recommendation

Start with the current temporal transformer family, but make it stronger before changing families.
Only switch architecture if the current family saturates.

## Loss Plan

Use a weighted combination of:

- latent MSE
- cosine alignment
- rollout loss

The goal is to avoid the trivial mean future and force the predictor to preserve direction and structure over time.

A practical default loss stack is:

$$
\mathcal{L} = \alpha \mathcal{L}_{\text{mse}} + \beta \mathcal{L}_{\text{cos}} + \gamma \mathcal{L}_{\text{rollout}}
$$

where `\alpha`, `\beta`, and `\gamma` are tuned after the first large stable run.

## Compute and Memory Plan

The main scaling factors are:

- number of videos
- number of sampled frames per clip
- context length
- future horizon
- batch size
- model depth and width

Guideline:

- increase data first if the GPU has room
- reduce batch size before cutting the horizon too aggressively
- only shrink the temporal window when the model cannot fit otherwise

If the model overfits or becomes unstable, reduce complexity in this order:

1. batch size
2. model width
3. horizon length
4. context length

## Evaluation Plan

Every meaningful run should report:

- train loss
- validation loss
- test loss
- latent MSE
- normalized latent MSE
- cosine similarity
- repeat-last baseline
- mean-context baseline
- per-horizon rollout error

Qualitative inspection should include:

- context path
- true future path
- predicted future path
- a handful of representative clips from each dataset

## Browser Plan

The browser remains the inspection surface for this project.

Keep the browser aligned with the latest checkpoint by ensuring it can:

- load dataset clips
- load local uploads
- replay clips from the start
- show PCA and t-SNE
- display the forecast path and metrics

For stronger models, expose more of the forecast details rather than less.

## Artifact Plan

Each major run should save:

- checkpoint files
- result JSON
- predictions CSV or equivalent artifact
- validation report
- short notes on what changed

Keep the artifact naming consistent so runs are easy to compare later.

## Experiment Order

Recommended order of execution:

1. stabilize the baseline
2. scale Something-Something V2
3. improve the predictor
4. add multi-scale supervision
5. scale to Kinetics-400
6. run the mixed dataset version
7. revisit the encoder only if the predictor is clearly not the bottleneck

## Risks

- the predictor may still learn a smooth average future
- the encoder may be the bottleneck rather than the temporal model
- multi-scale supervision may add complexity without enough gain
- mixed training may blur dataset-specific structure if the sampler is poorly balanced

## Decision Points

Before the full-scale run, decide:

- whether to start with single-dataset or mixed training
- whether to keep the encoder frozen for the entire first round
- whether to add stochastic outputs now or later
- whether to add a decoder immediately after latent improvement

## Recommended Next Action

Start with a medium-scale Something-Something V2 run using a stronger temporal predictor and a longer context/future window.
If that run improves the meaningful baselines and rollout quality, scale to Kinetics-400 and then the mixed-data run.
