# Latent Projection Browser Plan

## Goal

Build a browser-hosted latent projection view that makes the video world model easy to inspect.

The page should answer three questions:

1. What clip am I looking at?
2. Where do the observed latents move in 2D?
3. Does the predicted future latent trajectory follow the true future path?

This is the first live interface for the video world model. It is intentionally minimal: a clip viewer, a latent-space plot, and a small metric panel.

---

## Representation

Let the frozen video encoder map a clip of `T` sampled frames to a latent sequence

$$
Z = [\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_T],
\qquad \mathbf{z}_t \in \mathbb{R}^d.
$$

Here `d` is the encoder latent dimension.

For the browser map, each clip is summarized by the mean latent

$$
\bar{\mathbf{z}} = \frac{1}{T}\sum_{t=1}^T \mathbf{z}_t \in \mathbb{R}^d.
$$

That gives one point per clip for the background cloud.
The selected clip still keeps its full trajectory so the page can draw the context path, the true future path, and the predicted future path.

---

## Time To Frames

The UI accepts time in seconds because that is the natural way to think about video.

If the user chooses `context-seconds`, `future-seconds`, and `sample-fps`, then the code converts them to even frame counts:

$$
T_c = 2\left\lceil \frac{\text{context-seconds} \cdot \text{sample-fps}}{2} \right\rceil,
\qquad
T_f = 2\left\lceil \frac{\text{future-seconds} \cdot \text{sample-fps}}{2} \right\rceil,
\qquad
T = T_c + T_f.
$$

The even rounding matters because the encoder uses temporal tubelets of size 2.
The number of latent time steps is therefore

$$
L = \frac{T}{2}.
$$

So if you ask for 5 seconds of context and 1.5 seconds of future at 4 fps, the task becomes `20` context frames, `6` future frames, and `13` latent steps.

---

## CLI

### Training the latent world model

The world-model trainer is still the main training entrypoint:

```powershell
uv run python scripts/run_video_world_model.py --checkpoint logs/videomae_large/best_videomae.pt --data-root data --source-split train --subset-size 2000 --context-seconds 5.0 --future-seconds 1.5 --sample-fps 4.0 --feature-batch-size 1 --batch-size 4 --epochs 10 --output-dir logs/video_world_model --cache-dir logs/video_world_model/cache --profile
```

Key arguments:

- `--checkpoint`: frozen VideoMAE backbone used as the latent encoder.
- `--data-root`: root directory for the video dataset.
- `--source-split`: dataset split sampled into the latent bank.
- `--subset-size`: how many videos are sampled into the run.
- `--context-seconds`: history length in seconds.
- `--future-seconds`: prediction horizon in seconds.
- `--sample-fps`: sampling rate used to turn seconds into frames.
- `--feature-batch-size`: batch size used while encoding raw clips into cached latents.
- `--batch-size`: batch size used while training the temporal predictor on cached latents.
- `--epochs`: number of full passes over the latent training set.
- `--cache-dir`: cached latent bank location.
- `--output-dir`: directory for checkpoints and reports.
- `--profile`: enables `torch.profiler` inside the training run.

The model minimizes latent regression loss over the future horizon:

$$
\mathcal{L} = \lambda_{\mathrm{mse}} \cdot \frac{1}{K}\sum_{k=1}^{K} \|\hat{\mathbf{z}}_{t+k} - \mathbf{z}_{t+k}\|_2^2
+ \lambda_{\mathrm{norm}} \cdot \frac{1}{K}\sum_{k=1}^{K} \left\|\frac{\hat{\mathbf{z}}_{t+k}}{\|\hat{\mathbf{z}}_{t+k}\|} - \frac{\mathbf{z}_{t+k}}{\|\mathbf{z}_{t+k}\|}\right\|_2^2.
$$

The cosine similarity score is the average inner product between the normalized prediction and target latents:

$$
\frac{1}{K}\sum_{k=1}^{K} \left\langle \frac{\hat{\mathbf{z}}_{t+k}}{\|\hat{\mathbf{z}}_{t+k}\|}, \frac{\mathbf{z}_{t+k}}{\|\mathbf{z}_{t+k}\|} \right\rangle.
$$

### Serving the browser demo

```powershell
uv run python scripts/video/serve_video_latent_projection.py --world-model-checkpoint logs/video_world_model/latent_world_model_best_videomae_1400170f03_train_2000_224_26_20_6.pt --data-root data --source-split train --subset-size 2000 --context-seconds 5.0 --future-seconds 1.5 --sample-fps 4.0 --feature-batch-size 1 --projection-method pca --host 127.0.0.1 --port 8002
```

The server exposes:

- `/latent.html`: the browser page
- `/api/latent/examples`: clip list for the dropdown
- `/api/latent/analyze`: full latent trajectory analysis for one clip
- `/api/latent/default`: the default analysis payload
- `/api/latent/file/<video_id>.webm`: raw clip playback for the selected sample

---

## What The Page Shows

- a video player for the selected clip
- a 2D latent projection plot
- the observed context path
- the true future path
- the predicted future path
- baseline metrics for `repeat_last` and `mean_context`

The background cloud uses PCA by default because it is stable and fast enough for live inspection.
`t-SNE` is available for local structure inspection, but it is slower and less stable across reruns.

---

## Why This Helps

The page turns a latent world model into something inspectable.

Instead of only reading a loss number, you can watch the latent trajectory move and compare:

- how the true future continues
- how the predictor extrapolates
- whether the forecast collapses to a bland average
- whether the selected clip behaves differently from the background clips

That is the right level of feedback for iterating toward a stronger JEPA-style world model.

---

## Next Steps

1. add latent-space rollout comparison across multiple clips
2. add a small decoder or video reconstruction preview for predicted latents
3. add a temporal embedding animation view for longer clips
4. make the projection page accept arbitrary clip lengths while respecting the encoder window constraints

## Interaction Flow

The page now follows a strict order:

1. choose a clip
2. choose PCA or t-SNE
3. press load
4. press play to animate the latent forecast

Changing the clip or the projection after loading marks the view as stale until the clip is loaded again. That avoids showing an old forecast as if it belonged to the new selection.
