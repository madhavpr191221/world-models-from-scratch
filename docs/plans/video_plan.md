# Video SSL Roadmap for the JEPA World Models Repo

## Summary

This document is the planning track for moving from image SSL into video SSL
and latent dynamics. It assumes the current repo foundation is in place:

- VICReg is already implemented and understood at a working level.
- BYOL is understood as theory, but not implemented in this repo.
- The next step is to understand how representation learning changes when the
  input becomes time-based instead of a single image.

The goal is not to jump straight to a full world model. The goal is to build a
strong mental model for video SSL, then shape the repo toward a video-first
latent dynamics demo that can later grow into planning or action conditioning.

## Learning Goal

The main questions are:

1. What stays the same from image SSL to video SSL?
2. What changes when the model must handle time, motion, and order?
3. Which parts of the representation are about appearance, and which parts are
   about dynamics?

This plan treats video SSL as the bridge between representation learning and
world models.

## What I Already Know

The repo already has enough context to move past basic SSL theory:

- VICReg: learned invariance, variance, covariance; no collapse through
  explicit regularization.
- BYOL: online encoder, target encoder, predictor, no explicit negatives.
- Linear probing: freeze the encoder and test what is already readable in the
  representation.

The next learning step is to compare those ideas against video-specific
questions:

- How do we represent motion without forcing pixel reconstruction?
- How do we tell whether the model understands time or only appearance?
- How do we measure whether the representation supports future prediction?

## Three-Phase Roadmap

### Phase 1: Learn Video SSL Basics

Goal: understand how representation learning changes when the input is a clip
instead of a single image.

Repo deliverables:

- a small video dataset loader
- a simple clip encoder or clip sampler
- a baseline training script for video SSL
- a write-up that compares image SSL and video SSL
- a minimal evaluation script that checks temporal consistency

Questions this phase should answer:

1. What information stays stable across nearby frames?
2. What changes when motion is added?
3. How much of the learned space is appearance versus time?

### Phase 1 Data Design

Before any training, the repo needs a small but exact data contract for the
Something-Something dataset.

Known files:

- `data/20bn-something-something-download-package-labels/labels/labels.json`
- `data/20bn-something-something-download-package-labels/labels/train.json`
- `data/20bn-something-something-download-package-labels/labels/validation.json`
- `data/something_v2/20bn-something-something-v2/<video_id>.webm`

Observed schema:

- `train.json` is a list of records.
- each record has an `id` field such as `"78687"`.
- the matching video file is `78687.webm`.
- each record has a human-readable `label`.
- `labels.json` maps the human-readable label to a numeric class id.

Loader contract:

- input: one video id
- output: one tensor clip with shape `[T, C, H, W]`
- output: one numeric label
- output: the readable label string
- output: the video id

Clip sampling contract:

- sample a fixed number of frames, such as `8` or `16`
- keep the temporal order intact
- resize frames to a fixed spatial size
- normalize the frames with the same image pipeline style used elsewhere in
  the repo unless a video-specific transform is chosen later

Smoke-test contract:

1. Load one `.webm` file.
2. Sample one short clip.
3. Verify the returned tensor shape.
4. Verify the label lookup.
5. Visualize or save the sampled frames before training anything.

This is the first concrete implementation target. No model training should
start until this loader and smoke test are working.

### Phase 2: Build Latent Dynamics Evidence

Goal: test whether the frozen representation contains usable time structure.

Repo deliverables:

- a frozen-backbone temporal probe
- a latent future-prediction baseline
- a sequence-level retrieval or nearest-neighbor view
- cached video embeddings for repeatable evaluation
- plots or tables that compare encoder space and any projector-like space

Implementation shape for the first probe:

- input: clip tensor of shape `[T, C, H, W]`
- backbone: frozen video or frame encoder
- features: clip pooled embedding or concatenated frame embeddings
- head: linear classifier first, MLP only if needed
- targets: temporal order, future offset, or action template id
- output: accuracy or regression error on a held-out split

Questions this phase should answer:

1. Can a simple head recover future state better than chance?
2. Does temporal order matter?
3. Are nearby clips actually close in latent space for the right reasons?

## Why Temporal Probing Exists

Temporal probing asks whether the frozen video representation knows anything
about time, not just appearance.

Why we do it:

- An image model can recognize what is in a frame.
- A video model should ideally know something about order, motion, and change.
- Temporal probing checks whether that information is already present in the
  frozen embedding.

What it tells us:

- whether the model sees sequence or only objects
- whether motion is encoded in the latent space
- whether the representation is useful for later video understanding or
  world-model work

Concrete examples:

1. Forward vs reversed clip
- Input: the same clip in normal order and in reverse order.
- Question: can a frozen head tell which one is forward?
- Why it matters: if the model scores above chance, it is sensitive to
  temporal order.

2. Frame-drop or offset prediction
- Input: a clip with one frame removed, or a clip sampled from an
  earlier/later point in time.
- Question: can a frozen head tell that the timing changed?
- Why it matters: if the model notices this, it likely encodes more than
  static appearance.

### What A Temporal Probe Means

A temporal probe is the video version of a linear probe.

- Freeze the backbone.
- Turn each clip into a feature vector.
- Train a simple head on top of those frozen features.
- Measure whether the frozen representation already contains time information.

In this repo, the first temporal probe should be simple and honest:

1. Sample a short clip from one video.
2. Encode either the whole clip or each frame.
3. Train a small classifier or regressor on the frozen features.
4. Test whether the head can recover something about time.

Good first probe targets:

- clip order classification
- future frame or future clip offset prediction
- temporal direction prediction
- action template classification from frozen clip features

What this probe is not:

- it is not end-to-end video training
- it is not a full world model
- it is not a complicated sequence model
- it is not meant to prove planning yet

The point is to ask: what does the frozen representation already contain about
time?

### Phase 3: Make a Video Frontend

Goal: present the video model as a clear research demo.

Repo deliverables:

- a clip upload or clip selection page
- a latent trajectory viewer
- a perturbation view showing how the trajectory changes under crop, blur,
  occlusion, frame drop, or speed changes
- a nearest-neighbor clip panel
- a small rollout or prediction preview

Questions this phase should answer:

1. What does the model preserve over time?
2. Where does it become uncertain?
3. How does the latent path move through the clip?

## Video SSL Directions

### 1. Clip-Level SSL

Treat a short video clip like a structured version of an image.

- Input: 2 to 16 frames from the same clip.
- Task: learn a stable representation of the clip.
- Useful for: action recognition, temporal invariance, basic dynamics.

Pros:

- Simple starting point.
- Reuses many image-SSL instincts.
- Easy to explain in a frontend.

Cons:

- Can ignore fine-grained time structure.
- Often learns appearance more strongly than motion.

### 2. Latent Prediction

Predict a future latent representation instead of reconstructing pixels.

- Input: observed frames or clips.
- Task: encode the past, predict the future in embedding space.
- Useful for: world-model style reasoning.

Pros:

- Closer to the JEPA family of ideas.
- Better fit for dynamics than reconstruction-based models.
- Easier to connect to planning later.

Cons:

- Harder to evaluate than image SSL.
- Requires careful choice of temporal benchmark and evaluation protocol.

### 3. Factorized Appearance + Dynamics

Split the representation into two parts:

- appearance features
- motion/dynamics features

Pros:

- Matches how humans often reason about video.
- Good fit for geometry-heavy visualizations.
- Makes the latent space easier to explain.

Cons:

- More modeling choices.
- Can become overdesigned if the data and benchmarks are too small.

## Recommended Path

For this repo, the best sequence is:

1. Start with clip-level SSL on a small video dataset.
2. Add latent prediction in embedding space.
3. Separate appearance from motion in the visualization, not necessarily in the
   first model.
4. Build a frontend that shows latent trajectories and nearest neighbors over
   time.

This keeps the project honest and manageable.

## What Would Stand Out

If the goal is to stand out to a hiring manager or research team, the most
useful combination is:

1. A clear SSL learning track.
2. A video latent-space visualization.
3. A frozen-backbone probe or rollout test.
4. A clean explanation of what the model does and does not know.

That combination is stronger than a generic "AI demo" because it shows:

- actual representation learning understanding
- careful evaluation
- visual intuition for geometry and dynamics
- a credible path toward world models

## Risks

- Video SSL can become too broad too quickly.
- If the dataset is small, results may be noisy and overfit.
- A flashy video demo without a clear evaluation story will not help much.
- A world-model direction is stronger when the latent-space evidence is
  explicit and frozen-backbone tests are included.

## Next Implementation Shape

The next code phase should stay small and concrete:

1. add a minimal video data path
2. add one frozen latent dynamics test
3. add one clip visualization page

That is enough to prove the direction before growing the project further.

## Loader Design

The first dataset class should be intentionally boring.

Responsibilities:

- read `train.json` and `validation.json`
- map `id` to the `.webm` file on disk
- map the text label to a class id
- return a clip tensor and metadata

Not responsibilities:

- no training logic
- no augmentation policy experimentation
- no model-specific assumptions
- no frontend logic

If that loader works cleanly, the model and frontend layers can be built on top
of it without changing the data contract later.

## Planned Folder Structure

The raw data stays in `data/`. The code and scripts stay in the normal source
and `scripts/` locations. The plan is to keep the new video pieces isolated and
easy to find.

```text
data/
  something_v2/
    20bn-something-something-v2/
      <video_id>.webm
  20bn-something-something-download-package-labels/
    labels/
      labels.json
      train.json
      validation.json
      test.json
      test-answers.csv

src/jepa_world_models/
  data/
    video.py
      # implemented: video dataset helpers and clip sampling utilities
  analysis/
    video_probing.py
      # implemented: frozen latent probe / temporal evaluation helpers

scripts/
  inspect_video_sample.py
  # implemented: smoke test for one clip and frame sampling
  run_video_probing.py
    # implemented: video SSL / dynamics evaluation entry point

docs/
  plans/
    video_plan.md
      # this document
```

## Planned Dataset API

The first dataset helper should expose a small, predictable API:

- `__len__()`
- `__getitem__(idx)` returning:
  - clip tensor
  - numeric label
  - readable label
  - video id
- `get_clip_path(idx)` for debugging
- `sample_frames(video_id, num_frames)` or an internal equivalent

That keeps the rest of the code from depending on the raw JSON shape.

## Video Probe v1: Frozen Temporal Direction Test

### Summary

Build the first real video SSL evaluation on top of the existing
Something-Something V2 loader. The goal is to test whether a frozen image
encoder already contains temporal information when applied framewise to a short
clip.

Primary target: forward-vs-reversed clip classification using frozen clip
features. This is simple, label-free, and directly tests order sensitivity.

### Key Changes

- Add `scripts/run_video_probing.py` as the entrypoint for video probing.
- Add `src/jepa_world_models/analysis/video_probing.py` for reusable feature
  extraction and probe training.
- Reuse `src/jepa_world_models/data/video.py` as the dataset source.
- Use the current VicReg encoder as a frozen frame encoder.
- Compare two feature views:
  - sequence features: concatenate per-frame embeddings to preserve order
  - pooled features: average frame embeddings as an order-insensitive baseline
- Add a small optional sanity probe for action-template classification on the
  same frozen features.
- Write outputs to `logs/video_probing/`:
  - `summary.json`
  - `results.csv`
  - optional cached features for reruns

### Interface / Behavior

- Input: sampled clip tensor of shape `[T, C, H, W]`
- Backbone: frozen current encoder, applied per frame
- Probe target:
  - primary: `forward` vs `reversed`
  - secondary sanity check: template classification
- Head: linear classifier first
- No backbone updates
- No CLIP model dependency

### Test Plan

- Run the existing smoke test script on one `.webm` clip and confirm frame
  export still works.
- Run the video probing script on a small subset first.
- Confirm:
  - forward-vs-reversed accuracy is above chance
  - pooled baseline is weaker than sequence features
  - outputs are written under `logs/video_probing/`
- Verify the backbone stays frozen during probing.

### Assumptions

- The current VicReg encoder is the frozen backbone for v1.
- Something-Something V2 is the first video dataset.
- `av` remains the decoder backend because it already works in the repo.
- OpenCV is optional and not required for v1.
- The first probe should optimize for clarity and signal, not benchmark breadth.
### How These Examples Help Downstream

The point of the temporal examples is not just to get a probe score. The point is to see whether the frozen video features can help with tasks that depend on change over time.

Use the same style of clips to test whether the representation can support:

1. Action recognition.
   - Example: car turning left versus car turning right.
   - Example: person opening a door versus person closing a door.

2. Event ordering.
   - Example: traffic light red to green versus green to red.
   - Example: person walking into frame, reaching for an object, then leaving.

3. Motion-sensitive retrieval.
   - Example: if the query is a car braking, the nearest neighbors should be other braking clips, not just any car clips.
   - Example: if the query is a cup being filled, the neighbors should be other pouring or filling clips.

4. Future-state or next-step prediction.
   - Example: after a door starts opening, the model should help predict that the next state is more open.
   - Example: after a pedestrian steps into a crosswalk, the next likely state is that the person is farther across the road.

5. Simple anomaly checks.
   - Example: a clip where a car moves backward in a way that does not match the rest of the sequence.
   - Example: a clip where an object appears to jump positions without a smooth transition.

This is why the plan uses everyday objects as examples.
They are easy to understand, and they make it obvious whether the model is learning just appearance or also change over time.
## Frontend Plan For Video Latent Dynamics

The frontend should make one thing visible:

- as the video changes over time, the latent features should change too

Keep it simple. Do not try to explain everything at once.

### Page Layout

1. Top row
   - title
   - one short sentence explaining what the page shows
   - a small note that the encoder is frozen and the latent features come from the trained video model

2. Main video panel
   - video player
   - play / pause
   - scrubber
   - step forward / step back buttons

3. Latent dynamics panel
   - a 2D plot of the current clip embedding
   - a trail showing how the embedding moves frame by frame
   - the current point highlighted

4. Nearest-neighbor panel
   - nearest videos or nearest clips
   - update this as the clip plays
   - show filenames and short labels if available

5. Short explanation panel
   - one or two sentences in plain language
   - example: “The point moves as the clip changes. Similar motions stay near each other.”

### Components

1. `VideoPlayer`
   - shows the selected clip
   - drives the frame index

2. `LatentTrajectoryPlot`
   - takes embeddings for each frame or short window
   - draws the path through 2D space
   - highlights the current frame

3. `NeighborStrip`
   - shows top nearest neighbors for the current time step
   - updates as the clip changes

4. `MotionSummary`
   - gives a short human-readable explanation of what the animation is showing

5. `ClipSelector`
   - lets the user pick a video example
   - later can support upload

### Animation Ideas

Use animations that help the user understand change, not decoration.

1. Moving latent point
   - as each frame advances, the point moves
   - this is the clearest visual for latent dynamics

2. Fading trail
   - older points fade out
   - the path of motion stays visible

3. Smooth neighbor updates
   - neighbor cards shift when the current frame changes
   - this makes time-dependent similarity easy to see

4. Clip comparison
   - show forward and reversed clips side by side
   - compare their latent paths

5. State transitions
   - add simple labels like “start”, “middle”, “end”
   - use them only if they help explain the motion

### Why This Matters

This is the frontend version of the video probe.

It answers:

1. Does the embedding move as the clip changes?
2. Do similar motions land near each other?
3. Can the user see the difference between appearance and motion?

### Exact Repo Deliverables

1. `scripts/run_video_probing.py`
   - produces embeddings, probe results, and cache files

2. `scripts/serve_video_demo.py`
   - serves the frontend and the video-backed retrieval/latent views

3. `frontend/video.html`
   - page structure for the demo

4. `frontend/video.js`
   - handles playback, animation, and neighbor updates

5. `frontend/video.css`
   - visual styling for the page

6. `docs/plans/video_plan.md`
   - explains the plan and the meaning of latent dynamics

### What To Say In The Demo

1. “This point is the frozen embedding of the current clip.”
2. “As the clip changes, the point moves.”
3. “If the motion is similar, the points stay close.”
4. “This is one way to see latent dynamics without opening the model.”

## Frontend Implementation Plan

The concrete implementation checklist lives here:

- [docs/frontend_implementation_plan.md](../frontend_implementation_plan.md)

Use that doc to turn the video frontend idea into actual files, page structure, and animation work.

## Video-Level Objective

At the video level, the goal is simple:

- given a short clip, produce features that make it easy to separate different motion patterns and different temporal orderings

In plain language, this means:

1. The model should know what is happening over time.
2. The model should notice if the order changes.
3. The model should tell different kinds of motion apart.

Everyday examples:

1. A cup being filled versus a cup being emptied.
   - Same object.
   - Different motion.
   - Different outcome.

2. A door opening versus a door closing.
   - Same scene.
   - Opposite direction.
   - The clip should not look the same in feature space.

3. A car entering a frame versus leaving a frame.
   - Same vehicle.
   - Different temporal direction.
   - The representation should keep that difference.

4. A person walking forward versus walking backward.
   - Same person.
   - Same appearance.
   - Different motion pattern.

5. A ball rolling versus a ball bouncing.
   - Both involve movement.
   - The timing is different.
   - The features should reflect that difference.

Why this matters:

- if two clips show the same object but different motion, the model should not collapse them into the same thing
- if one clip is reversed, the model should notice that the order changed
- if two clips have similar motion, the features should end up close to each other

This is the simplest way to think about the video objective:

- image level: what is in the frame?
- video level: what is happening over time?

## Motion vs Order

These are related, but not the same.

1. Motion means the kind of change.
   - Example: a cup is being filled.
   - Example: a door is opening.
   - Example: a car is entering the frame.

2. Order means the sequence of states.
   - Example: empty cup -> half full -> full.
   - Example: closed door -> opening -> open.
   - Example: outside frame -> entering -> centered.

Why the difference matters:

- two clips can have the same object but different motion
- two clips can have the same frames but in the wrong order
- a good video representation should notice both

Concrete examples:

1. Cup being filled vs emptied.
   - Motion changes direction.
   - The order of states is reversed.

2. Door opening vs closing.
   - Same object.
   - Opposite temporal sequence.

3. Car entering vs leaving.
   - Same car.
   - Different direction relative to the frame.

The model should not treat these as the same clip.

Why this is useful in the demo:

- if the motion changes, the latent trail should change too
- if the order changes, the trajectory or probe output should also change
- if two clips look similar but one is reversed, the demo should make that difference visible

That is the point of the video frontend:

- show the change
- show the order
- show that the representation can tell them apart

## Temporal Probing With Math

Temporal probing asks whether a frozen video encoder keeps information about time order, not just appearance.

Let a clip be a sequence of frames:

$$
x = (x_1, x_2, \dots, x_T)
$$

Let the frozen encoder produce a feature for each frame:

$$
z_t = f_\theta(x_t)
$$

For a short clip, we can pool these frame features into one clip feature:

$$
z_{\text{clip}} = \frac{1}{T}\sum_{t=1}^{T} z_t
$$

Now define a temporal probe dataset with two versions of the same clip:

$$
x^{\text{fwd}} = (x_1, x_2, \dots, x_T)
$$

$$
x^{\text{rev}} = (x_T, x_{T-1}, \dots, x_1)
$$

If the representation keeps temporal structure, then a simple classifier should be able to separate them:

$$
\hat{y} = \mathrm{softmax}(W z_{\text{clip}} + b)
$$

with

$$
y = 
\begin{cases}
0 & \text{forward clip} \\
1 & \text{reversed clip}
\end{cases}
$$

Why this matters:

- if the encoder only stores appearance, forward and reversed clips may look too similar
- if the encoder stores time order, the probe should separate them better than chance
- this gives a simple test for whether the latent space has motion information

In the frontend, this becomes a direct visual question:

- does the forward clip trace a different path than the reversed clip?
- do the nearest neighbors also change when the order changes?
