# JEPA World Models

This repository is a working project for self-supervised vision, probing, and
latent-space inspection. The current focus is not a full world model. It is:

- training and checking a VICReg-style vision encoder
- probing the frozen representation with linear and low-shot classifiers
- using nearest-neighbor retrieval to inspect the latent space
- presenting the work through a simple research-facing frontend

The codebase is set up so the analysis code, docs, and demo reuse the same
artifacts instead of building parallel pipelines.

## What Is Built

### Model and training

- A VICReg training path with a ViT-based encoder.
- Training configuration and checkpoint loading for the current model setup.
- Utility code for running the encoder and projector on the same data pipeline.

### Probing and retrieval

- Linear probe evaluation on frozen embeddings.
- Low-shot probe evaluation for smaller labeled subsets.
- k-NN retrieval on the frozen embedding space.
- Feature-bank caching so probing and retrieval do not recompute embeddings on
  every run.
- Layerwise feature extraction for checking where semantics begin to appear.

### Demo and frontend

- A static, multi-page frontend for the project.
- Pages for Home, Method, Results, and Demo.
- A retrieval demo that accepts an uploaded image and returns nearest neighbors.
- A backend endpoint that reuses the trained encoder and the cached retrieval
  index.
- A downloaded image corpus under `data/test_images` with a manifest so the
  demo can show real filenames.

### Downloads and data

- A script for downloading class-balanced test images from Google Images using
  SerpApi.
- A manifest that records the saved file name, source URL, and query used for
  each downloaded image.
- STL-10 raw data already present under `data_raw`.

### Documentation

- A probing pipeline write-up.
- A frontend architecture document.
- A general world-model architecture document.
- A separate video SSL planning document.

## What The Repo Can Do Right Now

1. Train or reload the current VICReg checkpoint.
2. Run a probing suite on frozen embeddings.
3. Cache retrieval indices for encoder or projector space.
4. Serve a local frontend that explains the project.
5. Accept an uploaded image and return nearest neighbors with filenames,
   labels, thumbnails, and similarity scores.

## How To Run

### Probing

```powershell
uv run python scripts/run_probing.py --checkpoint checkpoints/vicreg/best.pt
```

### Static frontend

```powershell
uv run python scripts/serve_frontend.py
```

### Retrieval demo

```powershell
uv run python scripts/serve_retrieval_demo.py --checkpoint checkpoints/vicreg/best.pt
```

## What I Understand So Far

The current project is about checking whether the learned representation is
actually useful. The key idea is:

- freeze the encoder
- measure what the representation already contains
- compare encoder space and projector space
- inspect local neighborhood structure, not just one accuracy number

That is why the repo has probing, retrieval, and a frontend that explains the
method instead of hiding it behind a generic app.

## BYOL

BYOL is part of the learning background for this repo, but it is not currently
implemented in code. It matters here as theory and comparison material:

- it helps frame representation learning without negatives
- it becomes more relevant later if the repo expands beyond VICReg into other
  SSL variants
- it is useful as a conceptual reference for future video SSL work

## Going Forward

The next direction is video SSL and latent dynamics.

The plan is not to jump straight to a full world model. The next steps are:

1. learn how SSL changes when the input becomes a clip instead of a still
   image
2. build a latent-dynamics probe on video
3. make a small frontend that shows trajectories over time
4. only then move toward rollout, future prediction, and action-conditioned
   modeling

That direction is documented in [docs/plans/video_plan.md](docs/plans/video_plan.md).
# JEPA World Models

This repo currently contains:

- a VICReg-style self-supervised image encoder
- probing and retrieval utilities for STL-10
- a frontend for image retrieval
- a video latent-dynamics demo
- a temporal probing plan for video

## Video Temporal Probe

The video path is now focused on a simple temporal diagnostic:

- sample 32 frames from a short clip
- encode frames with the frozen VICReg ViT
- keep frame order
- train a small classifier to predict forward vs reversed

The goal is not full video action recognition.
It is to test whether the frozen representation keeps temporal order.

### Planned/available scripts

- `scripts/run_video_temporal_probe.py`
- `scripts/run_video_dynamics.py`
- `scripts/serve_video_demo.py`

### Planned docs

- `docs/video/video_plan.md`
- `docs/video/video_classification_thing.md`
- `docs/video/frontend_implementation_plan.md`
