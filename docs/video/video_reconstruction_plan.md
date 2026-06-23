# Video Reconstruction Demo Plan

## Summary

Build a masked reconstruction demo on top of the pretrained VideoMAE model.

The demo shows:

1. original clip
2. masked clip
3. reconstructed clip

The goal is a visible, explainable artifact that makes the model’s behavior easy to inspect.

## Reconstruction Strategy

This first version uses a tubelet bank:

- encode a pool of training clips with the pretrained VideoMAE encoder
- store token embeddings and the corresponding raw tubelets
- reconstruct masked tokens by retrieving the nearest raw tubelet from the bank

This is a practical way to turn the current representation model into a visible reconstruction demo.

## Frontend Flow

- user uploads a short video
- user chooses a mask ratio and mask mode
- backend reconstructs the clip
- frontend displays:
  - original video
  - masked video
  - reconstructed video
  - frame-strip previews

## Entry Points

- CLI demo:
  - `uv run python scripts/run_video_reconstruction.py --checkpoint logs/videomae_large/best_videomae.pt`
- Local UI server:
  - `uv run python scripts/serve_video_reconstruction.py --checkpoint logs/videomae_large/best_videomae.pt`

## Test Plan

- upload a known short clip
- confirm the masked region is visible
- confirm the reconstructed output is written and playable
- confirm the frontend renders the three video panes

## Assumptions

- reconstruction means masked reconstruction, not free-form generation
- the first demo is retrieval-assisted and frontend-visible
- a separate browser-based UI is worth it because this is the first time the model will produce an obviously visual output
