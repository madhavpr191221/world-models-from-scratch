# Video Reconstruction Demo Plan

## Summary

Build a masked reconstruction demo on top of the pretrained VideoMAE model.

The demo shows:

1. original clip
2. masked clip
3. reconstructed clip

The goal is a visible, explainable artifact that makes the model’s behavior easy to inspect.

## Reconstruction Strategy

This first version uses the VideoMAE decoder as the main path:

- encode a pool of training clips with the pretrained VideoMAE encoder
- run the decoder on masked clips to produce token-level predictions
- fit a small tubelet decoder head that maps decoded tokens back to tubelet pixels
- copy visible tubelets from the input clip and fill masked tubelets from the head output

The older tubelet-bank retrieval path stays available as a fallback/debug mode, but the default demo should show the decoder output.

## Frontend Flow

- user uploads a short video
- user chooses a mask ratio and mask mode
- backend reconstructs the clip
- frontend displays:
  - original video
  - masked video
  - reconstructed video
  - frame-strip previews
  - a mask map showing which tubelets were hidden

## Entry Points

- CLI demo:
  - `uv run python scripts/run_video_reconstruction.py --checkpoint logs/videomae_large/best_videomae.pt --reconstruction-mode decoder`
- Local UI server:
  - `uv run python scripts/serve_video_reconstruction.py --checkpoint logs/videomae_large/best_videomae.pt --reconstruction-mode decoder`

## Test Plan

- upload a known short clip
- confirm the masked region is visible
- confirm the reconstructed output is written and playable
- confirm the frontend renders the three video panes and the mask map

## Assumptions

- reconstruction means masked reconstruction, not free-form generation
- the first demo is decoder-first and frontend-visible
- a separate browser-based UI is worth it because this is the first time the model will produce an obviously visual output
