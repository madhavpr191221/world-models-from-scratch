# Scripts

The top-level entrypoints stay small:

- `run_training.py`
- `run_video_world_model.py`

Everything else is grouped by purpose:

- `scripts/dev/` for local debugging and profiling helpers
- `scripts/data/` for data download and collection helpers
- `scripts/video/` for video pretraining, reconstruction, demos, and probes
- `scripts/probing/` for retrieval and probing entrypoints
