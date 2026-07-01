# Tech Stack

## Current architecture

- Video data -> tubelet encoder -> latent cache -> temporal predictor -> rollout metrics and plots.

## Required properties

- Encoder and predictor must be pluggable.
- The temporal model may be a causal transformer, Mamba, TCN, GRU/LSTM, or another sequence model.
- The encoder may be a ViT-like video encoder, CNN, 3D CNN, Swin-style model, or another feature extractor.
- The training pipeline must expose clear CLI entrypoints.
- The outputs must be saved in stable run folders with metrics, plots, and validation reports.

## Preferred runtime behavior

- GPU first when available.
- CPU fallback must still work.
- Training should save checkpoints during the run, not only at the end.
- Profiling and plotting should be part of the workflow, not separate afterthoughts.

## Data and artifacts

- Data splits must be explicit and non-overlapping.
- Latent caches must record provenance, frame counts, context/future settings, and model fingerprint.
- Each experiment should write its outputs into a dedicated folder.

## Development standards

- Keep feature branches small and focused.
- Encode non-negotiables in specs, not memory.
- Treat validation as a first-class deliverable.

