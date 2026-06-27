# Specifications

This folder is the home for spec-driven development in `jepa_world_models`.

The goal is to keep experiments structured:

- write the spec before implementation when possible
- keep the spec close to the code it governs
- treat the spec as the contract for behavior, metrics, and acceptance criteria
- update the spec when the implementation meaningfully changes

## How To Use

1. Create a spec file for a new experiment or feature.
2. State the objective, scope, non-goals, inputs, outputs, and success criteria.
3. Link the spec to the code and docs that implement it.
4. Keep a short test plan in the same file.
5. Mark open questions explicitly instead of leaving them implicit.

## Recommended Practice

Use the `spec-anchored` style:

- the spec is the source of truth for the current work
- the implementation may evolve, but only in ways the spec allows
- tests, scripts, and docs should all point back to the same file

## Folder Layout

- `docs/specs/template.md`: reusable spec template
- `docs/specs/video/`: video-related specs
- `docs/specs/vision/`: future vision/image specs
- `docs/specs/research/`: broader research experiments
- `docs/specs/video/strong_latent_dynamics_full_data.md`: full-scale latent dynamics training spec

## Current Rule

If you are starting a new experiment, add a spec here first.
If you are modifying an existing experiment, update the matching spec before or alongside the code.
