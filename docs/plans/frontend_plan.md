# Frontend Plan for the JEPA World Models Repo

## Summary

Build a research-facing frontend that presents this repo as a serious
vision/representation project, not a generic demo. The site should combine
editorial-style documentation, measured probing results, and one interactive
retrieval experience so a hiring manager can understand the method, inspect
evidence, and interact with the embedding space.

Primary implementation choice: Astro + MDX for content pages, React islands for
interactivity, FastAPI-backed artifact endpoints for retrieval/results. The site
should emphasize latent-space geometry and "dynamics" through restrained
animations, especially in the retrieval and layer-comparison views.

## Key Changes

- Add a frontend site with 5 pages:
  - Home: project thesis, key results, architecture preview
  - Method: VICReg + probing explanation, encoder/projector distinction
  - Results: linear probe, low-shot, k-NN, layerwise comparisons
  - Retrieval Demo: image upload, nearest neighbors, similarity scores
  - About / Research Direction: connect this work to latent world-model ideas
- Keep the docs as the source of truth and make the frontend reflect them:
  - reuse the probing methodology already documented
  - reuse the existing architecture diagrams and expand them into
    page/component/data-flow diagrams
- Define a minimal backend contract for the UI:
  - summary artifact endpoint for probe results
  - retrieval endpoint for query images and nearest neighbors
  - static artifact loading for plots, tables, and cached outputs
- Use a strong visual language:
  - editorial layout
  - clear metric cards
  - diagram-led explanation
  - restrained color system
  - typography-forward presentation
- Add motion intentionally to visualize latent dynamics:
  - page and section entrance reveals
  - diagram transitions
  - retrieval result arrival animation
  - comparison tab transitions
  - light hover states on cards and thumbnails
- Keep the system modular:
  - shared shell/layout
  - reusable section headers, metric tiles, diagram containers, tables,
    preview cards, image cards, tabs
  - page-specific composition rather than duplicated UI

## Implementation Plan

- Define the frontend information architecture and routing.
- Finalize a shared component set for research pages and the demo.
- Build the static pages first so the site is immediately useful without the
  demo.
- Wire in probe/retrieval artifacts from the existing analysis pipeline.
- Implement the retrieval demo as the only major interactive surface.
- Add motion and accessibility as a finishing layer, not as the core structure.
- Deploy the frontend separately from the model-serving backend if needed.

## Tests and Acceptance Criteria

- Home page clearly explains the project in under one screen.
- Method page explains encoder vs projector, freezing, and probing without
  ambiguity.
- Results page shows probe metrics and makes encoder/projector comparison easy.
- Retrieval demo accepts an image and returns ranked neighbors with readable
  labels/scores.
- Page/component diagrams and data-flow diagrams exist in the docs and match
  the implementation.
- Animations are subtle, readable, and respect reduced-motion preferences.
- The site loads fast on desktop and mobile and does not feel like a generic ML
  dashboard.

## Assumptions

- The frontend should prioritize clarity and credibility over flashy
  interaction.
- Astro/MDX is the right default because the site is content-heavy and
  research-oriented.
- React islands are only needed for the upload/retrieval interaction and
  comparison controls.
- The repo should avoid duplicate demo functionality; the new frontend should
  reuse the existing probing/retrieval artifacts instead of introducing a second
  parallel pipeline.
