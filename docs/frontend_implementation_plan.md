# Frontend Implementation Plan

This document turns the frontend idea into concrete repo work.

Goal:

- show video latent dynamics clearly
- keep the interface simple
- make the result good enough for a hiring manager or imaging team

## 1. What The Frontend Must Do

1. Load a video clip.
2. Play it frame by frame.
3. Show a latent trajectory that moves as the clip changes.
4. Show nearest neighbors for the current clip or frame.
5. Keep the explanation short and readable.

## 2. Build Order

### Phase 1: Backend First

Goal: make sure the data and results exist before styling the page.

1. Expose a video probing endpoint or script output.
2. Return frame embeddings, a 2D trajectory, and nearest neighbors.
3. Include filenames and labels in the returned results.
4. Make sure the query clip is excluded from its own neighbors.
5. Save outputs in a predictable folder under `logs/`.

Done means:

- the frontend has real data to display
- the backend can be tested without the UI

### Phase 2: Frontend Second

Goal: make the results readable in the browser.

1. Build the page layout.
2. Add the video player.
3. Add the latent trajectory panel.
4. Add the neighbor panel.
5. Add short plain-language text blocks.

Done means:

- a user can open the page and understand what they are looking at
- the page works with one sample clip first

### Phase 3: Animation Polish

Goal: make the motion easy to see.

1. Animate the latent point as the clip plays.
2. Add a fading trail.
3. Update neighbors smoothly.
4. Add forward vs reversed comparison.
5. Tune timing and spacing so the page feels clear, not busy.

Done means:

- the motion is obvious
- the animations help understanding instead of distracting from it

## 3. Pages

1. Home
   - short project summary
   - links to Method, Results, and Demo

2. Method
   - what the encoder does
   - what the probe does
   - what temporal probing means

3. Results
   - probe results
   - retrieval examples
   - video probe summary

4. Demo
   - upload or choose a video
   - play it
   - watch the latent trajectory move
   - inspect nearest neighbors

## 4. Components

1. `VideoPlayer`
   - plays the clip
   - exposes play, pause, scrub, and step controls

2. `LatentTrajectoryPlot`
   - draws the moving point in 2D
   - shows the trail across time

3. `NeighborStrip`
   - shows nearest neighbors for the current clip state
   - includes filenames and short labels

4. `ClipSelector`
   - lets the user choose a sample clip
   - later can support upload

5. `MotionSummary`
   - prints a plain-language explanation of what the animation means

## 5. Data Flow

1. User selects or uploads a clip.
2. The backend extracts frame embeddings.
3. The backend returns a trajectory and neighbor results.
4. The frontend animates the trajectory while the clip plays.
5. The frontend updates the neighbor panel as time changes.

## 6. Animations To Build First

1. Moving latent point
2. Fading trail
3. Smooth neighbor updates
4. Forward vs reversed clip comparison

## 7. Files To Add Or Update

1. `frontend/video.html`
2. `frontend/video.js`
3. `frontend/video.css`
4. `scripts/serve_video_demo.py`
5. `src/jepa_world_models/analysis/video_probing.py`

## 8. Done Means

1. A user can open the demo page.
2. A user can play a clip.
3. The latent plot moves with time.
4. Neighbor results update clearly.
5. The page is understandable without jargon.
