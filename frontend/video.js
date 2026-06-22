const state = {
  examples: [],
  analysis: null,
  reverseAnalysis: null,
  scaledTrajectory: [],
  reverseTrajectory: [],
  currentFrame: 0,
  compareMode: false,
};

const els = {
  exampleSelect: document.getElementById("exampleSelect"),
  loadButton: document.getElementById("loadButton"),
  playButton: document.getElementById("playButton"),
  stepBackButton: document.getElementById("stepBackButton"),
  stepForwardButton: document.getElementById("stepForwardButton"),
  compareButton: document.getElementById("compareButton"),
  compareSection: document.getElementById("compareSection"),
  forwardSvg: document.getElementById("forwardSvg"),
  reverseSvg: document.getElementById("reverseSvg"),
  forwardMeta: document.getElementById("forwardMeta"),
  reverseMeta: document.getElementById("reverseMeta"),
  clipTitle: document.getElementById("clipTitle"),
  clipLabel: document.getElementById("clipLabel"),
  clipFrames: document.getElementById("clipFrames"),
  frameLabel: document.getElementById("frameLabel"),
  motionLabel: document.getElementById("motionLabel"),
  videoPlayer: document.getElementById("videoPlayer"),
  frameSlider: document.getElementById("frameSlider"),
  trajectorySvg: document.getElementById("trajectorySvg"),
  neighborStatus: document.getElementById("neighborStatus"),
  neighborList: document.getElementById("neighborList"),
  explanationText: document.getElementById("explanationText"),
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function niceScore(score) {
  return Number(score).toFixed(3);
}

function niceDistance(value) {
  return Number(value).toFixed(3);
}

function frameCount() {
  return state.analysis?.query?.num_frames ?? 0;
}

function selectedIndex() {
  return Number(els.exampleSelect.value || 0);
}

function scaleTrajectory(points, bounds) {
  const pad = 10;
  const width = 100;
  const height = 100;
  const xRange = Math.max(bounds.x_max - bounds.x_min, 1e-6);
  const yRange = Math.max(bounds.y_max - bounds.y_min, 1e-6);
  return points.map((point) => {
    const xNorm = (point.x - bounds.x_min) / xRange;
    const yNorm = (point.y - bounds.y_min) / yRange;
    return {
      x: pad + xNorm * (width - pad * 2),
      y: height - (pad + yNorm * (height - pad * 2)),
    };
  });
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function getInterpolatedPoint(points, frameIndex, videoTimeRatio) {
  const current = points[frameIndex] || points[points.length - 1];
  const next = points[Math.min(frameIndex + 1, points.length - 1)] || current;
  const t = clamp(videoTimeRatio, 0, 1);
  return {
    x: lerp(current.x, next.x, t),
    y: lerp(current.y, next.y, t),
  };
}

function renderTrajectoryOnSvg(svg, points, currentFrame, videoTimeRatio = 0) {
  if (!points || !points.length) {
    svg.innerHTML = "";
    return;
  }

  const trailPoints = points.slice(0, currentFrame + 1);
  const activePoint = getInterpolatedPoint(points, currentFrame, videoTimeRatio);
  const previousPoint = points[Math.max(currentFrame - 1, 0)] || activePoint;
  const direction = {
    x: activePoint.x - previousPoint.x,
    y: activePoint.y - previousPoint.y,
  };
  const angle = Math.atan2(direction.y, direction.x) * (180 / Math.PI);
  const arrowSize = 2.5;
  const isReverse = svg === els.reverseSvg;
  const arrowPolygon = `
    <polygon
      class="trajectory-arrowhead${isReverse ? " reverse" : ""}"
      points="0,-${arrowSize} ${arrowSize * 1.25},0 0,${arrowSize} -${arrowSize * 0.6},0"
      transform="translate(${activePoint.x.toFixed(2)}, ${activePoint.y.toFixed(2)}) rotate(${Number.isFinite(angle) ? angle : 0})"
    ></polygon>
  `;

  const dots = trailPoints
    .map((point, index) => {
      const age = trailPoints.length - 1 - index;
      const opacity = clamp(1 - age * 0.18, 0.18, 1);
      const radius = index === currentFrame ? 2.4 : clamp(2.0 - age * 0.08, 0.8, 2.0);
      return `<circle class="trajectory-dot${isReverse ? " reverse" : ""}" cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="${radius.toFixed(2)}" opacity="${opacity.toFixed(2)}"></circle>`;
    })
    .join("");

  const arrows =
    currentFrame > 0
      ? `
        <line class="trajectory-arrow" x1="${previousPoint.x.toFixed(2)}" y1="${previousPoint.y.toFixed(2)}" x2="${activePoint.x.toFixed(2)}" y2="${activePoint.y.toFixed(2)}" marker-end="url(#arrowHead)"></line>
      `
      : "";

  svg.innerHTML = `
    <defs>
      <marker id="arrowHead" markerWidth="4" markerHeight="4" refX="3" refY="2" orient="auto">
        <path d="M0,0 L4,2 L0,4 z" fill="rgba(209, 74, 31, 0.65)"></path>
      </marker>
    </defs>
    ${arrows}
    ${dots}
    ${arrowPolygon}
  `;
}

function renderCurrentMode(frameIndex, videoTimeRatio = 0) {
  renderTrajectoryOnSvg(els.trajectorySvg, state.scaledTrajectory, frameIndex, videoTimeRatio);
}

function renderComparison(frameIndex, videoTimeRatio = 0) {
  if (!state.compareMode || !state.reverseAnalysis) return;
  renderTrajectoryOnSvg(els.forwardSvg, state.scaledTrajectory, frameIndex, videoTimeRatio);
  const reverseFrame = Math.max(frameCount() - 1 - frameIndex, 0);
  renderTrajectoryOnSvg(els.reverseSvg, state.reverseTrajectory, reverseFrame, videoTimeRatio);
  els.forwardMeta.textContent = state.analysis?.query?.filename || "";
  els.reverseMeta.textContent = state.reverseAnalysis?.query?.filename || "";
}

function renderNeighbors(currentFrame) {
  const frameNeighbors = state.analysis?.frame_neighbors?.[currentFrame];
  const neighbors = frameNeighbors && frameNeighbors.length ? frameNeighbors : state.analysis?.global_neighbors || [];
  els.neighborList.replaceChildren();
  for (const neighbor of neighbors) {
    const card = document.createElement("article");
    card.className = "neighbor-card";

    const rank = document.createElement("div");
    rank.className = "rank";
    rank.textContent = `# ${neighbor.rank}`;

    const name = document.createElement("div");
    name.className = "name";
    name.textContent = neighbor.label_text || neighbor.filename;

    const filename = document.createElement("div");
    filename.className = "meta-line";
    filename.textContent = neighbor.filename;

    const indexLine = document.createElement("div");
    indexLine.className = "meta-line";
    indexLine.textContent = `Index: ${neighbor.index}`;

    const score = document.createElement("div");
    score.className = "meta-line";
    score.textContent = `Score: ${niceScore(neighbor.score)}`;

    card.append(rank, name, filename, indexLine, score);
    els.neighborList.appendChild(card);
  }
  els.neighborStatus.textContent = `Showing neighbors for frame ${currentFrame + 1} of ${frameCount() || 1}`;
}

function renderFrame(frameIndex, videoTimeRatio = 0) {
  if (!state.analysis) return;
  const totalFrames = frameCount();
  const current = clamp(frameIndex, 0, Math.max(totalFrames - 1, 0));
  state.currentFrame = current;

  const point = state.analysis.trajectory[current] || state.analysis.trajectory[state.analysis.trajectory.length - 1];
  const currentMotion = point?.step_distance || 0;
  let motionLabel = "Frozen latent space";
  if (currentMotion > 0.75) {
    motionLabel = "Motion is changing quickly";
  } else if (currentMotion > 0.25) {
    motionLabel = "Motion is changing";
  } else {
    motionLabel = "Motion is steady";
  }

  els.frameSlider.value = String(current);
  els.frameLabel.textContent = `Frame ${current + 1} / ${totalFrames}`;
  els.motionLabel.textContent = `${motionLabel} · change ${niceDistance(currentMotion)}`;
  renderCurrentMode(current, videoTimeRatio);
  renderComparison(current, videoTimeRatio);
  renderNeighbors(current);

  els.explanationText.textContent = state.compareMode
    ? "The left trail is the original clip. The right trail is the same clip in reverse. If the representation cares about order, the two should not look the same."
    : "The arrow is the current frame embedding. The fading trail shows recent frame embeddings. The arrow shows direction of change, not raw pixels.";
}

function seekVideoToFrame(frameIndex) {
  if (!state.analysis || !els.videoPlayer.duration) return;
  const totalFrames = frameCount();
  if (totalFrames <= 1) return;
  const clamped = clamp(frameIndex, 0, totalFrames - 1);
  const ratio = clamped / (totalFrames - 1);
  els.videoPlayer.currentTime = ratio * els.videoPlayer.duration;
  renderFrame(clamped, 0);
}

function syncFrameFromVideo() {
  if (!state.analysis || !els.videoPlayer.duration) return;
  const totalFrames = frameCount();
  if (totalFrames <= 1) {
    renderFrame(0, 0);
    return;
  }
  const ratio = clamp(els.videoPlayer.currentTime / els.videoPlayer.duration, 0, 1);
  const scaled = ratio * (totalFrames - 1);
  const frameIndex = Math.floor(scaled);
  const localT = scaled - frameIndex;
  renderFrame(frameIndex, localT);
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function populateExamples(examples) {
  els.exampleSelect.replaceChildren();
  for (const example of examples) {
    const option = document.createElement("option");
    option.value = String(example.index);
    option.textContent = `${example.label_text} (${example.video_id})`;
    els.exampleSelect.appendChild(option);
  }
}

async function loadExamples() {
  const payload = await fetchJson("/api/video/examples?limit=12");
  state.examples = payload.examples || [];
  populateExamples(state.examples);
}

async function loadAnalysis(index) {
  els.neighborStatus.textContent = "Finding neighbors...";
  const payload = await fetchJson(`/api/video/analyze?index=${encodeURIComponent(index)}&top_k=5`);
  const reversePayload = await fetchJson(`/api/video/analyze?index=${encodeURIComponent(index)}&top_k=5&reverse=1`);
  state.analysis = payload;
  state.reverseAnalysis = reversePayload;
  state.scaledTrajectory = scaleTrajectory(payload.trajectory, payload.bounds);
  state.reverseTrajectory = scaleTrajectory(reversePayload.trajectory, reversePayload.bounds);
  state.currentFrame = 0;

  const query = payload.query;
  els.clipTitle.textContent = query.filename;
  els.clipLabel.textContent = query.label_text;
  els.clipFrames.textContent = `${query.num_frames} frames`;
  els.videoPlayer.src = query.video_url;
  els.videoPlayer.load();
  els.frameSlider.min = "0";
  els.frameSlider.max = String(Math.max(query.num_frames - 1, 0));
  els.frameSlider.value = "0";

  if (state.compareMode) {
    els.compareSection.classList.remove("hidden");
  } else {
    els.compareSection.classList.add("hidden");
  }
  renderFrame(0, 0);
}

function updatePlayButton() {
  els.playButton.textContent = els.videoPlayer.paused ? "Play" : "Pause";
}

async function init() {
  await loadExamples();
  const initialIndex = state.examples[0]?.index ?? 0;
  els.exampleSelect.value = String(initialIndex);
  await loadAnalysis(initialIndex);

  els.loadButton.addEventListener("click", async () => {
    await loadAnalysis(selectedIndex());
  });

  els.exampleSelect.addEventListener("change", async () => {
    await loadAnalysis(selectedIndex());
  });

  els.playButton.addEventListener("click", async () => {
    if (els.videoPlayer.paused) {
      await els.videoPlayer.play();
    } else {
      els.videoPlayer.pause();
    }
    updatePlayButton();
  });

  els.compareButton.addEventListener("click", () => {
    state.compareMode = !state.compareMode;
    els.compareSection.classList.toggle("hidden", !state.compareMode);
    els.compareButton.classList.toggle("active", state.compareMode);
    els.compareButton.textContent = state.compareMode
      ? "Hide forward / reversed"
      : "Compare forward / reversed";
    if (state.analysis && state.reverseAnalysis) {
      renderFrame(state.currentFrame, 0);
    }
  });

  els.stepBackButton.addEventListener("click", () => {
    seekVideoToFrame(state.currentFrame - 1);
  });

  els.stepForwardButton.addEventListener("click", () => {
    seekVideoToFrame(state.currentFrame + 1);
  });

  els.frameSlider.addEventListener("input", () => {
    seekVideoToFrame(Number(els.frameSlider.value));
  });

  els.videoPlayer.addEventListener("timeupdate", syncFrameFromVideo);
  els.videoPlayer.addEventListener("play", updatePlayButton);
  els.videoPlayer.addEventListener("pause", updatePlayButton);
  els.videoPlayer.addEventListener("ended", () => {
    updatePlayButton();
    renderFrame(frameCount() - 1, 0);
  });
  els.videoPlayer.addEventListener("loadedmetadata", () => {
    updatePlayButton();
    syncFrameFromVideo();
  });
}

init().catch((error) => {
  console.error(error);
  els.neighborStatus.textContent = `Failed to load demo: ${error.message}`;
  els.explanationText.textContent = "The demo could not load its backend data.";
});
