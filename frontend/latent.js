const state = {
  examples: [],
  analysis: null,
  currentStep: 0,
  projectionMethod: "pca",
  rafId: null,
  isLoaded: false,
};

const els = {
  exampleSelect: document.getElementById("exampleSelect"),
  methodSelect: document.getElementById("methodSelect"),
  loadButton: document.getElementById("loadButton"),
  playButton: document.getElementById("playButton"),
  stepBackButton: document.getElementById("stepBackButton"),
  stepForwardButton: document.getElementById("stepForwardButton"),
  statusPill: document.getElementById("statusPill"),
  clipTitle: document.getElementById("clipTitle"),
  clipMeta: document.getElementById("clipMeta"),
  clipFrames: document.getElementById("clipFrames"),
  videoPlayer: document.getElementById("videoPlayer"),
  stepSlider: document.getElementById("stepSlider"),
  stepLabel: document.getElementById("stepLabel"),
  phaseLabel: document.getElementById("phaseLabel"),
  projectionSvg: document.getElementById("projectionSvg"),
  projectionWrap: document.querySelector(".projection-wrap"),
  projectionEmpty: document.getElementById("projectionEmpty"),
  projectionSummary: document.getElementById("projectionSummary"),
  projectionSummarySecondary: document.getElementById("projectionSummarySecondary"),
  latentMse: document.getElementById("latentMse"),
  normalizedLatentMse: document.getElementById("normalizedLatentMse"),
  cosineSimilarity: document.getElementById("cosineSimilarity"),
  repeatLastMse: document.getElementById("repeatLastMse"),
  meanContextMse: document.getElementById("meanContextMse"),
  backgroundCount: document.getElementById("backgroundCount"),
  explanationText: document.getElementById("explanationText"),
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function nice(value, digits = 4) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return "-";
  }
  return num.toFixed(digits);
}

function totalSteps() {
  if (!state.analysis) return 0;
  return Number(state.analysis.query.context_steps || 0) + Number(state.analysis.query.future_steps || 0);
}

function selectedIndex() {
  return Number(els.exampleSelect.value || 0);
}

function selectedMethod() {
  return els.methodSelect.value || "pca";
}

function fetchJson(url) {
  return fetch(url).then((response) => {
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    return response.json();
  });
}

function populateExamples(examples) {
  els.exampleSelect.replaceChildren();
  for (const example of examples) {
    const option = document.createElement("option");
    option.value = String(example.index);
    option.textContent = `${example.video_id} (${example.sample_index})`;
    els.exampleSelect.appendChild(option);
  }
}

function setMetric(el, value, digits = 4) {
  if (!el) return;
  el.textContent = nice(value, digits);
}

function setStatus(text) {
  if (els.statusPill) {
    els.statusPill.textContent = text;
  }
}

function setPlaybackEnabled(enabled) {
  els.playButton.disabled = !enabled;
  els.stepBackButton.disabled = !enabled;
  els.stepForwardButton.disabled = !enabled;
  els.stepSlider.disabled = !enabled;
}

function scalePoints(points, bounds) {
  const pad = 10;
  const width = 100;
  const height = 100;
  const xRange = Math.max(bounds.x_max - bounds.x_min, 1e-6);
  const yRange = Math.max(bounds.y_max - bounds.y_min, 1e-6);
  return points.map((point) => ({
    x: pad + ((point.x - bounds.x_min) / xRange) * (width - pad * 2),
    y: height - (pad + ((point.y - bounds.y_min) / yRange) * (height - pad * 2)),
    raw: point,
  }));
}

function pointPath(points) {
  if (!points.length) return "";
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");
}

function circleMarkup(point, className, radius = 1.2) {
  return `<circle class="${className}" cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="${radius}" />`;
}

function interpolatePoint(points, index, ratio) {
  const current = points[index] || points[points.length - 1];
  const next = points[Math.min(index + 1, points.length - 1)] || current;
  const t = clamp(ratio, 0, 1);
  return {
    x: current.x + (next.x - current.x) * t,
    y: current.y + (next.y - current.y) * t,
  };
}

function renderEmptyProjection() {
  els.projectionSvg.innerHTML = "";
  if (els.projectionWrap) {
    els.projectionWrap.classList.remove("is-loaded");
  }
  if (els.projectionEmpty) {
    els.projectionEmpty.hidden = false;
  }
}

function renderProjection(currentStep = 0, stepRatio = 0) {
  if (!state.analysis) {
    renderEmptyProjection();
    return;
  }

  if (els.projectionWrap) {
    els.projectionWrap.classList.add("is-loaded");
  }
  if (els.projectionEmpty) {
    els.projectionEmpty.hidden = true;
  }

  const bounds = state.analysis.bounds;
  const background = scalePoints(state.analysis.background_points || [], bounds);
  const context = scalePoints(state.analysis.context_trajectory || [], bounds);
  const futureTrue = scalePoints(state.analysis.future_true_trajectory || [], bounds);
  const futurePred = scalePoints(state.analysis.future_pred_trajectory || [], bounds);

  const contextSteps = context.length;
  const futureSteps = futureTrue.length;
  const futureIndex = Math.max(currentStep - contextSteps, 0);
  const phase = currentStep < contextSteps ? "context" : "future";

  const activeContext = currentStep < contextSteps
    ? interpolatePoint(context, currentStep, stepRatio)
    : context[context.length - 1];
  const activeTrue = currentStep < contextSteps
    ? null
    : interpolatePoint(futureTrue, Math.min(futureIndex, futureTrue.length - 1), stepRatio);
  const activePred = currentStep < contextSteps
    ? null
    : interpolatePoint(futurePred, Math.min(futureIndex, futurePred.length - 1), stepRatio);

  const boundaryPoint = context[context.length - 1] || activeContext;
  const backgroundMarkup = background
    .map((point) => circleMarkup(point, "latent-background-point", 0.55))
    .join("");
  const contextPath = pointPath(context);
  const futureTruePath = pointPath([boundaryPoint, ...futureTrue]);
  const futurePredPath = pointPath([boundaryPoint, ...futurePred]);

  const activeMarkup = [];
  if (phase === "context" && activeContext) {
    activeMarkup.push(circleMarkup(activeContext, "latent-active-context", 2.0));
  } else {
    if (activeTrue) {
      activeMarkup.push(circleMarkup(activeTrue, "latent-active-true", 2.0));
    }
    if (activePred) {
      activeMarkup.push(circleMarkup(activePred, "latent-active-pred", 2.0));
    }
  }

  els.projectionSvg.innerHTML = `
    ${backgroundMarkup}
    <path class="latent-context-line" d="${contextPath}"></path>
    <path class="latent-true-line" d="${futureTruePath}"></path>
    <path class="latent-pred-line" d="${futurePredPath}"></path>
    ${boundaryPoint ? circleMarkup(boundaryPoint, "latent-boundary", 1.8) : ""}
    ${activeMarkup.join("")}
  `;

  const phaseLabel = phase === "context"
    ? `Observed context step ${currentStep + 1} / ${contextSteps}`
    : `Forecast step ${futureIndex + 1} / ${futureSteps}`;
  els.stepLabel.textContent = `Step ${currentStep + 1} / ${totalSteps()}`;
  els.phaseLabel.textContent = phaseLabel;
}

function renderMetrics() {
  if (!state.analysis) return;
  const metrics = state.analysis.metrics || {};
  const baselines = state.analysis.baseline_metrics || {};
  setMetric(els.latentMse, metrics.latent_mse);
  setMetric(els.normalizedLatentMse, metrics.normalized_latent_mse);
  setMetric(els.cosineSimilarity, metrics.cosine_similarity);
  setMetric(els.repeatLastMse, baselines.repeat_last?.latent_mse);
  setMetric(els.meanContextMse, baselines.mean_context?.latent_mse);
  els.backgroundCount.textContent = String((state.analysis.background_points || []).length);
}

function renderHeader() {
  if (!state.analysis) return;
  const query = state.analysis.query;
  els.clipTitle.textContent = query.filename;
  els.clipMeta.textContent = query.video_id;
  els.clipFrames.textContent = `${query.context_steps + query.future_steps} latent steps`;
  els.projectionSummary.textContent = `${state.analysis.projection_method.toUpperCase()} · ${query.latent_dim}-D to 2-D`;
  els.projectionSummarySecondary.textContent = `${query.context_steps} context steps, ${query.future_steps} future steps`;
  els.explanationText.textContent = state.analysis.projection_method === "tsne"
    ? "t-SNE is a local projection. It is useful for inspection, but PCA is usually better for quick live playback and stable comparison."
    : "The background cloud is built from clip-level latent summaries. The orange trajectory is the observed context; green is the true future; red is the model forecast.";
}

function renderPlaceholderState() {
  renderEmptyProjection();
  els.clipTitle.textContent = "Select a clip to begin";
  els.clipMeta.textContent = "No clip loaded";
  els.clipFrames.textContent = "0 latent steps";
  els.projectionSummary.textContent = "Awaiting clip";
  if (els.projectionSummarySecondary) {
    els.projectionSummarySecondary.textContent = "Load a clip to see scores";
  }
  setStatus("Ready to load");
  setPlaybackEnabled(false);
  els.stepLabel.textContent = "Waiting for clip";
  els.phaseLabel.textContent = "Choose a clip, then load it";
}

function renderAll(step = 0, ratio = 0) {
  if (!state.analysis) return;
  const maxStep = Math.max(totalSteps() - 1, 0);
  const currentStep = clamp(step, 0, maxStep);
  state.currentStep = currentStep;
  els.stepSlider.value = String(currentStep);
  renderProjection(currentStep, ratio);
}

function seekStep(step) {
  if (!state.analysis) return;
  const maxStep = Math.max(totalSteps() - 1, 0);
  const current = clamp(step, 0, maxStep);
  state.currentStep = current;
  els.stepSlider.value = String(current);
  if (els.videoPlayer.duration && maxStep > 0) {
    els.videoPlayer.currentTime = (current / maxStep) * els.videoPlayer.duration;
  }
  renderProjection(current, 0);
}

function syncFromVideo() {
  if (!state.analysis || !els.videoPlayer.duration) return;
  const maxStep = Math.max(totalSteps() - 1, 0);
  if (maxStep <= 0) {
    renderProjection(0, 0);
    return;
  }
  const scaled = clamp((els.videoPlayer.currentTime / els.videoPlayer.duration) * maxStep, 0, maxStep);
  const currentStep = Math.floor(scaled);
  const ratio = scaled - currentStep;
  state.currentStep = currentStep;
  els.stepSlider.value = String(currentStep);
  renderProjection(currentStep, ratio);
}

function updatePlayButton() {
  els.playButton.textContent = els.videoPlayer.paused ? "Play and predict" : "Pause playback";
}

function schedulePlaybackSync() {
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
  const tick = () => {
    if (els.videoPlayer.paused || els.videoPlayer.ended) {
      state.rafId = null;
      return;
    }
    syncFromVideo();
    state.rafId = requestAnimationFrame(tick);
  };
  if (!els.videoPlayer.paused) {
    state.rafId = requestAnimationFrame(tick);
  }
}

async function loadExamples() {
  const payload = await fetchJson("/api/latent/examples?limit=12");
  state.examples = payload.examples || [];
  populateExamples(state.examples);
}

async function loadAnalysis(index) {
  const method = selectedMethod();
  setStatus(`Loading ${method.toUpperCase()} projection...`);
  const payload = await fetchJson(`/api/latent/analyze?index=${encodeURIComponent(index)}&method=${encodeURIComponent(method)}&background_sample_size=512&seed=0`);
  state.analysis = payload;
  state.isLoaded = true;
  renderHeader();
  renderMetrics();
  els.videoPlayer.src = payload.query.video_url;
  els.videoPlayer.load();
  els.stepSlider.min = "0";
  els.stepSlider.max = String(Math.max(totalSteps() - 1, 0));
  els.stepSlider.value = "0";
  state.currentStep = 0;
  setPlaybackEnabled(true);
  renderAll(0, 0);
  updatePlayButton();
  setStatus(`Loaded ${payload.query.filename}`);
}

async function init() {
  renderPlaceholderState();
  await loadExamples();
  const initialIndex = state.examples[0]?.index ?? 0;
  els.exampleSelect.value = String(initialIndex);
  setStatus("Ready to load");

  els.loadButton.addEventListener("click", async () => {
    await loadAnalysis(selectedIndex());
  });

  els.exampleSelect.addEventListener("change", () => {
    if (state.analysis) {
      setPlaybackEnabled(false);
      setStatus("Clip changed. Reload to update the map.");
      return;
    }
    setStatus("Clip selected. Press load to project.");
  });

  els.methodSelect.addEventListener("change", () => {
    state.projectionMethod = selectedMethod();
    if (state.analysis) {
      setPlaybackEnabled(false);
      setStatus(`Projection changed to ${state.projectionMethod.toUpperCase()}. Reload to update the map.`);
      return;
    }
    setStatus(`Projection set to ${state.projectionMethod.toUpperCase()}. Press load to project.`);
  });

  els.playButton.addEventListener("click", async () => {
    if (els.videoPlayer.paused) {
      await els.videoPlayer.play();
    } else {
      els.videoPlayer.pause();
    }
    updatePlayButton();
    schedulePlaybackSync();
  });

  els.stepBackButton.addEventListener("click", () => {
    seekStep(state.currentStep - 1);
  });

  els.stepForwardButton.addEventListener("click", () => {
    seekStep(state.currentStep + 1);
  });

  els.stepSlider.addEventListener("input", () => {
    seekStep(Number(els.stepSlider.value));
  });

  els.videoPlayer.addEventListener("timeupdate", syncFromVideo);
  els.videoPlayer.addEventListener("play", () => {
    updatePlayButton();
    schedulePlaybackSync();
  });
  els.videoPlayer.addEventListener("pause", () => {
    updatePlayButton();
    schedulePlaybackSync();
  });
  els.videoPlayer.addEventListener("ended", () => {
    updatePlayButton();
    schedulePlaybackSync();
    renderProjection(Math.max(totalSteps() - 1, 0), 0);
  });
  els.videoPlayer.addEventListener("loadedmetadata", () => {
    updatePlayButton();
    syncFromVideo();
  });
}

init().catch((error) => {
  console.error(error);
  setStatus("Failed to load demo");
  els.phaseLabel.textContent = `Failed to load demo: ${error.message}`;
  els.explanationText.textContent = "The latent projection browser could not load its backend data.";
});

