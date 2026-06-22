const input = document.getElementById("image-input");
const status = document.getElementById("file-status");
const requestStatus = document.getElementById("request-status");
const previewImage = document.getElementById("preview-image");
const previewCopy = document.getElementById("preview-copy");
const uploadCopy = document.getElementById("upload-copy");
const dropzone = document.getElementById("dropzone");
const neighborResults = document.getElementById("neighbor-results");
const loadingState = document.getElementById("loading-state");

let previewUrl = null;
let activeRequestId = 0;

function setLoading(isLoading) {
  if (!loadingState) {
    return;
  }
  loadingState.hidden = !isLoading;
  loadingState.style.display = isLoading ? "inline-flex" : "none";
}

function setPreview(file) {
  if (!file) {
    status.textContent = "No image selected yet.";
    requestStatus.textContent = "Waiting for upload.";
    previewImage.hidden = true;
    previewImage.removeAttribute("src");
    previewCopy.hidden = false;
    previewCopy.textContent = "The selected image will appear here.";
    uploadCopy.textContent = "Click here or drop an image";
    setLoading(false);
    renderEmptyResults("No results yet.");
    return;
  }

  status.textContent = `Selected: ${file.name}`;
  uploadCopy.textContent = "Image selected";
  previewCopy.hidden = true;
  previewImage.hidden = false;
  previewImage.alt = file.name;

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;

  void runRetrieval(file);
}

function renderEmptyResults(message) {
  if (!neighborResults) {
    return;
  }
  setLoading(false);
  neighborResults.innerHTML = `<p class="result-note">${message}</p>`;
}

function renderResults(items) {
  if (!neighborResults) {
    return;
  }
  setLoading(false);

  if (!items?.length) {
    renderEmptyResults("No neighbors returned.");
    return;
  }

  neighborResults.innerHTML = items
    .map((item) => {
      const rank = item.rank ?? item.position ?? "?";
      const label = item.label ?? "Unknown";
      const score = item.score ?? item.similarity ?? item.sim ?? "";
      const index = item.index ?? item.dataset_index ?? item.idx ?? "";
      const filename = item.filename ?? `sample_${String(index).padStart(4, "0")}.png`;
      const thumbnail = item.thumbnail ?? item.image ?? item.preview ?? "";
      const scoreText = formatScore(score);
      return `
        <article class="neighbor-item">
          <div class="neighbor-thumb">
            ${
              thumbnail
                ? `<img src="${escapeAttribute(thumbnail)}" alt="${escapeAttribute(label)}" />`
                : `<span>${escapeHtml(label).slice(0, 1).toUpperCase()}</span>`
            }
          </div>
          <div>
            <p class="neighbor-rank">#${rank}</p>
            <h4>${escapeHtml(label)}</h4>
            <p class="neighbor-filename">${escapeHtml(filename)}</p>
          </div>
          <div class="neighbor-meta">
            <p>${index !== "" ? `Index: ${index}` : "Index unavailable"}</p>
            <p>${scoreText ? `Score: ${scoreText}` : "Score unavailable"}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

async function runRetrieval(file) {
  if (!requestStatus || !neighborResults) {
    return;
  }

  const requestId = ++activeRequestId;
  requestStatus.textContent = "Sending image to retrieval backend...";
  setLoading(true);
  renderEmptyResults("Loading neighbors...");

  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch("/api/retrieve", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed with ${response.status}`);
    }

    const payload = await response.json();
    if (requestId !== activeRequestId) {
      return;
    }
    requestStatus.textContent = `Retrieved ${payload.neighbors?.length ?? 0} neighbors.`;
    renderResults(payload.neighbors);
  } catch (error) {
    if (requestId !== activeRequestId) {
      return;
    }
    requestStatus.textContent = `Retrieval failed: ${error.message}`;
    renderEmptyResults("No results because the request failed.");
  } finally {
    if (requestId === activeRequestId) {
      setLoading(false);
    }
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#96;");
}

function formatScore(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return "";
  }
  return num.toFixed(3);
}

if (input) {
  input.addEventListener("change", () => setPreview(input.files?.[0]));
}

if (dropzone) {
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("is-dragging");
  });

  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
    const file = event.dataTransfer.files?.[0];
    if (!file || !file.type.startsWith("image/")) {
      return;
    }

    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    input.files = dataTransfer.files;
    setPreview(file);
  });
}
