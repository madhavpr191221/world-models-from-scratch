const fileInput = document.getElementById("video-file");
const maskRatio = document.getElementById("mask-ratio");
const maskRatioValue = document.getElementById("mask-ratio-value");
const maskMode = document.getElementById("mask-mode");
const runBtn = document.getElementById("run-btn");
const statusText = document.getElementById("status-text");
const originalVideo = document.getElementById("original-video");
const maskedVideo = document.getElementById("masked-video");
const reconstructedVideo = document.getElementById("reconstructed-video");
const originalSheet = document.getElementById("original-sheet");
const maskedSheet = document.getElementById("masked-sheet");
const reconstructedSheet = document.getElementById("reconstructed-sheet");
const maskSheet = document.getElementById("mask-sheet");
const originalDownload = document.getElementById("original-download");
const maskedDownload = document.getElementById("masked-download");
const reconstructedDownload = document.getElementById("reconstructed-download");

maskRatio.addEventListener("input", () => {
  maskRatioValue.textContent = `${Math.round(Number(maskRatio.value) * 100)}%`;
});

function setStatus(text) {
  statusText.textContent = text;
}

function setMedia(videoEl, imgEl, urlVideo, urlImg) {
  videoEl.src = urlVideo;
  imgEl.src = urlImg;
}

runBtn.addEventListener("click", async () => {
  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Choose a video file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("mask_ratio", maskRatio.value);
  formData.append("mask_mode", maskMode.value);

  setStatus("Uploading and reconstructing...");
  runBtn.disabled = true;
  try {
    const response = await fetch("/api/reconstruct", { method: "POST", body: formData });
    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }
    const data = await response.json();
    setMedia(originalVideo, originalSheet, data.artifacts.original_gif, data.artifacts.original_sheet);
    setMedia(maskedVideo, maskedSheet, data.artifacts.masked_gif, data.artifacts.masked_sheet);
    setMedia(reconstructedVideo, reconstructedSheet, data.artifacts.reconstructed_gif, data.artifacts.reconstructed_sheet);
    maskSheet.src = data.artifacts.mask_sheet;
    originalDownload.href = data.artifacts.original_video;
    maskedDownload.href = data.artifacts.masked_video;
    reconstructedDownload.href = data.artifacts.reconstructed_video;
    const extra = data.metrics?.reconstruction_loss != null ? ` Reconstruction loss ${Number(data.metrics.reconstruction_loss).toFixed(4)}.` : "";
    setStatus(`Done. Run ${data.run_id} saved.${extra}`);
  } catch (err) {
    setStatus(`Reconstruction failed: ${err.message}`);
  } finally {
    runBtn.disabled = false;
  }
});
