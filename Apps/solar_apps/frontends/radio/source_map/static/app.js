"use strict";

const byId = (id) => document.getElementById(id);
const els = {};
for (const id of [
  "serverState", "openSectionBtn", "saveFolderBtn", "configInput", "sourcePathInput",
  "outputPathInput", "frequenciesField", "frequenciesInput", "polarizationInput",
  "imageTimeStartInput", "imageTimeEndInput",
  "cmapInput", "rangeModeInput", "fixedRangeFields", "vminInput", "vmaxInput",
  "radioUnitInput", "gaussianInput", "backgroundModeInput", "backgroundDisplayInput",
  "backgroundFitInput", "spectrogramInput", "spectrogramPathField", "spectrogramPathInput",
  "spectrogramUnitField", "spectrogramUnitInput", "advancedInput", "discoverBtn",
  "candidateInput", "generateBtn", "cancelBtn", "generationStatus", "undoBtn", "redoBtn",
  "sequenceStartInput", "sequenceEndInput", "prepareSequenceBtn", "cancelSequenceBtn",
  "sequenceProgressGroup", "sequenceProgress", "previousFrameBtn", "nextFrameBtn",
  "frameIndexInput", "frameCountLabel",
  "resetViewBtn", "zoomInput", "zoomValue", "canvasViewport", "emptyState", "mapCanvas",
  "mapStatus", "coordinateReadout", "artifactWarnings", "clearRoisBtn", "roiList",
  "roiNameInput", "roiColorInput", "roiWidthInput", "roiVisibleInput", "roiLabelInput",
  "updateRoiBtn", "deleteRoiBtn", "rectLeftInput", "rectRightInput", "rectBottomInput",
  "rectTopInput", "addNumericBtn", "downloadPngBtn", "downloadJsonBtn", "saveBundleBtn",
  "exportStatus", "openDialog", "openImageInput", "openSidecarInput", "openRoiInput",
  "openRoiTemplateInput", "openStatus", "openArtifactBtn", "exportSourceInput",
  "exportDirectoryField", "exportDirectoryInput", "exportScopeInput", "exportRangeFields",
  "exportStartInput", "exportEndInput", "exportContentInput", "exportRoiPathField",
  "exportRoiPathInput", "roiTemplateHint", "exportKindInput", "exportDestinationInput",
  "browseExportDestinationBtn", "videoOptions", "exportFpsInput", "exportQualityInput",
  "startExportBtn", "cancelExportBtn", "exportProgressGroup", "exportProgress"
]) els[id] = byId(id);

const state = {
  appConfig: null,
  discoveryId: null,
  candidates: [],
  jobId: null,
  sequenceJobId: null,
  preparedSequenceJobId: null,
  sequenceArtifacts: [],
  sequenceStartFrame: 1,
  sequenceEndFrame: 1,
  sequencePosition: -1,
  exportJobId: null,
  artifact: null,
  image: null,
  rois: [],
  selectedRoiId: null,
  history: [],
  future: [],
  draft: null,
  draftStart: null,
  drawingPanel: null,
  panStart: null,
  zoom: 100,
  fitScale: 1,
  clientId: `source-map-${Date.now()}-${Math.random().toString(16).slice(2)}`,
  heartbeatTimer: null,
  stopOnClose: true
};

const nativeExtensions = {
  sourcePathInput: [".fits", ".fit", ".fts"],
  spectrogramPathInput: [".fits", ".fit", ".fts"],
  openImageInput: [".png"],
  openSidecarInput: [".json"],
  openRoiInput: [".json"],
  exportRoiPathInput: [".json"],
  exportDirectoryInput: [],
  exportDestinationInput: [],
};

class ApiError extends Error {
  constructor(message, status, payload = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
    this.code = payload.code || "";
  }
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let payload;
  try { payload = await response.json(); } catch (_error) { payload = {ok: false, error: response.statusText}; }
  if (!response.ok || !payload.ok) {
    throw new ApiError(payload.error || `Request failed (${response.status})`, response.status, payload);
  }
  return payload;
}

function jsonRequest(body, method = "POST") {
  return {method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
}

function setStatus(element, text, error = false) {
  element.textContent = text || "";
  element.classList.toggle("error", error);
}

function selectedRadio(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value;
}

function requestPayload() {
  let advanced;
  try { advanced = JSON.parse(els.advancedInput.value || "{}"); }
  catch (error) { throw new Error(`Advanced JSON: ${error.message}`); }
  return {
    config: els.configInput.value.trim(),
    mode: selectedRadio("mode"),
    source_path: els.sourcePathInput.value.trim(),
    output_dir: els.outputPathInput.value.trim(),
    frequencies: els.frequenciesInput.value.trim(),
    polarization: els.polarizationInput.value,
    image_time_start: els.imageTimeStartInput.value.trim() || null,
    image_time_end: els.imageTimeEndInput.value.trim() || null,
    cmap: els.cmapInput.value,
    color_range_mode: els.rangeModeInput.value,
    fixed_vmin: els.vminInput.value,
    fixed_vmax: els.vmaxInput.value,
    radio_unit: els.radioUnitInput.value.trim(),
    gaussian_overlay: els.gaussianInput.checked,
    background_mode: els.backgroundModeInput.value,
    background_display: els.backgroundDisplayInput.checked,
    background_fit: els.backgroundFitInput.checked,
    spectrogram_panel: els.spectrogramInput.checked,
    spectrogram_path: els.spectrogramPathInput.value.trim(),
    spectrogram_unit: els.spectrogramUnitInput.value.trim(),
    advanced
  };
}

async function discover() {
  setStatus(els.generationStatus, "Scanning source data...");
  els.discoverBtn.disabled = true;
  try {
    const requested = requestPayload();
    const payload = await api("/api/source-maps/discover", jsonRequest(requested));
    if (requested.image_time_start !== (els.imageTimeStartInput.value.trim() || null) ||
        requested.image_time_end !== (els.imageTimeEndInput.value.trim() || null)) {
      throw new Error("Image UTC interval changed during scanning; scan source data again.");
    }
    state.discoveryId = payload.discovery_id;
    state.candidates = payload.candidates;
    els.candidateInput.replaceChildren();
    for (const candidate of state.candidates) {
      const option = document.createElement("option");
      option.value = candidate.id;
      const time = candidate.observation_time ? candidate.observation_time.replace("T", " ") : "time unknown";
      const frequencies = candidate.frequencies_mhz?.length ? `${candidate.frequencies_mhz.join(", ")} MHz` : "frequency unknown";
      option.textContent = `${candidate.title} | ${time} | ${frequencies} | ${candidate.pairing_status}`;
      els.candidateInput.append(option);
    }
    els.candidateInput.disabled = false;
    els.generateBtn.disabled = false;
    els.prepareSequenceBtn.disabled = state.candidates.length === 0;
    els.sequenceStartInput.disabled = state.candidates.length === 0;
    els.sequenceEndInput.disabled = state.candidates.length === 0;
    els.sequenceStartInput.max = String(Math.max(1, state.candidates.length));
    els.sequenceEndInput.max = String(Math.max(1, state.candidates.length));
    els.sequenceStartInput.value = "1";
    els.sequenceEndInput.value = String(Math.max(1, state.candidates.length));
    clearPreparedSequence();
    setStatus(els.generationStatus, `${state.candidates.length} candidate(s)`);
  } catch (error) {
    setStatus(els.generationStatus, error.message, true);
  } finally {
    els.discoverBtn.disabled = false;
  }
}

for (const field of [els.imageTimeStartInput, els.imageTimeEndInput]) {
  field.addEventListener("input", () => {
    state.discoveryId = null;
    state.candidates = [];
    els.candidateInput.replaceChildren();
    els.generateBtn.disabled = true;
    els.prepareSequenceBtn.disabled = true;
    clearPreparedSequence();
    setStatus(els.generationStatus, "Image UTC interval changed; scan source data again.");
  });
}

async function generate() {
  if (!state.discoveryId || !els.candidateInput.value) return;
  els.generateBtn.disabled = true;
  els.prepareSequenceBtn.disabled = true;
  els.discoverBtn.disabled = true;
  els.cancelBtn.disabled = false;
  setStatus(els.generationStatus, "Starting render...");
  try {
    const payload = await api("/api/render-jobs", jsonRequest({
      discovery_id: state.discoveryId,
      candidate_id: els.candidateInput.value
    }));
    state.jobId = payload.job.id;
    await pollJob(state.jobId);
  } catch (error) {
    setStatus(els.generationStatus, error.message, true);
  } finally {
    state.jobId = null;
    els.discoverBtn.disabled = false;
    els.generateBtn.disabled = !state.discoveryId;
    els.prepareSequenceBtn.disabled = !state.discoveryId;
    els.cancelBtn.disabled = true;
  }
}

async function pollJob(jobId) {
  while (state.jobId === jobId) {
    const payload = await api(`/api/render-jobs/${jobId}`);
    const job = payload.job;
    if (job.status === "completed") {
      setStatus(els.generationStatus, "Render completed");
      const artifactPayload = await api(`/api/artifacts/${job.artifact_id}/metadata`);
      await loadArtifact(artifactPayload.artifact);
      return;
    }
    if (job.status === "failed") throw new Error(job.error || job.stderr || "Render failed");
    if (job.status === "canceled") {
      setStatus(els.generationStatus, "Render canceled");
      return;
    }
    setStatus(els.generationStatus, job.status === "canceling" ? "Canceling..." : "Rendering...");
    await new Promise((resolve) => window.setTimeout(resolve, 800));
  }
}

async function cancelJob() {
  if (!state.jobId) return;
  try {
    await api(`/api/render-jobs/${state.jobId}`, {method: "DELETE"});
    setStatus(els.generationStatus, "Render canceled");
  } catch (error) {
    setStatus(els.generationStatus, error.message, true);
  }
}

function readFrameRange(startInput, endInput, maximum = null) {
  const start = Number(startInput.value);
  const end = Number(endInput.value);
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start) {
    throw new Error("Frame range must use positive, 1-based inclusive indexes");
  }
  if (maximum != null && end > maximum) {
    throw new Error(`End frame cannot exceed ${maximum}`);
  }
  return {start, end};
}

function clearPreparedSequence() {
  state.sequenceJobId = null;
  state.preparedSequenceJobId = null;
  state.sequenceArtifacts = [];
  state.sequencePosition = -1;
  state.sequenceStartFrame = 1;
  state.sequenceEndFrame = 1;
  els.sequenceProgressGroup.hidden = true;
  els.sequenceProgress.value = 0;
  els.previousFrameBtn.disabled = true;
  els.nextFrameBtn.disabled = true;
  els.frameIndexInput.disabled = true;
  els.frameIndexInput.value = "1";
  els.frameCountLabel.textContent = "/ 1";
  if (els.exportSourceInput.value === "sequence_job") els.exportSourceInput.value = "artifact";
  updateExportControls();
}

async function prepareSequence() {
  if (!state.discoveryId) return;
  let range;
  try {
    range = readFrameRange(els.sequenceStartInput, els.sequenceEndInput, state.candidates.length);
  } catch (error) {
    setStatus(els.generationStatus, error.message, true);
    return;
  }
  els.prepareSequenceBtn.disabled = true;
  els.generateBtn.disabled = true;
  els.discoverBtn.disabled = true;
  els.cancelSequenceBtn.disabled = false;
  els.sequenceProgressGroup.hidden = false;
  els.sequenceProgress.max = range.end - range.start + 1;
  els.sequenceProgress.value = 0;
  setStatus(els.generationStatus, `Preparing frames ${range.start}–${range.end}...`);
  try {
    const payload = await api("/api/sequence-jobs", jsonRequest({
      discovery_id: state.discoveryId,
      start_frame: range.start,
      end_frame: range.end,
    }));
    state.sequenceJobId = payload.job.id;
    state.sequenceStartFrame = range.start;
    state.sequenceEndFrame = range.end;
    await pollSequenceJob(state.sequenceJobId);
  } catch (error) {
    setStatus(els.generationStatus, error.message, true);
  } finally {
    state.sequenceJobId = null;
    els.discoverBtn.disabled = false;
    els.generateBtn.disabled = !state.discoveryId;
    els.prepareSequenceBtn.disabled = !state.discoveryId;
    els.cancelSequenceBtn.disabled = true;
  }
}

async function pollSequenceJob(jobId) {
  while (state.sequenceJobId === jobId) {
    const payload = await api(`/api/sequence-jobs/${jobId}`);
    const job = payload.job;
    const total = Math.max(1, Number(job.total) || (state.sequenceEndFrame - state.sequenceStartFrame + 1));
    const completed = Math.max(0, Number(job.completed) || 0);
    els.sequenceProgress.max = total;
    els.sequenceProgress.value = Math.min(completed, total);
    if (job.status === "completed") {
      const entries = job.artifact_ids || job.artifacts || job.result?.artifact_ids || job.result?.artifacts || [];
      if (!entries.length) throw new Error("Sequence completed without artifacts");
      state.sequenceArtifacts = entries;
      state.preparedSequenceJobId = jobId;
      state.sequencePosition = 0;
      els.sequenceProgress.value = total;
      els.sequenceProgressGroup.hidden = true;
      setStatus(els.generationStatus, `Prepared ${entries.length} frame(s)`);
      els.exportSourceInput.value = "sequence_job";
      els.exportScopeInput.value = entries.length > 1 ? "range" : "current";
      els.exportStartInput.value = String(state.sequenceStartFrame);
      els.exportEndInput.value = String(state.sequenceEndFrame);
      await loadSequenceFrame(0);
      updateExportControls();
      return;
    }
    if (job.status === "failed") throw new Error(job.error || "Sequence preparation failed");
    if (job.status === "canceled") {
      setStatus(els.generationStatus, "Sequence preparation canceled");
      els.sequenceProgressGroup.hidden = true;
      return;
    }
    const current = job.current_frame ?? job.current_index;
    const detail = current == null ? "" : ` (frame ${current})`;
    setStatus(els.generationStatus, job.status === "canceling"
      ? "Canceling sequence..."
      : `Preparing ${completed}/${total}${detail}`);
    await new Promise((resolve) => window.setTimeout(resolve, 800));
  }
}

async function cancelSequenceJob() {
  if (!state.sequenceJobId) return;
  try {
    await api(`/api/sequence-jobs/${state.sequenceJobId}`, {method: "DELETE"});
    setStatus(els.generationStatus, "Canceling sequence...");
  } catch (error) {
    setStatus(els.generationStatus, error.message, true);
  }
}

async function artifactFromSequenceEntry(entry) {
  if (entry && typeof entry === "object") {
    const artifact = entry.artifact || entry;
    if (artifact.metadata && artifact.image_url) return artifact;
    entry = entry.artifact_id || entry.id;
  }
  if (!entry) throw new Error("Prepared sequence contains an invalid artifact reference");
  const payload = await api(`/api/artifacts/${encodeURIComponent(entry)}/metadata`);
  return payload.artifact;
}

async function loadSequenceFrame(position) {
  if (!state.sequenceArtifacts.length) return;
  const bounded = Math.max(0, Math.min(Number(position), state.sequenceArtifacts.length - 1));
  const artifact = await artifactFromSequenceEntry(state.sequenceArtifacts[bounded]);
  await loadArtifact(artifact, {preserveRois: true});
  state.sequencePosition = bounded;
  const absoluteFrame = state.sequenceStartFrame + bounded;
  els.frameIndexInput.min = String(state.sequenceStartFrame);
  els.frameIndexInput.max = String(state.sequenceEndFrame);
  els.frameIndexInput.value = String(absoluteFrame);
  els.frameIndexInput.disabled = false;
  els.frameCountLabel.textContent = `/ ${state.sequenceEndFrame}`;
  els.previousFrameBtn.disabled = bounded === 0;
  els.nextFrameBtn.disabled = bounded >= state.sequenceArtifacts.length - 1;
  updateExportControls();
}

async function navigateSequenceFrame(delta = 0) {
  if (!state.sequenceArtifacts.length) return;
  const requested = delta
    ? state.sequencePosition + delta
    : Number(els.frameIndexInput.value) - state.sequenceStartFrame;
  try {
    if (!Number.isInteger(requested) || requested < 0 || requested >= state.sequenceArtifacts.length) {
      throw new Error(`Frame must be from ${state.sequenceStartFrame} to ${state.sequenceEndFrame}`);
    }
    await loadSequenceFrame(requested);
  } catch (error) {
    setStatus(els.generationStatus, error.message, true);
  }
}

async function openArtifact() {
  setStatus(els.openStatus, "Opening...");
  try {
    const payload = await api("/api/artifacts/open", jsonRequest({
      image_path: els.openImageInput.value.trim(),
      sidecar_path: els.openSidecarInput.value.trim(),
      roi_set_path: els.openRoiInput.value.trim(),
      roi_template_mode: els.openRoiTemplateInput.checked,
    }));
    await loadArtifact(payload.artifact, {preserveRois: false});
    clearPreparedSequence();
    els.openDialog.close();
    setStatus(els.openStatus, "");
  } catch (error) {
    setStatus(els.openStatus, error.message, true);
  }
}

async function loadArtifact(artifact, {preserveRois = false} = {}) {
  const image = new Image();
  image.decoding = "async";
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error("Could not load the generated PNG"));
    image.src = `${artifact.image_url}?v=${Date.now()}`;
  });
  const metadata = artifact.metadata;
  if (image.naturalWidth !== metadata.image.width || image.naturalHeight !== metadata.image.height) {
    throw new Error("Loaded PNG dimensions do not match sidecar metadata");
  }
  state.artifact = artifact;
  state.image = image;
  if (!preserveRois || !state.rois.length) {
    state.rois = clone(artifact.roi_set?.rois || []);
    state.selectedRoiId = state.rois[0]?.id || null;
  } else if (!state.rois.some((roi) => roi.id === state.selectedRoiId)) {
    state.selectedRoiId = state.rois[0]?.id || null;
  }
  if (!preserveRois) {
    state.history = [];
    state.future = [];
  }
  state.draft = null;
  state.draftStart = null;
  els.mapCanvas.width = image.naturalWidth;
  els.mapCanvas.height = image.naturalHeight;
  els.mapCanvas.hidden = false;
  els.emptyState.hidden = true;
  els.outputPathInput.value ||= "";
  setStatus(els.mapStatus, `${metadata.image.filename} | ${metadata.mode} | ${metadata.panels.length} panel(s)`);
  const warnings = metadata.warnings || [];
  els.artifactWarnings.hidden = warnings.length === 0;
  els.artifactWarnings.textContent = warnings.join(" ");
  state.zoom = 100;
  els.zoomInput.value = "100";
  refreshCanvasSize();
  refreshRoiUi();
  draw();
}

function clone(value) { return JSON.parse(JSON.stringify(value)); }

function commitRois(nextRois, selectedId) {
  state.history.push({rois: clone(state.rois), selectedRoiId: state.selectedRoiId});
  if (state.history.length > 100) state.history.shift();
  state.future = [];
  state.rois = clone(nextRois);
  state.selectedRoiId = selectedId;
  refreshRoiUi();
  draw();
}

function undo() {
  const previous = state.history.pop();
  if (!previous) return;
  state.future.push({rois: clone(state.rois), selectedRoiId: state.selectedRoiId});
  state.rois = previous.rois;
  state.selectedRoiId = previous.selectedRoiId;
  refreshRoiUi();
  draw();
}

function redo() {
  const next = state.future.pop();
  if (!next) return;
  state.history.push({rois: clone(state.rois), selectedRoiId: state.selectedRoiId});
  state.rois = next.rois;
  state.selectedRoiId = next.selectedRoiId;
  refreshRoiUi();
  draw();
}

function selectedRoi() { return state.rois.find((roi) => roi.id === state.selectedRoiId) || null; }

function refreshRoiUi() {
  els.roiList.replaceChildren();
  if (!state.rois.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No regions";
    els.roiList.append(empty);
  }
  for (const roi of state.rois) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `roi-item${roi.id === state.selectedRoiId ? " active" : ""}`;
    const swatch = document.createElement("span");
    swatch.className = "roi-swatch";
    swatch.style.background = roi.style.color;
    const name = document.createElement("span");
    name.textContent = roi.name;
    const kind = document.createElement("span");
    kind.className = "roi-kind";
    kind.textContent = roi.visible ? roi.type : "hidden";
    button.append(swatch, name, kind);
    button.addEventListener("click", () => {
      state.selectedRoiId = roi.id;
      refreshRoiUi();
      draw();
    });
    els.roiList.append(button);
  }
  const roi = selectedRoi();
  for (const element of [els.roiNameInput, els.roiColorInput, els.roiWidthInput, els.roiVisibleInput, els.roiLabelInput, els.updateRoiBtn, els.deleteRoiBtn]) {
    element.disabled = !roi;
  }
  if (roi) {
    els.roiNameInput.value = roi.name;
    els.roiColorInput.value = roi.style.color;
    els.roiWidthInput.value = String(roi.style.line_width);
    els.roiVisibleInput.checked = roi.visible;
    els.roiLabelInput.checked = roi.style.show_label;
    if (roi.type === "rectangle") {
      els.rectLeftInput.value = roi.geometry.left;
      els.rectRightInput.value = roi.geometry.right;
      els.rectBottomInput.value = roi.geometry.bottom;
      els.rectTopInput.value = roi.geometry.top;
    }
  }
  const hasArtifact = Boolean(state.artifact);
  const hasRois = state.rois.length > 0;
  els.addNumericBtn.disabled = !hasArtifact;
  els.clearRoisBtn.disabled = !hasRois;
  els.downloadPngBtn.disabled = !hasArtifact;
  els.downloadJsonBtn.disabled = !hasArtifact;
  els.saveBundleBtn.disabled = !hasArtifact;
  els.saveFolderBtn.disabled = !hasArtifact;
  els.undoBtn.disabled = state.history.length === 0;
  els.redoBtn.disabled = state.future.length === 0;
  updateExportControls();
}

function uniqueRoiName(base = "ROI") {
  const used = new Set(state.rois.map((roi) => roi.name.toLowerCase()));
  let index = state.rois.length + 1;
  let name = `${base} ${index}`;
  while (used.has(name.toLowerCase())) name = `${base} ${++index}`;
  return name;
}

function newRoi(type, geometry) {
  return {
    id: `roi-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    name: uniqueRoiName(),
    type,
    geometry,
    visible: true,
    style: {color: els.roiColorInput.value || "#00d4ff", line_width: 3, show_label: true}
  };
}

function addRoi(roi) { commitRois([...state.rois, roi], roi.id); }

function updateSelectedRoi() {
  const roi = selectedRoi();
  if (!roi) return;
  const name = els.roiNameInput.value.trim();
  if (!name) return setStatus(els.exportStatus, "ROI name is required", true);
  if (state.rois.some((item) => item.id !== roi.id && item.name.toLowerCase() === name.toLowerCase())) {
    return setStatus(els.exportStatus, "ROI names must be unique", true);
  }
  const width = Number(els.roiWidthInput.value);
  if (!Number.isFinite(width) || width < 1 || width > 12) return setStatus(els.exportStatus, "Line width must be between 1 and 12", true);
  const next = state.rois.map((item) => item.id === roi.id ? {
    ...item,
    name,
    visible: els.roiVisibleInput.checked,
    style: {color: els.roiColorInput.value, line_width: width, show_label: els.roiLabelInput.checked}
  } : item);
  commitRois(next, roi.id);
  setStatus(els.exportStatus, "");
}

function deleteSelectedRoi() {
  const roi = selectedRoi();
  if (!roi) return;
  const next = state.rois.filter((item) => item.id !== roi.id);
  commitRois(next, next[0]?.id || null);
}

function addNumericRectangle() {
  const values = [els.rectLeftInput, els.rectBottomInput, els.rectRightInput, els.rectTopInput].map((input) => Number(input.value));
  if (!values.every(Number.isFinite)) return setStatus(els.exportStatus, "Rectangle coordinates must be finite", true);
  const [x1, y1, x2, y2] = values;
  const geometry = {left: Math.min(x1, x2), bottom: Math.min(y1, y2), right: Math.max(x1, x2), top: Math.max(y1, y2)};
  if (geometry.left === geometry.right || geometry.bottom === geometry.top) return setStatus(els.exportStatus, "Rectangle must have positive area", true);
  addRoi(newRoi("rectangle", geometry));
  setStatus(els.exportStatus, "");
}

function panelBounds(panel) {
  const [left, top, right, bottom] = panel.bbox_normalized;
  return {left: left * els.mapCanvas.width, top: top * els.mapCanvas.height, right: right * els.mapCanvas.width, bottom: bottom * els.mapCanvas.height};
}

function dataToPixel(panel, point) {
  const box = panelBounds(panel);
  const [x0, x1] = panel.xlim_arcsec;
  const [y0, y1] = panel.ylim_arcsec;
  const fx = (point[0] - x0) / (x1 - x0);
  const fy = (point[1] - y0) / (y1 - y0);
  return [box.left + fx * (box.right - box.left), box.bottom - fy * (box.bottom - box.top)];
}

function pixelToData(panel, point) {
  const box = panelBounds(panel);
  const fx = (point[0] - box.left) / (box.right - box.left);
  const fy = (box.bottom - point[1]) / (box.bottom - box.top);
  const [x0, x1] = panel.xlim_arcsec;
  const [y0, y1] = panel.ylim_arcsec;
  return [x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)];
}

function panelAt(point) {
  if (!state.artifact) return null;
  return state.artifact.metadata.panels.find((panel) => {
    const box = panelBounds(panel);
    return point[0] >= box.left && point[0] <= box.right && point[1] >= box.top && point[1] <= box.bottom;
  }) || null;
}

function drawScene(ctx, includeDraft = true) {
  if (!state.image) return;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(state.image, 0, 0, ctx.canvas.width, ctx.canvas.height);
  for (const roi of state.rois) if (roi.visible) drawRoiAcrossPanels(ctx, roi, roi.id === state.selectedRoiId);
  if (includeDraft && state.draft) drawRoiAcrossPanels(ctx, state.draft, true, true);
}

function draw() {
  if (!state.image) return;
  drawScene(els.mapCanvas.getContext("2d"), true);
}

function drawRoiAcrossPanels(ctx, roi, selected = false, draft = false) {
  for (const panel of state.artifact.metadata.panels) {
    const box = panelBounds(panel);
    ctx.save();
    ctx.beginPath();
    ctx.rect(box.left, box.top, box.right - box.left, box.bottom - box.top);
    ctx.clip();
    ctx.beginPath();
    let labelPoint;
    if (roi.type === "rectangle") {
      const first = dataToPixel(panel, [roi.geometry.left, roi.geometry.top]);
      const second = dataToPixel(panel, [roi.geometry.right, roi.geometry.bottom]);
      ctx.rect(Math.min(first[0], second[0]), Math.min(first[1], second[1]), Math.abs(second[0] - first[0]), Math.abs(second[1] - first[1]));
      labelPoint = first;
    } else {
      const points = roi.geometry.points.map((point) => dataToPixel(panel, point));
      if (points.length) {
        ctx.moveTo(points[0][0], points[0][1]);
        for (const point of points.slice(1)) ctx.lineTo(point[0], point[1]);
        if (!draft) ctx.closePath();
        labelPoint = points[0];
      }
    }
    ctx.strokeStyle = roi.style.color;
    ctx.lineWidth = selected ? Number(roi.style.line_width) + 2 : Number(roi.style.line_width);
    ctx.setLineDash(draft ? [10, 7] : []);
    ctx.stroke();
    if (!draft && roi.style.show_label && labelPoint) drawRoiLabel(ctx, roi.name, labelPoint, roi.style.color, box);
    ctx.restore();
  }
}

function drawRoiLabel(ctx, text, point, color, box) {
  ctx.font = "600 16px Segoe UI, Arial, sans-serif";
  const width = ctx.measureText(text).width + 12;
  const x = Math.min(Math.max(point[0], box.left), Math.max(box.left, box.right - width));
  const y = Math.min(Math.max(point[1] - 24, box.top), Math.max(box.top, box.bottom - 22));
  ctx.fillStyle = "rgba(18, 25, 29, 0.78)";
  ctx.fillRect(x, y, width, 22);
  ctx.fillStyle = color;
  ctx.fillText(text, x + 6, y + 16);
}

function canvasPoint(event) {
  const rect = els.mapCanvas.getBoundingClientRect();
  return [
    (event.clientX - rect.left) * els.mapCanvas.width / rect.width,
    (event.clientY - rect.top) * els.mapCanvas.height / rect.height
  ];
}

function pointerDown(event) {
  if (!state.image) return;
  const tool = selectedRadio("tool");
  if (tool === "pan") {
    state.panStart = {x: event.clientX, y: event.clientY, left: els.canvasViewport.scrollLeft, top: els.canvasViewport.scrollTop};
    els.mapCanvas.setPointerCapture(event.pointerId);
    return;
  }
  const pixel = canvasPoint(event);
  const panel = panelAt(pixel);
  if (!panel) return;
  const point = pixelToData(panel, pixel);
  state.drawingPanel = panel;
  state.draftStart = point;
  if (tool === "rectangle") {
    state.draft = {name: "Draft", type: "rectangle", geometry: {left: point[0], right: point[0], bottom: point[1], top: point[1]}, style: {color: "#ffb000", line_width: 3, show_label: false}};
  } else {
    state.draft = {name: "Draft", type: "lasso", geometry: {points: [point]}, style: {color: "#ffb000", line_width: 3, show_label: false}};
  }
  els.mapCanvas.setPointerCapture(event.pointerId);
  draw();
}

function pointerMove(event) {
  if (!state.image) return;
  if (state.panStart) {
    els.canvasViewport.scrollLeft = state.panStart.left - (event.clientX - state.panStart.x);
    els.canvasViewport.scrollTop = state.panStart.top - (event.clientY - state.panStart.y);
    return;
  }
  const pixel = canvasPoint(event);
  const hoverPanel = panelAt(pixel);
  if (hoverPanel) {
    const [x, y] = pixelToData(hoverPanel, pixel);
    els.coordinateReadout.textContent = `HPLN ${x.toFixed(2)} / HPLT ${y.toFixed(2)} arcsec`;
  } else {
    els.coordinateReadout.textContent = "HPLN -- / HPLT --";
  }
  if (!state.draft || !state.drawingPanel) return;
  const point = pixelToData(state.drawingPanel, pixel);
  if (state.draft.type === "rectangle") {
    const start = state.draftStart;
    state.draft.geometry = {left: Math.min(start[0], point[0]), right: Math.max(start[0], point[0]), bottom: Math.min(start[1], point[1]), top: Math.max(start[1], point[1])};
  } else {
    const points = state.draft.geometry.points;
    const previous = points[points.length - 1];
    if (Math.hypot(point[0] - previous[0], point[1] - previous[1]) > 0.5) points.push(point);
  }
  draw();
}

function pointerUp(event) {
  if (state.panStart) {
    state.panStart = null;
    if (els.mapCanvas.hasPointerCapture(event.pointerId)) els.mapCanvas.releasePointerCapture(event.pointerId);
    return;
  }
  if (!state.draft) return;
  const draft = state.draft;
  state.draft = null;
  state.draftStart = null;
  state.drawingPanel = null;
  if (els.mapCanvas.hasPointerCapture(event.pointerId)) els.mapCanvas.releasePointerCapture(event.pointerId);
  if (draft.type === "rectangle") {
    if (draft.geometry.left !== draft.geometry.right && draft.geometry.bottom !== draft.geometry.top) addRoi(newRoi("rectangle", draft.geometry));
  } else if (new Set(draft.geometry.points.map((point) => `${point[0].toFixed(8)},${point[1].toFixed(8)}`)).size >= 3) {
    addRoi(newRoi("lasso", {points: draft.geometry.points}));
  }
  draw();
}

function refreshCanvasSize() {
  if (!state.image) return;
  const availableWidth = Math.max(200, els.canvasViewport.clientWidth - 32);
  const availableHeight = Math.max(200, els.canvasViewport.clientHeight - 32);
  state.fitScale = Math.min(availableWidth / state.image.naturalWidth, availableHeight / state.image.naturalHeight, 1);
  const scale = state.fitScale * state.zoom / 100;
  els.mapCanvas.style.width = `${Math.max(1, Math.round(state.image.naturalWidth * scale))}px`;
  els.mapCanvas.style.height = `${Math.max(1, Math.round(state.image.naturalHeight * scale))}px`;
  els.zoomValue.textContent = `${state.zoom}%`;
}

function roiPayload() {
  return {
    schema_version: 1,
    coordinate_system: "HPLN/HPLT arcsec",
    image_sha256: state.artifact.metadata.image.sha256,
    rois: clone(state.rois)
  };
}

function exportCanvas() {
  const canvas = document.createElement("canvas");
  canvas.width = state.image.naturalWidth;
  canvas.height = state.image.naturalHeight;
  drawScene(canvas.getContext("2d"), false);
  return canvas;
}

function exportStem() {
  return (state.artifact.suggested_filename || state.artifact.metadata.image.filename).replace(/\.png$/i, "");
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadPng() {
  const filename = state.artifact.annotated_suggested_filename || `${exportStem()}_annotated.png`;
  exportCanvas().toBlob((blob) => blob && downloadBlob(blob, filename), "image/png");
}

function downloadJson() {
  downloadBlob(new Blob([`${JSON.stringify(roiPayload(), null, 2)}\n`], {type: "application/json"}), `${exportStem()}.roi-set.json`);
}

async function saveBundle() {
  if (!state.artifact) return;
  const outputDir = els.outputPathInput.value.trim();
  if (!outputDir) return setStatus(els.exportStatus, "Output folder is required", true);
  setStatus(els.exportStatus, "Saving bundle...");
  try {
    const blob = await new Promise((resolve, reject) => exportCanvas().toBlob((value) => value ? resolve(value) : reject(new Error("PNG export failed")), "image/png"));
    const form = new FormData();
    form.append("artifact_id", state.artifact.id);
    form.append("output_dir", outputDir);
    form.append("roi_set", JSON.stringify(roiPayload()));
    form.append(
      "annotated_image",
      blob,
      state.artifact.annotated_suggested_filename || `${exportStem()}_annotated.png`,
    );
    const payload = await api("/api/exports/save", {method: "POST", body: form});
    setStatus(els.exportStatus, `${payload.annotated_image_path}\n${payload.roi_set_path}`);
  } catch (error) {
    setStatus(els.exportStatus, error.message, true);
  }
}

function updateExportControls() {
  const source = els.exportSourceInput.value;
  const kind = els.exportKindInput.value;
  const withRoi = els.exportContentInput.value === "roi";
  const sequenceOption = els.exportSourceInput.querySelector('option[value="sequence_job"]');
  if (sequenceOption) sequenceOption.disabled = !state.preparedSequenceJobId;
  els.exportDirectoryField.hidden = source !== "directory";
  els.exportRangeFields.hidden = source === "artifact";
  els.exportScopeInput.disabled = kind === "image" || source === "artifact";
  if (kind === "image" || source === "artifact") els.exportScopeInput.value = "current";
  const scope = els.exportScopeInput.value;
  const endLabel = els.exportEndInput.closest("label");
  if (endLabel) endLabel.hidden = scope !== "range";
  els.videoOptions.hidden = kind !== "video";
  els.exportRoiPathField.hidden = !withRoi;
  els.roiTemplateHint.hidden = !withRoi;
  const sourceReady = source === "artifact"
    ? Boolean(state.artifact)
    : source === "sequence_job"
      ? Boolean(state.preparedSequenceJobId && state.sequenceArtifacts.length)
      : Boolean(els.exportDirectoryInput.value.trim());
  const roiReady = !withRoi || state.rois.length > 0 || Boolean(els.exportRoiPathInput.value.trim());
  els.startExportBtn.disabled = Boolean(state.exportJobId)
    || !sourceReady
    || !roiReady
    || !els.exportDestinationInput.value.trim();
}

function currentAbsoluteFrame() {
  return state.sequencePosition >= 0
    ? state.sequenceStartFrame + state.sequencePosition
    : 1;
}

function exportRequestPayload(overwrite = false) {
  const sourceType = els.exportSourceInput.value;
  const exportKind = els.exportKindInput.value;
  const scope = (exportKind === "image" || sourceType === "artifact")
    ? "current"
    : els.exportScopeInput.value;
  const payload = {
    source_type: sourceType,
    scope,
    export_kind: exportKind,
    content: els.exportContentInput.value,
    destination: els.exportDestinationInput.value.trim(),
    fps: Number(els.exportFpsInput.value),
    quality: els.exportQualityInput.value,
    overwrite,
  };
  if (!payload.destination) throw new Error("Export destination is required");
  if (sourceType === "artifact") {
    if (!state.artifact) throw new Error("Load or generate an artifact first");
    payload.artifact_id = state.artifact.id;
    payload.current_frame = 1;
  } else if (sourceType === "sequence_job") {
    if (!state.preparedSequenceJobId) throw new Error("Prepare a sequence first");
    payload.sequence_job_id = state.preparedSequenceJobId;
  } else {
    payload.source_directory = els.exportDirectoryInput.value.trim();
    if (!payload.source_directory) throw new Error("External Source Map directory is required");
  }
  if (scope === "range") {
    const maximum = sourceType === "sequence_job" ? state.sequenceEndFrame : null;
    const range = readFrameRange(els.exportStartInput, els.exportEndInput, maximum);
    payload.start_frame = range.start;
    payload.end_frame = range.end;
  } else if (sourceType !== "artifact") {
    const frame = sourceType === "sequence_job"
      ? currentAbsoluteFrame()
      : Number(els.exportStartInput.value || 1);
    if (!Number.isInteger(frame) || frame < 1) throw new Error("Current frame must be a positive integer");
    payload.current_frame = frame;
  }
  if (payload.content === "roi") {
    const roiPath = els.exportRoiPathInput.value.trim();
    if (roiPath) {
      payload.roi_set_path = roiPath;
      payload.roi_template_mode = true;
    } else if (state.artifact && state.rois.length) {
      payload.roi_set = roiPayload();
      payload.roi_template_mode = sourceType !== "artifact";
    } else {
      throw new Error("Draw an ROI or select a historical ROI JSON template");
    }
  }
  if (exportKind === "video" && (!Number.isInteger(payload.fps) || payload.fps < 1 || payload.fps > 60)) {
    throw new Error("FPS must be an integer from 1 to 60");
  }
  return payload;
}

async function startExport(overwrite = false) {
  let requestPayload;
  try {
    requestPayload = exportRequestPayload(overwrite);
  } catch (error) {
    setStatus(els.exportStatus, error.message, true);
    return;
  }
  els.startExportBtn.disabled = true;
  els.cancelExportBtn.disabled = false;
  els.exportProgressGroup.hidden = false;
  els.exportProgress.value = 0;
  setStatus(els.exportStatus, "Starting export...");
  try {
    const payload = await api("/api/export-jobs", jsonRequest(requestPayload));
    state.exportJobId = payload.job.id;
    await pollExportJob(state.exportJobId, requestPayload);
  } catch (error) {
    const conflict = error.status === 409 || error.code === "target_exists";
    if (conflict && !overwrite && window.confirm(`${error.message}\n\nReplace the existing target?`)) {
      state.exportJobId = null;
      await startExport(true);
      return;
    }
    setStatus(els.exportStatus, error.message, true);
  } finally {
    state.exportJobId = null;
    els.cancelExportBtn.disabled = true;
    updateExportControls();
  }
}

function exportResultText(job) {
  const result = job.result || {};
  const lines = [];
  if (result.path) lines.push(result.path);
  if (result.output_dir) lines.push(result.output_dir);
  if (result.manifest_path) lines.push(`Manifest: ${result.manifest_path}`);
  if (Array.isArray(result.files)) lines.push(`${result.files.length} file(s)`);
  if (result.media) {
    const media = result.media;
    lines.push(`${media.frame_count ?? job.total ?? "?"} frames, ${media.fps ?? els.exportFpsInput.value} FPS`);
  }
  if (Array.isArray(job.warnings) && job.warnings.length) lines.push(`Warnings: ${job.warnings.join(" ")}`);
  return lines.join("\n") || "Export completed";
}

async function pollExportJob(jobId, originalRequest) {
  while (state.exportJobId === jobId) {
    const payload = await api(`/api/export-jobs/${jobId}`);
    const job = payload.job;
    const total = Math.max(1, Number(job.total) || 1);
    const completed = Math.max(0, Number(job.completed) || 0);
    els.exportProgress.max = total;
    els.exportProgress.value = Math.min(completed, total);
    if (job.status === "completed") {
      els.exportProgress.value = total;
      els.exportProgressGroup.hidden = true;
      setStatus(els.exportStatus, exportResultText(job));
      return;
    }
    if (job.status === "failed") {
      const error = new ApiError(job.error || "Export failed", 500, job);
      if ((job.code === "target_exists" || job.error_code === "target_exists") && !originalRequest.overwrite
          && window.confirm(`${error.message}\n\nReplace the existing target?`)) {
        state.exportJobId = null;
        await startExport(true);
        return;
      }
      throw error;
    }
    if (job.status === "canceled") {
      els.exportProgressGroup.hidden = true;
      setStatus(els.exportStatus, "Export canceled");
      return;
    }
    const current = job.current_index == null ? "" : ` (frame ${job.current_index})`;
    setStatus(els.exportStatus, job.status === "canceling"
      ? "Canceling export..."
      : `Exporting ${completed}/${total}${current}`);
    await new Promise((resolve) => window.setTimeout(resolve, 800));
  }
}

async function cancelExportJob() {
  if (!state.exportJobId) return;
  try {
    await api(`/api/export-jobs/${state.exportJobId}`, {method: "DELETE"});
    setStatus(els.exportStatus, "Canceling export...");
  } catch (error) {
    setStatus(els.exportStatus, error.message, true);
  }
}

function nativeStatusFor(targetId) {
  if (targetId.startsWith("open")) return els.openStatus;
  if (targetId.startsWith("export")) return els.exportStatus;
  return els.generationStatus;
}

async function openBrowser(targetId, mode = "open_file", options = {}) {
  const target = byId(targetId);
  const status = nativeStatusFor(targetId);
  const kind = mode === "select_directory" ? "folder" : mode === "save_file" ? "save" : "file";
  setStatus(status, `Opening Windows ${kind} dialog...`);
  try {
    if (!window.SolarNativePathDialog) throw new Error("Windows native path dialog is unavailable");
    const paths = await window.SolarNativePathDialog.select({
      mode,
      initialPath: target.value.trim(),
      title: options.title || (mode === "select_directory" ? "Select local folder" : mode === "save_file" ? "Save local file" : "Select local file"),
      extensions: options.extensions || nativeExtensions[targetId] || [],
      defaultSuffix: options.defaultSuffix || "",
      operation: options.operation || targetId.replace(/Input$/, "").replace(/[^A-Za-z0-9_.-]/g, "-") || "browse",
      field: targetId,
    });
    if (!paths.length) {
      setStatus(status, "Selection cancelled.");
      return;
    }
    target.value = paths[0];
    target.dispatchEvent(new Event("change", {bubbles: true}));
    if (targetId.startsWith("export")) updateExportControls();
    setStatus(status, paths[0]);
  } catch (error) {
    setStatus(status, error.message, true);
  }
}

async function browseExportDestination() {
  const kind = els.exportKindInput.value;
  if (kind === "image_sequence") {
    await openBrowser("exportDestinationInput", "select_directory", {title: "Select PNG sequence output folder", operation: "export-image-sequence"});
  } else if (kind === "video") {
    await openBrowser("exportDestinationInput", "save_file", {title: "Save Source Map video", extensions: [".mp4"], defaultSuffix: ".mp4", operation: "export-video"});
  } else {
    await openBrowser("exportDestinationInput", "save_file", {title: "Save Source Map image", extensions: [".png"], defaultSuffix: ".png", operation: "export-image"});
  }
}

function updateConditionalFields() {
  els.frequenciesField.hidden = selectedRadio("mode") !== "multi_band";
  els.fixedRangeFields.hidden = els.rangeModeInput.value !== "fixed";
  els.spectrogramPathField.hidden = !els.spectrogramInput.checked;
  els.spectrogramUnitField.hidden = !els.spectrogramInput.checked;
  updateExportControls();
}

async function initializeLifecycle() {
  try {
    const response = await fetch("/api/client-config");
    const config = await response.json();
    state.stopOnClose = Boolean(config.stop_on_close);
    await sendHeartbeat();
    state.heartbeatTimer = window.setInterval(sendHeartbeat, Number(config.heartbeat_interval_ms) || 5000);
  } catch (_error) { /* Health display already reports connectivity. */ }
}

async function sendHeartbeat() {
  try { await fetch("/api/client-heartbeat", jsonRequest({client_id: state.clientId})); } catch (_error) { /* Local shutdown race. */ }
}

function closeClient() {
  const body = JSON.stringify({client_id: state.clientId, stop_on_close: state.stopOnClose});
  navigator.sendBeacon("/api/client-close", new Blob([body], {type: "application/json"}));
}

async function initialize() {
  bindEvents();
  updateConditionalFields();
  try {
    state.appConfig = await api("/api/config");
    els.configInput.value = state.appConfig.default_config;
    for (const cmap of state.appConfig.colormaps) {
      const option = document.createElement("option");
      option.value = cmap;
      option.textContent = cmap;
      els.cmapInput.append(option);
    }
    els.cmapInput.value = "hot";
    els.outputPathInput.value = state.appConfig.allowed_roots[0] || "";
    els.serverState.textContent = "Local service online";
    els.serverState.classList.add("online");
  } catch (error) {
    els.serverState.textContent = error.message;
  }
  await initializeLifecycle();
}

function bindEvents() {
  els.discoverBtn.addEventListener("click", discover);
  els.generateBtn.addEventListener("click", generate);
  els.cancelBtn.addEventListener("click", cancelJob);
  els.prepareSequenceBtn.addEventListener("click", prepareSequence);
  els.cancelSequenceBtn.addEventListener("click", cancelSequenceJob);
  els.previousFrameBtn.addEventListener("click", () => navigateSequenceFrame(-1));
  els.nextFrameBtn.addEventListener("click", () => navigateSequenceFrame(1));
  els.frameIndexInput.addEventListener("change", () => navigateSequenceFrame(0));
  els.openSectionBtn.addEventListener("click", () => els.openDialog.showModal());
  els.openArtifactBtn.addEventListener("click", openArtifact);
  els.undoBtn.addEventListener("click", undo);
  els.redoBtn.addEventListener("click", redo);
  els.resetViewBtn.addEventListener("click", () => {
    state.zoom = 100;
    els.zoomInput.value = "100";
    refreshCanvasSize();
    els.canvasViewport.scrollTo({left: 0, top: 0});
  });
  els.zoomInput.addEventListener("input", () => {
    state.zoom = Number(els.zoomInput.value);
    refreshCanvasSize();
  });
  els.updateRoiBtn.addEventListener("click", updateSelectedRoi);
  els.deleteRoiBtn.addEventListener("click", deleteSelectedRoi);
  els.addNumericBtn.addEventListener("click", addNumericRectangle);
  els.clearRoisBtn.addEventListener("click", () => commitRois([], null));
  els.downloadPngBtn.addEventListener("click", downloadPng);
  els.downloadJsonBtn.addEventListener("click", downloadJson);
  els.saveBundleBtn.addEventListener("click", saveBundle);
  els.saveFolderBtn.addEventListener("click", saveBundle);
  els.startExportBtn.addEventListener("click", () => startExport(false));
  els.cancelExportBtn.addEventListener("click", cancelExportJob);
  els.browseExportDestinationBtn.addEventListener("click", browseExportDestination);
  els.mapCanvas.addEventListener("pointerdown", pointerDown);
  els.mapCanvas.addEventListener("pointermove", pointerMove);
  els.mapCanvas.addEventListener("pointerup", pointerUp);
  els.mapCanvas.addEventListener("pointercancel", pointerUp);
  els.rangeModeInput.addEventListener("change", updateConditionalFields);
  els.spectrogramInput.addEventListener("change", updateConditionalFields);
  for (const element of [
    els.exportSourceInput, els.exportScopeInput, els.exportContentInput, els.exportKindInput,
    els.exportDirectoryInput, els.exportRoiPathInput, els.exportDestinationInput,
  ]) {
    element.addEventListener(element.tagName === "SELECT" ? "change" : "input", updateExportControls);
  }
  els.exportKindInput.addEventListener("change", () => {
    els.exportDestinationInput.value = "";
    updateExportControls();
  });
  for (const input of document.querySelectorAll("input[name='mode']")) input.addEventListener("change", updateConditionalFields);
  for (const button of document.querySelectorAll("[data-browse]")) {
    button.addEventListener("click", () => openBrowser(
      button.dataset.browse,
      button.dataset.mode || (button.dataset.folder === "true" ? "select_directory" : "open_file"),
    ));
  }
  window.addEventListener("resize", refreshCanvasSize);
  window.addEventListener("beforeunload", closeClient);
  window.addEventListener("keydown", (event) => {
    const activeTag = document.activeElement?.tagName;
    if (activeTag === "INPUT" || activeTag === "TEXTAREA" || activeTag === "SELECT") return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      event.shiftKey ? redo() : undo();
    } else if (event.key === "Delete") deleteSelectedRoi();
  });
}

initialize();
