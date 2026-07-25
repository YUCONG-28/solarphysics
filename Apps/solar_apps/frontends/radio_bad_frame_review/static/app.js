"use strict";

const API = "/api";
const FRAME_PAGE_SIZE = 100;

function createClientId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return "client-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
}

const state = {
  config: null,
  discovery: null,
  review: null,
  reviewView: "candidates",
  selectedCandidateId: null,
  selectedFrameId: null,
  selectedFrameIndex: null,
  framePage: null,
  browserPath: null,
  clientId: createClientId(),
  serverStopOnClose: true,
  heartbeatTimer: null,
  scanTimer: null,
  scanStartedAt: null,
};

const $ = (selector, root = document) => root.querySelector(selector);

async function requestJson(url, options = {}) {
  const init = {...options};
  if (init.body && typeof init.body !== "string") {
    init.headers = {"Content-Type": "application/json", ...(init.headers || {})};
    init.body = JSON.stringify(init.body);
  }
  const response = await fetch(url, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    const error = new Error(payload.error || response.status + " " + response.statusText);
    error.stale = Boolean(payload.stale);
    throw error;
  }
  return payload;
}

function setServiceStatus(label, status) {
  const element = $("#service-status");
  element.textContent = label;
  element.dataset.state = status;
}

function setScanMessage(message, kind = "") {
  const element = $("#scan-message");
  element.textContent = message;
  element.dataset.kind = kind;
}

function selectedValues(containerId) {
  return [...document.querySelectorAll("#" + containerId + " input[type='checkbox']:checked")].map((input) => input.value);
}

function setFrequencyActionState(disabled) {
  $("#frequency-select-all").disabled = disabled;
  $("#frequency-clear-all").disabled = disabled;
}

function setFrequencySelection(checked) {
  const inputs = document.querySelectorAll("#frequency-options input[type='checkbox']");
  for (const input of inputs) input.checked = checked;
  updateScanEstimate();
}

function selectedFrameEstimate() {
  if (!state.discovery) return 0;
  const frequencies = new Set(selectedValues("frequency-options").map(Number));
  const polarizations = new Set(selectedValues("polarization-options"));
  const start = Math.max(0, Number($("#start-index").value || 0));
  const endText = $("#end-index").value.trim();
  const end = endText === "" ? null : Number(endText);
  let total = 0;
  for (const band of state.discovery.bands || []) {
    if (!frequencies.has(Number(band.frequency_mhz))) continue;
    for (const polarization of band.polarizations || []) {
      if (!polarizations.has(polarization.name)) continue;
      const count = Number(polarization.file_count || 0);
      const stop = end === null ? count : Math.min(Math.max(end, 0), count);
      total += Math.max(stop - Math.min(start, count), 0);
    }
  }
  return total;
}

function updateScanEstimate() {
  const estimate = selectedFrameEstimate();
  const output = $("#scan-estimate");
  if (!state.discovery?.bands?.length) {
    output.textContent = "Discover bands to estimate the scan size.";
    return;
  }
  const scope = $("#review-scope").value === "all_scanned" ? "all-frame browsing" : "candidate review";
  output.textContent = estimate.toLocaleString() + " FITS frame(s) selected for " + scope + ".";
}

function updatePreviewDisplayControls() {
  const fixed = $("#preview-range-mode").value === "fixed";
  const lower = $("#preview-pmin").value || "50";
  const upper = $("#preview-pmax").value || "99.7";
  const help = $("#preview-display-help");
  help.dataset.kind = "";
  const transformLabel = $("#preview-transform").value === "linear"
    ? "Linear raw uses FITS intensity and BUNIT. "
    : "Robust asinh preserves faint artifacts. ";
  help.textContent = fixed
    ? transformLabel + "Fixed reuses one P" + lower + "–P" + upper + " range sampled across the current review for this frequency."
    : transformLabel + "Auto calculates an independent P" + lower + "–P" + upper + " range for each FITS file.";
}

function previewDisplayParameters() {
  const params = new URLSearchParams({
    cmap: $("#preview-cmap").value,
    transform: $("#preview-transform").value,
    range_mode: $("#preview-range-mode").value,
  });
  const pminText = $("#preview-pmin").value.trim();
  const pmaxText = $("#preview-pmax").value.trim();
  const pmin = Number(pminText);
  const pmax = Number(pmaxText);
  if (!pminText || !pmaxText || !Number.isFinite(pmin) || !Number.isFinite(pmax) || pmin < 0 || pmax > 100 || pmin >= pmax) {
    const help = $("#preview-display-help");
    help.dataset.kind = "error";
    help.textContent = "Percentiles must satisfy 0 ≤ Lower < Upper ≤ 100.";
    return null;
  }
  params.set("pmin", String(pmin));
  params.set("pmax", String(pmax));
  return params;
}

function previewLoadingState(fallback) {
  return $("#preview-range-mode").value === "fixed"
    ? "Scanning fixed per-frequency intensity range"
    : fallback;
}

function previewImageUrl(path) {
  const params = previewDisplayParameters();
  if (!params) return null;
  params.set("v", String(state.review?.updated_at || "preview"));
  return path + "?" + params.toString();
}

function refreshActivePreview() {
  updatePreviewDisplayControls();
  if (!state.review) return;
  if (state.reviewView === "frames") {
    const frame = frameFromCurrentPage(state.selectedFrameIndex);
    if (frame) renderFramePreview(frame);
    return;
  }
  if (state.selectedCandidateId) renderCandidatePreview();
}

function configurePreviewDisplay(config) {
  const display = config.preview_display || {};
  const defaults = display.defaults || {};
  const select = $("#preview-cmap");
  const current = select.value;
  const colormaps = Array.isArray(display.colormaps) ? display.colormaps : [];
  if (colormaps.length) {
    select.replaceChildren(...colormaps.map((value) => new Option(value, value)));
    select.value = colormaps.includes(current) ? current : (defaults.cmap || "coolwarm");
  }
  const percentileDefaults = display.percentile_defaults || [50, 99.7];
  if (!$("#preview-pmin").value) $("#preview-pmin").value = String(percentileDefaults[0]);
  if (!$("#preview-pmax").value) $("#preview-pmax").value = String(percentileDefaults[1]);
  updatePreviewDisplayControls();
}

function renderDiscovery(discovery) {
  state.discovery = discovery;
  const frequencyContainer = $("#frequency-options");
  const polarizationContainer = $("#polarization-options");
  frequencyContainer.replaceChildren();
  polarizationContainer.replaceChildren();
  const bands = discovery.bands || [];
  if (!bands.length) {
    frequencyContainer.textContent = "No radio bands found";
    polarizationContainer.textContent = "No polarizations found";
    frequencyContainer.className = "option-grid empty-options";
    polarizationContainer.className = "option-grid empty-options";
    setFrequencyActionState(true);
    $("#scan-button").disabled = true;
    updateScanEstimate();
    return;
  }

  frequencyContainer.className = "option-grid";
  for (const band of bands) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = String(band.frequency_mhz);
    input.checked = true;
    input.addEventListener("change", updateScanEstimate);
    const count = (band.polarizations || []).reduce((sum, item) => sum + Number(item.file_count || 0), 0);
    label.append(input, document.createTextNode(band.label + " (" + count.toLocaleString() + ")"));
    frequencyContainer.append(label);
  }

  const polarizations = [...new Set(bands.flatMap((band) => (band.polarizations || []).map((item) => item.name)))].sort();
  polarizationContainer.className = "option-grid";
  for (const polarization of polarizations) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = polarization;
    input.checked = true;
    input.addEventListener("change", updateScanEstimate);
    label.append(input, document.createTextNode(polarization));
    polarizationContainer.append(label);
  }
  setFrequencyActionState(false);
  $("#scan-button").disabled = false;
  updateScanEstimate();
}

async function discoverBands() {
  const root = $("#root-input").value.trim();
  if (!root) return setScanMessage("Choose a radio root first.", "error");
  setScanMessage("Discovering radio bands...");
  $("#discover-button").disabled = true;
  try {
    const payload = await requestJson(API + "/discover", {method: "POST", body: {root}});
    renderDiscovery(payload);
    setScanMessage(
      payload.bands.length ? payload.bands.length + " frequency bands found." : "No <frequency>MHz/RR or LL folders were found.",
      payload.bands.length ? "success" : "error"
    );
  } catch (error) {
    setScanMessage(error.message, "error");
  } finally {
    $("#discover-button").disabled = false;
  }
}

function formatDuration(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  return Math.floor(whole / 60) + ":" + String(whole % 60).padStart(2, "0");
}

function startScanProgress(total) {
  state.scanStartedAt = Date.now();
  const progress = $("#scan-progress");
  progress.hidden = false;
  $("#scan-button").setAttribute("aria-busy", "true");
  $("#scan-progress-label").textContent =
    "Scanning " + total.toLocaleString() + " selected FITS frames; exact progress is unavailable.";
  const tick = () => {
    $("#scan-elapsed").textContent = formatDuration((Date.now() - state.scanStartedAt) / 1000);
  };
  tick();
  state.scanTimer = window.setInterval(tick, 1000);
  setServiceStatus("Scanning", "loading");
}

function stopScanProgress() {
  if (state.scanTimer) window.clearInterval(state.scanTimer);
  state.scanTimer = null;
  state.scanStartedAt = null;
  $("#scan-progress").hidden = true;
  $("#scan-button").removeAttribute("aria-busy");
  if (state.config) setServiceStatus("Local", "ready");
}

async function createReview() {
  const frequencies = selectedValues("frequency-options").map(Number);
  const polarizations = selectedValues("polarization-options");
  if (!frequencies.length || !polarizations.length) {
    return setScanMessage("Select at least one frequency and polarization.", "error");
  }
  const endValue = $("#end-index").value.trim();
  const total = selectedFrameEstimate();
  if (total < 1) return setScanMessage("The selected frame range contains no FITS files.", "error");
  const body = {
    root: $("#root-input").value.trim(),
    frequencies_mhz: frequencies,
    polarizations,
    start_index: Number($("#start-index").value || 0),
    end_index: endValue === "" ? null : Number(endValue),
    candidate_strategy: $("#candidate-strategy").value,
    sample_count: Number($("#sample-count").value || 1200),
    review_scope: $("#review-scope").value,
  };
  $("#scan-button").disabled = true;
  setScanMessage("Scan started. The page remains responsive while FITS files are analysed.");
  startScanProgress(total);
  try {
    const payload = await requestJson(API + "/reviews", {method: "POST", body});
    await openReview(payload.review);
    await loadReviewList(state.review.review_id);
    const summary = state.review.summary || {};
    const candidateCount = Number(summary.candidate_count || 0);
    let message =
      "Scan complete: " + Number(summary.scanned_file_count || 0).toLocaleString() +
      " frame(s) scanned; " + candidateCount.toLocaleString() + " automatic review candidate(s).";
    if (candidateCount === 0) {
      message += state.review.input?.review_scope === "all_scanned"
        ? " Continue the all-frame visual audit; zero automatic candidates does not prove every frame is good."
        : " Zero automatic candidates is a valid result; create an all-frame review for manual coverage.";
    }
    setScanMessage(message, "success");
  } catch (error) {
    setScanMessage(error.message, "error");
  } finally {
    stopScanProgress();
    $("#scan-button").disabled = !state.discovery?.bands?.length;
  }
}

function setCount(id, value) {
  $(id).textContent = Number(value || 0).toLocaleString();
}

function formatReason(reason) {
  const value = Array.isArray(reason) ? reason.join("; ") : String(reason || "unknown");
  return value.replaceAll(";", "; ").replaceAll(":", ": ");
}

function renderReviewSummary() {
  const review = state.review;
  if (!review) return;
  const summary = review.summary || {};
  const finalized = review.status !== "draft";
  const allFrames = review.input?.review_scope === "all_scanned";
  $("#review-status").textContent = review.status;
  $("#review-status").dataset.status = review.status;
  $("#review-id").textContent = review.review_id + " | " + (review.input?.root || "");
  setCount("#count-scanned", summary.scanned_file_count);
  setCount("#count-candidates", summary.candidate_count);
  setCount("#count-auto-bad", summary.automatic_bad_count || summary.auto_bad_count);
  setCount("#count-auto-uncertain", summary.automatic_uncertain_count || summary.auto_uncertain_count);
  setCount("#count-sampled-good", summary.sampled_good_count);
  setCount("#count-pending", summary.pending_count);
  setCount("#count-bad", review.status === "skipped" ? summary.final_bad_count : summary.confirmed_bad_count);
  setCount("#count-degraded", summary.degraded_count);
  setCount("#count-keep", summary.kept_count);
  setCount("#count-viewed", summary.viewed_frame_count);
  setCount("#count-remaining", summary.remaining_frame_count);
  $("#complete-review").disabled =
    finalized ||
    Number(summary.pending_count) > 0 ||
    Number(summary.uncertain_count) > 0 ||
    (allFrames && Number(summary.remaining_frame_count) > 0);
  $("#skip-review").disabled = finalized;
  $("#download-json").href = API + "/reviews/" + review.review_id + "/manifest.json";
  $("#download-csv").href = API + "/reviews/" + review.review_id + "/table.csv";
  $("#download-audit").hidden = !allFrames;
  $("#download-audit").href = allFrames ? API + "/reviews/" + review.review_id + "/audit.csv" : "#";
  $("#coverage-message").textContent = allFrames
    ? Number(summary.viewed_frame_count || 0).toLocaleString() + " viewed / " +
      Number(summary.scanned_file_count || 0).toLocaleString() + " scanned"
    : "Automatic queue only";
}

function renderCandidateTable() {
  const rows = $("#candidate-rows");
  rows.replaceChildren();
  const candidates = state.review?.candidates || [];
  const noCandidates = $("#no-candidates");
  noCandidates.hidden = candidates.length > 0;
  noCandidates.textContent = state.review?.input?.review_scope === "all_scanned"
    ? "No automatic candidates. Continue the all-frame visual audit."
    : "Scan completed with zero automatic candidates. Use an all-frame review for manual coverage.";
  candidates.forEach((candidate, index) => {
    const row = document.createElement("tr");
    row.dataset.candidateId = candidate.candidate_id;
    row.tabIndex = 0;
    if (candidate.candidate_id === state.selectedCandidateId) row.classList.add("selected");
    const values = [
      String(index + 1),
      candidate.time || "Unknown",
      Number(candidate.frequency_mhz).toLocaleString(),
      candidate.polarization,
      candidate.relative_path,
      (candidate.automatic_decision || candidate.algorithm_flag || "unknown") + "\n" +
        formatReason(candidate.automatic_reasons || candidate.algorithm_reason),
    ];
    values.forEach((value, cellIndex) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (cellIndex === 4) cell.className = "path-cell";
      if (cellIndex === 5) cell.className = "reason-cell";
      row.append(cell);
    });
    const humanCell = document.createElement("td");
    const humanValue = candidate.human_label?.quality_label || candidate.human_decision || "Pending";
    humanCell.textContent = humanValue;
    humanCell.className = "quality-cell " + String(humanValue).toLowerCase();
    row.append(humanCell);
    const mlCell = document.createElement("td");
    const prediction = candidate.ml_prediction;
    mlCell.textContent = prediction
      ? (prediction.predicted_label || "prediction") + (prediction.ood ? " (OOD)" : "")
      : "—";
    mlCell.className = prediction?.ood ? "ml-ood-cell" : "";
    row.append(mlCell);
    row.addEventListener("click", () => selectCandidate(candidate.candidate_id));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectCandidate(candidate.candidate_id);
      }
    });
    rows.append(row);
  });
}

function setReviewView(view) {
  state.reviewView = view;
  const showAll = view === "frames";
  $("#candidate-table-region").hidden = showAll;
  $("#all-frame-region").hidden = !showAll;
  $("#show-candidates").classList.toggle("active", !showAll);
  $("#show-candidates").setAttribute("aria-pressed", String(!showAll));
  $("#show-all-frames").classList.toggle("active", showAll);
  $("#show-all-frames").setAttribute("aria-pressed", String(showAll));
}

function renderReview() {
  const review = state.review;
  $("#empty-state").hidden = Boolean(review);
  $("#review-panel").hidden = !review;
  if (!review) return;
  const allFrames = review.input?.review_scope === "all_scanned";
  $("#show-all-frames").hidden = !allFrames;
  if (!allFrames && state.reviewView === "frames") state.reviewView = "candidates";
  setReviewView(state.reviewView);
  renderReviewSummary();
  renderCandidateTable();
  if (state.reviewView === "candidates") {
    const candidates = review.candidates || [];
    if (!state.selectedCandidateId && candidates[0]) {
      state.selectedCandidateId = candidates[0].candidate_id;
    }
    if (state.selectedCandidateId && candidates.some((item) => item.candidate_id === state.selectedCandidateId)) {
      renderCandidatePreview();
    } else {
      showEmptyPreview("No candidate preview is required.");
    }
  } else if (!state.selectedFrameId) {
    showEmptyPreview("Loading the all-frame review...");
  }
}

function showEmptyPreview(message) {
  const image = $("#preview-image");
  image.onload = null;
  image.onerror = null;
  image.hidden = true;
  $("#preview-placeholder").hidden = false;
  $("#preview-placeholder").textContent = message;
  $("#preview-meta").textContent = "";
  $("#preview-state").textContent = "";
  clearAssessment();
}

function selectCandidate(candidateId) {
  state.selectedCandidateId = candidateId;
  for (const row of document.querySelectorAll("#candidate-rows tr")) {
    row.classList.toggle("selected", row.dataset.candidateId === candidateId);
  }
  renderCandidatePreview();
}

function renderCandidatePreview() {
  const candidate = state.review?.candidates?.find((item) => item.candidate_id === state.selectedCandidateId);
  if (!candidate) return;
  $("#preview-title").textContent = candidate.relative_path;
  $("#preview-meta").textContent =
    Number(candidate.frequency_mhz).toLocaleString() + " MHz | " +
    candidate.polarization + " | " + (candidate.time || "unknown time");
  $("#preview-state").textContent = previewLoadingState("Loading preview");
  renderAssessment(candidate);
  $("#preview-placeholder").hidden = true;
  const image = $("#preview-image");
  image.hidden = false;
  image.onload = () => { $("#preview-state").textContent = ""; };
  image.onerror = previewError;
  const url = previewImageUrl(
    API + "/reviews/" + state.review.review_id + "/candidates/" +
    candidate.candidate_id + "/preview"
  );
  if (!url) {
    image.hidden = true;
    $("#preview-placeholder").hidden = false;
    $("#preview-placeholder").textContent = "Enter a valid percentile range.";
    $("#preview-state").textContent = "Display settings incomplete";
    return;
  }
  image.src = url;
}

function previewError() {
  const image = $("#preview-image");
  image.hidden = true;
  $("#preview-placeholder").hidden = false;
  $("#preview-placeholder").textContent = "Preview could not be rendered. Check the display settings or rescan if the data changed.";
  $("#preview-state").textContent = "Preview failed";
}

function clearAssessment() {
  $("#automatic-decision").textContent = "Not selected";
  $("#automatic-reasons").textContent = "Rules are read-only and never become human truth.";
  $("#automatic-version").textContent = "";
  $("#ml-decision").textContent = "No published model";
  $("#ml-probabilities").textContent = "ML never changes the final bad-frame list.";
  $("#ml-ood").textContent = "";
  for (const input of document.querySelectorAll("#quality-labels input, #event-tags input, #artifact-tags input")) {
    input.checked = false;
    input.disabled = true;
  }
  $("#save-label").disabled = true;
}

function renderAssessment(item) {
  $("#automatic-decision").textContent = item.automatic_decision || item.algorithm_flag || "unknown";
  $("#automatic-reasons").textContent = formatReason(item.automatic_reasons || item.algorithm_reason);
  $("#automatic-version").textContent = item.automatic_rule_version || item.rule_version || "legacy rule";
  const human = item.human_label || {};
  const finalized = state.review?.status !== "draft";
  for (const input of document.querySelectorAll("input[name='quality-label']")) {
    input.checked = input.value === (human.quality_label || item.human_decision || "");
    input.disabled = finalized;
  }
  for (const input of document.querySelectorAll("#event-tags input")) {
    input.checked = (human.event_tags || []).includes(input.value);
    input.disabled = finalized;
  }
  for (const input of document.querySelectorAll("#artifact-tags input")) {
    input.checked = (human.artifact_tags || []).includes(input.value);
    input.disabled = finalized;
  }
  $("#save-label").disabled = finalized;
  const ml = item.ml_prediction;
  if (!ml) {
    $("#ml-decision").textContent = "No published model";
    $("#ml-probabilities").textContent = "ML never changes the final bad-frame list.";
    $("#ml-ood").textContent = "";
    return;
  }
  $("#ml-decision").textContent = (ml.predicted_label || "prediction") + " · shadow mode";
  const probabilities = ml.probabilities || {};
  $("#ml-probabilities").textContent = ["good", "degraded", "bad"]
    .filter((label) => Number.isFinite(Number(probabilities[label])))
    .map((label) => label + " " + (100 * Number(probabilities[label])).toFixed(1) + "%")
    .join(" · ") || "No calibrated probabilities";
  $("#ml-ood").textContent = ml.ood
    ? "OOD: " + ((ml.ood_reasons || []).join(", ") || "outside training domain")
    : "Model " + (ml.model_id || "unknown");
}

async function loadFramePage(offset) {
  const payload = await requestJson(
    API + "/reviews/" + state.review.review_id + "/frames?offset=" + offset + "&limit=" + FRAME_PAGE_SIZE
  );
  state.framePage = payload;
  renderFrameTable();
  return payload;
}

function frameFromCurrentPage(index) {
  return state.framePage?.frames?.find((frame) => Number(frame.index) === Number(index)) || null;
}

async function initializeAllFrameView() {
  setScanMessage("Loading all-frame audit...");
  let page = await loadFramePage(0);
  const target = page.first_unviewed_index === null ? 0 : Number(page.first_unviewed_index);
  if (target >= page.offset + page.frames.length) {
    page = await loadFramePage(Math.floor(target / FRAME_PAGE_SIZE) * FRAME_PAGE_SIZE);
  }
  const frame = frameFromCurrentPage(target) || page.frames[0];
  if (frame) selectFrame(frame);
  setScanMessage("All-frame audit loaded.", "success");
}

async function goToFrameIndex(index) {
  if (!state.framePage) return;
  const total = Number(state.framePage.total || 0);
  if (!total) return;
  const target = Math.max(0, Math.min(Number(index), total - 1));
  let frame = frameFromCurrentPage(target);
  if (!frame) {
    await loadFramePage(Math.floor(target / FRAME_PAGE_SIZE) * FRAME_PAGE_SIZE);
    frame = frameFromCurrentPage(target);
  }
  if (frame) selectFrame(frame);
}

function selectFrame(frame) {
  state.selectedFrameId = frame.file_id;
  state.selectedFrameIndex = Number(frame.index);
  for (const row of document.querySelectorAll("#frame-rows tr")) {
    row.classList.toggle("selected", row.dataset.fileId === frame.file_id);
  }
  const total = Number(state.framePage?.total || 0);
  $("#frame-jump").value = String(frame.ordinal);
  $("#frame-jump").max = String(Math.max(total, 1));
  $("#frame-position").textContent = frame.ordinal.toLocaleString() + " / " + total.toLocaleString();
  $("#previous-frame").disabled = frame.index <= 0;
  $("#next-frame").disabled = frame.index + 1 >= total;
  renderFramePreview(frame);
}

function renderFrameTable() {
  const rows = $("#frame-rows");
  rows.replaceChildren();
  for (const frame of state.framePage?.frames || []) {
    const row = document.createElement("tr");
    row.dataset.fileId = frame.file_id;
    row.tabIndex = 0;
    if (frame.file_id === state.selectedFrameId) row.classList.add("selected");
    const human = frame.human_label?.quality_label || "—";
    const values = [
      String(frame.ordinal),
      frame.viewed ? "Viewed" : "Not viewed",
      frame.time || "Unknown",
      Number(frame.frequency_mhz).toLocaleString(),
      frame.polarization,
      frame.relative_path,
      (frame.automatic_decision || "unknown") + "\n" + formatReason(frame.automatic_reasons),
      human,
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 1) cell.className = frame.viewed ? "frame-viewed" : "frame-unviewed";
      if (index === 5) cell.className = "path-cell";
      if (index === 6) cell.className = "reason-cell";
      if (index === 7) cell.className = "quality-cell " + String(human).toLowerCase();
      row.append(cell);
    });
    row.addEventListener("click", () => selectFrame(frame));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectFrame(frame);
      }
    });
    rows.append(row);
  }
}

function renderFramePreview(frame) {
  $("#preview-title").textContent = frame.relative_path;
  $("#preview-meta").textContent =
    Number(frame.frequency_mhz).toLocaleString() + " MHz | " +
    frame.polarization + " | " + (frame.time || "unknown time");
  $("#preview-state").textContent = previewLoadingState(frame.viewed ? "Previously viewed" : "Loading preview");
  renderAssessment(frame);
  $("#preview-placeholder").hidden = true;
  const image = $("#preview-image");
  const expectedFileId = frame.file_id;
  image.hidden = false;
  image.onload = () => {
    if (state.selectedFrameId !== expectedFileId) return;
    $("#preview-state").textContent = frame.viewed ? "Viewed" : "Preview loaded";
    void markFrameViewed(frame);
  };
  image.onerror = previewError;
  const url = previewImageUrl(
    API + "/reviews/" + state.review.review_id + "/frames/" +
    frame.file_id + "/preview"
  );
  if (!url) {
    image.hidden = true;
    $("#preview-placeholder").hidden = false;
    $("#preview-placeholder").textContent = "Enter a valid percentile range.";
    $("#preview-state").textContent = "Display settings incomplete";
    return;
  }
  image.src = url;
}

async function markFrameViewed(frame) {
  if (frame.viewed || state.review?.status !== "draft") return;
  try {
    const payload = await requestJson(
      API + "/reviews/" + state.review.review_id + "/frames/" + frame.file_id + "/viewed",
      {method: "POST"}
    );
    frame.viewed = true;
    state.review = payload.review;
    renderReviewSummary();
    renderFrameTable();
    $("#preview-state").textContent = "Viewed";
  } catch (error) {
    setScanMessage(error.message, "error");
  }
}

async function saveDecision(candidateId, decision) {
  if (state.review?.status !== "draft") return;
  setScanMessage("Saving decision...");
  try {
    const payload = await requestJson(API + "/reviews/" + state.review.review_id, {
      method: "PATCH",
      body: {decisions: {[candidateId]: decision}},
    });
    state.review = payload.review;
    renderReview();
    setScanMessage("Decision saved.", "success");
  } catch (error) {
    setScanMessage(error.message, "error");
  }
}

async function saveLabel() {
  if (state.review?.status !== "draft") return;
  const selected = $("input[name='quality-label']:checked");
  if (!selected) return setScanMessage("Choose a human quality label.", "error");
  const label = {
    quality_label: selected.value,
    event_tags: [...document.querySelectorAll("#event-tags input:checked")].map((input) => input.value),
    artifact_tags: [...document.querySelectorAll("#artifact-tags input:checked")].map((input) => input.value),
  };
  setScanMessage("Saving human label...");
  try {
    let payload;
    if (state.reviewView === "frames") {
      if (!state.selectedFrameId) return;
      payload = await requestJson(
        API + "/reviews/" + state.review.review_id + "/frames/" + state.selectedFrameId,
        {method: "PATCH", body: {label}}
      );
      state.review = payload.review;
      const selectedIndex = state.selectedFrameIndex;
      await loadFramePage(Math.floor(selectedIndex / FRAME_PAGE_SIZE) * FRAME_PAGE_SIZE);
      const refreshed = frameFromCurrentPage(selectedIndex);
      if (refreshed) selectFrame(refreshed);
      renderCandidateTable();
      renderReviewSummary();
    } else {
      if (!state.selectedCandidateId) return;
      payload = await requestJson(API + "/reviews/" + state.review.review_id, {
        method: "PATCH",
        body: {labels: {[state.selectedCandidateId]: label}},
      });
      state.review = payload.review;
      renderReview();
    }
    setScanMessage("Human label saved.", "success");
  } catch (error) {
    setScanMessage(error.message, "error");
  }
}

async function finalizeReview(mode) {
  if (!state.review || state.review.status !== "draft") return;
  if (
    mode === "skipped" &&
    !window.confirm("Skip manual review? Automatic decisions remain weak labels and will never enter supervised training.")
  ) return;
  setScanMessage(mode === "skipped" ? "Skipping review..." : "Completing review...");
  try {
    const payload = await requestJson(API + "/reviews/" + state.review.review_id + "/finalize", {
      method: "POST",
      body: {mode},
    });
    state.review = payload.review;
    renderReview();
    await loadReviewList(state.review.review_id);
    setScanMessage("Review " + mode + ".", "success");
  } catch (error) {
    setScanMessage(error.message, "error");
  }
}

async function loadReviewList(selectedId = "") {
  const payload = await requestJson(API + "/reviews");
  const select = $("#recent-review");
  select.replaceChildren(new Option(payload.reviews.length ? "Choose a saved review" : "No saved reviews", ""));
  for (const review of payload.reviews) {
    const summary = review.summary || {};
    const scope = review.input?.review_scope === "all_scanned" ? "all frames" : "candidates";
    const label =
      review.status + " | " + Number(summary.candidate_count || 0).toLocaleString() +
      " queued | " + scope + " | " + review.created_at;
    select.add(new Option(label, review.review_id));
  }
  if (selectedId) select.value = selectedId;
}

async function openReview(review) {
  state.review = review;
  state.selectedCandidateId = review.candidates?.[0]?.candidate_id || null;
  state.selectedFrameId = null;
  state.selectedFrameIndex = null;
  state.framePage = null;
  state.reviewView = review.input?.review_scope === "all_scanned" ? "frames" : "candidates";
  renderReview();
  if (state.reviewView === "frames") await initializeAllFrameView();
}

async function loadReview(reviewId) {
  if (!reviewId) return;
  setScanMessage("Loading saved review...");
  try {
    const payload = await requestJson(API + "/reviews/" + reviewId);
    await openReview(payload.review);
    setScanMessage("Saved review loaded.", "success");
  } catch (error) {
    setScanMessage(error.message, "error");
  }
}

async function switchReviewView(view) {
  if (!state.review) return;
  if (view === "frames" && state.review.input?.review_scope !== "all_scanned") return;
  setReviewView(view);
  if (view === "frames") {
    if (!state.framePage) await initializeAllFrameView();
    else {
      const frame = frameFromCurrentPage(state.selectedFrameIndex);
      if (frame) selectFrame(frame);
    }
  } else {
    renderCandidateTable();
    const candidates = state.review.candidates || [];
    if (!state.selectedCandidateId && candidates[0]) state.selectedCandidateId = candidates[0].candidate_id;
    if (state.selectedCandidateId) renderCandidatePreview();
    else showEmptyPreview("No candidate preview is required.");
  }
}

async function openNativeRootDialog() {
  setScanMessage("Opening Windows folder dialog...");
  try {
    const paths = await window.SolarNativePathDialog.select({
      mode: "select_directory",
      initialPath: $("#root-input").value,
      title: "Select radio root",
      operation: "scan-observation",
      field: "root-input",
    });
    if (!paths.length) {
      setScanMessage("Selection cancelled.");
      return;
    }
    $("#root-input").value = paths[0];
    await discoverBands();
  } catch (error) {
    await openInAppRootDialog(error);
  }
}

async function openInAppRootDialog(error) {
  setScanMessage(
    "Windows folder dialog unavailable: " + error.message + " Opened the in-app folder browser instead.",
    "error",
  );
  const dialog = $("#folder-dialog");
  if (!dialog.open) dialog.showModal();
  await loadDirectories($("#root-input").value.trim());
}

async function loadDirectories(path) {
  const query = path ? "?path=" + encodeURIComponent(path) : "";
  try {
    const payload = await requestJson(API + "/files" + query);
    state.browserPath = payload.path;
    $("#folder-path").textContent = payload.path || "Allowed roots";
    $("#folder-parent").disabled = !payload.parent;
    $("#folder-parent").dataset.path = payload.parent || "";
    $("#folder-choose").disabled = !payload.path;
    const list = $("#folder-list");
    list.replaceChildren();
    for (const directory of payload.directories || []) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "folder-item";
      button.textContent = directory.name;
      button.addEventListener("click", () => loadDirectories(directory.path));
      list.append(button);
    }
    if (!list.childElementCount) {
      const empty = document.createElement("p");
      empty.textContent = "No child folders";
      list.append(empty);
    }
  } catch (error) {
    setScanMessage(error.message, "error");
  }
}

async function initializeLifecycle() {
  try {
    const config = await requestJson(API + "/client-config");
    state.serverStopOnClose = Boolean(config.stop_on_close);
    const interval = Math.max(1000, Number(config.heartbeat_interval_ms || 5000));
    await sendHeartbeat();
    state.heartbeatTimer = window.setInterval(sendHeartbeat, interval);
  } catch (_error) {
    state.serverStopOnClose = false;
  }
}

async function sendHeartbeat() {
  try {
    await fetch(API + "/client-heartbeat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({client_id: state.clientId}),
      keepalive: true,
    });
  } catch (_error) {
    // The local server may already be stopping.
  }
}

function closeClient() {
  if (state.heartbeatTimer) window.clearInterval(state.heartbeatTimer);
  const body = JSON.stringify({client_id: state.clientId, stop_on_close: state.serverStopOnClose});
  if (navigator.sendBeacon) {
    navigator.sendBeacon(API + "/client-close", new Blob([body], {type: "application/json"}));
  } else {
    fetch(API + "/client-close", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body,
      keepalive: true,
    });
  }
}

async function initialize() {
  $("#discover-button").addEventListener("click", discoverBands);
  $("#frequency-select-all").addEventListener("click", () => setFrequencySelection(true));
  $("#frequency-clear-all").addEventListener("click", () => setFrequencySelection(false));
  $("#scan-button").addEventListener("click", createReview);
  $("#browse-root").addEventListener("click", openNativeRootDialog);
  $("#start-index").addEventListener("input", updateScanEstimate);
  $("#end-index").addEventListener("input", updateScanEstimate);
  $("#review-scope").addEventListener("change", updateScanEstimate);
  $("#preview-cmap").addEventListener("change", refreshActivePreview);
  $("#preview-range-mode").addEventListener("change", refreshActivePreview);
  $("#preview-transform").addEventListener("change", refreshActivePreview);
  $("#preview-pmin").addEventListener("change", refreshActivePreview);
  $("#preview-pmax").addEventListener("change", refreshActivePreview);
  window.addEventListener("solar-ui-state-restored", refreshActivePreview);
  $("#folder-parent").addEventListener("click", (event) => loadDirectories(event.currentTarget.dataset.path));
  $("#folder-choose").addEventListener("click", () => {
    if (!state.browserPath) return;
    $("#root-input").value = state.browserPath;
    $("#folder-dialog").close();
    discoverBands();
  });
  $("#recent-review").addEventListener("change", (event) => loadReview(event.target.value));
  $("#complete-review").addEventListener("click", () => finalizeReview("completed"));
  $("#skip-review").addEventListener("click", () => finalizeReview("skipped"));
  $("#save-label").addEventListener("click", saveLabel);
  $("#show-candidates").addEventListener("click", () => switchReviewView("candidates"));
  $("#show-all-frames").addEventListener("click", () => switchReviewView("frames"));
  $("#previous-frame").addEventListener("click", () => goToFrameIndex(Number(state.selectedFrameIndex) - 1));
  $("#next-frame").addEventListener("click", () => goToFrameIndex(Number(state.selectedFrameIndex) + 1));
  $("#frame-jump").addEventListener("change", (event) => goToFrameIndex(Number(event.target.value) - 1));
  document.addEventListener("keydown", (event) => {
    if (state.reviewView !== "frames" || event.altKey || event.ctrlKey || event.metaKey) return;
    if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName)) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      goToFrameIndex(Number(state.selectedFrameIndex) - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      goToFrameIndex(Number(state.selectedFrameIndex) + 1);
    }
  });
  window.addEventListener("pagehide", closeClient, {once: true});

  try {
    const config = await requestJson(API + "/config");
    state.config = config;
    configurePreviewDisplay(config);
    if (config.allowed_roots?.length === 1) $("#root-input").value = config.allowed_roots[0];
    const shadowOption = $("#candidate-strategy option[value='shadow']");
    shadowOption.disabled = !config.active_model_id;
    shadowOption.textContent = config.active_model_id
      ? "Rules + ML shadow (" + config.active_model_id + ")"
      : "Rules + ML shadow (no published model)";
    await loadReviewList();
    setServiceStatus("Local", "ready");
    if (config.model_warning) setScanMessage(config.model_warning, "error");
  } catch (error) {
    setServiceStatus("Unavailable", "error");
    setScanMessage(error.message, "error");
  }
  await initializeLifecycle();
}

document.addEventListener("DOMContentLoaded", initialize);
