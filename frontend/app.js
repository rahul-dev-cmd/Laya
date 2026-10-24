/*
 * Laya Speech Coaching Dashboard Application Logic
 * Strict adherence to Design Rules:
 * - Plain JavaScript with Plotly.js
 * - No gradients, no emoji or icons, no copy buttons, no slogans
 * - No long dashes (em-dash or en-dash)
 * - All numbers computed strictly from data
 */

// Application state
const state = {
  currentExample: "librivox_01_pitch_tier5",
  data: null,
  ladders: null,
  activeStressTab: "pitch",
  selectedRegionIndex: 0,
  isPlaying: false,
  currentTime: 0.0,
  duration: 95.0,
  animationFrameId: null,
  currentReport: null,
  chatHistory: [],
  chatSessionId: Math.random().toString(36).substring(2, 10),
};

// DOM element references
const elements = {
  modeBadge: document.getElementById("modeBadge"),
  exampleSelect: document.getElementById("exampleSelect"),
  btnLoadExample: document.getElementById("btnLoadExample"),
  uploadForm: document.getElementById("uploadForm"),
  btnSubmitAnalyze: document.getElementById("btnSubmitAnalyze"),
  uploadStatus: document.getElementById("uploadStatus"),
  resultSentenceText: document.getElementById("resultSentenceText"),
  statRegionCount: document.getElementById("statRegionCount"),
  statRegionType: document.getElementById("statRegionType"),
  statFlaggedDuration: document.getElementById("statFlaggedDuration"),
  statTotalDuration: document.getElementById("statTotalDuration"),
  statPeakZ: document.getElementById("statPeakZ"),
  deliveryScoreCard: document.getElementById("deliveryScoreCard"),
  rubricReliabilityNote: document.getElementById("rubricReliabilityNote"),
  rubricOverallScore: document.getElementById("rubricOverallScore"),
  rubricEmptyState: document.getElementById("rubricEmptyState"),
  rubricDimensionsList: document.getElementById("rubricDimensionsList"),
  rubricCalculationDetails: document.getElementById("rubricCalculationDetails"),
  rubricMethodText: document.getElementById("rubricMethodText"),
  rubricCriteriaList: document.getElementById("rubricCriteriaList"),
  rubricMethodNote: document.getElementById("rubricMethodNote"),
  injectedLegendBadge: document.getElementById("injectedLegendBadge"),
  legendInjectedItem: document.getElementById("legendInjectedItem"),
  audioPlayer: document.getElementById("audioPlayer"),
  btnPlayPause: document.getElementById("btnPlayPause"),
  btnStop: document.getElementById("btnStop"),
  audioTimeDisplay: document.getElementById("audioTimeDisplay"),
  audioStatusDisplay: document.getElementById("audioStatusDisplay"),
  stackedTracksPlot: document.getElementById("stackedTracksPlot"),
  regionsTableBody: document.getElementById("regionsTableBody"),
  whyFlaggedContainer: document.getElementById("whyFlaggedContainer"),
  tabPitchLadder: document.getElementById("tabPitchLadder"),
  tabRateLadder: document.getElementById("tabRateLadder"),
  stressLadderPlot: document.getElementById("stressLadderPlot"),
  stressTableHead: document.getElementById("stressTableHead"),
  stressTableBody: document.getElementById("stressTableBody"),
  transcriptReviewCard: document.getElementById("transcriptReviewCard"),
  transcriptReviewLabel: document.getElementById("transcriptReviewLabel"),
  transcriptReviewText: document.getElementById("transcriptReviewText"),
  btnRerun: document.getElementById("btnRerun"),
  whisperModel: document.getElementById("whisperModel"),
  transcriptSourceBadge: document.getElementById("transcriptSourceBadge"),
  lowConfidenceSection: document.getElementById("lowConfidenceSection"),
  lowConfidenceCount: document.getElementById("lowConfidenceCount"),
  lowConfidenceChips: document.getElementById("lowConfidenceChips"),
  btnExplainReport: document.getElementById("btnExplainReport"),
  explainLoadingSpinner: document.getElementById("explainLoadingSpinner"),
  explanationCard: document.getElementById("explanationCard"),
  explanationCardTitle: document.getElementById("explanationCardTitle"),
  explanationSourceBadge: document.getElementById("explanationSourceBadge"),
  explanationSummary: document.getElementById("explanationSummary"),
  explanationWhatHappened: document.getElementById("explanationWhatHappened"),
  explanationWhyMatters: document.getElementById("explanationWhyMatters"),
  explanationPracticeTips: document.getElementById("explanationPracticeTips"),
  explanationCaveats: document.getElementById("explanationCaveats"),
  explanationCaveatsBlock: document.getElementById("explanationCaveatsBlock"),
  chatCard: document.getElementById("chatCard"),
  chatMessagesContainer: document.getElementById("chatMessagesContainer"),
  chatEmptyState: document.getElementById("chatEmptyState"),
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
  btnSendChat: document.getElementById("btnSendChat"),
  chatStatusText: document.getElementById("chatStatusText"),
};

/**
 * Safely escape HTML characters.
 */
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/**
 * Focus and highlight target word in the editable transcript box.
 */
function selectWordInTextarea(word) {
  const textarea = elements.transcriptReviewText;
  if (!textarea) return;
  const text = textarea.value;
  const cleanTarget = word.replace(/[^\w]/g, "").toLowerCase();
  if (!cleanTarget) return;

  const words = text.split(/\s+/);
  let charIdx = 0;
  for (const w of words) {
    const cleanW = w.replace(/[^\w]/g, "").toLowerCase();
    const pos = text.indexOf(w, charIdx);
    if (pos !== -1 && cleanW === cleanTarget) {
      textarea.focus();
      textarea.setSelectionRange(pos, pos + w.length);
      return;
    }
    if (pos !== -1) {
      charIdx = pos + w.length;
    }
  }
}

/**
 * Format timestamp in M:SS.S notation (e.g. 54.58 -> 0:54.6)
 */
function formatTime(sec) {
  if (isNaN(sec) || sec < 0) sec = 0.0;
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

/**
 * Initialize application lifecycle and event listeners.
 */
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadExamplesList();
  loadLaddersData();
});

function setupEventListeners() {
  // Load example button
  elements.btnLoadExample.addEventListener("click", () => {
    state.currentExample = elements.exampleSelect.value;
    loadExample(state.currentExample);
  });

  // Example select change
  elements.exampleSelect.addEventListener("change", (e) => {
    state.currentExample = e.target.value;
    loadExample(state.currentExample);
  });

  // Upload analysis form
  elements.uploadForm.addEventListener("submit", handleUploadSubmit);

  // Re-run analysis button
  if (elements.btnRerun) {
    elements.btnRerun.addEventListener("click", handleRerun);
  }

  // Audio Play / Pause
  elements.btnPlayPause.addEventListener("click", togglePlayback);
  elements.btnStop.addEventListener("click", resetPlayback);

  // Explain in simple words button
  if (elements.btnExplainReport) {
    elements.btnExplainReport.addEventListener("click", handleExplainClick);
  }

  // Chat form submit
  if (elements.chatForm) {
    elements.chatForm.addEventListener("submit", handleChatSubmit);
  }

  // Chat suggested question buttons
  const suggestionButtons = document.querySelectorAll(".btn-suggestion");
  suggestionButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-question");
      if (q) {
        sendChatMessage(q);
      }
    });
  });

  // Audio element events
  elements.audioPlayer.addEventListener("timeupdate", onAudioTimeUpdate);
  elements.audioPlayer.addEventListener("ended", onAudioEnded);
  elements.audioPlayer.addEventListener("error", onAudioError);

  // Stress test tabs
  elements.tabPitchLadder.addEventListener("click", () => {
    setStressTab("pitch");
  });
  elements.tabRateLadder.addEventListener("click", () => {
    setStressTab("rate");
  });

  // Window resize to fit Plotly
  window.addEventListener("resize", () => {
    if (document.getElementById("stackedTracksPlot")) {
      Plotly.Plots.resize("stackedTracksPlot");
    }
    if (document.getElementById("stressLadderPlot")) {
      Plotly.Plots.resize("stressLadderPlot");
    }
  });
}

/**
 * Load available examples from backend API.
 */
async function loadExamplesList() {
  try {
    const res = await fetch("/api/examples");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const list = await res.json();
    if (Array.isArray(list) && list.length > 0) {
      elements.exampleSelect.innerHTML = "";
      list.forEach((item) => {
        const id = typeof item === "string" ? item : item.id;
        const name = typeof item === "string" ? item : item.name;
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = name;
        if (id === "librivox_01_pitch_tier5") {
          opt.selected = true;
        }
        elements.exampleSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.error("Failed to load examples list:", err);
  }

  // Load default selected example
  loadExample(elements.exampleSelect.value || state.currentExample);
}

/**
 * Load example data from backend API.
 */
async function loadExample(name) {
  try {
    elements.uploadStatus.textContent = "Loading " + name + "...";
    const res = await fetch(`/api/example/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.data = data;
    state.selectedRegionIndex = 0;
    renderDashboard(data);
    elements.uploadStatus.textContent = "";
    prefetchExampleReport(name);
  } catch (err) {
    elements.uploadStatus.textContent = "Error loading example: " + err.message;
    console.error("Error loading example:", err);
  }
}

/**
 * Handle custom file upload analysis.
 */
async function handleUploadSubmit(e) {
  e.preventDefault();
  const fileAudio = document.getElementById("fileAudio").files[0];
  const fileRef = document.getElementById("fileReference").files[0];
  const transcriptText = document.getElementById("transcriptText").value.trim();
  const whisperModelEl = elements.whisperModel || document.getElementById("whisperModel");

  if (!fileAudio) {
    alert("Please select a participant WAV audio file.");
    return;
  }

  const formData = new FormData();
  formData.append("audio_file", fileAudio);
  if (transcriptText) {
    formData.append("transcript_text", transcriptText);
  }
  if (fileRef) {
    formData.append("reference_file", fileRef);
  }
  if (whisperModelEl) {
    formData.append("whisper_model", whisperModelEl.value);
  }

  elements.btnSubmitAnalyze.disabled = true;
  elements.uploadStatus.textContent = transcriptText
    ? "Running forced alignment and flaw detection..."
    : "Transcribing audio with Whisper and detecting flaws...";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `HTTP ${res.status}`);
    }
    const result = await res.json();
    state.data = result;
    state.selectedRegionIndex = 0;
    renderDashboard(result);
    elements.uploadStatus.textContent = "Analysis complete.";
  } catch (err) {
    elements.uploadStatus.textContent = "Analysis failed: " + err.message;
    alert("Analysis failed: " + err.message);
  } finally {
    elements.btnSubmitAnalyze.disabled = false;
  }
}

/**
 * Handle re-running analysis with the updated transcript from the review card.
 */
async function handleRerun() {
  const rerunText = elements.transcriptReviewText.value.trim();
  const fileAudioInput = document.getElementById("fileAudio");
  const fileRefInput = document.getElementById("fileReference");
  const whisperModelEl = elements.whisperModel || document.getElementById("whisperModel");

  const formData = new FormData();
  if (rerunText) {
    formData.append("transcript_text", rerunText);
  }
  formData.append("transcript_source", "edited");
  if (whisperModelEl) {
    formData.append("whisper_model", whisperModelEl.value);
  }

  if (fileAudioInput.files && fileAudioInput.files.length > 0) {
    formData.append("audio_file", fileAudioInput.files[0]);
    if (fileRefInput.files && fileRefInput.files.length > 0) {
      formData.append("reference_file", fileRefInput.files[0]);
    }
  } else if (state.data && state.data.audio_file) {
    elements.btnRerun.disabled = true;
    elements.uploadStatus.textContent = "Loading audio files for re-run...";
    try {
      const audioBlob = await fetch(`/api/audio?path=${encodeURIComponent(state.data.audio_file)}`).then((r) => {
        if (!r.ok) throw new Error("Could not load participant audio file.");
        return r.blob();
      });
      formData.append("audio_file", audioBlob, "participant.wav");

      if (state.data.reference_file) {
        const refBlob = await fetch(`/api/audio?path=${encodeURIComponent(state.data.reference_file)}`).then((r) => {
          if (!r.ok) throw new Error("Could not load reference audio file.");
          return r.blob();
        });
        formData.append("reference_file", refBlob, "reference.wav");
      }
    } catch (err) {
      elements.uploadStatus.textContent = "Re-run failed: " + err.message;
      alert("Re-run failed: " + err.message);
      elements.btnRerun.disabled = false;
      return;
    }
  } else {
    alert("Please select or load an audio file first.");
    return;
  }

  elements.btnRerun.disabled = true;
  elements.btnSubmitAnalyze.disabled = true;
  elements.uploadStatus.textContent = "Re-running analysis with updated transcript...";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `HTTP ${res.status}`);
    }
    const result = await res.json();
    state.data = result;
    state.selectedRegionIndex = 0;
    renderDashboard(result);
    elements.uploadStatus.textContent = "Re-run complete.";
  } catch (err) {
    elements.uploadStatus.textContent = "Re-run failed: " + err.message;
    alert("Re-run failed: " + err.message);
  } finally {
    elements.btnRerun.disabled = false;
    elements.btnSubmitAnalyze.disabled = false;
  }
}

/**
 * Load stress test ladders benchmark data.
 */
async function loadLaddersData() {
  try {
    const res = await fetch("/api/ladders");
    if (!res.ok) return;
    const ladders = await res.json();
    state.ladders = ladders;
    renderStressTest();
  } catch (err) {
    console.error("Failed to load ladders:", err);
  }
}

/**
 * Render the entire dashboard from dataset JSON.
 */
function renderDashboard(data) {
  if (!data) return;

  // Reset explanation state
  state.currentReport = null;
  if (elements.explanationCard) {
    elements.explanationCard.style.display = "none";
  }

  // Reset and enable chat box for loaded analysis
  resetChatForNewAnalysis();

  // 1. Mode Badge
  const isPaired = data.mode === "paired";
  let modeLabel = isPaired
    ? "Mode: Paired (ideal recording available)"
    : "Mode: Experimental (no ideal recording)";
  if (data.mode_note) {
    modeLabel += ` (${data.mode_note})`;
  }
  elements.modeBadge.textContent = modeLabel;
  elements.modeBadge.className = `mode-badge ${isPaired ? "paired" : "experimental"}`;

  // 1b. Transcript Review Box
  if (elements.transcriptReviewCard) {
    elements.transcriptReviewCard.style.display = "block";
    const src = data.transcript_source || "auto";
    if (elements.transcriptSourceBadge) {
      elements.transcriptSourceBadge.textContent = src;
      elements.transcriptSourceBadge.className = `transcript-source-badge ${src}`;
    }
    if (elements.transcriptReviewLabel) {
      if (src === "edited") {
        elements.transcriptReviewLabel.textContent = "Edited transcript";
      } else if (src === "provided") {
        elements.transcriptReviewLabel.textContent = "Provided transcript";
      } else {
        elements.transcriptReviewLabel.textContent = "Auto-generated, check for errors";
      }
    }
    elements.transcriptReviewText.value = data.transcript || "";

    // Low confidence words (< 0.6) highlighting with label "Check these words"
    const words = data.words || [];
    const lowConfWords = words.filter(
      (w) => typeof w.confidence === "number" && w.confidence < 0.6
    );
    if (elements.lowConfidenceSection && elements.lowConfidenceChips) {
      if (lowConfWords.length > 0) {
        elements.lowConfidenceSection.style.display = "block";
        if (elements.lowConfidenceCount) {
          elements.lowConfidenceCount.textContent = `${lowConfWords.length} word${lowConfWords.length === 1 ? "" : "s"} below 0.6 confidence`;
        }
        elements.lowConfidenceChips.innerHTML = "";
        lowConfWords.forEach((item) => {
          const pill = document.createElement("button");
          pill.type = "button";
          pill.className = "low-conf-pill";
          pill.title = `Confidence: ${item.confidence.toFixed(2)} (${item.start}s to ${item.end}s). Click to select in text.`;
          pill.innerHTML = `<span>${escapeHtml(item.word)}</span><span class="conf-val">${item.confidence.toFixed(2)}</span>`;
          pill.addEventListener("click", () => {
            selectWordInTextarea(item.word);
          });
          elements.lowConfidenceChips.appendChild(pill);
        });
      } else {
        elements.lowConfidenceSection.style.display = "none";
        elements.lowConfidenceChips.innerHTML = "";
      }
    }
  }

  // 2. Audio Player setup
  setupAudio(data);

  // 3. Results Hero Strip (Headline sentence & 3 data numbers)
  renderResultsHero(data);

  // 3b. Delivery Score Rubric
  renderDeliveryRubric(data);

  // 4. Ground Truth legend indicator
  const hasGroundTruth = Boolean(data.ground_truth);
  elements.injectedLegendBadge.style.display = hasGroundTruth ? "inline-flex" : "none";
  elements.legendInjectedItem.style.display = hasGroundTruth ? "flex" : "none";

  // 5. Stacked Tracks Plot
  renderStackedTracksPlot(data);

  // 6. Detected Regions Table
  renderRegionsTable(data);

  // 7. Why It Was Flagged Card
  renderWhyFlagged(data);
}

/**
 * Setup HTML5 Audio element with backend audio route.
 */
function setupAudio(data) {
  resetPlayback();
  const audioPath = data.audio_file;
  if (!audioPath) {
    elements.audioStatusDisplay.textContent = "No audio file path in dataset";
    return;
  }

  // Set audio source via streaming API
  const audioUrl = `/api/audio?path=${encodeURIComponent(audioPath)}`;
  elements.audioPlayer.src = audioUrl;
  elements.audioPlayer.load();

  state.duration = data.duration || 95.0;
  elements.audioTimeDisplay.textContent = `0:00.0 / ${formatTime(state.duration)}`;
  elements.audioStatusDisplay.textContent = audioPath.split("/").pop() || "Audio loaded";
}

function togglePlayback() {
  if (!elements.audioPlayer.src) return;
  if (state.isPlaying) {
    elements.audioPlayer.pause();
    state.isPlaying = false;
    elements.btnPlayPause.textContent = "Play";
  } else {
    elements.audioPlayer.play().then(() => {
      state.isPlaying = true;
      elements.btnPlayPause.textContent = "Pause";
      trackPlayback();
    }).catch((err) => {
      elements.audioStatusDisplay.textContent = "Playback failed: " + err.message;
    });
  }
}

function resetPlayback() {
  if (state.animationFrameId) {
    cancelAnimationFrame(state.animationFrameId);
  }
  elements.audioPlayer.pause();
  elements.audioPlayer.currentTime = 0;
  state.currentTime = 0;
  state.isPlaying = false;
  elements.btnPlayPause.textContent = "Play";
  elements.audioTimeDisplay.textContent = `0:00.0 / ${formatTime(state.duration)}`;
  updatePlayheadLine(0);
}

function onAudioTimeUpdate() {
  state.currentTime = elements.audioPlayer.currentTime;
  elements.audioTimeDisplay.textContent = `${formatTime(state.currentTime)} / ${formatTime(state.duration)}`;
  updatePlayheadLine(state.currentTime);
}

function onAudioEnded() {
  state.isPlaying = false;
  elements.btnPlayPause.textContent = "Play";
}

function onAudioError() {
  elements.audioStatusDisplay.textContent = "Audio missing or cannot be streamed.";
}

function trackPlayback() {
  if (!state.isPlaying) return;
  state.currentTime = elements.audioPlayer.currentTime;
  elements.audioTimeDisplay.textContent = `${formatTime(state.currentTime)} / ${formatTime(state.duration)}`;
  updatePlayheadLine(state.currentTime);
  state.animationFrameId = requestAnimationFrame(trackPlayback);
}

function seekToTime(seconds) {
  if (seconds < 0) seconds = 0;
  if (seconds > state.duration) seconds = state.duration;
  elements.audioPlayer.currentTime = seconds;
  state.currentTime = seconds;
  elements.audioTimeDisplay.textContent = `${formatTime(state.currentTime)} / ${formatTime(state.duration)}`;
  updatePlayheadLine(seconds);
}

/**
 * Render Results Hero Strip (Result Sentence and 3 numbers computed from data).
 */
function renderResultsHero(data) {
  const flaws = data.flaw_regions || [];
  const duration = data.duration || 95.0;

  if (flaws.length === 0) {
    elements.resultSentenceText.textContent = "No speech flaws detected across the recording.";
    elements.statRegionCount.textContent = "0";
    elements.statRegionType.textContent = "no flaws detected";
    elements.statFlaggedDuration.textContent = "0.0 s";
    elements.statTotalDuration.textContent = `of ${duration.toFixed(1)} s total`;
    elements.statPeakZ.textContent = "0.00";
    return;
  }

  // Find primary flaw by maximum absolute peak_z
  let primary = flaws[0];
  for (let i = 1; i < flaws.length; i++) {
    if (Math.abs(flaws[i].peak_z) > Math.abs(primary.peak_z)) {
      primary = flaws[i];
    }
  }

  // 1. Result sentence
  const sStr = primary.start.toFixed(2);
  const eStr = primary.end.toFixed(2);
  if (primary.type === "pitch") {
    elements.resultSentenceText.textContent = `Pitch intonation was monotone from ${sStr} s to ${eStr} s.`;
  } else if (primary.type === "rate") {
    const dir = primary.peak_z >= 0 ? "accelerated" : "decelerated";
    elements.resultSentenceText.textContent = `Speech rate ${dir} from ${sStr} s to ${eStr} s.`;
  } else if (primary.type === "pause") {
    const dir = primary.peak_z >= 0 ? "extended" : "shortened";
    elements.resultSentenceText.textContent = `Pause duration was ${dir} from ${sStr} s to ${eStr} s.`;
  } else {
    elements.resultSentenceText.textContent = `Anomaly flagged between ${sStr} s and ${eStr} s.`;
  }

  // 2. Three data numbers:
  // Counter 1: Region count
  elements.statRegionCount.textContent = flaws.length.toString();
  elements.statRegionType.textContent = flaws.length === 1 ? "1 region flagged" : `${flaws.length} regions flagged`;

  // Counter 2: Total flagged seconds out of total duration
  let flaggedSec = 0.0;
  flaws.forEach((r) => {
    flaggedSec += Math.max(0.0, r.end - r.start);
  });
  elements.statFlaggedDuration.textContent = `${flaggedSec.toFixed(1)} s`;
  elements.statTotalDuration.textContent = `of ${duration.toFixed(1)} s total`;

  // Counter 3: Peak z-score
  const sign = primary.peak_z >= 0 ? "+" : "";
  elements.statPeakZ.textContent = `${sign}${primary.peak_z.toFixed(2)}`;
}

/**
 * Render Delivery Score Rubric Card.
 * Adheres strictly to design rules:
 * Flat styling, no gradients, no emoji, no icons, no animation.
 * All values rendered straight from data.
 */
function renderDeliveryRubric(data) {
  const rubric = data.rubric;
  const card = elements.deliveryScoreCard;
  if (!card) return;

  if (!rubric) {
    if (elements.rubricEmptyState) elements.rubricEmptyState.style.display = "block";
    if (elements.rubricDimensionsList) elements.rubricDimensionsList.style.display = "none";
    if (elements.rubricCalculationDetails) elements.rubricCalculationDetails.style.display = "none";
    if (elements.rubricOverallScore) elements.rubricOverallScore.textContent = "-";
    if (elements.rubricReliabilityNote) elements.rubricReliabilityNote.textContent = "";
    return;
  }

  if (elements.rubricEmptyState) elements.rubricEmptyState.style.display = "none";
  if (elements.rubricDimensionsList) elements.rubricDimensionsList.style.display = "flex";
  if (elements.rubricCalculationDetails) elements.rubricCalculationDetails.style.display = "block";

  // 1. Overall Score as large number
  if (rubric.overall !== null && rubric.overall !== undefined) {
    elements.rubricOverallScore.textContent = rubric.overall.toFixed(1);
  } else {
    elements.rubricOverallScore.textContent = "null";
  }

  // 2. Reliability note in plain text
  if (elements.rubricReliabilityNote) {
    elements.rubricReliabilityNote.textContent = rubric.reliability_note || "";
  }

  // 3. Dimensions list
  if (elements.rubricDimensionsList) {
    elements.rubricDimensionsList.innerHTML = "";
    const dims = rubric.dimensions || [];
    const duration = data.duration || 0.0;

    dims.forEach((d) => {
      const row = document.createElement("div");
      row.className = "rubric-dim-row";

      let scoreDisplay = "Not scored yet";
      let barWidth = 0;
      if (d.scored && d.score !== null && d.score !== undefined) {
        scoreDisplay = d.score.toFixed(1);
        barWidth = Math.max(0, Math.min(100, d.score));
      }

      let shortLine = "";
      if (d.name === "Pace" || d.name === "Pitch variation") {
        const flagSec = (d.flagged_seconds !== null && d.flagged_seconds !== undefined)
          ? d.flagged_seconds.toFixed(1)
          : "0.0";
        const durStr = duration.toFixed(1);
        if (d.region_count === 0) {
          shortLine = `0.0 s of ${durStr} s flagged, no flaws detected`;
        } else {
          const zStr = (d.worst_z !== null && d.worst_z !== undefined)
            ? `, worst z ${d.worst_z.toFixed(2)}`
            : "";
          const sevStr = d.worst_severity ? `, ${d.worst_severity}` : "";
          shortLine = `${flagSec} s of ${durStr} s flagged${zStr}${sevStr}`;
        }
      } else if (d.name === "Pauses") {
        if (d.scored && d.score !== null) {
          const examinedStr = d.pauses_examined ? ` of ${d.pauses_examined}` : "";
          const zStr = (d.worst_z !== null && d.worst_z !== undefined)
            ? `, worst z ${d.worst_z.toFixed(2)}`
            : "";
          const sevStr = d.worst_severity ? `, ${d.worst_severity}` : "";
          shortLine = `${d.flagged_pauses}${examinedStr} pauses flagged${zStr}${sevStr}`;
        } else {
          const pauseCount = d.flagged_pauses !== undefined ? d.flagged_pauses : (d.region_count || 0);
          const zStr = (d.worst_z !== null && d.worst_z !== undefined)
            ? `, worst z ${d.worst_z.toFixed(2)}`
            : "";
          const sevStr = d.worst_severity ? `, ${d.worst_severity}` : "";
          const reasonStr = d.reason ? ` Reason: ${d.reason}` : "";
          shortLine = `${pauseCount} flagged pauses${zStr}${sevStr}.${reasonStr}`;
        }
      } else if (d.name === "Loudness") {
        const reasonStr = d.reason ? ` Reason: ${d.reason}` : "";
        shortLine = `Not scored yet.${reasonStr}`;
      } else {
        shortLine = d.reason || "";
      }

      row.innerHTML = `
        <div class="rubric-dim-header">
          <span class="rubric-dim-name">${escapeHtml(d.name)}</span>
          <span class="rubric-dim-score ${d.scored ? "scored" : "unscored"}">${escapeHtml(scoreDisplay)}</span>
        </div>
        <div class="rubric-bar-track">
          <div class="rubric-bar-fill" style="width: ${barWidth}%;"></div>
        </div>
        <div class="rubric-dim-shortline">${escapeHtml(shortLine)}</div>
      `;
      elements.rubricDimensionsList.appendChild(row);
    });
  }

  // 4. Collapsible section: method, criteria, method_note straight from data
  if (elements.rubricMethodText) {
    elements.rubricMethodText.textContent = rubric.method || "";
  }

  if (elements.rubricCriteriaList) {
    elements.rubricCriteriaList.innerHTML = "";
    (rubric.dimensions || []).forEach((d) => {
      const li = document.createElement("li");
      li.className = "rubric-criteria-item";
      li.innerHTML = `<strong>${escapeHtml(d.name)}:</strong> ${escapeHtml(d.criterion || "")}`;
      elements.rubricCriteriaList.appendChild(li);
    });
  }

  if (elements.rubricMethodNote) {
    elements.rubricMethodNote.textContent = rubric.method_note || "";
  }
}

/**
 * Render Stacked Tracks Plot sharing one time axis using Plotly.
 */
function renderStackedTracksPlot(data) {
  const curves = data.curves || {};
  const pCurves = curves.participant || {};
  const rCurves = curves.reference || null;
  const flaws = data.flaw_regions || [];
  const gt = data.ground_truth || null;
  const duration = data.duration || 95.0;

  // Participant series
  const pTime = pCurves.time || curves.timestamps || [];
  const pF0 = pCurves.f0_semitones || curves.f0_semitones || [];
  const pEnergy = pCurves.energy_z || curves.energy_z || [];
  const pRate = pCurves.speech_rate || curves.speech_rate || [];

  // Approximate waveform envelope trace from Energy
  const pWave = pEnergy.map((v) => (v !== null ? Math.max(0.02, (v + 3.0) / 4.0) : 0.02));

  // Reference series (if available)
  const rTime = rCurves ? rCurves.time || [] : [];
  const rF0 = rCurves ? rCurves.f0_semitones || [] : [];
  const rEnergy = rCurves ? rCurves.energy_z || [] : [];
  const rRate = rCurves ? rCurves.speech_rate || [] : [];

  // Traces setup across 4 subplots
  const traces = [];

  // Track 1: Waveform / Acoustic Amplitude (Row 1)
  traces.push({
    x: pTime,
    y: pWave,
    xaxis: "x",
    yaxis: "y1",
    name: "Waveform Envelope",
    type: "scatter",
    mode: "lines",
    line: { color: "#141a1e", width: 1.0 },
    hoverinfo: "x+y",
  });

  // Track 2: Pitch F0 in Semitones (Row 2)
  if (rCurves && rTime.length > 0) {
    traces.push({
      x: rTime,
      y: rF0,
      xaxis: "x",
      yaxis: "y2",
      name: "Reference Pitch (F0 st)",
      type: "scatter",
      mode: "lines",
      line: { color: "#888d92", dash: "dash", width: 1.5 },
      hoverinfo: "x+y",
    });
  }
  traces.push({
    x: pTime,
    y: pF0,
    xaxis: "x",
    yaxis: "y2",
    name: "Participant Pitch (F0 st)",
    type: "scatter",
    mode: "lines",
    line: { color: "#141a1e", width: 1.7 },
    hoverinfo: "x+y",
  });

  // Track 3: Energy RMS in dB z-score (Row 3)
  if (rCurves && rTime.length > 0) {
    traces.push({
      x: rTime,
      y: rEnergy,
      xaxis: "x",
      yaxis: "y3",
      name: "Reference Energy (z)",
      type: "scatter",
      mode: "lines",
      line: { color: "#888d92", dash: "dash", width: 1.5 },
      hoverinfo: "x+y",
    });
  }
  traces.push({
    x: pTime,
    y: pEnergy,
    xaxis: "x",
    yaxis: "y3",
    name: "Participant Energy (z)",
    type: "scatter",
    mode: "lines",
    line: { color: "#141a1e", width: 1.7 },
    hoverinfo: "x+y",
  });

  // Track 4: Speech Rate in words/s (Row 4)
  if (rCurves && rTime.length > 0) {
    traces.push({
      x: rTime,
      y: rRate,
      xaxis: "x",
      yaxis: "y4",
      name: "Reference Rate (wps)",
      type: "scatter",
      mode: "lines",
      line: { color: "#888d92", dash: "dash", width: 1.5 },
      hoverinfo: "x+y",
    });
  }
  traces.push({
    x: pTime,
    y: pRate,
    xaxis: "x",
    yaxis: "y4",
    name: "Participant Rate (wps)",
    type: "scatter",
    mode: "lines",
    line: { color: "#141a1e", width: 1.7 },
    hoverinfo: "x+y",
  });

  // Shapes & Annotations for Flagged Regions and Injected Ground Truth
  const shapes = [];
  const annotations = [];

  // Flagged regions: Shaded column spanning all tracks
  flaws.forEach((flaw) => {
    let color = "rgba(184, 67, 40, 0.16)";
    let border = "#b84328";
    if (flaw.type === "rate") {
      color = "rgba(175, 110, 20, 0.16)";
      border = "#af6e14";
    } else if (flaw.type === "pause") {
      color = "rgba(30, 105, 135, 0.16)";
      border = "#1e6987";
    }

    shapes.push({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: flaw.start,
      x1: flaw.end,
      y0: 0,
      y1: 1,
      fillcolor: color,
      line: { color: border, width: 1 },
    });

    // Tab tag annotation at top
    annotations.push({
      x: (flaw.start + flaw.end) / 2.0,
      y: 1.03,
      xref: "x",
      yref: "paper",
      text: `${flaw.type.toUpperCase()}: ${flaw.start.toFixed(1)}s - ${flaw.end.toFixed(1)}s`,
      showarrow: false,
      font: { family: "monospace", size: 10, color: border },
      bgcolor: "#ffffff",
      bordercolor: border,
      borderwidth: 1,
      borderpad: 2,
    });
  });

  // Injected ground truth bracket if exists
  if (gt && gt.start !== undefined && gt.end !== undefined) {
    shapes.push({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: gt.start,
      x1: gt.end,
      y0: 0.01,
      y1: 0.99,
      fillcolor: "rgba(31, 107, 77, 0.04)",
      line: { color: "#1f6b4d", width: 2, dash: "dot" },
    });

    annotations.push({
      x: (gt.start + gt.end) / 2.0,
      y: -0.05,
      xref: "x",
      yref: "paper",
      text: `Injected (ground truth): ${gt.start.toFixed(1)}s - ${gt.end.toFixed(1)}s`,
      showarrow: false,
      font: { family: "monospace", size: 10, color: "#1f6b4d" },
      bgcolor: "#eaf5ef",
      bordercolor: "#1f6b4d",
      borderwidth: 1,
      borderpad: 2,
    });
  }

  // Vertical Playhead Line shape (at state.currentTime)
  shapes.push({
    name: "playhead",
    type: "line",
    xref: "x",
    yref: "paper",
    x0: state.currentTime,
    x1: state.currentTime,
    y0: 0,
    y1: 1,
    line: { color: "#b84328", width: 2 },
  });

  // Plotly Subplots Layout
  const layout = {
    grid: { rows: 4, columns: 1, pattern: "coupled", roworder: "top to bottom" },
    margin: { l: 70, r: 25, t: 30, b: 40 },
    showlegend: false,
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    hovermode: "x",
    xaxis: {
      range: [0, duration],
      showgrid: true,
      gridcolor: "#f0ece1",
      zeroline: false,
      title: { text: "Time (seconds)", font: { size: 11, family: "monospace" } },
    },
    yaxis1: {
      title: { text: "Waveform", font: { size: 10, family: "monospace" } },
      showgrid: false,
      zeroline: false,
      showticklabels: false,
    },
    yaxis2: {
      title: { text: "Pitch (st)", font: { size: 10, family: "monospace" } },
      showgrid: true,
      gridcolor: "#f0ece1",
      zeroline: true,
      zerolinecolor: "#d0c8b8",
    },
    yaxis3: {
      title: { text: "Energy (z)", font: { size: 10, family: "monospace" } },
      showgrid: true,
      gridcolor: "#f0ece1",
      zeroline: true,
      zerolinecolor: "#d0c8b8",
    },
    yaxis4: {
      title: { text: "Rate (wps)", font: { size: 10, family: "monospace" } },
      showgrid: true,
      gridcolor: "#f0ece1",
      zeroline: false,
    },
    shapes: shapes,
    annotations: annotations,
  };

  const config = {
    responsive: true,
    displayModeBar: false,
  };

  Plotly.newPlot("stackedTracksPlot", traces, layout, config).then(() => {
    // Add click handler to jump playhead on plot click
    const plotEl = document.getElementById("stackedTracksPlot");
    plotEl.on("plotly_click", (eventData) => {
      if (eventData && eventData.points && eventData.points.length > 0) {
        const clickedTime = eventData.points[0].x;
        seekToTime(clickedTime);
      }
    });
  });
}

/**
 * Update the playhead line shape on the Plotly plot without full redraw.
 */
function updatePlayheadLine(timeSec) {
  const plotEl = document.getElementById("stackedTracksPlot");
  if (!plotEl || !plotEl.layout || !plotEl.layout.shapes) return;

  const shapes = plotEl.layout.shapes;
  const playheadIdx = shapes.findIndex((s) => s.name === "playhead");
  if (playheadIdx !== -1) {
    shapes[playheadIdx].x0 = timeSec;
    shapes[playheadIdx].x1 = timeSec;
    Plotly.relayout("stackedTracksPlot", { shapes: shapes });
  }
}

/**
 * Render Detected Regions Table.
 */
function renderRegionsTable(data) {
  const flaws = data.flaw_regions || [];
  elements.regionsTableBody.innerHTML = "";

  if (flaws.length === 0) {
    elements.regionsTableBody.innerHTML = `
      <tr>
        <td colspan="6" class="empty-state">No flaw regions flagged by frozen detector rules.</td>
      </tr>
    `;
    return;
  }

  flaws.forEach((flaw, idx) => {
    const tr = document.createElement("tr");
    tr.className = `interactive-row ${idx === state.selectedRegionIndex ? "selected-row" : ""}`;

    const sign = flaw.peak_z >= 0 ? "+" : "";
    const peakZStr = `${sign}${flaw.peak_z.toFixed(2)}`;
    const durStr = `${(flaw.end - flaw.start).toFixed(2)} s`;

    tr.innerHTML = `
      <td><span class="tag tag-${flaw.type}">${flaw.type}</span></td>
      <td>${flaw.start.toFixed(2)} s - ${flaw.end.toFixed(2)} s</td>
      <td>${durStr}</td>
      <td style="font-family: monospace;">${peakZStr}</td>
      <td>${flaw.severity}</td>
      <td><button class="btn btn-secondary btn-sm" type="button">Jump</button></td>
    `;

    tr.addEventListener("click", () => {
      state.selectedRegionIndex = idx;
      seekToTime(flaw.start);
      renderWhyFlagged(data);
      // Update selected row style
      const rows = elements.regionsTableBody.querySelectorAll("tr");
      rows.forEach((r, rIdx) => {
        r.classList.toggle("selected-row", rIdx === idx);
      });
    });

    elements.regionsTableBody.appendChild(tr);
  });
}

/**
 * Render "Why It Was Flagged" Card showing exact numbers and formula.
 */
function renderWhyFlagged(data) {
  const flaws = data.flaw_regions || [];
  if (flaws.length === 0) {
    elements.whyFlaggedContainer.innerHTML = `
      <div class="empty-state">No speech flaws detected. All metrics remained within standard deviations.</div>
    `;
    return;
  }

  const selected = flaws[state.selectedRegionIndex] || flaws[0];
  const unit = selected.unit || "";
  const obsStr = selected.observed_value !== null ? `${selected.observed_value} ${unit}` : "null";
  const baseMeanStr = selected.baseline_mean !== null ? `${selected.baseline_mean} ${unit}` : "null";
  const baseSpreadStr = selected.baseline_spread !== null ? `${selected.baseline_spread} ${unit}` : "null";
  const threshStr = selected.threshold !== null ? selected.threshold : "null";
  const refStr = selected.reference_value !== null ? `${selected.reference_value} ${unit}` : "null (unpaired mode)";
  const winStartStr = selected.peak_window_start !== null ? `${selected.peak_window_start} s` : "null";
  const winEndStr = selected.peak_window_end !== null ? `${selected.peak_window_end} s` : "null";

  const sign = selected.peak_z >= 0 ? "+" : "";
  const peakZStr = `${sign}${selected.peak_z.toFixed(3)}`;
  const zStr = `${sign}${selected.z.toFixed(3)}`;

  let nullReasonHtml = "";
  if (selected.null_reason) {
    nullReasonHtml = `
      <div class="calc-row">
        <span>Window Note:</span>
        <span style="color: var(--color-text-muted);">${selected.null_reason}</span>
      </div>
    `;
  }

  elements.whyFlaggedContainer.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <span class="tag tag-${selected.type}">${selected.type.toUpperCase()} FLAW</span>
      <span style="font-family: monospace; font-size: var(--font-size-sm); font-weight: 600;">
        Region: ${selected.start.toFixed(2)}s to ${selected.end.toFixed(2)}s
      </span>
    </div>

    <div class="calc-box">
      <div class="calc-row">
        <span>Peak Window Start / End:</span>
        <span>${winStartStr} to ${winEndStr}</span>
      </div>
      <div class="calc-row">
        <span>Observed Value (${unit}):</span>
        <span style="font-weight: 700;">${obsStr}</span>
      </div>
      <div class="calc-row">
        <span>Reference Take Value:</span>
        <span>${refStr}</span>
      </div>
      <div class="calc-row">
        <span>Baseline Mean:</span>
        <span>${baseMeanStr}</span>
      </div>
      <div class="calc-row">
        <span>Baseline Spread (denominator):</span>
        <span>${baseSpreadStr}</span>
      </div>
      <div class="calc-row">
        <span>Detection Threshold:</span>
        <span>${threshStr}</span>
      </div>
      ${nullReasonHtml}
      <div class="calc-formula">
        Calculation: z = (${selected.observed_value} - ${selected.baseline_mean}) / ${selected.baseline_spread} = ${zStr} (peak z = ${peakZStr}, check = ${selected.check ? "VALID" : "MISMATCH"})
      </div>
    </div>

    <div class="explanation-quote">
      <strong>Grounded Explanation:</strong><br>
      ${selected.explanation}
    </div>
  `;
}

/**
 * Handle Stress Test tab switching between pitch and rate.
 */
function setStressTab(tab) {
  state.activeStressTab = tab;
  elements.tabPitchLadder.classList.toggle("active", tab === "pitch");
  elements.tabRateLadder.classList.toggle("active", tab === "rate");
  renderStressTest();
}

/**
 * Render Stress Test ladder benchmark data and Plotly chart.
 */
function renderStressTest() {
  if (!state.ladders) return;
  const isPitch = state.activeStressTab === "pitch";
  const ladderObj = isPitch ? state.ladders.pitch : state.ladders.rate;
  if (!ladderObj) return;

  const baseline = ladderObj.baseline || {};
  const summary = ladderObj.ladder_summary || [];
  const threshold = baseline.z_threshold || (isPitch ? -2.0 : 2.0);

  // Update table header
  elements.stressTableHead.innerHTML = `
    <th>Clip File</th>
    <th>${isPitch ? "Pitch Range (k)" : "Speed Factor"}</th>
    <th>Peak z</th>
    <th>Detected</th>
    <th>IoU</th>
  `;

  // Populate table body
  elements.stressTableBody.innerHTML = "";
  summary.forEach((item) => {
    const tr = document.createElement("tr");
    const paramVal = isPitch ? `k = ${item.k}` : `${item.factor}x`;
    const peakZVal = item.peak_z !== null ? item.peak_z.toFixed(3) : "-";
    const iouVal = item.refined_iou !== undefined ? item.refined_iou.toFixed(3) : item.iou !== undefined ? item.iou.toFixed(3) : "-";
    const isDetected = item.detected === "yes";

    tr.innerHTML = `
      <td style="font-family: monospace;">${item.file}</td>
      <td>${paramVal}</td>
      <td style="font-family: monospace;">${peakZVal}</td>
      <td style="font-weight: 600; color: ${isDetected ? "var(--color-accent)" : "var(--color-text-muted)"};">${item.detected}</td>
      <td style="font-family: monospace;">${iouVal}</td>
    `;
    elements.stressTableBody.appendChild(tr);
  });

  // Plotly chart for Ladder Peak z values vs threshold line
  const xLabels = summary.map((item) => (isPitch ? `k=${item.k}` : `${item.factor}x`));
  const yValues = summary.map((item) => item.peak_z);
  const colors = summary.map((item) => (item.detected === "yes" ? "#b84328" : "#888d92"));

  const traceBars = {
    x: xLabels,
    y: yValues,
    type: "bar",
    marker: { color: colors },
    name: "Peak z",
  };

  const layout = {
    margin: { l: 50, r: 20, t: 30, b: 40 },
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    showlegend: false,
    xaxis: {
      title: { text: isPitch ? "Pitch Compression Parameter (k)" : "Rate Speedup Factor", font: { size: 10, family: "monospace" } },
    },
    yaxis: {
      title: { text: "Peak z-score", font: { size: 10, family: "monospace" } },
      zeroline: true,
      zerolinecolor: "#cfc8b8",
    },
    shapes: [
      {
        type: "line",
        xref: "paper",
        yref: "y",
        x0: 0,
        x1: 1,
        y0: threshold,
        y1: threshold,
        line: { color: "#b84328", width: 2, dash: "dot" },
      },
    ],
    annotations: [
      {
        xref: "paper",
        yref: "y",
        x: 0.98,
        y: threshold,
        text: `Threshold: ${threshold}`,
        showarrow: false,
        font: { size: 10, color: "#b84328", family: "monospace" },
        bgcolor: "#ffffff",
      },
    ],
  };

  Plotly.newPlot("stressLadderPlot", [traceBars], layout, { responsive: true, displayModeBar: false });
}

/**
 * Handle "Explain in simple words" button click.
 */
async function handleExplainClick() {
  if (!state.data) return;
  if (state.currentReport) {
    renderExplanationCard(state.currentReport);
    if (elements.explanationCard) {
      elements.explanationCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
    return;
  }
  await loadAndRenderReport(true);
}

/**
 * Prefetch report for precomputed example so click is instantaneous.
 */
async function prefetchExampleReport(exampleId) {
  if (!state.data) return;
  try {
    const payload = {
      example_id: exampleId,
      duration: state.data.duration,
      mode: state.data.mode,
      transcript_source: state.data.transcript_source,
      flaw_regions: state.data.flaw_regions || [],
    };
    const res = await fetch("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      state.currentReport = await res.json();
    }
  } catch (err) {
    console.warn("Report prefetch skipped:", err);
  }
}

/**
 * Request plain-language explanation report from backend.
 */
async function loadAndRenderReport(scrollToCard = false) {
  if (!state.data) return;

  if (elements.explainLoadingSpinner) {
    elements.explainLoadingSpinner.style.display = "inline";
  }
  if (elements.btnExplainReport) {
    elements.btnExplainReport.disabled = true;
  }

  try {
    const payload = {
      example_id: state.currentExample,
      duration: state.data.duration,
      mode: state.data.mode,
      transcript_source: state.data.transcript_source,
      flaw_regions: state.data.flaw_regions || [],
    };

    const res = await fetch("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const report = await res.json();
    state.currentReport = report;
    renderExplanationCard(report);

    if (scrollToCard && elements.explanationCard) {
      elements.explanationCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  } catch (err) {
    console.error("Failed to generate explanation:", err);
    alert("Could not load explanation: " + err.message);
  } finally {
    if (elements.explainLoadingSpinner) {
      elements.explainLoadingSpinner.style.display = "none";
    }
    if (elements.btnExplainReport) {
      elements.btnExplainReport.disabled = false;
    }
  }
}

/**
 * Render explanation report card.
 */
function renderExplanationCard(report) {
  if (!report || !elements.explanationCard) return;

  const isAi = report.source === "ai";

  // Label: "AI-written explanation of the measurements above" or "Template explanation"
  if (elements.explanationCardTitle) {
    elements.explanationCardTitle.textContent = isAi
      ? "AI-written explanation of the measurements above"
      : "Template explanation";
  }

  // Source Badge
  if (elements.explanationSourceBadge) {
    elements.explanationSourceBadge.textContent = isAi ? "AI" : "Template";
    elements.explanationSourceBadge.className = `source-badge ${isAi ? "ai" : "template"}`;
  }

  // Summary
  if (elements.explanationSummary) {
    elements.explanationSummary.textContent = report.summary || "";
  }

  // What Happened (one short paragraph per flagged region)
  if (elements.explanationWhatHappened) {
    elements.explanationWhatHappened.innerHTML = "";
    const items = Array.isArray(report.what_happened)
      ? report.what_happened
      : [report.what_happened];
    items.forEach((pText) => {
      if (!pText) return;
      const p = document.createElement("p");
      p.className = "explanation-paragraph";
      p.textContent = pText;
      elements.explanationWhatHappened.appendChild(p);
    });
  }

  // Why It Matters
  if (elements.explanationWhyMatters) {
    elements.explanationWhyMatters.textContent = report.why_it_matters || "";
  }

  // Practice Tips (three practice tips per region type)
  if (elements.explanationPracticeTips) {
    elements.explanationPracticeTips.innerHTML = "";
    const tips = report.practice_tips || {};

    if (Array.isArray(tips)) {
      const block = document.createElement("div");
      block.className = "practice-type-block";
      const ul = document.createElement("ul");
      ul.className = "practice-tips-list";
      tips.forEach((tipText) => {
        const li = document.createElement("li");
        li.textContent = tipText;
        ul.appendChild(li);
      });
      block.appendChild(ul);
      elements.explanationPracticeTips.appendChild(block);
    } else {
      for (const [regType, tipsList] of Object.entries(tips)) {
        const block = document.createElement("div");
        block.className = "practice-type-block";

        const heading = document.createElement("div");
        heading.className = "practice-type-heading";
        heading.textContent = `${regType.toUpperCase()} DELIVERY TIPS`;
        block.appendChild(heading);

        const ul = document.createElement("ul");
        ul.className = "practice-tips-list";
        if (Array.isArray(tipsList)) {
          tipsList.forEach((tipText) => {
            const li = document.createElement("li");
            li.textContent = tipText;
            ul.appendChild(li);
          });
        }
        block.appendChild(ul);
        elements.explanationPracticeTips.appendChild(block);
      }
    }
  }

  // Caveats
  if (elements.explanationCaveats && elements.explanationCaveatsBlock) {
    if (report.caveats) {
      elements.explanationCaveatsBlock.style.display = "block";
      elements.explanationCaveats.textContent = report.caveats;
    } else {
      elements.explanationCaveatsBlock.style.display = "none";
    }
  }

  elements.explanationCard.style.display = "block";
}

/**
 * Reset chat card state when a new analysis is loaded.
 */
function resetChatForNewAnalysis() {
  state.chatHistory = [];
  state.chatSessionId = Math.random().toString(36).substring(2, 10);

  if (elements.chatInput) {
    elements.chatInput.disabled = false;
    elements.chatInput.value = "";
  }
  if (elements.btnSendChat) {
    elements.btnSendChat.disabled = false;
  }
  document.querySelectorAll(".btn-suggestion").forEach((btn) => {
    btn.disabled = false;
  });

  if (elements.chatMessagesContainer) {
    elements.chatMessagesContainer.innerHTML =
      '<div class="empty-state" id="chatEmptyState">No questions asked yet. Choose a suggested question or enter your own below.</div>';
  }
  if (elements.chatStatusText) {
    elements.chatStatusText.style.display = "none";
    elements.chatStatusText.textContent = "";
  }
}

/**
 * Handle submission of the chat input form.
 */
function handleChatSubmit(e) {
  e.preventDefault();
  if (!elements.chatInput) return;
  const question = elements.chatInput.value.trim();
  if (!question) return;
  sendChatMessage(question);
}

/**
 * Send a question to /api/chat and render user and assistant turns.
 */
async function sendChatMessage(question) {
  if (!state.data) return;
  const cleanQ = question.trim().substring(0, 300);
  if (!cleanQ) return;

  if (elements.chatInput) {
    elements.chatInput.value = "";
  }

  const emptyState = document.getElementById("chatEmptyState");
  if (emptyState) {
    emptyState.remove();
  }

  // Append user message
  appendChatMessage("user", cleanQ);

  // Disable controls while waiting
  setChatControlsDisabled(true);

  // Show plain loading state
  const loadingEl = appendChatLoading();

  try {
    const curves = state.data.curves || {};
    const pCurves = curves.participant || curves;
    let curvesSummary = null;
    if (pCurves && (pCurves.f0_semitones || pCurves.speech_rate || pCurves.energy_z)) {
      curvesSummary = {};
      for (const k of ["f0_semitones", "energy_z", "speech_rate"]) {
        const arr = pCurves[k];
        if (Array.isArray(arr) && arr.length > 0) {
          const valid = arr.filter((v) => v !== null && !isNaN(v));
          if (valid.length > 0) {
            let min = valid[0];
            let max = valid[0];
            let sum = 0;
            for (const v of valid) {
              if (v < min) min = v;
              if (v > max) max = v;
              sum += v;
            }
            curvesSummary[k] = {
              min: Math.round(min * 100) / 100,
              max: Math.round(max * 100) / 100,
              mean: Math.round((sum / valid.length) * 100) / 100,
            };
          }
        }
      }
    }

    const payload = {
      analysis: {
        flaw_regions: state.data.flaw_regions || [],
        mode: state.data.mode || "experimental",
        duration: state.data.duration || 0.0,
        transcript_source: state.data.transcript_source || "provided",
        curves_summary: curvesSummary,
        report: state.currentReport || null,
      },
      question: cleanQ,
      history: state.chatHistory.slice(-4),
      session_id: state.chatSessionId,
    };

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    const answer = data.answer || "No response received.";
    const source = data.source || "template";

    if (loadingEl && loadingEl.parentNode) {
      loadingEl.remove();
    }

    appendChatMessage("assistant", answer, source);

    state.chatHistory.push({ role: "user", content: cleanQ });
    state.chatHistory.push({ role: "assistant", content: answer });
    if (state.chatHistory.length > 8) {
      state.chatHistory = state.chatHistory.slice(-8);
    }
  } catch (err) {
    if (loadingEl && loadingEl.parentNode) {
      loadingEl.remove();
    }
    appendChatMessage(
      "assistant",
      "Network error: could not connect to chat endpoint. Please try again.",
      "template"
    );
  } finally {
    setChatControlsDisabled(false);
    if (elements.chatInput) {
      elements.chatInput.focus();
    }
  }
}

/**
 * Disable or enable chat input and buttons.
 */
function setChatControlsDisabled(disabled) {
  if (elements.chatInput) elements.chatInput.disabled = disabled;
  if (elements.btnSendChat) elements.btnSendChat.disabled = disabled;
  document.querySelectorAll(".btn-suggestion").forEach((btn) => {
    btn.disabled = disabled;
  });
}

/**
 * Append a chat message bubble to the messages log.
 */
function appendChatMessage(role, text, source = "template") {
  if (!elements.chatMessagesContainer) return;

  const msgDiv = document.createElement("div");
  msgDiv.className = `chat-msg chat-msg-${role}`;

  const badge = document.createElement("div");
  badge.className = `chat-msg-badge ${role === "user" ? "user" : source}`;
  if (role === "user") {
    badge.textContent = "Question";
  } else {
    badge.textContent =
      source === "ai"
        ? "AI answer based only on the measurements above"
        : "Template answer";
  }
  msgDiv.appendChild(badge);

  const bubble = document.createElement("div");
  bubble.className = "chat-msg-bubble";
  bubble.textContent = text;
  msgDiv.appendChild(bubble);

  elements.chatMessagesContainer.appendChild(msgDiv);
  elements.chatMessagesContainer.scrollTop = elements.chatMessagesContainer.scrollHeight;
}

/**
 * Display a plain text loading state indicator in the chat container.
 */
function appendChatLoading() {
  if (!elements.chatMessagesContainer) return null;

  const msgDiv = document.createElement("div");
  msgDiv.className = "chat-msg chat-msg-assistant chat-loading";

  const badge = document.createElement("div");
  badge.className = "chat-msg-badge template";
  badge.textContent = "Analyzing measurements...";
  msgDiv.appendChild(badge);

  const bubble = document.createElement("div");
  bubble.className = "chat-msg-bubble chat-loading-text";
  bubble.textContent = "Checking speech data and verified tips...";
  msgDiv.appendChild(bubble);

  elements.chatMessagesContainer.appendChild(msgDiv);
  elements.chatMessagesContainer.scrollTop = elements.chatMessagesContainer.scrollHeight;
  return msgDiv;
}

