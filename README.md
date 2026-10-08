# Second Take: Contrastive Speech Analytics & Temporal Flaw Grounding

> **Multimodal AI Hackathon 2026 — Track C (Contrastive Speech Analytics & Temporal Flaw Grounding)**  
> English speech analytics tool that pinpoints exact temporal regions where delivery deviates from a baseline and explains flaws with measured numbers.

---

## 🎯 Project Overview

**Second Take** takes an English speech recording and its reference transcript, aligns words down to millisecond-level timestamps using WhisperX, extracts acoustic features (F0/pitch, energy/RMS, MFCCs, spectrogram, HNR, speech rate, and pauses), and computes speaker-normalized contrastive deviations (z-scores) relative to a reference baseline. Flaws are temporally bounded and articulated with template-based quantitative feedback.

### End-to-End Pipeline
```
[ Audio + Reference Transcript ]
               │
               ▼
[ WhisperX Word-Level Forced Alignment ]
               │
               ▼
[ Acoustic Feature Extraction (16 kHz, 10ms hop, 25ms window) ]
  • F0 (semitones rel. to speaker median)
  • Energy / RMS (z-scored per speaker)
  • Spectral / MFCCs / HNR
  • Pause distribution & local articulation rate
               │
               ▼
[ Contrastive Temporal Grounding (|z| >= 2.0) ]
               │
               ▼
[ Template-Based Diagnostic Explanations ]
               │
               ▼
[ Interactive Contrastive Dashboard (Next.js + WaveSurfer.js) ]
```

---

## 📐 Project Rules & Standards

All code adheres strictly to hackathon specifications:
- **Python**: 3.10 or 3.11 (`Python 3.11.9` recommended).
- **Audio standard**: 16,000 Hz sample rate, single-channel (mono), 16-bit PCM WAV / float32 in `[-1.0, 1.0]`.
- **STFT / Framing standard**:
  - Analysis window: **25 ms** (400 samples at 16 kHz)
  - Hop size: **10 ms** (160 samples at 16 kHz)
- **Speaker Normalization**:
  - F0 computed in semitones relative to speaker median.
  - Energy / RMS z-scored per speaker.
- **Anomaly Criterion**: Flag a temporal region as flawed only if $|z| \ge 2.0$.
- **Explanations**: Template-based with measured numbers (deterministic and interpretable; no uncontrolled LLM hallucination).
- **Reproducibility**: Fixed random seeds everywhere (`seed=42`). Pinned dependencies.
- **Compute footprint**: Lightweight CPU inference by default; optional GPU acceleration via `--device cuda`.
- **Data safety**: Never commit audio binaries, API keys, or `.env` files. Strictly verified public domain / CC-BY speech sources only.

---

## 📁 Repository Structure

```
Laya/
├── .gitignore                     # Audio binaries, secrets, virtualenvs excluded
├── LICENSE                        # MIT License
├── README.md                      # Project documentation and guide
├── requirements.txt               # Pinned dependencies for Python 3.11
├── data/
│   ├── raw/                       # Original audio files (gitignored)
│   ├── processed/                 # Standardized 16 kHz mono WAVs (gitignored)
│   ├── transcripts/               # Speech transcripts & source metadata
│   ├── corrupted/                 # Synthetically perturbed copies (Phase 3)
│   └── held_out/                  # Real self-recorded held-out evaluation set
├── src/
│   ├── __init__.py
│   ├── config.py                  # Global constants (16 kHz, 10ms hop, 25ms window, seed)
│   ├── audio/
│   │   ├── __init__.py
│   │   └── preprocess.py          # Audio loading, 16 kHz conversion, framing, RMS
│   ├── alignment/
│   │   ├── __init__.py
│   │   └── aligner.py             # WhisperX word-level alignment engine
│   └── utils/
│       ├── __init__.py
│       ├── seed.py                # Reproducibility seed setter across torch/numpy/python
│       └── io.py                  # JSON and transcript loading/saving
├── scripts/
│   ├── download_sample_speech.py  # Fetches verified LibriVox speech (Gettysburg Address)
│   └── run_alignment.py           # CLI tool to run WhisperX word alignment
└── tests/
    ├── __init__.py
    ├── test_audio_preprocess.py   # Unit tests for 16 kHz mono loading & framing
    └── test_io.py                 # Unit tests for seed reproducibility & I/O
```

---

## 🚀 Phase 1: Environment & Alignment Setup

### 1. Environment Setup (Python 3.11)
```powershell
# Create virtual environment
py -3.11 -m venv .venv

# Activate environment
.venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Run Unit Tests
```powershell
pytest -v
```

### 3. Fetch Public-Domain Sample Speech
Downloads Abraham Lincoln's Gettysburg Address (read by John Greenman for LibriVox, hosted on Internet Archive under Public Domain dedication), writes the reference transcript, and converts audio to standardized 16 kHz mono WAV:
```powershell
python scripts/download_sample_speech.py
```

### 4. Run Word-Level Alignment
Aligns the audio recording against speech frames using WhisperX to produce millisecond-accurate word start and end boundaries:
```powershell
# CPU inference (default)
python scripts/run_alignment.py

# Optional: with GPU acceleration if available
python scripts/run_alignment.py --device cuda
```

Output is saved to `data/processed/gettysburg_address_16k_aligned.json` with structured word intervals:
```json
{
  "audio_file": ".../data/processed/gettysburg_address_16k.wav",
  "duration_seconds": 158.03,
  "sample_rate": 16000,
  "device": "cpu",
  "total_words": 272,
  "words": [
    { "word": "Four", "start": 12.45, "end": 12.82, "score": 0.94 },
    { "word": "score", "start": 12.86, "end": 13.24, "score": 0.98 },
    { "word": "and", "start": 13.26, "end": 13.41, "score": 0.96 },
    { "word": "seven", "start": 13.43, "end": 13.85, "score": 0.97 }
  ]
}
```

---

## ⚡ Rate Flaw Ladder Mapping Table (`librivox_01`)

The speech rate / tempo flaw ladder evaluates local articulation rate perturbations across 5 speed-up tiers and 3 slow-down tiers, anchored to words 174–191 (*"However this may be it is certain that he soon became domesticated in the family of Colonel Syme"*) in `librivox_01.wav` (`53.90s` to `58.84s`):

| File Name | Tier Name | Direction | Factor | Description | Flaw Region (Corrupted Timeline) | Flaw Duration | Time Shift |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| `librivox_01_rate_tier1.wav` | Tier 1 | Speed-up | **1.05** | Subtle speed-up (+5%) | `[53.900s, 58.605s]` | 4.705 s | -0.235 s |
| `librivox_01_rate_tier2.wav` | Tier 2 | Speed-up | **1.10** | Mild speed-up (+10%) | `[53.900s, 58.391s]` | 4.491 s | -0.449 s |
| `librivox_01_rate_tier3.wav` | Tier 3 | Speed-up | **1.25** | Moderate speed-up (+25%) | `[53.900s, 57.852s]` | 3.952 s | -0.988 s |
| `librivox_01_rate_tier4.wav` | Tier 4 | Speed-up | **1.60** | Strong speed-up (+60%) | `[53.900s, 56.988s]` | 3.088 s | -1.853 s |
| `librivox_01_rate_tier5.wav` | Tier 5 | Speed-up | **2.00** | Extreme speed-up (+100%) | `[53.900s, 56.370s]` | 2.470 s | -2.470 s |
| `librivox_01_rate_slow_tier1.wav` | Slow Tier 1 | Slow-down | **0.95** | Subtle slow-down (-5%) | `[53.900s, 59.100s]` | 5.200 s | +0.260 s |
| `librivox_01_rate_slow_tier2.wav` | Slow Tier 2 | Slow-down | **0.80** | Moderate slow-down (-20%) | `[53.900s, 60.075s]` | 6.175 s | +1.235 s |
| `librivox_01_rate_slow_tier3.wav` | Slow Tier 3 | Slow-down | **0.60** | Strong slow-down (-40%) | `[53.900s, 62.133s]` | 8.233 s | +3.293 s |
| `librivox_01_control.wav` | Control | Neutral | **1.00** | Resynthesized control | *None* | — | 0.000 s |

*Note: Existing `librivox_01_rate_t3.wav` (factor 1.25) and `librivox_01_rate_t5.wav` (factor 1.60) are preserved without modification.*

### Rate Ladder Detection Summary (`rate_ladder_results.json`)

| File | Factor | Detected | Peak $z$ | Start Error | End Error | IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `librivox_01_rate_tier1.wav` | 1.05 | no | $+1.921$ | N/A | N/A | 0.0000 |
| `librivox_01_rate_tier2.wav` | 1.10 | yes | $+2.476$ | 0.8460s | 0.6239s | 0.6727 |
| `librivox_01_rate_tier3.wav` | 1.25 | yes | $+3.587$ | 0.0370s | 0.0350s | 0.9819 |
| `librivox_01_rate_tier4.wav` | 1.60 | yes | $+4.697$ | 0.0040s | 0.0375s | 0.9867 |
| `librivox_01_rate_tier5.wav` | 2.00 | yes | $+5.253$ | 0.0140s | 0.0360s | 0.9802 |
| `librivox_01_rate_slow_tier1.wav` | 0.95 | no | $-1.411$ | N/A | N/A | 0.0000 |
| `librivox_01_rate_slow_tier2.wav` | 0.80 | no | $-1.966$ | N/A | N/A | 0.0000 |
| `librivox_01_rate_slow_tier3.wav` | 0.60 | yes | $-2.521$ | 3.3080s | 0.0253s | 0.5951 |
| `librivox_01_control.wav` | 1.00 | no | $+1.921$ | N/A | N/A | 0.0000 |

---

## 🗺️ Project Roadmap
- [x] **Phase 1**: Repo skeleton, reproducible environment, audio preprocessing, and WhisperX word alignment.
- [x] **Phase 2**: Forced alignment and temporal rate-flaw grounding module.
- [x] **Phase 3**: Full 8-point speech tempo flaw ladder generation (5 speed-up + 3 slow-down).
- [ ] **Phase 4**: Acoustic feature extraction (F0 semitones, energy z-scores, MFCCs, HNR, pauses).
- [ ] **Phase 5**: Interactive Contrastive Dashboard (Next.js + WaveSurfer.js).
