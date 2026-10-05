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

## 🗺️ Project Roadmap
- [x] **Phase 1 (Current)**: Repo skeleton, reproducible environment, audio preprocessing, and WhisperX word alignment on LibriVox sample speech.
- [ ] **Phase 2**: Acoustic feature extraction (F0 semitones, energy z-scores, MFCCs, HNR, pauses).
- [ ] **Phase 3**: Corruption engine (4 flaw types × 5 severity tiers).
- [ ] **Phase 4**: Temporal flaw grounding ($|z| \ge 2$) & template explanations.
- [ ] **Phase 5**: Interactive Contrastive Dashboard (Next.js + WaveSurfer.js).
