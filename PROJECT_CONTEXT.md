# Laya: Project Context

## What we're building
Laya takes a speech recording plus its transcript, finds the time regions where delivery
deviates from a baseline, and explains each flaw with measured numbers. For Multimodal AI
Hackathon 2026, Track C (Contrastive Speech Analytics & Temporal Flaw Grounding).
English only for now.

## Pipeline
audio + transcript -> WhisperX word alignment -> features (F0, energy, MFCCs, FFT, HNR,
speech rate, pauses) -> speaker-normalized comparison to baseline (z-scores) -> flaw regions
with start/end times -> template-based explanations -> dashboard overlay.

## Dataset
- About 15 public-domain / CC-BY English speeches (LibriVox first). No TED.
- Flaw types: pitch, pause, rate (energy later), several severity tiers each, parameters logged.
- Labels are JSON; flaw regions are in the CORRUPTED file's own timeline.
- 5-10 real self-recorded held-out samples, never used for tuning.

## Rules for all code
- Python 3.10/3.11, 16 kHz mono, 10 ms hop, 25 ms window. Fixed seeds. Pinned requirements.
- F0 in semitones relative to the speaker's median; energy z-scored per speaker.
- Flag at |z| >= 2 (pitch detector has its own frozen thresholds in src/detection/config.py).
- Explanations are template-based, not LLM.
- CPU by default. Never commit audio, API keys or .env. Never guess licenses or URLs.
- One module at a time, simple commented code.

## Evaluation discipline (important)
- Detector thresholds are FROZEN. Do not change them or tune on clips 01 and 02.
- docs/preregistered_pause_baseline.md is a committed design; evaluate it only on
  speeches added after 9 Oct 2026 (clips 03 onward). Treat those as TEST data: do not run
  any detector on them until the full set is assembled.
- Report Mode A (own original baseline) as an upper bound. Report means AND ranges.

## Current phase
Pitch and pause detectors built; rate ladder built on clip 01. Adding speeches 03 to 10 using
scripts/prep_clip.py. Next: more flaw types (energy), Mode B baselines, explanations.
Do not build the frontend yet.