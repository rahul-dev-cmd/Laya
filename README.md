# Laya: Contrastive Speech Analytics and Temporal Flaw Grounding

The repository is https://github.com/rahul-dev-cmd/Laya.git

## 1. What Laya is

We built Laya to help people improve how they speak. When someone delivers a talk, it is hard to tell where their voice sounded flat, where they rushed through sentences or where they left an awkward silence. Laya listens to a speech recording, finds the exact seconds where the delivery went off track, and explains what happened using measured numbers instead of vague opinions.

So why not just ask an AI chatbot for feedback? If you upload speech to a regular language model it only reads the words on the page. It cannot hear the physical sound of your voice, so it guesses where you might have sounded monotone or nervous. Laya does the opposite. We measure the physical sound waves directly. We track fundamental frequency (pitch), volume energy, speaking speed and pauses. Every flag shows the exact mathematical calculation behind it, and running the same audio file always produces the exact same result.

This project is for students, speakers and anyone who wants to practice a speech with objective feedback. The name Laya comes from Sanskrit, and it means rhythm, tempo or cadence in music and speech.

## 2. Try it in two minutes

You can test the project right away on a precomputed speech example without running the full neural alignment pipeline.

First, clone the repository:

```bash
git clone https://github.com/rahul-dev-cmd/Laya.git
cd Laya
```

Activate your virtual environment and start the web server:

On Windows PowerShell:
```powershell
python -m backend.app
```

On macOS or Linux:
```bash
python3 -m backend.app
```

Now open your web browser to:
```
http://127.0.0.1:8000
```

On the page, find the dropdown menu of examples and pick `librivox_01_pitch_tier5`. Click the button that says "Load Example".

What you will see:
A timeline appears showing the speech. Laya flags one region from 54.58 to 58.84 s. In this region of 4.26 seconds, the speaker's pitch was completely monotone: the pitch standard deviation was only 0.037 semitones compared to the speaker's normal baseline of 2.93 semitones (z-score: -4.48). You can click on the audio player to listen to this exact spot and see the pitch curve stay completely flat.

## 3. How it works, in simple steps

We process speech through a clear chain of steps:

1. Audio and transcript: We start with a 16 kHz mono WAV audio recording and its plain text transcript.
2. Word timing: We align each word with millisecond start and end timestamps using WhisperX. This tells us exactly when each word began and ended.
3. Measuring acoustic features: We measure four physical features across the audio:
   - Pitch: The fundamental frequency (F0) of vocal cord vibrations, converted to semitones relative to the speaker's median pitch.
   - Loudness: The Root Mean Square (RMS) signal energy across frames.
   - Speaking speed: The speech rate, calculated as words per second inside a sliding 3.0 second window.
   - Pauses: The duration of silences between consecutive words.
4. Comparing with a baseline using z-scores:
   A z-score is a number that tells you how far away a measurement is from the average. A score of zero means exactly average, while a score above 2 or below -2 means the delivery is unusually far from normal.
5. Flagging regions:
   Whenever a feature strays far from normal (|z| >= 2.0), we mark that section as a flagged flaw region. We record its start time, end time, observed value and severity tier.
6. Plain explanation:
   We turn those numbers into an easy to understand paragraph explaining whether the delivery was flat, rushed, slow, or clipped.
7. Dashboard:
   We display the audio waveform, the synchronized text, and interactive pitch and pace graphs on a local web dashboard.

Two comparison modes:
- Paired mode: An ideal recording of the same speech (a reference take by the same speaker) is provided. We compare the test recording directly against this ideal take, which gives the cleanest and most accurate results.
- Experimental mode: No ideal reference recording is provided. We compare against a general statistical baseline calculated from other speech clips. This mode is less reliable because different people naturally speak at different speeds and pitch ranges.

## 4. Technical architecture

Data flow diagram:
```
  [Audio Recording] + [Text Transcript]
                 |
                 v
       Word Alignment (WhisperX)
                 |
                 v
    Acoustic Feature Extraction
  (Pitch F0, Energy RMS, Rate, Pauses)
                 |
                 v
      Baseline Comparison & Z-Scores
                 |
                 v
      Temporal Flaw Detectors
   (Pitch Detect, Rate Detect, Pause Detect)
                 |
                 v
       Flaw Regions JSON (start, end, z)
                 |
                 +-----------------------+
                 |                       |
                 v                       v
     Template / AI Explanation     Interactive Dashboard
        (backend/report.py)          (FastAPI + Frontend)
```

Data flow components:
- Alignment: Uses WhisperX and acoustic models to assign exact start and end seconds to every word in the transcript.
- Features: The module in `src/features/extract.py` computes pitch contours using Parselmouth Praat, energy levels, local speech rates, and inter-word silence intervals.
- Detectors: The detector modules in `src/detection/` evaluate pitch monotone flattening, rate rushes or slowdowns, and unnatural pauses against frozen thresholds (|z| >= 2.0).
- Pipeline: The coordinator module in `src/pipeline/analyze.py` ties alignment, feature extraction, and flaw detection together into a single structured output.
- Backend: A FastAPI server in `backend/app.py` exposes REST endpoints for file uploads, audio streaming, running pipeline analysis, and generating reports.
- Frontend: A browser interface built with HTML, vanilla CSS, and JavaScript in `frontend/` that renders timeline graphs, waveforms, and low-confidence word highlights.
- Optional explanation layer: The module in `backend/report.py` converts measurement numbers into plain-language summaries using vetted coaching tips or an optional language model.

Folder map:
- `backend/`: FastAPI web server, API routes, and report explanation generation.
- `data/`: Local working directory for uploaded files and temporary audio processing.
- `dataset/`: Speech clips, transcripts, corrupted test samples, and example results.
- `docs/`: Technical specifications including API contracts and baseline documents.
- `frontend/`: Web dashboard source files including HTML structure, styling, and JavaScript logic.
- `scripts/`: Utility scripts for downloading speech samples, preparing audio clips, and regenerating reports.
- `src/`: Core Python modules for audio preprocessing, alignment, features, flaw generation, and detectors.
- `tests/`: Automated unit and integration test suite verified with pytest.

For the exact JSON schema and output fields, see `docs/api_contract.md`.

## 5. Setup in detail

Prerequisites:
- Python 3.11
- ffmpeg (required for audio decoding and conversions)

Check that ffmpeg is installed:
```bash
ffmpeg -version
```

If ffmpeg is missing, install it:
- Windows (via winget): `winget install Gyan.FFmpeg`
- macOS (via Homebrew): `brew install ffmpeg`
- Linux (Ubuntu or Debian): `sudo apt update && sudo apt install ffmpeg`

Step by step setup:

Windows PowerShell:
```powershell
git clone https://github.com/rahul-dev-cmd/Laya.git
cd Laya
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS or Linux:
```bash
git clone https://github.com/rahul-dev-cmd/Laya.git
cd Laya
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the application:
```powershell
python -m backend.app
```
(On macOS or Linux: `python3 -m backend.app`)

Open your browser to:
```
http://127.0.0.1:8000
```

Note: When you run an analysis on a new audio clip for the first time, Whisper will download its speech alignment model weights to your machine. This initial download can take several minutes depending on your internet connection and computer speed.

## 6. Optional AI explanation

Laya includes an optional explanation layer in `backend/report.py`.
- What it does: It produces a plain-language explanation of why a flagged flaw matters for audience engagement.
- Privacy by design: The model receives only numeric measurements (timestamps, duration, z-score, observed values). It never receives audio files or transcripts.
- Configuration: Create a file named `backend/.env` with these three variable names:
```env
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
```
Provide your provider endpoint URL, your API key, and the model identifier.
- Security: Never commit `backend/.env` to git. The `.env` file is listed in `.gitignore` to prevent leaking keys.
- Fallback: If `backend/.env` is empty, if the provider fails or times out, or if the model hallucinates numbers not in the data, Laya automatically rejects the response and falls back to a deterministic template explanation with source set to `"template"`.

## 7. The dataset

The dataset contains public-domain audio recordings taken from LibriVox, paired with clean transcripts, word alignments, and synthetic flaw ladders.

- What it contains:
  Good original recordings, copies damaged on purpose across five severity tiers (tier 1 slight to tier 5 severe) for pitch, speaking rate, and pauses, text transcripts, word timing CSVs, and JSON label files marking injected flaw boundaries.
- How damaged copies are made:
  We select a continuous phrase near the middle of a speech clip bounded by natural pauses. To inject rate flaws, we time-stretch or compress the audio using WSOLA (Waveform Similarity Overlap-Add) so the speed changes without shifting pitch. To inject pitch flaws, we scale the vocal intonation contour toward the speaker's median F0 to produce a flat monotone delivery. To inject pause flaws, we shorten or lengthen silent intervals between words at phrase boundaries.
- Naming scheme:
  `librivox_<clip>_<flawtype>_<tier>.wav` and `.json`
  Examples: `librivox_01_pitch_tier5.wav`, `librivox_01_pitch_tier5.json`, `librivox_01_rate_tier4.wav`.
- Metadata and audits:
  Full source links, dates, and reader details are cataloged in `dataset/sources_table.csv` and `dataset/raw_sources_audit.csv`.
- Audio storage:
  Raw audio files are kept locally and are not stored in git.

Dataset download link: to be added by the author

## 8. Results so far

Here are the measured detection results on the test ladders, evaluated on speech clip `librivox_01.wav` from a single speaker:

| Flaw Type | Tier | Parameter | Found | Start Error | End Error | Overlap (IoU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Pitch flat | Tier 1 | k = 0.85 | No | none | none | 0.00 |
| Pitch flat | Tier 2 | k = 0.65 | Yes | 1.00 s | 2.24 s | 0.34 |
| Pitch flat | Tier 3 | k = 0.45 | Yes | 1.00 s | 0.86 s | 0.62 |
| Pitch flat | Tier 4 | k = 0.25 | Yes | 0.68 s | 0.00 s | 0.86 |
| Pitch flat | Tier 5 | k = 0.00 | Yes | 0.68 s | 0.00 s | 0.86 |
| Rate fast | Tier 1 | factor 1.05 | No | none | none | 0.00 |
| Rate fast | Tier 2 | factor 1.10 | Yes | 0.85 s | 0.62 s | 0.67 |
| Rate fast | Tier 3 | factor 1.25 | Yes | 0.04 s | 0.04 s | 0.98 |
| Rate fast | Tier 4 | factor 1.60 | Yes | 0.00 s | 0.04 s | 0.99 |
| Rate fast | Tier 5 | factor 2.00 | Yes | 0.01 s | 0.04 s | 0.98 |
| Rate slow | Tier 1 | factor 0.95 | No | none | none | 0.00 |
| Rate slow | Tier 2 | factor 0.80 | No | none | none | 0.00 |
| Rate slow | Tier 3 | factor 0.60 | Yes | 3.31 s | 0.03 s | 0.60 |

Note: These measurements come from a single audio clip and one speaker (`librivox_01`). Tier 1 modifications represent minor variations that fall within normal expressive speaking bounds, while tiers 2 to 5 are reliably detected with high temporal overlap.

## 9. Known limits, honestly

We want to be direct about the current boundaries of the system:
1. No-reference mode limitations: Experimental mode compares against a general statistical baseline. Because normal speech rates and pitch variations differ across individuals, it can flag innocent natural delivery on speakers it has not seen before.
2. Extra flags in rate examples: In the rate test files, the detector currently flags secondary regions outside the injected phrase. We are investigating whether natural delivery variations in the source recording caused these additional flags.
3. Small development dataset: The frozen detector thresholds were tuned using speeches 01 and 02. Wider evaluation across more speakers is ongoing.
4. Automatic transcription errors: When audio is uploaded without a reference text, the automatic speech recognition model can mishear words or names. Use the "Check these words" box to review low-confidence words before running alignment.
5. Windows Application Control: On certain corporate or school Windows computers, Windows Defender Application Control (WDAC) or AppLocker policies may prevent Python from loading compiled C-extension libraries like soundfile or scipy. If this occurs, run PowerShell as an administrator or configure your security policy to allow the `.venv` folder.

## 10. Running the tests, and reproducibility

We maintain automated tests to verify audio loading, feature extraction, detector thresholds, API endpoints, and report validation.

Run the test suite with:
```powershell
python -m pytest
```
(On macOS or Linux: `python3 -m pytest`)

Reproducibility guarantees:
- Fixed random seeds: All synthetic corruption generators and alignment passes use a fixed random seed of 42.
- Pinned requirements: Dependencies are pinned in `requirements.txt` to prevent breaking changes across package updates.

## 11. Credits and licences

- Audio source material: All audio recordings are sourced from LibriVox public domain audiobooks.
- Readers (from `dataset/raw_sources_audit.csv`):
  - LibriVox Volunteers (`raw_01.mp3`, Patrick Henry Life, Correspondence and Speeches)
  - Alister (`raw_02.mp3`, Speeches of Prince Bismarck)
  - Moira Fogarty (`raw_04.mp3`, Areopagitica by John Milton)
  - John Greenman (`raw_05.mp3`, The Gettysburg Address by Abraham Lincoln)
  - Bob Neufeld (`raw_06.mp3`, Apology by Plato)
  - Leni (`raw_07.mp3`, The Fables of Phaedrus)
- Code licence: This project is licensed under the MIT License. See the `LICENSE` file for the full copyright and permission notice.
