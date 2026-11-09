#!/usr/bin/env python3
"""FastAPI backend application for the Laya Speech Coaching Dashboard.

Endpoints:
- GET /: Serves frontend/index.html
- GET /api/examples: Lists all precomputed example JSON filenames in dataset/examples/
- GET /api/example/{name}: Loads and returns a specific example JSON (with ground truth if present)
- GET /api/ladders: Returns pitch and rate stress test ladder data from dataset/corrupted/
- GET /api/audio: Serves WAV audio files with HTTP range streaming and clear error if missing
- POST /api/analyze: Runs forced alignment, feature extraction, and flaw detection
"""

import csv
import json
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import soundfile as sf

from src.audio.convert import (
    MAX_UPLOAD_BYTES,
    MAX_DURATION_SECONDS,
    validate_audio_content,
    convert_to_wav16k_mono,
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer project's virtual environment when running under another Python executable
venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and sys.executable.lower() != str(venv_python).lower():
    import subprocess
    argv0 = sys.argv[0] if sys.argv else ""
    cmd_name = Path(argv0).stem.lower() if argv0 else ""
    if "uvicorn" in cmd_name:
        venv_uvicorn = PROJECT_ROOT / ".venv" / "Scripts" / "uvicorn.exe"
        if venv_uvicorn.exists():
            sys.exit(subprocess.call([str(venv_uvicorn)] + sys.argv[1:]))
        else:
            sys.exit(subprocess.call([str(venv_python), "-m", "uvicorn"] + sys.argv[1:]))
    elif argv0:
        sys.exit(subprocess.call([str(venv_python)] + sys.argv))

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.chat import router as chat_router
from backend.gnani_stt import (
    DEFAULT_KEYTERMS,
    compute_script_mix,
    merge_keyterms,
    transcribe_audio_gnani,
)
from backend.report import router as report_router
from src.pipeline.analyze import analyze, warmup_whisper_model

app = FastAPI(
    title="Laya",
    description="Contrastive Speech Analytics and Temporal Flaw Grounding Dashboard",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount AI report and chat endpoints
app.include_router(report_router)
app.include_router(chat_router)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
EXAMPLES_DIR = PROJECT_ROOT / "dataset" / "examples"
CORRUPTED_DIR = PROJECT_ROOT / "dataset" / "corrupted"
SOURCES_DIR = PROJECT_ROOT / "dataset" / "sources"
SCRATCH_DIR = PROJECT_ROOT / "scratch" / "uploads"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# Pre-load Whisper model at startup so first upload is fast
print("[Laya] Warming up Whisper model...")
warmup_whisper_model()
print("[Laya] Whisper model ready.")


def find_audio_file(file_str: str) -> Optional[Path]:
    """Resolve an audio file path against scratch, corrupted, sources, and examples directories."""
    raw_path = Path(file_str)
    if raw_path.exists() and raw_path.is_file():
        return raw_path

    clean_name = raw_path.name
    candidates = [
        PROJECT_ROOT / file_str,
        SCRATCH_DIR / clean_name,
        CORRUPTED_DIR / clean_name,
        SOURCES_DIR / clean_name,
        PROJECT_ROOT / "dataset" / "examples" / clean_name,
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    if SCRATCH_DIR.exists():
        for sub in SCRATCH_DIR.rglob(clean_name):
            if sub.is_file():
                return sub
    return None


@app.get("/")
def get_index():
    """Serve the single-page frontend interface."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="frontend/index.html not found. Please verify project structure.",
        )
    return FileResponse(index_file, media_type="text/html")


@app.get("/style.css")
def get_stylesheet():
    """Serve the primary stylesheet containing all CSS variables."""
    css_file = FRONTEND_DIR / "style.css"
    if not css_file.exists():
        raise HTTPException(status_code=404, detail="frontend/style.css not found.")
    return FileResponse(css_file, media_type="text/css")


@app.get("/app.js")
def get_javascript():
    """Serve the frontend application logic."""
    js_file = FRONTEND_DIR / "app.js"
    if not js_file.exists():
        raise HTTPException(status_code=404, detail="frontend/app.js not found.")
    return FileResponse(js_file, media_type="application/javascript")


@app.get("/api/examples")
def list_examples(include_hidden: bool = False) -> List[Dict[str, Any]]:
    """List available example JSON datasets from dataset/examples/manifest.json."""
    manifest_path = EXAMPLES_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        examples = []
        for item in manifest:
            if not include_hidden and not item.get("visible", True):
                continue
            ex = dict(item)
            ex.setdefault("id", item["name"])
            ex.setdefault("filename", f"{item['name']}.json")
            examples.append(ex)
        return examples

    if not EXAMPLES_DIR.exists():
        return []
    examples = []
    for p in sorted(EXAMPLES_DIR.glob("*.json")):
        if p.name == "manifest.json":
            continue
        stem = p.stem
        parts = stem.split("_")
        title = " ".join(parts).title()
        examples.append({"id": stem, "name": title, "filename": p.name, "visible": True})
    return examples


@app.get("/api/passages")
def list_passages() -> List[Dict[str, Any]]:
    """List practice passages having verified transcript (.txt) and reference WAV in dataset/sources/."""
    if not SOURCES_DIR.exists():
        return []

    # Map clip_id to raw_mp3 and duration from dataset/sources_table.csv
    clip_to_raw = {}
    clip_to_dur = {}
    table_csv = PROJECT_ROOT / "dataset" / "sources_table.csv"
    if table_csv.exists():
        try:
            with open(table_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("clip_id", "").strip()
                    raw = row.get("raw_mp3", "").strip()
                    dur_str = row.get("duration_seconds", "").strip()
                    if cid:
                        clip_to_raw[cid] = raw
                        if dur_str:
                            try:
                                clip_to_dur[cid] = round(float(dur_str), 1)
                            except ValueError:
                                pass
        except Exception:
            pass

    # Map raw_mp3 to source description from dataset/raw_sources_audit.csv
    raw_to_source = {}
    audit_csv = PROJECT_ROOT / "dataset" / "raw_sources_audit.csv"
    if audit_csv.exists():
        try:
            with open(audit_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_file = row.get("file", "").strip()
                    url = row.get("url", "").strip()
                    notes = row.get("notes", "").strip()
                    book = ""
                    for part in notes.split(";"):
                        if "Book:" in part:
                            book = part.split("Book:")[-1].strip()
                    source_str = book or (url if url != "UNKNOWN" else "") or "unknown"
                    if raw_file:
                        raw_to_source[raw_file] = source_str
        except Exception:
            pass

    passages = []
    for wav_path in sorted(SOURCES_DIR.glob("librivox_*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        # Only include passages that have BOTH wav and txt
        if not txt_path.exists():
            continue

        clip_id = wav_path.stem
        dur = clip_to_dur.get(clip_id)
        if dur is None:
            try:
                info = sf.info(str(wav_path))
                dur = round(float(info.duration), 1)
            except Exception:
                dur = 0.0

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text_content = f.read().strip()
        except Exception:
            text_content = ""

        raw_mp3 = clip_to_raw.get(clip_id, "")
        source_val = raw_to_source.get(raw_mp3, "unknown")

        passages.append({
            "id": clip_id,
            "title": clip_id,
            "source": source_val,
            "duration": dur,
            "text": text_content,
            "wav_file": wav_path.name,
        })

    return passages


@app.get("/api/example/{name}")
def get_example(name: str):
    """Retrieve an example JSON payload with optional ground truth metadata."""
    clean_name = name.replace(".json", "")
    target = EXAMPLES_DIR / f"{clean_name}.json"
    if not target.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Example file '{clean_name}.json' not found in dataset/examples/.",
        )

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Attach ground truth injected region if dataset/corrupted/<name>.json exists
    gt_file = CORRUPTED_DIR / f"{clean_name}.json"
    if gt_file.exists():
        try:
            with open(gt_file, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            fr = gt_data.get("flaw_region") or {}
            ct = gt_data.get("corrupted_timeline") or {}
            s = fr.get("start_seconds") if "start_seconds" in fr else ct.get("flaw_start_seconds")
            e = fr.get("end_seconds") if "end_seconds" in fr else ct.get("flaw_end_seconds")
            if s is not None and e is not None:
                data["ground_truth"] = {
                    "flaw_type": gt_data.get("flaw_type", ""),
                    "start": round(float(s), 3),
                    "end": round(float(e), 3),
                    "duration": round(float(e - s), 3),
                    "tier": gt_data.get("tier"),
                    "description": gt_data.get("description", "Injected flaw (ground truth)"),
                }
        except Exception:
            pass

    # Ensure transcript and transcript_source exist for downstream interface display
    if "transcript" not in data or not data["transcript"]:
        data["transcript"] = " ".join(w["word"] for w in data.get("words", []))
    if "transcript_source" not in data or not data["transcript_source"]:
        data["transcript_source"] = "provided"

    return data


@app.get("/api/ladders")
def get_ladders():
    """Retrieve pitch and rate stress test ladder benchmark summaries."""
    pitch_file = CORRUPTED_DIR / "pitch_ladder_results.json"
    rate_file = CORRUPTED_DIR / "rate_ladder_results.json"

    pitch_data = None
    if pitch_file.exists():
        with open(pitch_file, "r", encoding="utf-8") as f:
            pitch_data = json.load(f)

    rate_data = None
    if rate_file.exists():
        with open(rate_file, "r", encoding="utf-8") as f:
            rate_data = json.load(f)

    return {
        "pitch": pitch_data,
        "rate": rate_data,
    }


@app.get("/api/audio")
def get_audio(path: str = Query(..., description="File path or filename of WAV audio")):
    """Serve WAV audio with HTTP byte-range request streaming and missing file warning."""
    resolved = find_audio_file(path)
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Audio file '{path}' could not be found on the server. "
                f"Expected location in dataset/corrupted/ or dataset/sources/."
            ),
        )
    return FileResponse(
        resolved,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"},
    )


@app.post("/api/analyze")
async def run_analysis(
    audio_file: UploadFile = File(..., description="Participant audio WAV, WebM, Ogg, or MP4 file"),
    passage_id: Optional[str] = Form(None, description="Optional practice passage identifier"),
    reference_passage: Optional[str] = Form(None, description="Alias for passage_id"),
    transcript_text: Optional[str] = Form(None, description="Transcript text content"),
    transcript_file: Optional[UploadFile] = File(None, description="Transcript text TXT file"),
    reference_file: Optional[UploadFile] = File(None, description="Optional reference WAV file"),
    whisper_model: str = Form("tiny.en", description="Whisper model size: tiny.en, base.en, small.en, medium.en"),
    transcript_source: Optional[str] = Form(None, description="Transcript provenance label: provided, auto, edited"),
    language_mode: str = Form("english", description="Language mode: english, hindi, hinglish"),
    keyterms: Optional[str] = Form(None, description="Newline-delimited user keyterms for Hindi/Hinglish"),
):
    """Run full analysis pipeline on uploaded audio and transcript.
    
    Accepts WAV, WebM, Ogg, and MP4. Validates file content and duration (<= 150 s, <= 25 MB).
    Converts audio with ffmpeg to 16 kHz mono WAV.
    Temporary files are generated with random names and deleted immediately after analysis.
    """
    valid_models = {"tiny.en", "base.en", "small.en", "medium.en"}
    if whisper_model not in valid_models:
        whisper_model = "small.en"

    # 1. Read uploaded audio bytes with 25 MB limit check
    raw_bytes = await audio_file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
        )

    # 2. Create isolated random temp folder for this upload
    random_id = secrets.token_hex(12)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"rec_{random_id}_", dir=str(SCRATCH_DIR)))

    try:
        # Determine suffix from original filename or default based on magic bytes
        orig_suffix = Path(audio_file.filename or "").suffix.lower()
        if not orig_suffix:
            orig_suffix = ".webm" if raw_bytes.startswith(b"\x1a\x45\xdf\xa3") else ".audio"

        temp_raw = temp_dir / f"upload_{secrets.token_hex(8)}{orig_suffix}"
        temp_raw.write_bytes(raw_bytes)

        # 3. Content and duration validation
        is_valid, duration, err_msg = validate_audio_content(
            temp_raw,
            max_bytes=MAX_UPLOAD_BYTES,
            max_duration=MAX_DURATION_SECONDS,
        )
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Audio validation failed: {err_msg}"
            )

        # 4. Check ffmpeg availability and convert to 16 kHz mono WAV
        if not shutil.which("ffmpeg"):
            raise HTTPException(
                status_code=500,
                detail="ffmpeg is required for audio conversion but was not found on PATH. Please install ffmpeg."
            )

        converted_wav = temp_dir / f"conv_{secrets.token_hex(8)}.wav"
        try:
            convert_to_wav16k_mono(temp_raw, converted_wav)
        except RuntimeError as conv_err:
            raise HTTPException(status_code=500, detail=str(conv_err))

        # Remove raw upload file after conversion
        if temp_raw.exists():
            temp_raw.unlink()

        # 5. Handle transcript and reference audio (passage vs uploaded)
        pid = passage_id or reference_passage
        ref_path = None
        txt_path = None

        if pid:
            passage_wav = SOURCES_DIR / f"{pid}.wav"
            passage_txt = SOURCES_DIR / f"{pid}.txt"
            if passage_wav.exists() and passage_wav.is_file():
                ref_path = passage_wav
            if not transcript_text and passage_txt.exists() and passage_txt.is_file():
                txt_path = passage_txt
                transcript_source = transcript_source or "provided"

        if not ref_path and reference_file and reference_file.filename:
            ref_bytes = await reference_file.read()
            ref_path = temp_dir / f"ref_{secrets.token_hex(8)}.wav"
            ref_path.write_bytes(ref_bytes)

        is_indic = language_mode.lower() in ("hindi", "hinglish")
        gnani_chunks = []

        if is_indic and not txt_path:
            # Transcribe via Gnani STT with chunking (<= 25s) and keyterms
            gnani_res = transcribe_audio_gnani(
                wav_path=converted_wav,
                language_mode=language_mode,
                custom_keyterms=keyterms,
            )
            if not gnani_res.get("success"):
                raise HTTPException(
                    status_code=502,
                    detail=f"Gnani STT transcription failed: {gnani_res.get('error')}"
                )

            t_text = gnani_res["transcript"]
            candidate_path = temp_dir / f"transcript_{secrets.token_hex(8)}.txt"
            candidate_path.write_text(t_text, encoding="utf-8")
            txt_path = candidate_path
            transcript_source = "auto"
            gnani_chunks = gnani_res.get("chunks", [])

            # Write internal alignment CSV only for acoustic curve generation
            raw_words = t_text.split()
            if raw_words and duration > 0:
                step = duration / len(raw_words)
                align_csv = temp_dir / f"{converted_wav.stem}_alignment.csv"
                csv_lines = ["word,start_seconds,end_seconds,confidence"]
                for i, w in enumerate(raw_words):
                    csv_lines.append(f"{w},{round(i * step, 3)},{round((i + 1) * step, 3)},0.95")
                align_csv.write_text("\n".join(csv_lines), encoding="utf-8")

        elif is_indic and txt_path:
            # User edited transcript or provided text for Hindi/Hinglish
            with open(txt_path, "r", encoding="utf-8") as f:
                edited_text = f.read().strip()
            transcript_source = transcript_source or "edited"
            raw_words = edited_text.split()
            if raw_words and duration > 0:
                step = duration / len(raw_words)
                align_csv = temp_dir / f"{converted_wav.stem}_alignment.csv"
                csv_lines = ["word,start_seconds,end_seconds,confidence"]
                for i, w in enumerate(raw_words):
                    csv_lines.append(f"{w},{round(i * step, 3)},{round((i + 1) * step, 3)},0.95")
                align_csv.write_text("\n".join(csv_lines), encoding="utf-8")

            # Build chunk level times from audio duration (at most 25s chunks)
            chunk_list = []
            c_start = 0.0
            c_idx = 0
            while c_start < duration:
                c_end = min(c_start + 25.0, duration)
                chunk_list.append({
                    "chunk_index": c_idx,
                    "chunk_start": round(c_start, 3),
                    "chunk_end": round(c_end, 3),
                    "chunk_duration": round(c_end - c_start, 3),
                    "chunk_start_seconds": round(c_start, 3),
                    "chunk_end_seconds": round(c_end, 3),
                    "chunk_duration_seconds": round(c_end - c_start, 3),
                    "transcript": edited_text,
                    "words": [{"word": w} for w in raw_words],
                })
                c_start = c_end
                c_idx += 1
            gnani_chunks = chunk_list

        # 6. Run pipeline analysis
        result = analyze(
            audio_wav=converted_wav,
            transcript_txt=txt_path,
            reference_wav=ref_path,
            whisper_model=whisper_model,
            transcript_source=transcript_source,
            use_precomputed_alignment=is_indic,
        )

        if is_indic:
            result["language_mode"] = language_mode.lower()
            result["script_mix"] = compute_script_mix(result.get("transcript", ""))
            result["keyterms"] = merge_keyterms(keyterms)
            result["chunks"] = gnani_chunks
            result["timing_level"] = "chunk"
            # Prisma returns no word times. Remove any per-word start or end times for Hindi and Hinglish.
            # Keep only chunk level times (chunk start and end from our own chunking).
            result["words"] = [{"word": w["word"] if isinstance(w, dict) else str(w)} for w in result.get("words", [])]

        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")
    finally:
        # Delete temporary files after the analysis; never store recordings permanently
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/transcribe-gnani")
async def transcribe_gnani_endpoint(
    audio_file: UploadFile = File(..., description="Audio file for Gnani STT"),
    language_mode: str = Form("hinglish", description="Language mode: hindi or hinglish"),
    keyterms: Optional[str] = Form(None, description="Optional user keyterms (one per line)"),
):
    """Direct STT transcription via Gnani Prisma v3.0 REST with chunking (<= 25s)."""
    raw_bytes = await audio_file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
        )

    random_id = secrets.token_hex(12)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"stt_{random_id}_", dir=str(SCRATCH_DIR)))
    try:
        orig_suffix = Path(audio_file.filename or "").suffix.lower() or ".audio"
        temp_raw = temp_dir / f"upload_{secrets.token_hex(8)}{orig_suffix}"
        temp_raw.write_bytes(raw_bytes)

        if not shutil.which("ffmpeg"):
            raise HTTPException(status_code=500, detail="ffmpeg is required for audio conversion.")

        converted_wav = temp_dir / f"conv_{secrets.token_hex(8)}.wav"
        convert_to_wav16k_mono(temp_raw, converted_wav)

        res = transcribe_audio_gnani(
            wav_path=converted_wav,
            language_mode=language_mode,
            custom_keyterms=keyterms,
        )
        if not res.get("success"):
            raise HTTPException(status_code=502, detail=res.get("error", "Gnani STT failed."))
        return JSONResponse(res)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Mount static assets if frontend folder has extra files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def main():
    """Run backend server via uvicorn."""
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[Laya Dashboard] Starting server at http://127.0.0.1:{port}")
    uvicorn.run("backend.app:app", host="127.0.0.1", port=port, reload=True)


if __name__ == "__main__":
    main()
