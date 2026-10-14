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

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from src.pipeline.analyze import analyze

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

FRONTEND_DIR = PROJECT_ROOT / "frontend"
EXAMPLES_DIR = PROJECT_ROOT / "dataset" / "examples"
CORRUPTED_DIR = PROJECT_ROOT / "dataset" / "corrupted"
SOURCES_DIR = PROJECT_ROOT / "dataset" / "sources"
SCRATCH_DIR = PROJECT_ROOT / "scratch" / "uploads"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


def find_audio_file(file_str: str) -> Optional[Path]:
    """Resolve an audio file path against corrupted and sources directories."""
    clean_name = Path(file_str).name
    candidates = [
        PROJECT_ROOT / file_str,
        CORRUPTED_DIR / clean_name,
        SOURCES_DIR / clean_name,
        PROJECT_ROOT / "dataset" / "examples" / clean_name,
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
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
def list_examples() -> List[Dict[str, str]]:
    """List all available example JSON datasets in dataset/examples/."""
    if not EXAMPLES_DIR.exists():
        return []
    examples = []
    for p in sorted(EXAMPLES_DIR.glob("*.json")):
        stem = p.stem
        # Clean title for selector
        parts = stem.split("_")
        title = " ".join(parts).title()
        examples.append({"id": stem, "name": title, "filename": p.name})
    return examples


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
    audio_file: UploadFile = File(..., description="Participant audio WAV file"),
    transcript_text: Optional[str] = Form(None, description="Transcript text content"),
    transcript_file: Optional[UploadFile] = File(None, description="Transcript text TXT file"),
    reference_file: Optional[UploadFile] = File(None, description="Optional reference WAV file"),
    whisper_model: str = Form("small.en", description="Whisper model size: tiny.en, base.en, small.en, medium.en"),
    transcript_source: Optional[str] = Form(None, description="Transcript provenance label: provided, auto, edited"),
):
    """Run full analysis pipeline on uploaded audio and transcript."""
    valid_models = {"tiny.en", "base.en", "small.en", "medium.en"}
    if whisper_model not in valid_models:
        whisper_model = "small.en"

    # Create isolated temp folder for this upload
    temp_dir = Path(tempfile.mkdtemp(dir=str(SCRATCH_DIR)))
    try:
        # 1. Save participant audio
        audio_path = temp_dir / audio_file.filename
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)

        # 2. Save transcript if provided (optional)
        txt_path = None
        if transcript_file and transcript_file.filename:
            candidate_path = temp_dir / "transcript.txt"
            with open(candidate_path, "wb") as f:
                shutil.copyfileobj(transcript_file.file, f)
            txt_path = candidate_path
        elif transcript_text and transcript_text.strip():
            candidate_path = temp_dir / "transcript.txt"
            with open(candidate_path, "w", encoding="utf-8") as f:
                f.write(transcript_text.strip())
            txt_path = candidate_path

        # 3. Optional reference audio
        ref_path = None
        if reference_file and reference_file.filename:
            ref_path = temp_dir / reference_file.filename
            with open(ref_path, "wb") as f:
                shutil.copyfileobj(reference_file.file, f)

        # 4. Run pipeline analysis
        result = analyze(
            audio_wav=audio_path,
            transcript_txt=txt_path,
            reference_wav=ref_path,
            whisper_model=whisper_model,
            transcript_source=transcript_source,
        )

        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")
    finally:
        # Cleanup uploaded files from disk
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
