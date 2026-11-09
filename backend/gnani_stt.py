#!/usr/bin/env python3
"""Gnani Prisma STT and Timbre TTS Integration for Laya.

Adheres strictly to the following rules:
1. Do NOT use the STT Batch endpoint at all. Use only the quick endpoint
   https://api.vachana.ai/stt/v3 on chunks of at most 25 s. Timing comes from chunk offsets.
2. Always send keyterms. Default list: ["Laya", "Gnani", "Prisma", "Timbre"].
   Accepts optional user words, merged with defaults.
3. Hinglish default format is "codemix" with language_code "hi-IN,en-IN".
   Hindi uses "hi-IN" with "codemix" as well.
4. Language mix strip does NOT use API's detected_language_code.
   Computed directly from the transcript: classifies each word as Devanagari or
   Latin script, reporting share and switch positions. Labeled "Script mix of the transcript".
5. Timbre WAV duration must come from ffprobe, never from the WAV header.
6. The transcript is editable in the UI with the disclaimer:
   "Transcript is automatic and can contain mistakes. You can edit it."
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.convert import get_audio_stream_info

# Default keyterms list per specification
DEFAULT_KEYTERMS: List[str] = ["Laya", "Gnani", "Prisma", "Timbre"]
MAX_CHUNK_DURATION_SEC: float = 25.0
STT_V3_URL: str = "https://api.vachana.ai/stt/v3"
TTS_INFERENCE_URL: str = "https://api.vachana.ai/api/v1/tts/inference"

# Persistent cache directories keyed by SHA-256 per Stage 3 specification
CACHE_DIR: Path = PROJECT_ROOT / "scratch" / "cache"
REGION_TEXT_CACHE_DIR: Path = CACHE_DIR / "region_text"
SPEAK_CACHE_DIR: Path = CACHE_DIR / "speak"
REGION_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SPEAK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_gnani_api_key() -> str:
    """Load GNANI_API_KEY from backend/.env or environment without leaking."""
    env_paths = [
        PROJECT_ROOT / "backend" / ".env",
        PROJECT_ROOT / ".env",
    ]
    loaded: Dict[str, str] = {}
    for p in env_paths:
        if p.exists() and p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k not in loaded:
                            loaded[k] = v
            except Exception:
                pass

    key = loaded.get("GNANI_API_KEY", "")
    if not key and "GNANI_API_KEY" in os.environ:
        key = os.environ["GNANI_API_KEY"].strip().strip("'\"")

    return key


def get_timbre_duration_ffprobe(audio_path: Union[str, Path]) -> float:
    """Calculate audio duration using ffprobe (never from the WAV header).

    In the spike, the WAV header reported 44739 s for Timbre streaming WAVs,
    whereas ffprobe reports the true container stream duration.
    """
    try:
        info = get_audio_stream_info(audio_path)
        return float(info.get("duration", 0.0))
    except Exception:
        ffprobe_bin = shutil.which("ffprobe")
        if not ffprobe_bin:
            return 0.0
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            return float(proc.stdout.strip())
        except ValueError:
            return 0.0


def classify_word_script(word: str) -> str:
    """Classify a word as 'devanagari', 'latin', or 'other'.

    Unicode ranges:
    - Devanagari: U+0900 to U+097F
    - Latin: Basic Latin (A-Z, a-z) + Latin-1 Supplement + Latin Extended
    """
    clean = re.sub(r"[^\w]", "", word)
    if not clean:
        return "other"

    dev_count = 0
    lat_count = 0
    for ch in clean:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            dev_count += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            lat_count += 1

    if dev_count > 0 and dev_count >= lat_count:
        return "devanagari"
    if lat_count > 0:
        return "latin"
    return "other"


def compute_script_mix(transcript: str) -> Dict[str, Any]:
    """Compute script mix from the transcript directly (NOT from detected_language_code).

    Returns share of Devanagari vs Latin words and details on every script switch.
    """
    if not transcript or not transcript.strip():
        return {
            "label": "Script mix of the transcript",
            "devanagari_count": 0,
            "latin_count": 0,
            "total_words": 0,
            "devanagari_pct": 0.0,
            "latin_pct": 0.0,
            "switches": [],
            "switch_count": 0,
            "switch_positions": [],
        }

    raw_words = transcript.strip().split()
    classified_words = []
    for idx, w in enumerate(raw_words):
        script = classify_word_script(w)
        if script != "other":
            classified_words.append((idx, w, script))

    total = len(classified_words)
    dev_count = sum(1 for _, _, s in classified_words if s == "devanagari")
    lat_count = sum(1 for _, _, s in classified_words if s == "latin")

    dev_pct = round(100.0 * dev_count / total, 1) if total > 0 else 0.0
    lat_pct = round(100.0 * lat_count / total, 1) if total > 0 else 0.0

    switches = []
    for i in range(1, len(classified_words)):
        prev_idx, prev_w, prev_s = classified_words[i - 1]
        curr_idx, curr_w, curr_s = classified_words[i]
        if curr_s != prev_s:
            switches.append({
                "word_index": curr_idx,
                "word": curr_w,
                "from_script": prev_s,
                "to_script": curr_s,
            })

    return {
        "label": "Script mix of the transcript",
        "devanagari_count": dev_count,
        "latin_count": lat_count,
        "total_words": total,
        "devanagari_pct": dev_pct,
        "latin_pct": lat_pct,
        "switches": switches,
        "switch_count": len(switches),
        "switch_positions": [s["word_index"] for s in switches],
    }


def merge_keyterms(user_keyterms: Optional[Union[List[str], str]] = None) -> List[str]:
    """Merge default keyterms with user provided keyterms, eliminating duplicates."""
    merged = list(DEFAULT_KEYTERMS)
    if not user_keyterms:
        return merged

    if isinstance(user_keyterms, str):
        candidates = [line.strip() for line in user_keyterms.splitlines() if line.strip()]
    else:
        candidates = [str(k).strip() for k in user_keyterms if str(k).strip()]

    existing_lower = {k.lower() for k in merged}
    for cand in candidates:
        if cand.lower() not in existing_lower:
            merged.append(cand)
            existing_lower.add(cand.lower())

    return merged


def chunk_wav_audio(
    wav_path: Union[str, Path],
    max_chunk_sec: float = MAX_CHUNK_DURATION_SEC,
    temp_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Split an audio file into chunks of at most max_chunk_sec (default 25.0 s).

    Returns list of dicts with:
    - 'chunk_path': Path to temporary chunk WAV
    - 'start_seconds': float start offset
    - 'end_seconds': float end offset
    - 'duration': float duration in seconds
    """
    path = Path(wav_path)
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    total_samples = len(audio)
    total_duration = total_samples / float(sr)

    if total_duration <= max_chunk_sec:
        return [{
            "chunk_path": path,
            "start_seconds": 0.0,
            "end_seconds": round(total_duration, 4),
            "duration": round(total_duration, 4),
            "is_temp": False,
        }]

    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="gnani_chunks_"))

    chunk_samples = int(round(max_chunk_sec * sr))
    chunks = []
    chunk_idx = 0

    for start_samp in range(0, total_samples, chunk_samples):
        end_samp = min(start_samp + chunk_samples, total_samples)
        chunk_data = audio[start_samp:end_samp]
        s_sec = round(start_samp / float(sr), 4)
        e_sec = round(end_samp / float(sr), 4)
        dur = round((end_samp - start_samp) / float(sr), 4)

        out_chunk_path = temp_dir / f"chunk_{chunk_idx:03d}_{int(s_sec*1000)}ms.wav"
        sf.write(str(out_chunk_path), chunk_data, sr, subtype="PCM_16")

        chunks.append({
            "chunk_path": out_chunk_path,
            "start_seconds": s_sec,
            "end_seconds": e_sec,
            "duration": dur,
            "is_temp": True,
        })
        chunk_idx += 1

    return chunks


def parse_gnani_error(response: requests.Response) -> str:
    """Safely extract error message without exposing credentials."""
    status_code = response.status_code
    try:
        body = response.json()
    except Exception:
        return f"HTTP {status_code}: {response.text[:200]}"

    if isinstance(body, dict):
        if "error" in body:
            err = body["error"]
            if isinstance(err, dict):
                return f"HTTP {status_code} [{err.get('type', 'ERROR')}]: {err.get('message', '')}"
            return f"HTTP {status_code}: {err}"
        if "detail" in body:
            return f"HTTP {status_code}: {body['detail']}"
        if "message" in body:
            return f"HTTP {status_code}: {body['message']}"

    return f"HTTP {status_code}: {str(body)[:200]}"


def request_with_retry(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    attempts: int = 4,
    timeout: int = 60,
) -> Tuple[Optional[requests.Response], Optional[str]]:
    """Execute HTTP request with exponential backoff on 429 and 5xx errors."""
    retryable_statuses = {429, 500, 503, 504}
    delay = 1.0

    for attempt in range(1, attempts + 1):
        wait = delay
        try:
            r = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                files=files,
                json=json_data,
                timeout=timeout,
            )

            if r.status_code not in retryable_statuses:
                if r.status_code >= 400:
                    err_msg = parse_gnani_error(r)
                    return r, err_msg
                return r, None

            retry_after = r.headers.get("Retry-After", "")
            if r.status_code == 429 and retry_after.isdigit():
                wait = max(wait, float(retry_after))

            err_msg = parse_gnani_error(r)
            print(f"[Gnani API Retry] Attempt {attempt}/{attempts} (Status {r.status_code}). Waiting {wait:.1f}s.")

        except (requests.ConnectionError, requests.Timeout) as exc:
            err_msg = f"Network error: {type(exc).__name__}"
            print(f"[Gnani API Network Error] Attempt {attempt}/{attempts}. Waiting {wait:.1f}s.")

        if attempt < attempts:
            time.sleep(wait)
            delay *= 2.0

    return None, f"Request failed after {attempts} attempts. Last error: {err_msg}"


def transcribe_prisma_chunk(
    api_key: str,
    wav_path: Path,
    language_code: str = "hi-IN,en-IN",
    format_style: str = "codemix",
    keyterms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Transcribe an audio chunk using Prisma v3.0 REST endpoint (/stt/v3)."""
    headers = {"X-API-Key-ID": api_key}
    data = {
        "model": "prisma-v3.0",
        "language_code": language_code,
        "format": format_style,
    }
    effective_keyterms = merge_keyterms(keyterms)
    data["keyterms"] = json.dumps(effective_keyterms, ensure_ascii=False)

    t0 = time.time()
    with open(wav_path, "rb") as f:
        files = {"audio_file": (wav_path.name, f, "audio/wav")}
        r, err = request_with_retry("POST", STT_V3_URL, headers=headers, data=data, files=files)

    wall_time = time.time() - t0

    if err or r is None or not r.ok:
        return {
            "success": False,
            "error": err or "Unknown STT error",
            "wall_time": wall_time,
            "transcript": "",
        }

    body = r.json()
    return {
        "success": True,
        "transcript": body.get("transcript", "").strip(),
        "detected_language_code": body.get("detected_language_code", "None"),
        "processing_time": body.get("processing_time"),
        "end_to_end_latency": body.get("end_to_end_latency"),
        "wall_time": wall_time,
    }


def transcribe_audio_gnani(
    wav_path: Union[str, Path],
    language_mode: str = "hinglish",
    custom_keyterms: Optional[Union[List[str], str]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe audio file using Gnani Prisma REST STT with chunking (<= 25s).

    Adheres strictly to spike rules:
    - Chunks audio into <= 25 s segments
    - Language code: 'hi-IN,en-IN' for Hinglish, 'hi-IN' for Hindi
    - Format: 'codemix' for both
    - Always sends keyterms (defaults merged with custom)
    - Computes script mix from transcript (never detected_language_code)
    - Interpolates word timestamps from chunk offsets
    """
    key = api_key or load_gnani_api_key()
    if not key:
        return {
            "success": False,
            "error": "GNANI_API_KEY is not configured in backend/.env",
            "transcript": "",
            "words": [],
            "script_mix": compute_script_mix(""),
        }

    # Set parameters per Correction #3
    is_hindi = language_mode.lower() == "hindi"
    language_code = "hi-IN" if is_hindi else "hi-IN,en-IN"
    format_style = "codemix"

    keyterms = merge_keyterms(custom_keyterms)

    # Chunk audio to <= 25.0 s per Correction #1
    temp_dir = Path(tempfile.mkdtemp(prefix="gnani_stt_"))
    try:
        chunks = chunk_wav_audio(wav_path, max_chunk_sec=MAX_CHUNK_DURATION_SEC, temp_dir=temp_dir)
        transcript_parts = []
        all_words = []
        chunk_reports = []

        for idx, ch in enumerate(chunks):
            res = transcribe_prisma_chunk(
                api_key=key,
                wav_path=ch["chunk_path"],
                language_code=language_code,
                format_style=format_style,
                keyterms=keyterms,
            )

            if not res["success"]:
                return {
                    "success": False,
                    "error": f"Chunk {idx} ({ch['start_seconds']:.1f}s - {ch['end_seconds']:.1f}s) failed: {res.get('error')}",
                    "transcript": " ".join(transcript_parts),
                    "words": all_words,
                    "script_mix": compute_script_mix(" ".join(transcript_parts)),
                }

            part_text = res.get("transcript", "").strip()
            if part_text:
                transcript_parts.append(part_text)

            words_in_part = part_text.split()
            # Prisma returns no word times. Remove any per-word start or end times.
            for word_str in words_in_part:
                all_words.append({
                    "word": word_str,
                })

            # Keep only chunk level times (chunk start and end from our own chunking).
            # Anything shown as a timing in the UI or API for Hindi and Hinglish must say "chunk" and never "word".
            chunk_reports.append({
                "chunk_index": idx,
                "chunk_start": ch["start_seconds"],
                "chunk_end": ch["end_seconds"],
                "chunk_duration": ch["duration"],
                "chunk_start_seconds": ch["start_seconds"],
                "chunk_end_seconds": ch["end_seconds"],
                "chunk_duration_seconds": ch["duration"],
                "transcript": part_text,
                "words": [{"word": w} for w in words_in_part],
                "chunk_processing_time": res.get("processing_time"),
            })

        full_transcript = " ".join(transcript_parts).strip()
        script_mix = compute_script_mix(full_transcript)

        return {
            "success": True,
            "transcript": full_transcript,
            "words": all_words,
            "script_mix": script_mix,
            "chunks": chunk_reports,
            "language_mode": "hindi" if is_hindi else "hinglish",
            "language_code": language_code,
            "format": format_style,
            "keyterms": keyterms,
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_region_audio(
    audio_path: Union[str, Path],
    start_sec: float,
    end_sec: float,
    max_cap_sec: float = 8.0,
) -> Tuple[np.ndarray, int, float, float]:
    """Slice audio from 1 s before start to 1 s after end, capped at max_cap_sec.

    Returns:
        Tuple of (audio_samples, sample_rate, cut_start, cut_end).
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")

    with sf.SoundFile(str(audio_path)) as sf_file:
        sr = sf_file.samplerate
        total_frames = len(sf_file)
        total_duration = total_frames / float(sr)

        # Cut 1 s before start to 1 s after end per Stage 3 rule
        cut_start = max(0.0, float(start_sec) - 1.0)
        cut_end = min(total_duration, float(end_sec) + 1.0)

        # Cap total slice duration at 8 s
        if (cut_end - cut_start) > max_cap_sec:
            cut_end = min(total_duration, cut_start + max_cap_sec)

        start_frame = int(cut_start * sr)
        end_frame = max(start_frame + 1, int(cut_end * sr))

        sf_file.seek(start_frame)
        samples = sf_file.read(end_frame - start_frame, dtype="float32")

    # Ensure mono
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    return samples, sr, cut_start, cut_end


def get_region_text(
    audio_path: Union[str, Path],
    start_sec: float,
    end_sec: float,
    language_mode: str = "hindi",
    custom_keyterms: Optional[Union[List[str], str]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe a cut speech interval around a flaw flag with SHA-256 caching.

    Rules applied:
    - Cut 1 s before start to 1 s after end, capped at 8 s.
    - Cached by SHA-256 of clip bytes plus language.
    - Temporary audio deleted immediately after use.
    - Sent to Prisma v3.0 REST with keyterms.
    """
    samples, sr, cut_start, cut_end = extract_region_audio(
        audio_path=audio_path,
        start_sec=start_sec,
        end_sec=end_sec,
        max_cap_sec=8.0,
    )

    clip_duration = round(cut_end - cut_start, 3)

    # Compute cache key from SHA-256 of raw clip samples + language mode
    hasher = hashlib.sha256()
    hasher.update(samples.tobytes())
    hasher.update(language_mode.lower().strip().encode("utf-8"))
    clip_hash = hasher.hexdigest()

    cache_file = REGION_TEXT_CACHE_DIR / f"{clip_hash}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            cached_data["cached"] = True
            return cached_data
        except Exception:
            pass

    # Temporary file for sending to Prisma STT
    temp_dir = Path(tempfile.mkdtemp(prefix="region_stt_"))
    temp_wav = temp_dir / f"region_{clip_hash[:12]}.wav"

    try:
        sf.write(str(temp_wav), samples, sr)

        key = api_key or load_gnani_api_key()
        if not key:
            return {
                "success": False,
                "error": "GNANI_API_KEY is not configured",
                "text": "",
                "start": round(cut_start, 3),
                "end": round(cut_end, 3),
                "duration": clip_duration,
                "language_mode": language_mode,
                "cached": False,
                "cache_key": clip_hash,
            }

        is_hindi = language_mode.lower() == "hindi"
        lang_code = "hi-IN" if is_hindi else "hi-IN,en-IN"
        keyterms = merge_keyterms(custom_keyterms)

        res = transcribe_prisma_chunk(
            api_key=key,
            wav_path=temp_wav,
            language_code=lang_code,
            format_style="codemix",
            keyterms=keyterms,
        )

        if not res.get("success"):
            return {
                "success": False,
                "error": res.get("error", "Prisma region transcription failed"),
                "text": "",
                "start": round(cut_start, 3),
                "end": round(cut_end, 3),
                "duration": clip_duration,
                "language_mode": language_mode,
                "cached": False,
                "cache_key": clip_hash,
            }

        text = res.get("transcript", "").strip()
        result_payload = {
            "success": True,
            "text": text,
            "start": round(cut_start, 3),
            "end": round(cut_end, 3),
            "duration": clip_duration,
            "language_mode": language_mode,
            "cached": False,
            "cache_key": clip_hash,
        }

        # Store in persistent cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result_payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return result_payload

    finally:
        # Delete temporary audio after use
        shutil.rmtree(temp_dir, ignore_errors=True)


def synthesize_timbre_speech(
    text: str,
    language: str = "hi-IN",
    voice: str = "Shivani",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Synthesize speech using Timbre REST endpoint with SHA-256 caching.

    Rules applied:
    - Caches output audio by SHA-256 of text plus language.
    - Duration measured via ffprobe (never from WAV header).
    - Temporary conversion audio deleted immediately after use.
    """
    text_clean = text.strip()
    if not text_clean:
        return {"success": False, "error": "Text cannot be empty"}

    lang = language.strip() or "hi-IN"
    v = voice.strip() or "Shivani"

    # Compute cache key from SHA-256 of text + language + voice
    hasher = hashlib.sha256()
    hasher.update(text_clean.encode("utf-8"))
    hasher.update(lang.lower().encode("utf-8"))
    hasher.update(v.lower().encode("utf-8"))
    speak_hash = hasher.hexdigest()

    cache_path = SPEAK_CACHE_DIR / f"{speak_hash}.wav"
    if cache_path.exists():
        duration = get_timbre_duration_ffprobe(cache_path)
        return {
            "success": True,
            "audio_path": str(cache_path),
            "duration": duration,
            "cached": True,
            "cache_key": speak_hash,
            "voice": v,
            "language": lang,
        }

    key = api_key or load_gnani_api_key()
    if not key:
        return {"success": False, "error": "GNANI_API_KEY is not configured"}

    headers = {
        "Content-Type": "application/json",
        "X-API-Key-ID": key,
    }
    payload = {
        "text": text_clean,
        "voice": v,
        "model": "timbre-v3.0",
        "language": lang,
        "speed": 1.0,
        "audio_config": {
            "sample_rate": 48000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }

    t0 = time.time()
    r, err = request_with_retry("POST", TTS_INFERENCE_URL, headers=headers, json_data=payload)
    call_time = time.time() - t0

    if err or r is None or not r.ok:
        return {"success": False, "error": err or "Timbre TTS request failed", "call_time": call_time}

    # Write synthesized WAV to cache
    temp_dir = Path(tempfile.mkdtemp(prefix="timbre_tts_"))
    temp_file = temp_dir / f"timbre_{speak_hash[:12]}.wav"
    try:
        temp_file.write_bytes(r.content)
        # Move or copy into permanent cache
        shutil.copyfile(str(temp_file), str(cache_path))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    duration = get_timbre_duration_ffprobe(cache_path)

    return {
        "success": True,
        "audio_path": str(cache_path),
        "duration": duration,
        "call_time": call_time,
        "cached": False,
        "cache_key": speak_hash,
        "voice": v,
        "language": lang,
    }

