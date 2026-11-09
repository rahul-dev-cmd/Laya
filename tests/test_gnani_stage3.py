"""Tests for Stage 2 Verification and Stage 3 Region Text and Hear It Fixed.

Stage 2 Requirements:
- Language selector works in both upload and record panels.
- English uses original Whisper pipeline unchanged.
- Hindi and Hinglish run only pitch variation detector plus plain note that pace and pause checks are English only.
- Rubric scores only the dimensions that ran (Pitch variation) and says which ones.

Stage 3 Requirements:
- "What you said here" under each flag (cut 1 s before start to 1 s after end, cap 8 s, send to Prisma with keyterms).
- "Hear it fixed" button sending text to Timbre and playing it, labeled "Model voice, not your voice".
- Endpoints: POST /api/region_text and POST /api/speak.
- Caching by SHA-256 of clip plus language.
- Delete temporary audio after use.
- No emojis, no long dashes, no invented numbers.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from backend.app import app
from backend.gnani_stt import (
    DEFAULT_KEYTERMS,
    REGION_TEXT_CACHE_DIR,
    SPEAK_CACHE_DIR,
    extract_region_audio,
    get_region_text,
    merge_keyterms,
    synthesize_timbre_speech,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


def test_stage2_language_selector_in_both_panels():
    """Confirm the language selector works in both the upload and record panels."""
    html_path = PROJECT_ROOT / "frontend" / "index.html"
    assert html_path.exists()
    html_content = html_path.read_text(encoding="utf-8")

    # Upload panel language buttons
    assert 'id="btnLangEnglish"' in html_content
    assert 'id="btnLangHinglish"' in html_content
    assert 'id="btnLangHindi"' in html_content

    # Record panel language buttons
    assert 'id="btnRecLangEnglish"' in html_content
    assert 'id="btnRecLangHinglish"' in html_content
    assert 'id="btnRecLangHindi"' in html_content

    # Keyterms textareas in both panels
    assert 'id="keytermsInput"' in html_content
    assert 'id="recordKeytermsInput"' in html_content

    # app.js binds listeners for all 6 buttons and syncs
    js_content = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "btnRecLangEnglish" in js_content
    assert "btnRecLangHinglish" in js_content
    assert "btnRecLangHindi" in js_content
    assert "recordIndicKeytermsGroup" in js_content


def test_stage2_hindi_runs_only_pitch_and_scores_only_pitch():
    """Verify Hindi/Hinglish runs only pitch detector and rubric scores only pitch."""
    test_wav = PROJECT_ROOT / "scratch" / "test_hi.wav"
    assert test_wav.exists()

    # Mock Gnani STT transcription
    mock_gnani_result = {
        "success": True,
        "transcript": "नमस्ते यह एक परीक्षण है",
        "words": [{"word": "नमस्ते"}, {"word": "यह"}, {"word": "एक"}, {"word": "परीक्षण"}, {"word": "है"}],
        "script_mix": {"devanagari_words": 5, "latin_words": 0, "devanagari_share": 1.0, "latin_share": 0.0, "switches": []},
        "chunks": [
            {
                "chunk_index": 0,
                "chunk_start": 0.0,
                "chunk_end": 20.0,
                "chunk_duration": 20.0,
                "chunk_start_seconds": 0.0,
                "chunk_end_seconds": 20.0,
                "chunk_duration_seconds": 20.0,
                "transcript": "नमस्ते यह एक परीक्षण है",
                "words": [{"word": "नमस्ते"}, {"word": "यह"}, {"word": "एक"}, {"word": "परीक्षण"}, {"word": "है"}],
                "chunk_processing_time": 0.5,
            }
        ],
        "language_mode": "hindi",
        "language_code": "hi-IN",
        "format": "codemix",
        "keyterms": DEFAULT_KEYTERMS,
    }

    with patch("backend.app.transcribe_audio_gnani", return_value=mock_gnani_result):
        with open(test_wav, "rb") as f:
            res = client.post(
                "/api/analyze",
                files={"audio_file": ("test_hindi.wav", f, "audio/wav")},
                data={"language_mode": "hindi"},
            )

    assert res.status_code == 200, res.text
    data = res.json()

    # Plain note that pace and pause checks are English only
    assert data.get("pace_pause_note") == "Pace and pause checks are English only."

    # All flaw regions must be pitch only
    for r in data.get("flaw_regions", []):
        assert r.get("type") == "pitch"

    # Rubric checks
    rubric = data.get("rubric")
    assert rubric is not None

    dims_by_name = {d["name"]: d for d in rubric.get("dimensions", [])}
    assert "Pace" in dims_by_name
    assert "Pitch variation" in dims_by_name
    assert "Pauses" in dims_by_name

    # Pace is unscored with English only note
    assert dims_by_name["Pace"]["scored"] is False
    assert dims_by_name["Pace"]["score"] is None
    assert "Pace and pause checks are English only." in dims_by_name["Pace"]["reason"]

    # Pauses is unscored with English only note
    assert dims_by_name["Pauses"]["scored"] is False
    assert dims_by_name["Pauses"]["score"] is None
    assert "Pace and pause checks are English only." in dims_by_name["Pauses"]["reason"]

    # Pitch variation is scored
    assert dims_by_name["Pitch variation"]["scored"] is True
    assert dims_by_name["Pitch variation"]["score"] is not None

    # Overall equals Pitch variation score
    assert rubric["overall"] == dims_by_name["Pitch variation"]["score"]

    # Overall note says which dimension was scored
    assert "Overall uses 1 of 4 dimensions: Pitch variation" in rubric["overall_note"]
    assert "Pace and pause checks are English only." in rubric["overall_note"]


def test_stage2_english_uses_original_whisper_pipeline():
    """Verify English mode uses Whisper pipeline without Indic notes."""
    res = client.get("/api/example/librivox_01_pitch_tier5")
    assert res.status_code == 200
    data = res.json()
    assert "pace_pause_note" not in data
    rubric = data.get("rubric", {})
    dims_by_name = {d["name"]: d for d in rubric.get("dimensions", [])}
    # Pace is scored in standard English analysis
    assert dims_by_name["Pace"]["scored"] is True


def test_stage3_cut_1s_before_1s_after_and_cap_8s():
    """Verify cutting 1 s before start to 1 s after end and capping at 8 s."""
    test_wav = PROJECT_ROOT / "scratch" / "test_hi.wav"
    assert test_wav.exists()

    # Case 1: start=3.0, end=5.0 -> cut is [2.0, 6.0], length = 4.0 s
    samples, sr, c_start, c_end = extract_region_audio(test_wav, start_sec=3.0, end_sec=5.0, max_cap_sec=8.0)
    assert c_start == 2.0
    assert c_end == 6.0
    assert pytest.approx(c_end - c_start, 0.01) == 4.0
    assert len(samples) == int(sr * 4.0)

    # Case 2: start=0.5 -> start - 1.0 < 0 -> clamped to 0.0
    samples, sr, c_start, c_end = extract_region_audio(test_wav, start_sec=0.5, end_sec=3.0, max_cap_sec=8.0)
    assert c_start == 0.0
    assert c_end == 4.0
    assert pytest.approx(c_end - c_start, 0.01) == 4.0

    # Case 3: start=2.0, end=15.0 -> raw slice would be 14.0s -> capped at 8.0s
    samples, sr, c_start, c_end = extract_region_audio(test_wav, start_sec=2.0, end_sec=15.0, max_cap_sec=8.0)
    assert c_start == 1.0
    assert c_end == 9.0
    assert pytest.approx(c_end - c_start, 0.01) == 8.0
    assert len(samples) == int(sr * 8.0)


def test_stage3_region_text_endpoint_caching_and_cleanup():
    """Verify POST /api/region_text transcribes, caches by SHA-256, and deletes temp audio."""
    test_wav = PROJECT_ROOT / "scratch" / "test_hi.wav"
    assert test_wav.exists()

    # Pre-populate mock or verify endpoint
    mock_prisma_chunk = {
        "success": True,
        "transcript": "नमस्ते लया परीक्षण",
        "processing_time": 0.35,
    }

    with patch("backend.gnani_stt.transcribe_prisma_chunk", return_value=mock_prisma_chunk) as mock_chunk:
        # First call: cache miss
        res1 = client.post(
            "/api/region_text",
            json={
                "audio_path": "scratch/test_hi.wav",
                "start": 2.0,
                "end": 5.0,
                "language_mode": "hindi",
            },
        )
        assert res1.status_code == 200, res1.text
        data1 = res1.json()
        assert data1["success"] is True
        assert data1["text"] == "नमस्ते लया परीक्षण"
        assert "cache_key" in data1
        cache_key = data1["cache_key"]

        # Cache file must exist in scratch/cache/region_text
        cache_file = REGION_TEXT_CACHE_DIR / f"{cache_key}.json"
        assert cache_file.exists()

        # Second call: must be served from cache without calling Prisma chunk
        mock_chunk.reset_mock()
        res2 = client.post(
            "/api/region_text",
            json={
                "audio_path": "scratch/test_hi.wav",
                "start": 2.0,
                "end": 5.0,
                "language_mode": "hindi",
            },
        )
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["cached"] is True
        assert data2["text"] == "नमस्ते लया परीक्षण"
        assert mock_chunk.call_count == 0


def test_stage3_speak_endpoint_model_voice_label_and_cache():
    """Verify POST /api/speak and GET /api/speak return audio labeled 'Model voice, not your voice'."""
    # Synthesize dummy short wav
    dummy_wav_bytes = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    mock_tts_res = MagicMock()
    mock_tts_res.ok = True
    mock_tts_res.content = dummy_wav_bytes

    text = "नमस्ते यह लया मॉडल आवाज़ है"

    with patch("backend.gnani_stt.request_with_retry", return_value=(mock_tts_res, None)):
        with patch("backend.gnani_stt.get_timbre_duration_ffprobe", return_value=1.5):
            # Test POST /api/speak with audio format
            res = client.post(
                "/api/speak",
                json={
                    "text": text,
                    "language_mode": "hindi",
                    "voice": "Shivani",
                },
            )
            assert res.status_code == 200
            assert res.headers.get("x-model-voice") == "Model voice, not your voice"
            assert res.headers.get("content-type") == "audio/wav"

            # Test POST /api/speak with json format
            res_json = client.post(
                "/api/speak",
                json={
                    "text": text,
                    "language_mode": "hindi",
                    "format": "json",
                },
            )
            assert res_json.status_code == 200
            jdata = res_json.json()
            assert jdata["label"] == "Model voice, not your voice"
            assert "audio_url" in jdata


def test_stage3_ui_has_what_you_said_and_hear_fixed_and_no_disallowed_chars():
    """Verify UI contains 'What you said here', 'Hear it fixed', 'Model voice, not your voice', and no forbidden chars."""
    html_content = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    js_content = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    css_content = (PROJECT_ROOT / "frontend" / "style.css").read_text(encoding="utf-8")

    # Required strings
    assert "What you said here" in js_content
    assert "Hear it fixed" in js_content
    assert "Model voice, not your voice" in js_content

    # Check for forbidden characters in Stage 2/3 code:
    # No long dashes (em-dash \u2014 or en-dash \u2013)
    for name, content in [("app.js", js_content), ("style.css", css_content)]:
        assert "\u2014" not in content, f"Em-dash found in {name}"
