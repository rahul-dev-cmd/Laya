#!/usr/bin/env python3
"""Stage B tests: Prisma as an English transcript option with measured comparison.

Tests:
1. clean_prisma_text: lowercase, remove punctuation except apostrophes inside words,
   collapse spaces, detect digits.
2. Chunk joining: joins chunk texts into one transcript with language_code en-IN and codemix.
3. Digits note: sets numbers_note when cleaned text contains digits.
4. Whisper remains the default STT engine when stt_engine is omitted or set to 'whisper'.
5. Analysis with Gnani Prisma attaches caveat and numbers_note appropriately.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import soundfile as sf
import numpy as np
from starlette.testclient import TestClient

from backend.app import app
from backend.gnani_stt import (
    DEFAULT_KEYTERMS,
    clean_prisma_text,
    transcribe_audio_gnani,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_clean_prisma_text_rules():
    """Verify clean_prisma_text follows all cleaning specifications:
    - lowercase
    - remove punctuation except apostrophes inside words
    - collapse spaces
    """
    raw = "  'Hello', WORLD! It's an 18-minute talk, isn't it? Don't stop... \"quoted\" text!  "
    cleaned, has_digits = clean_prisma_text(raw)

    # 1. Lowercase
    assert cleaned == cleaned.lower()

    # 2. Internal apostrophes kept
    assert "it's" in cleaned
    assert "isn't" in cleaned
    assert "don't" in cleaned

    # 3. Outer quotes removed from 'Hello' and "quoted"
    assert "hello" in cleaned
    assert "'hello'" not in cleaned
    assert "quoted" in cleaned
    assert '"quoted"' not in cleaned

    # 4. Punctuation removed (no exclamation, comma, ellipsis, hyphens)
    for p in ["!", ",", "...", "?", '"', "-"]:
        assert p not in cleaned

    # 5. Spaces collapsed
    assert "  " not in cleaned
    assert cleaned.startswith("hello")

    # 6. Digits detected
    assert has_digits is True


def test_clean_prisma_text_digits_flag():
    """Verify digits detection correctly separates pure text from text containing digits."""
    no_digits_text = "The quick brown fox jumps over the lazy dog it's great"
    cleaned_no, has_digits_no = clean_prisma_text(no_digits_text)
    assert has_digits_no is False
    assert cleaned_no == no_digits_text.lower()

    with_digits_text = "In the year 2026 we tested 91 models"
    cleaned_with, has_digits_with = clean_prisma_text(with_digits_text)
    assert has_digits_with is True
    assert "2026" in cleaned_with
    assert "91" in cleaned_with


def test_prisma_english_chunk_joining():
    """Verify English Prisma STT uses language_code 'en-IN', format 'codemix',
    chunking <= 25s, default keyterms, and joins chunks into one transcript.
    """
    # Create temporary 1-second synthetic WAV in memory
    sample_rate = 16000
    duration = 2.0
    audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    sf.write(str(tmp_wav), audio_data, sample_rate, subtype="PCM_16")

    try:
        mock_chunks = [
            {"chunk_path": tmp_wav, "start_seconds": 0.0, "end_seconds": 1.0, "duration": 1.0, "is_temp": False},
            {"chunk_path": tmp_wav, "start_seconds": 1.0, "end_seconds": 2.0, "duration": 1.0, "is_temp": False},
        ]

        def mock_transcribe_chunk(api_key, wav_path, language_code, format_style, keyterms):
            assert language_code == "en-IN"
            assert format_style == "codemix"
            assert "Laya" in keyterms
            assert "Gnani" in keyterms
            return {
                "success": True,
                "transcript": "chunk text part" if "0.0" in str(wav_path) or True else "chunk text part two",
                "detected_language_code": "en-IN",
                "processing_time": 0.12,
            }

        with patch("backend.gnani_stt.chunk_wav_audio", return_value=mock_chunks), \
             patch("backend.gnani_stt.transcribe_prisma_chunk", side_effect=mock_transcribe_chunk), \
             patch("backend.gnani_stt.load_gnani_api_key", return_value="dummy_key"):

            res = transcribe_audio_gnani(tmp_wav, language_mode="english")

            assert res["success"] is True
            assert res["language_mode"] == "english"
            assert res["language_code"] == "en-IN"
            assert res["format"] == "codemix"
            assert len(res["chunks"]) == 2
            # Chunks are joined into one transcript
            assert res["transcript"] == "chunk text part chunk text part"
    finally:
        tmp_wav.unlink(missing_ok=True)


def test_whisper_stays_default_in_analysis():
    """Verify that Whisper stays the default STT engine when stt_engine is omitted or 'whisper'."""
    client = TestClient(app)

    # Use existing test audio
    wav_path = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.wav"
    if not wav_path.exists():
        pytest.skip("dataset/sources/librivox_01.wav not found")

    mock_analysis_result = {
        "mode": "experimental",
        "duration": 5.0,
        "transcript": "this is a whisper transcript",
        "words": [{"word": "this"}, {"word": "is"}],
        "flaw_regions": [],
        "rubric": {"dimensions": [], "overall": 90.0},
    }

    with patch("backend.app.analyze", return_value=dict(mock_analysis_result)), \
         patch("backend.app.transcribe_audio_gnani") as mock_gnani:

        # 1. Without stt_engine parameter
        with open(wav_path, "rb") as f:
            resp = client.post(
                "/api/analyze",
                data={"language_mode": "english"},
                files={"audio_file": ("test.wav", f, "audio/wav")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stt_engine"] == "whisper"
        assert data.get("prisma_caveat") is None
        assert data.get("numbers_note") is None
        # Gnani STT must not be called when Whisper is default
        mock_gnani.assert_not_called()

        # 2. With explicit stt_engine="whisper"
        with open(wav_path, "rb") as f:
            resp2 = client.post(
                "/api/analyze",
                data={"language_mode": "english", "stt_engine": "whisper"},
                files={"audio_file": ("test.wav", f, "audio/wav")},
            )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["stt_engine"] == "whisper"
        assert data2.get("prisma_caveat") is None
        assert data2.get("numbers_note") is None
        mock_gnani.assert_not_called()


def test_prisma_english_in_analysis_with_caveat_and_digits_note():
    """Verify English analysis with Gnani Prisma attaches caveat and numbers_note when digits present."""
    client = TestClient(app)

    wav_path = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.wav"
    if not wav_path.exists():
        pytest.skip("dataset/sources/librivox_01.wav not found")

    mock_gnani_res = {
        "success": True,
        "transcript": "Hello, world! We tested 42 speech clips in 2026.",
        "chunks": [],
        "language_mode": "english",
    }

    mock_analysis_result = {
        "mode": "experimental",
        "duration": 5.0,
        "transcript": "hello world we tested 42 speech clips in 2026",
        "words": [{"word": "hello"}],
        "flaw_regions": [],
        "rubric": {"dimensions": [], "overall": 88.0},
    }

    with patch("backend.app.transcribe_audio_gnani", return_value=mock_gnani_res) as mock_gnani, \
         patch("backend.app.analyze", return_value=dict(mock_analysis_result)) as mock_analyze:

        with open(wav_path, "rb") as f:
            resp = client.post(
                "/api/analyze",
                data={"language_mode": "english", "stt_engine": "gnani_prisma"},
                files={"audio_file": ("test.wav", f, "audio/wav")},
            )

        assert resp.status_code == 200
        data = resp.json()

        mock_gnani.assert_called_once()
        assert data["stt_engine"] == "gnani_prisma"
        assert data["prisma_caveat"] == "Transcript from Gnani Prisma; timing found by forced alignment."
        assert data["numbers_note"] == "The transcript contains numbers, which can affect timing."


def test_prisma_english_in_analysis_without_digits():
    """Verify English analysis with Gnani Prisma does not show numbers_note when transcript has no digits."""
    client = TestClient(app)

    wav_path = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.wav"
    if not wav_path.exists():
        pytest.skip("dataset/sources/librivox_01.wav not found")

    mock_gnani_res = {
        "success": True,
        "transcript": "Hello, world! It's a wonderful speech today.",
        "chunks": [],
        "language_mode": "english",
    }

    mock_analysis_result = {
        "mode": "experimental",
        "duration": 5.0,
        "transcript": "hello world it's a wonderful speech today",
        "words": [{"word": "hello"}],
        "flaw_regions": [],
        "rubric": {"dimensions": [], "overall": 92.0},
    }

    with patch("backend.app.transcribe_audio_gnani", return_value=mock_gnani_res) as mock_gnani, \
         patch("backend.app.analyze", return_value=dict(mock_analysis_result)) as mock_analyze:

        with open(wav_path, "rb") as f:
            resp = client.post(
                "/api/analyze",
                data={"language_mode": "english", "stt_engine": "gnani_prisma"},
                files={"audio_file": ("test.wav", f, "audio/wav")},
            )

        assert resp.status_code == 200
        data = resp.json()

        mock_gnani.assert_called_once()
        assert data["stt_engine"] == "gnani_prisma"
        assert data["prisma_caveat"] == "Transcript from Gnani Prisma; timing found by forced alignment."
        assert data.get("numbers_note") is None
