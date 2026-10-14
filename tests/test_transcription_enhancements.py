"""Tests for enhanced automatic transcription, model configuration, and word confidence."""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.app import app
from src.pipeline.analyze import analyze, transcribe_audio_whisper
from align import align_transcript_to_audio

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_word_confidence_present_in_analyze():
    """Verify that every word returned by analyze() contains a numeric confidence score."""
    audio_wav = PROJECT_ROOT / "dataset" / "corrupted" / "librivox_01_rate_tier4.wav"
    transcript_txt = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.txt"
    reference_wav = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.wav"

    if not audio_wav.exists() or not transcript_txt.exists():
        pytest.skip("Audio or transcript files missing for test.")

    result = analyze(
        audio_wav=audio_wav,
        transcript_txt=transcript_txt,
        reference_wav=reference_wav,
        whisper_model="tiny.en",
    )

    assert "words" in result
    assert len(result["words"]) > 0
    for w in result["words"]:
        assert "word" in w
        assert "start" in w
        assert "end" in w
        assert "confidence" in w
        assert isinstance(w["confidence"], float)
        assert 0.0 <= w["confidence"] <= 1.0


def test_transcript_source_edited_preserved():
    """Verify that passing transcript_source='edited' sets the contract field accordingly."""
    audio_wav = PROJECT_ROOT / "dataset" / "corrupted" / "librivox_01_rate_tier4.wav"
    transcript_txt = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.txt"

    if not audio_wav.exists() or not transcript_txt.exists():
        pytest.skip("Audio or transcript files missing for test.")

    result = analyze(
        audio_wav=audio_wav,
        transcript_txt=transcript_txt,
        transcript_source="edited",
        whisper_model="tiny.en",
    )

    assert result["transcript_source"] == "edited"


def test_api_analyze_with_whisper_model_and_edited_source():
    """Verify backend endpoint handles configurable whisper_model and transcript_source."""
    client = TestClient(app)
    audio_wav = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.wav"
    if not audio_wav.exists():
        pytest.skip("Audio file missing for test.")

    with open(audio_wav, "rb") as f_audio:
        response = client.post(
            "/api/analyze",
            files={"audio_file": ("librivox_01.wav", f_audio, "audio/wav")},
            data={
                "whisper_model": "tiny.en",
                "transcript_text": "A edited test sentence.",
                "transcript_source": "edited",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["transcript_source"] == "edited"
    assert "words" in data
    assert len(data["words"]) > 0
    assert "confidence" in data["words"][0]
