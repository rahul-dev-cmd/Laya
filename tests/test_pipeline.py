"""Tests for the analysis pipeline and dashboard contract."""

from pathlib import Path
import pytest

from src.pipeline.analyze import analyze

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_analyze_contract_structure():
    """Verify that analyze() returns a JSON-serializable dictionary with all contract keys."""
    audio_wav = PROJECT_ROOT / "dataset" / "corrupted" / "librivox_01_rate_tier4.wav"
    transcript_txt = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.txt"
    reference_wav = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.wav"

    if not audio_wav.exists() or not transcript_txt.exists() or not reference_wav.exists():
        pytest.skip("Required audio or transcript files not present for integration test.")

    result = analyze(
        audio_wav=audio_wav,
        transcript_txt=transcript_txt,
        reference_wav=reference_wav,
    )

    # 1. Top-level keys
    assert "mode" in result
    assert result["mode"] == "paired"
    assert "duration" in result
    assert isinstance(result["duration"], float)
    assert result["duration"] > 0.0
    assert "transcript_source" in result
    assert result["transcript_source"] == "provided"
    assert "transcript" in result
    assert isinstance(result["transcript"], str)
    assert len(result["transcript"]) > 0

    # 2. Word alignment list
    assert "words" in result
    assert isinstance(result["words"], list)
    assert len(result["words"]) > 0
    first_word = result["words"][0]
    assert "word" in first_word
    assert "start" in first_word
    assert "end" in first_word
    assert first_word["end"] >= first_word["start"]

    # 3. 10 ms curves
    assert "curves" in result
    curves = result["curves"]
    assert "participant" in curves
    assert "reference" in curves
    assert curves["reference"] is not None

    p_curves = curves["participant"]
    for k in ["time", "f0_semitones", "energy_z", "speech_rate"]:
        assert k in p_curves
        assert isinstance(p_curves[k], list)
        assert len(p_curves[k]) > 0

    # 4. Flaw regions
    assert "flaw_regions" in result
    assert isinstance(result["flaw_regions"], list)
    assert len(result["flaw_regions"]) > 0

    for reg in result["flaw_regions"]:
        assert "type" in reg
        assert reg["type"] in ["rate", "pitch", "pause"]
        assert "start" in reg
        assert "end" in reg
        assert "peak_z" in reg
        assert "severity" in reg
        assert "explanation" in reg
        assert isinstance(reg["explanation"], str)
        assert len(reg["explanation"]) > 0
        # Check that explanation embeds timestamps
        assert f"{reg['start']:.2f}" in reg["explanation"] or f"{reg['end']:.2f}" in reg["explanation"]
