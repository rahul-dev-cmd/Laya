"""Tests for alignment output validation and schema compliance."""

from pathlib import Path
import pytest
from src.utils.io import load_json

def test_alignment_json_schema():
    """Verify aligned JSON output contains all expected fields and valid timestamps."""
    aligned_file = Path("data/processed/gettysburg_address_16k_aligned.json")
    if not aligned_file.is_file():
        pytest.skip("Alignment file not generated yet.")

    data = load_json(aligned_file)
    assert "audio_file" in data
    assert "duration_seconds" in data
    assert "sample_rate" in data
    assert data["sample_rate"] == 16000
    assert "words" in data
    assert "segments" in data
    assert data["total_words"] > 0

    words = data["words"]
    for i, w in enumerate(words):
        assert "word" in w
        assert "start" in w
        assert "end" in w
        assert "score" in w
        assert w["end"] >= w["start"], f"Invalid word interval at index {i}: {w}"
        assert 0.0 <= w["score"] <= 1.0, f"Score out of bounds at index {i}: {w}"
