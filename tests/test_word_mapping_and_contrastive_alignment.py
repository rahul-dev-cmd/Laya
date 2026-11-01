"""Tests for word-level contrastive alignment, 1:1 mapping skips, and Run D regression.

Verifies:
1. Pause detector skips boundaries where adjacent words are not matched 1:1 in p_to_r.
2. Rate detector skips windows where adjacent words are not matched 1:1 in p_to_r.
3. analyze() exposes mapping_confidence, skipped_boundaries, skipped_windows, and warning.
4. Run D (librivox_01_pitch_tier5 auto-transcript) has 0 pause flags between 65s and 93s.

No long dashes in any text, comments or docstrings.
"""
from pathlib import Path
import csv
import tempfile
import pytest

from src.pipeline.analyze import analyze
from src.detection.pause_detect import detect_pause_flaws
from src.detection.rate_detect import detect_flaw_regions

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_pause_detector_skips_non_1to1_boundary():
    """Verify detect_pause_flaws skips boundary when words are not mapped 1:1."""
    # Create temporary alignment CSV with 3 words
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "start_seconds", "end_seconds"])
        writer.writerow(["hello", "0.0", "0.5"])
        writer.writerow(["brave", "0.9", "1.3"]) # 400ms pause between hello and brave
        writer.writerow(["world", "1.4", "1.8"]) # 100ms pause between brave and world
        tmp_csv = Path(f.name)

    baseline_stats = {
        "boundary_map": {
            0: {"boundary_index": 0, "duration_seconds": 0.05},
            1: {"boundary_index": 1, "duration_seconds": 0.05},
        },
        "std_seconds": 0.10,
        "median_seconds": 0.05,
        "scaled_mad_seconds": 0.10,
    }

    try:
        # Case A: p_to_r has boundary 0 not matched 1:1 (e.g. word 1 maps to ref word 5)
        p_to_r_unmatched = {0: 0, 1: 5, 2: 6}
        res_unmatched = detect_pause_flaws(
            test_alignment_csv_path=tmp_csv,
            baseline_stats=baseline_stats,
            z_threshold=2.0,
            eval_mode="A",
            p_to_r=p_to_r_unmatched,
        )
        # Boundary 0 (between word 0 and word 1) should be skipped because 5 != 0 + 1
        assert res_unmatched["skipped_boundaries"] == 1
        # Boundary 1 (between word 1 and word 2) is 5 -> 6 (contiguous), evaluated but gap=0.10 not flagged
        assert len(res_unmatched["flagged_boundaries"]) == 0

        # Case B: p_to_r is matched 1:1 (0 -> 0, 1 -> 1, 2 -> 2)
        p_to_r_matched = {0: 0, 1: 1, 2: 2}
        res_matched = detect_pause_flaws(
            test_alignment_csv_path=tmp_csv,
            baseline_stats=baseline_stats,
            z_threshold=2.0,
            eval_mode="A",
            p_to_r=p_to_r_matched,
        )
        assert res_matched["skipped_boundaries"] == 0
        # Boundary 0 had a 400ms pause vs 50ms baseline (diff = +350ms, z = +3.5), flagged as too long
        assert len(res_matched["flagged_boundaries"]) == 1
        assert res_matched["flagged_boundaries"][0]["boundary_index"] == 0
    finally:
        if tmp_csv.exists():
            tmp_csv.unlink()


def test_rate_detector_skips_non_1to1_window():
    """Verify detect_flaw_regions skips windows where adjacent words are not matched 1:1."""
    test_words = [
        {"word": "word0", "start": 0.10, "end": 0.30, "mid": 0.20},
        {"word": "word1", "start": 0.35, "end": 0.55, "mid": 0.45},
        {"word": "word2", "start": 0.60, "end": 0.80, "mid": 0.70},
    ]
    orig_words = [
        {"word": "word0", "start": 0.10, "end": 0.60},
        {"word": "word1", "start": 0.65, "end": 1.15},
        {"word": "word2", "start": 1.20, "end": 1.70},
    ]

    # Non-1:1 mapping: word 1 is not contiguous with word 0 in reference
    p_to_r_unmatched = {0: 0, 1: 2, 2: 3}
    regions_unmatched, skipped_unmatched = detect_flaw_regions(
        test_words=test_words,
        total_duration=1.0,
        baseline_mean=1.5,
        baseline_std=0.2,
        orig_words=orig_words,
        z_threshold=2.0,
        window_duration=1.0,
        hop_duration=0.1,
        p_to_r=p_to_r_unmatched,
        return_skipped=True,
    )
    assert skipped_unmatched > 0
    assert len(regions_unmatched) == 0

    # 1:1 mapping
    p_to_r_matched = {0: 0, 1: 1, 2: 2}
    regions_matched, skipped_matched = detect_flaw_regions(
        test_words=test_words,
        total_duration=1.0,
        baseline_mean=1.5,
        baseline_std=0.2,
        orig_words=orig_words,
        z_threshold=2.0,
        window_duration=1.0,
        hop_duration=0.1,
        p_to_r=p_to_r_matched,
        return_skipped=True,
    )
    assert skipped_matched == 0


def test_analyze_mapping_confidence_and_word_count_warning():
    """Verify analyze() result includes mapping_confidence and warning when counts differ."""
    audio_wav = PROJECT_ROOT / "dataset" / "corrupted" / "librivox_01_rate_tier4.wav"
    transcript_txt = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.txt"
    reference_wav = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.wav"

    if not audio_wav.exists() or not transcript_txt.exists() or not reference_wav.exists():
        pytest.skip("Audio or transcript files missing for test.")

    result = analyze(
        audio_wav=audio_wav,
        transcript_txt=transcript_txt,
        reference_wav=reference_wav,
    )

    assert "mapping_confidence" in result
    assert isinstance(result["mapping_confidence"], float)
    assert 0.0 <= result["mapping_confidence"] <= 1.0

    assert "skipped_boundaries" in result
    assert isinstance(result["skipped_boundaries"], int)
    assert "skipped_windows" in result
    assert isinstance(result["skipped_windows"], int)

    # When participant and reference have identical word counts (provided transcript)
    assert result["mapping_confidence"] == 1.0
    assert result.get("warning") is None


def test_run_d_regression_suppresses_pause_flags():
    """Regression test for Run D: pause flags between 65s and 93s must disappear."""
    audio_wav = PROJECT_ROOT / "scratch" / "simulated_pitch_tier5_from_webm.wav"
    if not audio_wav.exists():
        # Fallback to source wav if webm conversion cache is absent
        audio_wav = PROJECT_ROOT / "dataset" / "corrupted" / "librivox_01_pitch_tier5.wav"

    reference_wav = PROJECT_ROOT / "dataset" / "sources" / "librivox_01.wav"

    if not audio_wav.exists() or not reference_wav.exists():
        pytest.skip("Audio files missing for Run D regression test.")

    result = analyze(
        audio_wav=audio_wav,
        transcript_txt=None, # auto transcript
        reference_wav=reference_wav,
        whisper_model="small.en",
        seed=42,
    )

    # Mapping confidence and warning check
    assert "mapping_confidence" in result
    assert result["mapping_confidence"] > 0.90
    if len(result["words"]) != len(result.get("ref_words", [])):
        assert result.get("warning") is not None

    # Pause flags between 65 and 93 seconds must be 0
    pause_flags_65_93 = [
        r for r in result.get("flaw_regions", [])
        if r.get("type") == "pause" and 65.0 <= r.get("start", 0.0) <= 93.0
    ]
    assert len(pause_flags_65_93) == 0

    # The ground truth pitch flaw at 51s to 58s should still be detected
    pitch_flags = [
        r for r in result.get("flaw_regions", [])
        if r.get("type") == "pitch" and 50.0 <= r.get("start", 0.0) <= 60.0
    ]
    assert len(pitch_flags) >= 1
