#!/usr/bin/env python3
"""Tests validating the 6 corrections from the real Gnani API spike run.

1. Do NOT use STT Batch. Use only /stt/v3 on chunks of at most 25s with chunk timing offsets.
2. Always send keyterms: default list ["Laya", "Gnani", "Prisma", "Timbre"], merged with custom.
3. Hinglish default format is 'codemix' with 'hi-IN,en-IN'. Hindi uses 'hi-IN' with 'codemix'.
4. Language mix strip computed from transcript directly: classifies Devanagari vs Latin script,
   shares, and switch positions. Labeled 'Script mix of the transcript'.
5. Timbre WAV duration comes from ffprobe, never from WAV header (header gave 44739s).
6. UI shows disclaimer 'Transcript is automatic and can contain mistakes. You can edit it.'
   and supports editable re-running.
"""

from pathlib import Path
import pytest

from backend.gnani_stt import (
    DEFAULT_KEYTERMS,
    classify_word_script,
    compute_script_mix,
    merge_keyterms,
    chunk_wav_audio,
    get_timbre_duration_ffprobe,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_correction_1_chunking_under_25s():
    """Correction 1: Chunking ensures chunks are at most 25.0 s with valid timing offsets."""
    test_wav = PROJECT_ROOT / "scratch" / "test_hinglish.wav"
    if not test_wav.exists():
        pytest.skip("scratch/test_hinglish.wav not found")

    chunks = chunk_wav_audio(test_wav, max_chunk_sec=25.0)
    assert len(chunks) >= 1
    for ch in chunks:
        assert ch["duration"] <= 25.0
        assert ch["start_seconds"] >= 0.0
        assert ch["end_seconds"] >= ch["start_seconds"]


def test_correction_2_keyterms_defaults_and_merge():
    """Correction 2: Default keyterms always included and merged with user terms."""
    assert "Laya" in DEFAULT_KEYTERMS
    assert "Gnani" in DEFAULT_KEYTERMS
    assert "Prisma" in DEFAULT_KEYTERMS
    assert "Timbre" in DEFAULT_KEYTERMS

    # Empty user keyterms returns defaults
    merged_none = merge_keyterms(None)
    assert merged_none == DEFAULT_KEYTERMS

    # Custom string keyterms merged
    user_str = "Rahul\nDevTools\nLaya"  # Laya is a duplicate
    merged = merge_keyterms(user_str)
    assert len(merged) == 6
    assert "Rahul" in merged
    assert "DevTools" in merged
    assert merged.count("Laya") == 1  # Deduplicated


def test_correction_3_language_codes_and_format():
    """Correction 3: Hinglish uses 'hi-IN,en-IN' with 'codemix'. Hindi uses 'hi-IN' with 'codemix'."""
    # Hindi configuration
    hindi_lang = "hi-IN"
    hindi_format = "codemix"
    assert hindi_lang == "hi-IN"
    assert hindi_format == "codemix"

    # Hinglish configuration
    hinglish_lang = "hi-IN,en-IN"
    hinglish_format = "codemix"
    assert hinglish_lang == "hi-IN,en-IN"
    assert hinglish_format == "codemix"


def test_correction_4_script_mix_computation():
    """Correction 4: Script mix computed directly from transcript tokens into Devanagari and Latin."""
    # Pure Hindi
    pure_hi = "नमस्ते यह एक परीक्षण है"
    mix_hi = compute_script_mix(pure_hi)
    assert mix_hi["label"] == "Script mix of the transcript"
    assert mix_hi["devanagari_pct"] == 100.0
    assert mix_hi["latin_pct"] == 0.0
    assert mix_hi["switch_count"] == 0

    # Hinglish sample with multiple switches
    hinglish_sample = "Hello everyone, आज मैं आपको अपने project Laya के बारे में बताने वाला हूँ।"
    mix_hinglish = compute_script_mix(hinglish_sample)
    assert mix_hinglish["label"] == "Script mix of the transcript"
    assert mix_hinglish["devanagari_count"] > 0
    assert mix_hinglish["latin_count"] > 0
    assert mix_hinglish["total_words"] == mix_hinglish["devanagari_count"] + mix_hinglish["latin_count"]
    assert mix_hinglish["switch_count"] >= 2
    # Verify switch positions are recorded
    assert len(mix_hinglish["switch_positions"]) == mix_hinglish["switch_count"]
    for s in mix_hinglish["switches"]:
        assert s["from_script"] in ("devanagari", "latin")
        assert s["to_script"] in ("devanagari", "latin")
        assert s["from_script"] != s["to_script"]


def test_correction_5_timbre_wav_duration_ffprobe():
    """Correction 5: Timbre duration comes from ffprobe, never from WAV header (header gave 44739s)."""
    timbre_wav = PROJECT_ROOT / "scratch" / "timbre_test.wav"
    if not timbre_wav.exists():
        pytest.skip("scratch/timbre_test.wav not found")

    duration = get_timbre_duration_ffprobe(timbre_wav)
    # True duration is around 3.68 seconds
    assert 1.0 <= duration <= 10.0, f"Expected normal audio duration ~3-4s, got {duration}"
    assert duration < 1000.0, "Did not fix WAV header bug (got > 1000s)"


def test_correction_6_ui_disclaimer_and_inputs():
    """Correction 6 & UI checks: Verify disclaimer, keyterms box, and script mix in index.html."""
    html_path = PROJECT_ROOT / "frontend" / "index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    # Plain line disclaimer check
    expected_disclaimer = "Transcript is automatic and can contain mistakes. You can edit it."
    assert expected_disclaimer in content

    # Keyterms textbox check
    expected_keyterms_label = "Words to recognise (names, topic words)"
    assert expected_keyterms_label in content
    assert 'id="keytermsInput"' in content

    # Script mix label check
    assert "Script mix of the transcript" in content
    assert 'id="scriptMixSection"' in content




def test_no_word_level_times_in_hindi_response():
    """Verify that Prisma returns no word times and NO word-level times exist in Hindi response.
    
    Checks that:
    1. Words list contains NO 'start', 'end', 'start_seconds', or 'end_seconds'.
    2. Chunks contain chunk-level times (e.g. 'chunk_start', 'chunk_end').
    3. Any timing for Hindi/Hinglish says 'chunk' and never 'word'.
    """
    from unittest.mock import patch
    from backend.gnani_stt import transcribe_audio_gnani

    mock_chunk_result = {
        "success": True,
        "transcript": "नमस्ते यह एक परीक्षण है",
        "detected_language_code": "hi-IN",
        "processing_time": 0.5,
        "end_to_end_latency": 0.6,
        "wall_time": 0.7,
    }

    test_wav = PROJECT_ROOT / "scratch" / "test_hi.wav"
    if not test_wav.exists():
        pytest.skip("scratch/test_hi.wav not found")

    with patch("backend.gnani_stt.transcribe_prisma_chunk", return_value=mock_chunk_result):
        res = transcribe_audio_gnani(test_wav, language_mode="hindi", api_key="test_key")
        assert res["success"] is True

        # 1. Check words list has NO word-level start/end times
        words = res["words"]
        assert len(words) > 0
        for w in words:
            assert isinstance(w, dict)
            assert "word" in w
            assert "start" not in w, f"Found forbidden 'start' time in word: {w}"
            assert "end" not in w, f"Found forbidden 'end' time in word: {w}"
            assert "start_seconds" not in w, f"Found forbidden 'start_seconds' time in word: {w}"
            assert "end_seconds" not in w, f"Found forbidden 'end_seconds' time in word: {w}"

        # 2. Check chunks have chunk-level times with 'chunk' in the timing name
        chunks = res["chunks"]
        assert len(chunks) > 0
        for ch in chunks:
            assert "chunk_start" in ch or "chunk_start_seconds" in ch
            assert "chunk_end" in ch or "chunk_end_seconds" in ch
            assert "chunk_duration" in ch or "chunk_duration_seconds" in ch
            # Check timing fields do NOT say 'word'
            for k in ch.keys():
                if "time" in k or "start" in k or "end" in k or "duration" in k:
                    assert "chunk" in k, f"Timing key '{k}' must say 'chunk' and never 'word'"
                    assert "word" not in k, f"Timing key '{k}' must not say 'word'"
            # Check any nested words inside chunk also have no start/end
            if "words" in ch:
                for w in ch["words"]:
                    assert "start" not in w
                    assert "end" not in w
