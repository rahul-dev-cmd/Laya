"""Tests for Stage 4: Practice tips translation, Timbre audio synthesis and UI components."""

import hashlib
import json
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from starlette.testclient import TestClient

from backend.app import app
from backend.tips import (
    PRACTICE_TIPS_TABLE,
    PRACTICE_TIPS_TABLE_HI,
    get_direction_key,
    get_practice_tips_for_regions,
    get_tips_speech_text,
    is_hindi_language,
)


def test_hindi_practice_tips_table_completeness():
    """Verify that PRACTICE_TIPS_TABLE_HI matches the English table 1-to-1."""
    expected_keys = [
        "flat pitch",
        "over-varied pitch",
        "too fast",
        "too slow",
        "pause too short",
        "pause too long",
    ]
    assert set(PRACTICE_TIPS_TABLE_HI.keys()) == set(expected_keys)
    assert set(PRACTICE_TIPS_TABLE_HI.keys()) == set(PRACTICE_TIPS_TABLE.keys())

    devanagari_regex = re.compile(r"[\u0900-\u097F]")

    for key, tips in PRACTICE_TIPS_TABLE_HI.items():
        assert len(tips) == 3, f"Category '{key}' must have exactly 3 tips."
        for tip in tips:
            assert isinstance(tip, str)
            assert len(tip.strip()) > 10, f"Tip in '{key}' is too short: {tip}"
            # Ensure it contains Devanagari script
            assert devanagari_regex.search(tip) is not None, f"Tip must be in Devanagari: {tip}"
            # Ensure no forbidden characters (em-dash, en-dash, emojis)
            assert "\u2014" not in tip, f"Em-dash forbidden in tip: {tip}"
            assert "\u2013" not in tip, f"En-dash forbidden in tip: {tip}"
            for ch in tip:
                assert ord(ch) <= 0x1F000, f"Emoji forbidden in tip: {tip}"


def test_hindi_tips_direction_and_meaning():
    """Verify Hindi tips match the exact direction and meaning of English counterparts."""
    # Flat pitch: lift pitch on key words
    flat_pitch_tips = " ".join(PRACTICE_TIPS_TABLE_HI["flat pitch"])
    assert "आवाज़ उठाएं" in flat_pitch_tips or "ज़ोर दें" in flat_pitch_tips
    assert "सुर बदलें" in flat_pitch_tips

    # Over-varied pitch: natural comfortable voice, calm contour within narrower band
    over_pitch_tips = " ".join(PRACTICE_TIPS_TABLE_HI["over-varied pitch"])
    assert "सामान्य, सहज आवाज़" in over_pitch_tips
    assert "सबसे ज़्यादा ज़ोर" in over_pitch_tips
    assert "शांत" in over_pitch_tips or "संतुलित" in over_pitch_tips

    # Too fast: steady rhythm, pause for breath
    fast_tips = " ".join(PRACTICE_TIPS_TABLE_HI["too fast"])
    assert "सांस" in fast_tips or "संतुलित गति" in fast_tips

    # Too slow: conversational tempo, shorten gaps
    slow_tips = " ".join(PRACTICE_TIPS_TABLE_HI["too slow"])
    assert "गति बढ़ाने" in slow_tips or "सहज" in slow_tips

    # Pause too short: full breath pause without specific seconds
    pause_short_tips = " ".join(PRACTICE_TIPS_TABLE_HI["pause too short"])
    assert "गहरी सांस" in pause_short_tips
    assert "ठहराव" in pause_short_tips
    assert "दो सेकंड" not in pause_short_tips

    # Pause too long: keep breath pause short without specific seconds
    pause_long_tips = " ".join(PRACTICE_TIPS_TABLE_HI["pause too long"])
    assert "छोटा रखें" in pause_long_tips
    assert "लय न टूटे" in pause_long_tips
    assert "एक शांत सेकंड" not in pause_long_tips

    # English table checks: no specific numbers of seconds
    en_pause_short = PRACTICE_TIPS_TABLE["pause too short"][0]
    en_pause_long = PRACTICE_TIPS_TABLE["pause too long"][0]
    assert "take a full breath of pause at periods and when starting a new topic" in en_pause_short.lower()
    assert "keep the breath pause between sentences short so the rhythm does not break" in en_pause_long.lower()
    assert "two-second" not in " ".join(PRACTICE_TIPS_TABLE["pause too short"]).lower()
    assert "one calm second" not in " ".join(PRACTICE_TIPS_TABLE["pause too long"]).lower()


def test_deterministic_tips_retrieval_and_no_runtime_ai():
    """Verify that tips retrieval is deterministic and uses only fixed tables."""
    flaw_regions_flat = [{"type": "pitch", "z": -2.5, "peak_z": -2.5}]
    tips_hi = get_practice_tips_for_regions(flaw_regions_flat, language="hindi")
    tips_hi_repeat = get_practice_tips_for_regions(flaw_regions_flat, language="hindi")

    assert tips_hi == tips_hi_repeat
    assert "pitch" in tips_hi
    assert len(tips_hi["pitch"]) == 3
    assert tips_hi["pitch"] == PRACTICE_TIPS_TABLE_HI["flat pitch"]

    # English tips remain unchanged
    tips_en = get_practice_tips_for_regions(flaw_regions_flat, language="english")
    assert tips_en["pitch"] == PRACTICE_TIPS_TABLE["flat pitch"]

    # Combined speech text
    speech_text_hi = get_tips_speech_text(flaw_regions_flat, language="hindi")
    assert "हर वाक्य" in speech_text_hi
    assert len(speech_text_hi.split()) > 10


def test_sha256_cache_key_generation_and_audio_caching(tmp_path):
    """Verify audio caching by SHA-256 of text, language, and voice."""
    from backend.gnani_stt import synthesize_timbre_speech

    sample_text = "हर वाक्य के मुख्य शब्द पर ज़ोर दें।"
    lang = "hi-IN"
    voice = "Shivani"

    # Compute expected SHA-256
    hasher = hashlib.sha256()
    hasher.update(sample_text.strip().encode("utf-8"))
    hasher.update(lang.lower().encode("utf-8"))
    hasher.update(voice.lower().encode("utf-8"))
    expected_hash = hasher.hexdigest()

    cache_dir = Path("scratch/cache/speak")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fake_wav = cache_dir / f"{expected_hash}.wav"
    fake_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    try:
        with patch("backend.gnani_stt.get_timbre_duration_ffprobe", return_value=3.25):
            res = synthesize_timbre_speech(text=sample_text, language=lang, voice=voice)
            assert res["success"] is True
            assert res["cached"] is True
            assert res["cache_key"] == expected_hash
            assert res["voice"] == voice
            assert res["language"] == lang
    finally:
        fake_wav.unlink(missing_ok=True)


def test_api_tips_speak_endpoint():
    """Test the POST /api/tips/speak endpoint with JSON response."""
    client = TestClient(app)

    fake_res = {
        "success": True,
        "audio_path": "scratch/cache/speak/test_tips_audio.wav",
        "duration": 5.4,
        "cached": True,
        "cache_key": "dummy_tips_hash",
        "voice": "Shivani",
        "language": "hi-IN",
    }

    dummy_path = Path("scratch/cache/speak/test_tips_audio.wav")
    dummy_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    try:
        with patch("backend.app.synthesize_timbre_speech", return_value=fake_res):
            resp = client.post(
                "/api/tips/speak",
                json={
                    "flaw_regions": [{"type": "pitch", "z": -2.0, "peak_z": -2.0}],
                    "language_mode": "hindi",
                    "voice": "Shivani",
                    "format": "json",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["label"] == "Model voice, not your voice"
            assert data["language"] == "hi-IN"
            assert "audio_url" in data
            assert data["cached"] is True
            assert "हर वाक्य" in data["text"]
    finally:
        dummy_path.unlink(missing_ok=True)


def test_ui_has_listen_tips_button_and_disclaimer():
    """Verify that frontend/index.html and app.js contain the 'Listen to the tips' button."""
    index_html = Path("frontend/index.html").read_text(encoding="utf-8")
    app_js = Path("frontend/app.js").read_text(encoding="utf-8")

    assert "Listen to the tips" in index_html
    assert 'id="btnListenTips"' in index_html
    assert 'id="btnListenTipsMain"' in index_html
    assert 'id="tipsAudioPlayerContainer"' in index_html

    assert "handleListenTipsClick" in app_js
    assert "Model voice, not your voice" in app_js
    assert "/api/tips/speak" in app_js

