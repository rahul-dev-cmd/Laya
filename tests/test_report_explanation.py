"""Unit and integration tests for the AI explanation layer and number validation."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.report import (
    call_llm_provider,
    compute_cache_key,
    extract_allowed_numbers,
    generate_report,
    generate_template_report,
    validate_report_numbers,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def pitch_tier5_payload() -> Dict[str, Any]:
    target = PROJECT_ROOT / "dataset" / "examples" / "librivox_01_pitch_tier5.json"
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def rate_tier4_payload() -> Dict[str, Any]:
    target = PROJECT_ROOT / "dataset" / "examples" / "librivox_01_rate_tier4.json"
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def test_extract_allowed_numbers(pitch_tier5_payload: Dict[str, Any]) -> None:
    allowed = extract_allowed_numbers(pitch_tier5_payload)
    # Check key numbers exist in allowed mapping
    assert 54.58 in allowed
    assert 58.84 in allowed
    assert 0.037 in allowed
    assert -4.478 in allowed
    assert 94.98 in allowed
    assert 1.0 in allowed  # region count


def test_validate_report_numbers_valid(pitch_tier5_payload: Dict[str, Any]) -> None:
    allowed = extract_allowed_numbers(pitch_tier5_payload)
    template_rep = generate_template_report(pitch_tier5_payload)
    is_valid, hallucinations, matched = validate_report_numbers(template_rep, allowed)

    assert is_valid is True
    assert len(hallucinations) == 0
    assert len(matched) > 0


def test_validate_report_numbers_hallucinated(pitch_tier5_payload: Dict[str, Any]) -> None:
    allowed = extract_allowed_numbers(pitch_tier5_payload)
    fake_report = {
        "summary": "We detected a pitch flaw at 120.5 Hz with 85% confidence.",
        "what_happened": ["At 35.2 seconds, pitch dropped significantly."],
        "why_it_matters": "Good pitch is important.",
        "practice_tips": {"pitch": ["Tip one", "Tip two", "Tip three"]},
        "caveats": "None",
    }
    is_valid, hallucinations, matched = validate_report_numbers(fake_report, allowed)

    assert is_valid is False
    assert "120.5" in hallucinations or any("120" in h for h in hallucinations)
    assert "85.0" in hallucinations or "85" in hallucinations


def test_template_report_structure(pitch_tier5_payload: Dict[str, Any]) -> None:
    rep = generate_template_report(pitch_tier5_payload)
    assert rep["source"] == "template"
    assert "summary" in rep
    assert isinstance(rep["what_happened"], list)
    assert len(rep["what_happened"]) == 1
    assert "why_it_matters" in rep
    assert "practice_tips" in rep
    assert "pitch" in rep["practice_tips"]
    assert len(rep["practice_tips"]["pitch"]) == 3
    assert "caveats" in rep


def test_generate_report_fallback_when_no_key(pitch_tier5_payload: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "")
    rep = generate_report(pitch_tier5_payload, example_id="test_no_key")
    assert rep["source"] == "template"
    assert len(rep["what_happened"]) == 1


def test_api_report_endpoint(pitch_tier5_payload: Dict[str, Any]) -> None:
    client = TestClient(app)
    req_body = {
        "duration": pitch_tier5_payload["duration"],
        "mode": pitch_tier5_payload["mode"],
        "transcript_source": pitch_tier5_payload.get("transcript_source"),
        "flaw_regions": pitch_tier5_payload.get("flaw_regions", []),
        "example_id": "librivox_01_pitch_tier5",
    }
    response = client.post("/api/report", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert "source" in data
    assert "summary" in data
    assert "what_happened" in data
    assert "why_it_matters" in data
    assert "practice_tips" in data
    assert "caveats" in data


def test_pregenerated_example_report_loaded(pitch_tier5_payload: Dict[str, Any]) -> None:
    # Pre-generated report for librivox_01_pitch_tier5 should be loaded directly
    rep = generate_report(pitch_tier5_payload, example_id="librivox_01_pitch_tier5")
    assert rep is not None
    assert "summary" in rep
    assert len(rep["what_happened"]) == 1


def test_ai_source_validation_success(pitch_tier5_payload: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    allowed = extract_allowed_numbers(pitch_tier5_payload)
    mock_ai_output = {
        "summary": "We analyzed 94.98 seconds of speech and found 1 pitch flaw region.",
        "what_happened": [
            "From 54.58s to 58.84s (duration 4.26s), the vocal inflection dropped to 0.04 semitones with deviation z = -4.5."
        ],
        "why_it_matters": "Monotone inflection makes speech difficult to follow.",
        "practice_tips": {
            "pitch": [
                "Mark key emphasis words in your notes.",
                "Read short paragraphs aloud with expressive inflection.",
                "Ensure declarative statements end with gentle downward pitch."
            ]
        },
        "caveats": "Compared against a reference take."
    }
    is_valid, hallucinations, matched = validate_report_numbers(mock_ai_output, allowed)
    assert is_valid is True
    assert len(hallucinations) == 0
    # Every number should be accounted for
    for expected_num in [94.98, 1.0, 54.58, 58.84, 4.26, 0.04, -4.5]:
        assert any(abs(n - expected_num) < 0.01 for n in matched)


def test_compute_region_meaning():
    from backend.report import compute_region_meaning

    m_flat = compute_region_meaning("pitch", -3.2)
    assert "flat or monotone" in m_flat
    assert "standard deviation" in m_flat

    m_pitch_high = compute_region_meaning("pitch", 2.5)
    assert "pitch moved far more than the baseline" in m_pitch_high

    m_fast = compute_region_meaning("rate", 2.1)
    assert "speech was faster than the baseline" in m_fast

    m_slow = compute_region_meaning("rate", -2.1)
    assert "speech was slower than the baseline" in m_slow

    m_short_pause = compute_region_meaning("pause", -2.0)
    assert "shorter than normal" in m_short_pause

    m_long_pause = compute_region_meaning("pause", 2.0)
    assert "longer than normal" in m_long_pause


def test_practice_tips_table_completeness():
    from backend.tips import PRACTICE_TIPS_TABLE, get_direction_key

    expected_keys = [
        "flat pitch",
        "over-varied pitch",
        "too fast",
        "too slow",
        "pause too short",
        "pause too long",
    ]
    for key in expected_keys:
        assert key in PRACTICE_TIPS_TABLE
        assert len(PRACTICE_TIPS_TABLE[key]) == 3
        for tip in PRACTICE_TIPS_TABLE[key]:
            assert isinstance(tip, str)
            assert len(tip) > 10

    assert get_direction_key("pitch", -2.0) == "flat pitch"
    assert get_direction_key("pitch", 2.0) == "over-varied pitch"
    assert get_direction_key("rate", 2.0) == "too fast"
    assert get_direction_key("rate", -2.0) == "too slow"
    assert get_direction_key("pause", -2.0) == "pause too short"
    assert get_direction_key("pause", 2.0) == "pause too long"


def test_build_caveats_text():
    from backend.report import build_caveats_text

    c1 = build_caveats_text("paired", None)
    assert c1 == "Compared with an ideal recording of the same speech"

    c2 = build_caveats_text("paired", "auto")
    assert "Compared with an ideal recording of the same speech" in c2
    assert "automatically" in c2

    c3 = build_caveats_text("experimental", None)
    assert c3 == "Compared with a general baseline, so less reliable"

    c4 = build_caveats_text("experimental", "auto")
    assert "Compared with a general baseline, so less reliable" in c4
    assert "automatically" in c4


def test_meaning_contradiction_check():
    from backend.report import check_meaning_contradictions

    pitch_flat_region = [{"type": "pitch", "z": -3.5}]

    # Flat pitch: reject drop, steady pitch, smooth, reduce variation
    bad_reply_drop = {
        "summary": "Pitch dropped significantly in the recording.",
        "what_happened": ["The tone dropped low."],
        "why_it_matters": "Drops make it hard to hear.",
    }
    ok_drop, reason_drop = check_meaning_contradictions(bad_reply_drop, pitch_flat_region)
    assert ok_drop is False
    assert "drop" in reason_drop

    bad_reply_steady = {
        "summary": "The speech was good.",
        "what_happened": ["Speaker maintained a steady pitch throughout."],
        "why_it_matters": "Pitch matters.",
    }
    ok_steady, reason_steady = check_meaning_contradictions(bad_reply_steady, pitch_flat_region)
    assert ok_steady is False
    assert "steady pitch" in reason_steady

    bad_reply_smooth = {
        "summary": "Smooth pitch was detected.",
        "what_happened": ["The delivery was smooth."],
        "why_it_matters": "Smoothness matters.",
    }
    ok_smooth, reason_smooth = check_meaning_contradictions(bad_reply_smooth, pitch_flat_region)
    assert ok_smooth is False
    assert "smooth" in reason_smooth

    bad_reply_reduce = {
        "summary": "The speaker needs to reduce variation.",
        "what_happened": ["Observed value was low."],
        "why_it_matters": "Important.",
    }
    ok_reduce, reason_reduce = check_meaning_contradictions(bad_reply_reduce, pitch_flat_region)
    assert ok_reduce is False
    assert "reduce variation" in reason_reduce

    # Valid reply describing monotone delivery without contradiction
    good_reply = {
        "summary": "The speech delivery was flat and monotone in one section.",
        "what_happened": ["The pitch variation was too small with observed standard deviation of 0.04 semitones."],
        "why_it_matters": "Lack of pitch inflection reduces listener interest.",
    }
    ok_good, reason_good = check_meaning_contradictions(good_reply, pitch_flat_region)
    assert ok_good is True
    assert reason_good is None


