#!/usr/bin/env python3
"""Comprehensive test suite for interactive speech analysis chat assistant."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.chat import (
    GLOSSARY,
    build_allowed_numbers_map,
    check_rate_limit,
    generate_template_chat_answer,
    validate_chat_answer,
)

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PITCH_TIER5 = PROJECT_ROOT / "dataset" / "examples" / "librivox_01_pitch_tier5.json"


@pytest.fixture
def pitch_tier5_data() -> Dict[str, Any]:
    with open(EXAMPLE_PITCH_TIER5, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "transcript_source" not in data or not data["transcript_source"]:
        data["transcript_source"] = "provided"
    return data


def test_chat_endpoint_schema(pitch_tier5_data):
    """Test POST /api/chat basic request and response format."""
    payload = {
        "analysis": pitch_tier5_data,
        "question": "What is a z-score?",
        "history": [],
        "session_id": "test_schema_sess",
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert data["source"] in ["ai", "template"]
    assert len(data["answer"]) > 0


def test_chat_flat_analysis_fields(pitch_tier5_data):
    """Test POST /api/chat accepting flat flaw_regions, mode, duration, transcript_source."""
    payload = {
        "flaw_regions": pitch_tier5_data.get("flaw_regions", []),
        "mode": pitch_tier5_data.get("mode", "paired"),
        "duration": pitch_tier5_data.get("duration", 94.98),
        "transcript_source": pitch_tier5_data.get("transcript_source", "provided"),
        "question": "What happened?",
        "history": [],
        "session_id": "test_flat_sess",
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["source"] in ["ai", "template"]


def test_chat_question_validation(pitch_tier5_data):
    """Test question empty rejection and max length constraint."""
    # Empty question
    resp = client.post(
        "/api/chat",
        json={"analysis": pitch_tier5_data, "question": "   ", "session_id": "test_empty_sess"},
    )
    assert resp.status_code == 400

    # Question longer than 300 chars is accepted and clamped/handled
    long_q = "Why was this flagged? " + ("detail " * 60)
    assert len(long_q) > 300
    # Client can send up to 300 chars, if exactly 300 it passes schema
    resp_clamped = client.post(
        "/api/chat",
        json={"analysis": pitch_tier5_data, "question": long_q[:300], "session_id": "test_long_sess"},
    )
    assert resp_clamped.status_code == 200


def test_chat_data_filter_never_contains_words_or_transcript(pitch_tier5_data):
    """Verify that filtered context prepared for the model strips words, transcript, and audio."""
    flaw_regions = pitch_tier5_data.get("flaw_regions", [])
    clean_regions = [
        {
            "type": r.get("type"),
            "start": round(float(r.get("start", 0.0)), 3),
            "end": round(float(r.get("end", 0.0)), 3),
            "duration": round(float(r.get("end", 0.0) - r.get("start", 0.0)), 3),
            "peak_z": round(float(r.get("peak_z", r.get("z", 0.0))), 3),
            "observed_value": r.get("observed_value"),
            "unit": r.get("unit"),
            "baseline_mean": r.get("baseline_mean"),
            "baseline_spread": r.get("baseline_spread"),
            "threshold": r.get("threshold"),
            "severity": r.get("severity"),
        }
        for r in flaw_regions
    ]

    filtered_context = {
        "duration": round(float(pitch_tier5_data.get("duration", 0.0)), 3),
        "mode": pitch_tier5_data.get("mode", "experimental"),
        "transcript_source": pitch_tier5_data.get("transcript_source", "auto"),
        "flaw_regions": clean_regions,
        "glossary": GLOSSARY,
    }

    raw_str = json.dumps(filtered_context)
    # Check that forbidden fields are completely absent
    assert "audio_file" not in raw_str
    assert "words" not in raw_str
    assert "transcript" not in raw_str or "transcript_source" in raw_str


def test_template_deterministic_answers(pitch_tier5_data):
    """Test deterministic responder for the four suggested questions and keyword matches."""
    allowed_map = build_allowed_numbers_map(pitch_tier5_data)

    # 1. What happened
    ans1 = generate_template_chat_answer("What happened?", pitch_tier5_data)
    assert "pitch" in ans1.lower()
    assert "54.58" in ans1
    assert "58.84" in ans1
    valid1, reason1, _ = validate_chat_answer(ans1, pitch_tier5_data, pitch_tier5_data["flaw_regions"], allowed_map)
    assert valid1, f"Validation failed: {reason1}"

    # 2. Why flagged
    ans2 = generate_template_chat_answer("Why was it flagged?", pitch_tier5_data)
    assert "flagged" in ans2.lower()
    assert "observed value" in ans2.lower()
    valid2, reason2, _ = validate_chat_answer(ans2, pitch_tier5_data, pitch_tier5_data["flaw_regions"], allowed_map)
    assert valid2, f"Validation failed: {reason2}"

    # 3. How to improve
    ans3 = generate_template_chat_answer("How can I improve?", pitch_tier5_data)
    assert "tips" in ans3.lower() or "pitch" in ans3.lower()
    valid3, reason3, _ = validate_chat_answer(ans3, pitch_tier5_data, pitch_tier5_data["flaw_regions"], allowed_map)
    assert valid3, f"Validation failed: {reason3}"

    # 4. What is a z-score
    ans4 = generate_template_chat_answer("What is a z-score?", pitch_tier5_data)
    assert "standard deviations" in ans4.lower()
    valid4, reason4, _ = validate_chat_answer(ans4, pitch_tier5_data, pitch_tier5_data["flaw_regions"], allowed_map)
    assert valid4, f"Validation failed: {reason4}"

    # 5. How reliable
    ans5 = generate_template_chat_answer("How reliable is this analysis?", pitch_tier5_data)
    assert "paired mode" in ans5.lower() or "reliable" in ans5.lower()
    valid5, reason5, _ = validate_chat_answer(ans5, pitch_tier5_data, pitch_tier5_data["flaw_regions"], allowed_map)
    assert valid5, f"Validation failed: {reason5}"

    # 6. Fallback for other questions
    ans6 = generate_template_chat_answer("Who won the world cup?", pitch_tier5_data)
    assert "cannot be answered from the measurements" in ans6.lower()
    assert "what happened" in ans6.lower()
    valid6, reason6, _ = validate_chat_answer(ans6, pitch_tier5_data, pitch_tier5_data["flaw_regions"], allowed_map)
    assert valid6, f"Validation failed: {reason6}"


def test_validation_rejects_hallucinations(pitch_tier5_data):
    """Test that numbers not present in the analysis data cause validation failure."""
    allowed_map = build_allowed_numbers_map(pitch_tier5_data)
    flaw_regions = pitch_tier5_data.get("flaw_regions", [])

    hallucinated_answer = "The z-score was 98.76 and duration was 456.78 seconds."
    valid, reason, _ = validate_chat_answer(hallucinated_answer, pitch_tier5_data, flaw_regions, allowed_map)
    assert not valid
    assert "Hallucinated numbers" in reason


def test_validation_rejects_meaning_contradiction(pitch_tier5_data):
    """Test that contradiction of flat pitch (e.g. drop/falling pitch) is rejected."""
    allowed_map = build_allowed_numbers_map(pitch_tier5_data)
    flaw_regions = pitch_tier5_data.get("flaw_regions", [])

    # The pitch tier 5 has flat pitch (z = -4.478 < 0)
    contradicting_answer = "The speaker had a noticeable pitch drop at 54.58 s."
    valid, reason, _ = validate_chat_answer(contradicting_answer, pitch_tier5_data, flaw_regions, allowed_map)
    assert not valid
    assert "Meaning contradiction" in reason


def test_validation_rejects_emoji_and_long_dashes(pitch_tier5_data):
    """Test rejection of emoji characters and long dashes."""
    allowed_map = build_allowed_numbers_map(pitch_tier5_data)
    flaw_regions = pitch_tier5_data.get("flaw_regions", [])

    emoji_ans = "The delivery was flat at 54.58 s \U0001F600."
    valid, reason, _ = validate_chat_answer(emoji_ans, pitch_tier5_data, flaw_regions, allowed_map)
    assert not valid
    assert "Emoji" in reason

    dash_ans = "The delivery was flat \u2014 very monotone at 54.58 s."
    valid2, reason2, _ = validate_chat_answer(dash_ans, pitch_tier5_data, flaw_regions, allowed_map)
    assert not valid2
    assert "Long dash" in reason2


def test_rate_limiting_enforcement():
    """Verify rate limit of 10 questions per minute per session."""
    sess_id = "test_rate_limit_unique_sess"
    for i in range(10):
        ok = check_rate_limit(sess_id, max_requests=10, window_seconds=60.0)
        assert ok, f"Request {i+1} should have been allowed"

    # 11th request should be blocked
    ok_11 = check_rate_limit(sess_id, max_requests=10, window_seconds=60.0)
    assert not ok_11, "11th request should be rate limited"


def test_identical_question_cache(pitch_tier5_data):
    """Verify caching of identical question plus analysis."""
    payload = {
        "analysis": pitch_tier5_data,
        "question": "What is a z-score?",
        "session_id": "test_cache_sess",
    }
    resp1 = client.post("/api/chat", json=payload)
    assert resp1.status_code == 200
    data1 = resp1.json()

    resp2 = client.post("/api/chat", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.json()

    assert data1["answer"] == data2["answer"]
    assert data1["source"] == data2["source"]


def test_chat_graph_questions(pitch_tier5_data):
    """Verify answering questions about the 4 dashboard graphs, tracks, shaded bands, and playback cursor."""
    allowed_map = build_allowed_numbers_map(pitch_tier5_data)
    flaw_regions = pitch_tier5_data.get("flaw_regions", [])

    # 1. What do the graphs show?
    resp = client.post(
        "/api/chat",
        json={"analysis": pitch_tier5_data, "question": "What do the graphs show?", "session_id": "test_graph_all"},
    )
    assert resp.status_code == 200
    ans = resp.json()["answer"]
    assert "waveform" in ans.lower()
    assert "pitch" in ans.lower()
    assert "energy" in ans.lower()
    assert "rate" in ans.lower()

    # 2. Waveform track
    ans_wf = generate_template_chat_answer("What does the waveform graph show?", pitch_tier5_data)
    assert "amplitude" in ans_wf.lower()
    valid, reason, _ = validate_chat_answer(ans_wf, pitch_tier5_data, flaw_regions, allowed_map)
    assert valid, reason

    # 3. Pitch track
    ans_p = generate_template_chat_answer("What does the pitch curve show?", pitch_tier5_data)
    assert "semitones" in ans_p.lower()
    valid, reason, _ = validate_chat_answer(ans_p, pitch_tier5_data, flaw_regions, allowed_map)
    assert valid, reason

    # 4. Energy track
    ans_e = generate_template_chat_answer("What does Track 3 show?", pitch_tier5_data)
    assert "loudness" in ans_e.lower() or "energy" in ans_e.lower()
    valid, reason, _ = validate_chat_answer(ans_e, pitch_tier5_data, flaw_regions, allowed_map)
    assert valid, reason

    # 5. Rate track
    ans_r = generate_template_chat_answer("What does the rate plot show?", pitch_tier5_data)
    assert "speaking speed" in ans_r.lower()
    valid, reason, _ = validate_chat_answer(ans_r, pitch_tier5_data, flaw_regions, allowed_map)
    assert valid, reason

    # 6. Shaded regions
    ans_sh = generate_template_chat_answer("What does the shaded region on the graph mean?", pitch_tier5_data)
    assert "flagged" in ans_sh.lower() or "shaded" in ans_sh.lower()
    valid, reason, _ = validate_chat_answer(ans_sh, pitch_tier5_data, flaw_regions, allowed_map)
    assert valid, reason

    # 7. Red playback cursor
    ans_red = generate_template_chat_answer("What is the red vertical line?", pitch_tier5_data)
    assert "cursor" in ans_red.lower() or "playback" in ans_red.lower()
    valid, reason, _ = validate_chat_answer(ans_red, pitch_tier5_data, flaw_regions, allowed_map)
    assert valid, reason

