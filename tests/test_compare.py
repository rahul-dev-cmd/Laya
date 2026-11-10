"""Tests for Retake Comparison (Stage A)."""

import pytest
from starlette.testclient import TestClient

from backend.app import app
from backend.compare import METHOD_NOTE, compare_takes, regions_overlap_50pct


def make_dummy_analysis(
    duration: float = 10.0,
    language_mode: str = "english",
    mode: str = "paired",
    overall: float = 85.0,
    dimensions: list = None,
    flaw_regions: list = None,
    transcript: str = "Hello world from test speech.",
    passage_id: str = None,
):
    if dimensions is None:
        dimensions = [
            {"name": "Pace", "scored": True, "score": 80.0},
            {"name": "Pitch variation", "scored": True, "score": 90.0},
            {"name": "Pauses", "scored": False, "score": None},
        ]
    if flaw_regions is None:
        flaw_regions = [
            {"type": "pitch", "start": 2.0, "end": 4.0, "peak_z": -2.5, "severity": "tier 3"},
            {"type": "rate", "start": 6.0, "end": 8.0, "peak_z": 2.2, "severity": "tier 2"},
        ]

    data = {
        "duration": duration,
        "language_mode": language_mode,
        "mode": mode,
        "flaw_regions": flaw_regions,
        "rubric": {
            "overall": overall,
            "dimensions": dimensions,
        },
    }
    if transcript is not None:
        data["transcript"] = transcript
    if passage_id is not None:
        data["passage_id"] = passage_id

    return data


def test_identical_takes_give_zero_change():
    """Verify that comparing identical takes results in zero delta across all scored dimensions."""
    take1 = make_dummy_analysis()
    res = compare_takes(take1, take1)

    assert res["comparable"] is True
    assert res["method_note"] == METHOD_NOTE
    assert len(res["dimensions"]) == 2

    for dim in res["dimensions"]:
        assert dim["change"] == 0.0
        assert dim["score_1"] == dim["score_2"]
        assert "+0.0" in dim["signed_change"] or "0.0" in dim["signed_change"]

    assert res["overall"]["overall_change"] == 0.0
    assert res["overall"]["overall_1"] == res["overall"]["overall_2"]

    # All take 1 flags are still flagged since take 2 is identical
    for flag in res["take_1_flags"]:
        assert flag["status"] == "still flagged"

    # No new flags in take 2
    assert len(res["take_2_new_flags"]) == 0


def test_score_rise_gives_positive_change():
    """Verify that a score rise gives a positive change and score drop gives negative change."""
    take1 = make_dummy_analysis(
        overall=80.0,
        dimensions=[
            {"name": "Pitch variation", "scored": True, "score": 80.0},
        ],
    )
    take2 = make_dummy_analysis(
        overall=85.5,
        dimensions=[
            {"name": "Pitch variation", "scored": True, "score": 85.5},
        ],
    )

    res = compare_takes(take1, take2)
    assert res["comparable"] is True

    pitch_dim = next(d for d in res["dimensions"] if d["name"] == "Pitch variation")
    assert pitch_dim["change"] > 0
    assert pitch_dim["change"] == 5.5
    assert pitch_dim["signed_change"] == "+5.5"

    assert res["overall"]["overall_change"] > 0
    assert res["overall"]["overall_change"] == 5.5
    assert res["overall"]["signed_change"] == "+5.5"

    # Conversely, take 2 having lower score gives negative change
    res_drop = compare_takes(take2, take1)
    pitch_drop = next(d for d in res_drop["dimensions"] if d["name"] == "Pitch variation")
    assert pitch_drop["change"] < 0
    assert pitch_drop["change"] == -5.5
    assert pitch_drop["signed_change"] == "-5.5"


def test_non_comparable_cases():
    """Verify non-comparable cases: different language, mode, or duration differing by >20%."""
    base = make_dummy_analysis(duration=10.0, language_mode="english", mode="paired")

    # 1. Different language
    diff_lang = make_dummy_analysis(duration=10.0, language_mode="hindi", mode="paired")
    res_lang = compare_takes(base, diff_lang)
    assert res_lang["comparable"] is False
    assert any("language" in r.lower() for r in res_lang["reasons"])
    assert "dimensions" not in res_lang

    # 2. Different mode (paired vs experimental)
    diff_mode = make_dummy_analysis(duration=10.0, language_mode="english", mode="experimental")
    res_mode = compare_takes(base, diff_mode)
    assert res_mode["comparable"] is False
    assert any("mode" in r.lower() for r in res_mode["reasons"])
    assert "dimensions" not in res_mode

    # 3. Duration differs by > 20 percent (10.0s vs 13.0s is 30% diff)
    diff_dur = make_dummy_analysis(duration=13.0, language_mode="english", mode="paired")
    res_dur = compare_takes(base, diff_dur)
    assert res_dur["comparable"] is False
    assert any("duration" in r.lower() for r in res_dur["reasons"])
    assert "dimensions" not in res_dur

    # 4. Duration within 20 percent (10.0s vs 11.5s is 15% diff) is comparable
    within_dur = make_dummy_analysis(duration=11.5, language_mode="english", mode="paired")
    res_ok = compare_takes(base, within_dur)
    assert res_ok["comparable"] is True


def test_flag_overlap_tracking():
    """Verify 50% overlap rule: still flagged vs not flagged in take 2 vs new in take 2."""
    take1 = make_dummy_analysis(
        flaw_regions=[
            {"type": "pitch", "start": 2.0, "end": 4.0},  # length 2.0s
            {"type": "rate", "start": 6.0, "end": 8.0},   # length 2.0s
        ]
    )

    take2 = make_dummy_analysis(
        flaw_regions=[
            # Pitch [2.5, 4.5]: overlaps [2.0, 4.0] on [2.5, 4.0] (1.5s). 1.5 / 2.0 = 75% >= 50%
            {"type": "pitch", "start": 2.5, "end": 4.5},
            # Pause [10.0, 11.0]: new flag not overlapping anything in take 1
            {"type": "pause", "start": 10.0, "end": 11.0},
        ]
    )

    res = compare_takes(take1, take2)
    assert res["comparable"] is True

    # Take 1 flags check
    t1_pitch = next(f for f in res["take_1_flags"] if f["type"] == "pitch")
    assert t1_pitch["status"] == "still flagged"

    t1_rate = next(f for f in res["take_1_flags"] if f["type"] == "rate")
    assert t1_rate["status"] == "not flagged in take 2"

    # Take 2 new flags check
    assert len(res["take_2_new_flags"]) == 1
    t2_pause = res["take_2_new_flags"][0]
    assert t2_pause["type"] == "pause"
    assert t2_pause["status"] == "new in take 2"

    # Strict check: never say 'fixed' or 'improved' anywhere
    dump = str(res).lower()
    assert "fixed" not in dump
    assert "improved" not in dump


def test_indic_word_diff_count():
    """Verify word-level diff count using difflib for Hindi/Hinglish transcripts."""
    take1 = make_dummy_analysis(
        language_mode="hindi",
        transcript="यह लया के लिए एक अच्छा परीक्षण है",
    )
    take2 = make_dummy_analysis(
        language_mode="hindi",
        transcript="यह लया के लिए एक नया और अच्छा परीक्षण है",
    )

    res = compare_takes(take1, take2)
    assert res["comparable"] is True
    assert res["transcripts"] is not None
    assert res["transcripts"]["words_differ"] > 0
    assert "words differ" in res["transcripts"]["label"]
    assert "mistakes" not in res["transcripts"]["label"].lower()

    # English comparison does not compute transcript word diff
    take_en1 = make_dummy_analysis(language_mode="english")
    take_en2 = make_dummy_analysis(language_mode="english")
    res_en = compare_takes(take_en1, take_en2)
    assert res_en["transcripts"] is None


def test_api_compare_endpoint_validation():
    """Verify that POST /api/compare validates input and rejects non-analysis objects."""
    client = TestClient(app)

    # 1. Valid analysis results return 200
    take1 = make_dummy_analysis()
    take2 = make_dummy_analysis()
    resp = client.post("/api/compare", json={"take1": take1, "take2": take2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["comparable"] is True
    assert "dimensions" in data

    # 2. Missing take2 returns 400
    resp_missing = client.post("/api/compare", json={"take1": take1})
    assert resp_missing.status_code == 400

    # 3. Invalid object (e.g. missing required rubric/flaw_regions/duration) returns 400
    invalid_take = {"foo": "bar"}
    resp_invalid = client.post("/api/compare", json={"take1": take1, "take2": invalid_take})
    assert resp_invalid.status_code == 400
    assert "valid analysis result objects" in resp_invalid.json().get("detail", "")

    # 4. Non-JSON Content-Type returns 400
    resp_bad_type = client.post("/api/compare", content=b"plain text", headers={"Content-Type": "text/plain"})
    assert resp_bad_type.status_code == 400


def test_different_passage_ids_not_comparable():
    """Verify that takes using different passages are not comparable."""
    take1 = make_dummy_analysis(passage_id="librivox_01")
    take2 = make_dummy_analysis(passage_id="librivox_02")

    res = compare_takes(take1, take2)
    assert res["comparable"] is False
    assert "The two takes use different passages." in res["reasons"]
    assert "dimensions" not in res

    # Same passage id is comparable
    take2_same = make_dummy_analysis(passage_id="librivox_01")
    res_same = compare_takes(take1, take2_same)
    assert res_same["comparable"] is True

    # One take has passage_id, other doesn't: does not trigger different passages error
    take2_no_pass = make_dummy_analysis(passage_id=None)
    res_one = compare_takes(take1, take2_no_pass)
    assert res_one["comparable"] is True


def test_transcript_similarity_rules():
    """Verify transcript comparison rules: different speeches reject, close match accepts, missing skips."""
    # 1. Two clearly different transcripts are not comparable (< 0.6 ratio)
    t1_diff = make_dummy_analysis(
        transcript="The Gettysburg address dedicated a portion of the field as a final resting place for the brave soldiers."
    )
    t2_diff = make_dummy_analysis(
        transcript="Quantum mechanics describes the physical properties of nature at the scale of atoms and subatomic particles."
    )
    res_diff = compare_takes(t1_diff, t2_diff)
    assert res_diff["comparable"] is False
    assert "The two takes look like different speeches, so a comparison would not be meaningful." in res_diff["reasons"]

    # 2. Same transcript with a few words changed is still comparable (ratio >= 0.6)
    t1_close = make_dummy_analysis(
        transcript="The quick brown fox jumps over the lazy dog in the green garden near the river."
    )
    t2_close = make_dummy_analysis(
        transcript="The fast brown fox leaps over the lazy dog in the green meadow near the river."
    )
    res_close = compare_takes(t1_close, t2_close)
    assert res_close["comparable"] is True

    # 3. Results without a transcript skip the transcript check
    t1_none = make_dummy_analysis(transcript=None)
    t2_none = make_dummy_analysis(transcript=None)
    res_none = compare_takes(t1_none, t2_none)
    assert res_none["comparable"] is True

    # One has transcript, other has None
    res_one_none = compare_takes(t1_close, t2_none)
    assert res_one_none["comparable"] is True


def test_unscored_rubric_dimensions_are_skipped():
    """Verify that rubric dimensions unscored in either take are skipped from comparison."""
    take1 = make_dummy_analysis(
        dimensions=[
            {"name": "Pace", "scored": True, "score": 80.0},
            {"name": "Pitch variation", "scored": True, "score": 90.0},
            {"name": "Pauses", "scored": False, "score": None},
            {"name": "Loudness", "scored": False, "score": None},
        ]
    )
    take2 = make_dummy_analysis(
        dimensions=[
            {"name": "Pace", "scored": False, "score": None},  # Unscored in take 2
            {"name": "Pitch variation", "scored": True, "score": 95.0},  # Scored in both
            {"name": "Pauses", "scored": True, "score": 85.0},  # Unscored in take 1
            {"name": "Loudness", "scored": False, "score": None},  # Unscored in both
        ]
    )

    res = compare_takes(take1, take2)
    assert res["comparable"] is True
    assert len(res["dimensions"]) == 1
    assert res["dimensions"][0]["name"] == "Pitch variation"
    assert res["dimensions"][0]["score_1"] == 90.0
    assert res["dimensions"][0]["score_2"] == 95.0
    assert res["dimensions"][0]["change"] == 5.0


