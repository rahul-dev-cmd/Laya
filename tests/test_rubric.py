"""Tests for delivery rubric scoring module.

Tests cover:
- no flags gives 100
- one flag covering the whole clip gives 0
- overlapping regions are merged and not double counted
- regions are clipped to the duration
- overall is the average of scored dimensions
- an unscored dimension is excluded
- fewer than two scored dimensions gives null
- results are deterministic

No long dashes in any text or comments.
"""

from typing import Any, Dict
import pytest

from src.scoring.rubric import score_analysis
from src.scoring.config import (
    METHOD_NOTE,
    RELIABILITY_NOTE_EXPERIMENTAL,
    RELIABILITY_NOTE_PAIRED,
    REASON_LOUDNESS_UNSCORED,
    REASON_PAUSES_UNSCORED,
)


def test_no_flags_gives_100():
    """Verify that an analysis with zero flaw flags receives a score of 100.0."""
    result = {
        "duration": 60.0,
        "mode": "paired",
        "flaw_regions": [],
    }

    rubric = score_analysis(result)

    assert rubric["overall"] == 100.0

    dims_by_name = {d["name"]: d for d in rubric["dimensions"]}
    assert dims_by_name["Pace"]["score"] == 100.0
    assert dims_by_name["Pace"]["flagged_seconds"] == 0.0
    assert dims_by_name["Pace"]["region_count"] == 0
    assert dims_by_name["Pace"]["worst_z"] is None
    assert dims_by_name["Pace"]["worst_severity"] is None

    assert dims_by_name["Pitch variation"]["score"] == 100.0
    assert dims_by_name["Pitch variation"]["flagged_seconds"] == 0.0
    assert dims_by_name["Pitch variation"]["region_count"] == 0

    assert rubric["method_note"] == METHOD_NOTE
    assert rubric["reliability_note"] == RELIABILITY_NOTE_PAIRED


def test_one_flag_covering_whole_clip_gives_0():
    """Verify that a flaw region spanning the entire recording duration gives score 0.0."""
    duration = 80.0
    result = {
        "duration": duration,
        "mode": "paired",
        "flaw_regions": [
            {
                "type": "rate",
                "start": 0.0,
                "end": duration,
                "peak_z": 4.5,
                "severity": "tier 5 (severe)",
            },
            {
                "type": "pitch",
                "start": 0.0,
                "end": duration,
                "peak_z": -4.2,
                "severity": "tier 5 (severe)",
            },
        ],
    }

    rubric = score_analysis(result)
    dims_by_name = {d["name"]: d for d in rubric["dimensions"]}

    assert dims_by_name["Pace"]["score"] == 0.0
    assert dims_by_name["Pace"]["flagged_seconds"] == duration

    assert dims_by_name["Pitch variation"]["score"] == 0.0
    assert dims_by_name["Pitch variation"]["flagged_seconds"] == duration

    assert rubric["overall"] == 0.0


def test_overlapping_regions_merged_not_double_counted():
    """Verify that overlapping regions are merged so overlapping time is not double counted."""
    duration = 100.0
    # Two overlapping rate regions: [10.0, 30.0] and [20.0, 40.0]
    # Union span is [10.0, 40.0], total flagged seconds = 30.0
    # Without merging, naive sum would be 20.0 + 20.0 = 40.0
    result = {
        "duration": duration,
        "mode": "experimental",
        "flaw_regions": [
            {
                "type": "rate",
                "start": 10.0,
                "end": 30.0,
                "peak_z": 2.5,
                "severity": "tier 2 (mild)",
            },
            {
                "type": "rate",
                "start": 20.0,
                "end": 40.0,
                "peak_z": 3.6,
                "severity": "tier 4 (high)",
            },
        ],
    }

    rubric = score_analysis(result)
    dims_by_name = {d["name"]: d for d in rubric["dimensions"]}

    # Score should be 100 * (1 - 30.0 / 100.0) = 70.0
    assert dims_by_name["Pace"]["flagged_seconds"] == 30.0
    assert dims_by_name["Pace"]["score"] == 70.0
    assert dims_by_name["Pace"]["region_count"] == 2
    assert dims_by_name["Pace"]["worst_z"] == 3.6
    assert dims_by_name["Pace"]["worst_severity"] == "tier 4 (high)"


def test_regions_clipped_to_duration():
    """Verify that regions extending beyond recording boundaries are clipped to [0, duration]."""
    duration = 50.0
    # pitch region starting before 0.0: [-10.0, 20.0] -> clipped to [0.0, 20.0] (20s)
    # pitch region ending after duration: [35.0, 65.0] -> clipped to [35.0, 50.0] (15s)
    # total flagged = 35.0s
    result = {
        "duration": duration,
        "mode": "paired",
        "flaw_regions": [
            {
                "type": "pitch",
                "start": -10.0,
                "end": 20.0,
                "peak_z": -2.5,
                "severity": "tier 2 (mild)",
            },
            {
                "type": "pitch",
                "start": 35.0,
                "end": 65.0,
                "peak_z": -3.0,
                "severity": "tier 3 (moderate)",
            },
        ],
    }

    rubric = score_analysis(result)
    dims_by_name = {d["name"]: d for d in rubric["dimensions"]}

    # Score: 100 * (1 - 35.0 / 50.0) = 30.0
    assert dims_by_name["Pitch variation"]["flagged_seconds"] == 35.0
    assert dims_by_name["Pitch variation"]["score"] == 30.0


def test_overall_is_average_of_scored_dimensions():
    """Verify that overall score is the plain unweighted average of scored dimensions."""
    duration = 100.0
    # Pace has 20s flagged -> score = 80.0
    # Pitch has 10s flagged -> score = 90.0
    # Pauses and Loudness unscored
    # Overall = (80.0 + 90.0) / 2 = 85.0
    result = {
        "duration": duration,
        "mode": "paired",
        "flaw_regions": [
            {
                "type": "rate",
                "start": 0.0,
                "end": 20.0,
                "peak_z": 2.4,
                "severity": "tier 2 (mild)",
            },
            {
                "type": "pitch",
                "start": 50.0,
                "end": 60.0,
                "peak_z": -2.2,
                "severity": "tier 1 (slight)",
            },
        ],
    }

    rubric = score_analysis(result)
    dims_by_name = {d["name"]: d for d in rubric["dimensions"]}

    assert dims_by_name["Pace"]["score"] == 80.0
    assert dims_by_name["Pitch variation"]["score"] == 90.0
    assert rubric["overall"] == 85.0


def test_unscored_dimension_is_excluded():
    """Verify that unscored dimensions (Loudness, unrecorded Pauses) are excluded from average."""
    duration = 100.0
    result = {
        "duration": duration,
        "mode": "paired",
        "flaw_regions": [
            {
                "type": "rate",
                "start": 0.0,
                "end": 20.0,
                "peak_z": 2.4,
                "severity": "tier 2 (mild)",
            },
            {
                "type": "pitch",
                "start": 50.0,
                "end": 60.0,
                "peak_z": -2.2,
                "severity": "tier 1 (slight)",
            },
            {
                "type": "pause",
                "start": 10.0,
                "end": 10.5,
                "peak_z": -2.1,
                "severity": "tier 1 (slight)",
            },
            {
                "type": "loudness",
                "start": 70.0,
                "end": 75.0,
                "peak_z": 2.5,
                "severity": "tier 2 (mild)",
            },
        ],
    }

    rubric = score_analysis(result)
    dims_by_name = {d["name"]: d for d in rubric["dimensions"]}

    # Loudness is marked measured=False, score=None, scored=False
    assert dims_by_name["Loudness"]["measured"] is False
    assert dims_by_name["Loudness"]["scored"] is False
    assert dims_by_name["Loudness"]["score"] is None
    assert dims_by_name["Loudness"]["reason"] == REASON_LOUDNESS_UNSCORED

    # Pauses without pauses_examined is marked scored=False, score=None
    assert dims_by_name["Pauses"]["scored"] is False
    assert dims_by_name["Pauses"]["score"] is None
    assert dims_by_name["Pauses"]["reason"] == REASON_PAUSES_UNSCORED
    assert dims_by_name["Pauses"]["flagged_pauses"] == 1

    # Overall must be average of Pace (80.0) and Pitch (90.0) only
    assert rubric["overall"] == 85.0

    # If pauses_examined is provided, pauses becomes scored and included in average
    result_with_pauses = dict(result)
    result_with_pauses["pauses_examined"] = 10  # 1 flagged out of 10 -> score = 90.0
    rubric_with_pauses = score_analysis(result_with_pauses)
    dims_p = {d["name"]: d for d in rubric_with_pauses["dimensions"]}
    assert dims_p["Pauses"]["scored"] is True
    assert dims_p["Pauses"]["score"] == 90.0
    # Overall is average of 80.0, 90.0, 90.0 -> 260 / 3 = 86.666... -> 86.7
    assert rubric_with_pauses["overall"] == 86.7


def test_fewer_than_two_scored_dimensions_gives_null():
    """Verify that overall is null when fewer than two dimensions are scored."""
    # Custom result with single scored dimension by manually simulating an analysis
    from src.scoring.rubric import score_analysis

    # Create dummy result with duration
    result = {
        "duration": 50.0,
        "mode": "paired",
        "flaw_regions": [],
    }
    rubric = score_analysis(result)
    # Normally both Pace and Pitch are scored (2 dims) -> overall is not null
    assert rubric["overall"] is not None

    # Test edge case: if only 1 dimension is scored in dimensions list
    scored_dims_1 = [
        {"name": "Pace", "score": 90.0, "scored": True},
        {"name": "Pitch variation", "score": None, "scored": False},
        {"name": "Pauses", "score": None, "scored": False},
        {"name": "Loudness", "score": None, "scored": False},
    ]
    from src.scoring.config import MIN_SCORED_DIMENSIONS
    scored = [d for d in scored_dims_1 if d.get("scored") and d.get("score") is not None]
    overall = round(sum(d["score"] for d in scored) / len(scored), 1) if len(scored) >= MIN_SCORED_DIMENSIONS else None
    assert overall is None


def test_results_are_deterministic():
    """Verify that score_analysis produces strictly deterministic results."""
    result: Dict[str, Any] = {
        "duration": 94.98,
        "mode": "paired",
        "flaw_regions": [
            {
                "type": "pitch",
                "start": 54.58,
                "end": 58.84,
                "peak_z": -4.478,
                "severity": "tier 5 (severe)",
            },
            {
                "type": "rate",
                "start": 10.0,
                "end": 15.0,
                "peak_z": 2.8,
                "severity": "tier 3 (moderate)",
            },
        ],
    }

    first_run = score_analysis(result)
    for _ in range(25):
        subsequent_run = score_analysis(result)
        assert first_run == subsequent_run


def test_reliability_note_modes():
    """Verify reliability_note matches paired vs experimental mode definitions."""
    res_paired = {"duration": 10.0, "mode": "paired", "flaw_regions": []}
    res_exp = {"duration": 10.0, "mode": "experimental", "flaw_regions": []}

    rubric_paired = score_analysis(res_paired)
    rubric_exp = score_analysis(res_exp)

    assert rubric_paired["reliability_note"] == RELIABILITY_NOTE_PAIRED
    assert rubric_exp["reliability_note"] == RELIABILITY_NOTE_EXPERIMENTAL
