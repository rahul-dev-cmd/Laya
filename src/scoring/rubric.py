"""Rubric scoring module for Laya delivery analysis.

Computes normalized delivery scores for pace, pitch variation, pauses,
and loudness, along with an overall delivery score.

No long dashes in any text or comments.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.scoring.config import (
    CRITERION_LOUDNESS,
    CRITERION_PACE,
    CRITERION_PAUSES,
    CRITERION_PITCH,
    METHOD_FORMULA_TEXT,
    METHOD_NOTE,
    MIN_SCORED_DIMENSIONS,
    REASON_LOUDNESS_UNSCORED,
    REASON_PAUSES_UNSCORED,
    RELIABILITY_NOTE_EXPERIMENTAL,
    RELIABILITY_NOTE_PAIRED,
    RUBRIC_VERSION,
)


def merge_and_clip_regions(
    regions: List[Dict[str, Any]],
    duration: float,
) -> Tuple[List[Tuple[float, float]], float]:
    """Clip regions to [0.0, duration] and merge overlapping intervals.

    Returns:
        Tuple of (merged_intervals, total_flagged_seconds).
    """
    if duration <= 0:
        return [], 0.0

    clipped: List[Tuple[float, float]] = []
    for r in regions:
        s = float(r.get("start", 0.0))
        e = float(r.get("end", 0.0))
        clipped_s = max(0.0, min(duration, s))
        clipped_e = max(0.0, min(duration, e))
        if clipped_e > clipped_s:
            clipped.append((clipped_s, clipped_e))

    if not clipped:
        return [], 0.0

    # Sort intervals chronologically by start timestamp
    clipped.sort(key=lambda iv: (iv[0], iv[1]))

    merged: List[List[float]] = []
    for s, e in clipped:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    total_seconds = sum(iv[1] - iv[0] for iv in merged)
    return [(iv[0], iv[1]) for iv in merged], total_seconds


def find_worst_deviation(
    regions: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[str]]:
    """Find the peak absolute deviation and corresponding severity among regions."""
    if not regions:
        return None, None

    worst_reg: Optional[Dict[str, Any]] = None
    max_abs_z = -1.0

    for r in regions:
        z_val = r.get("peak_z")
        if z_val is None:
            z_val = r.get("z")
        if z_val is not None:
            abs_z = abs(float(z_val))
            if abs_z > max_abs_z:
                max_abs_z = abs_z
                worst_reg = r

    if worst_reg is None:
        return None, None

    raw_z = worst_reg.get("peak_z")
    if raw_z is None:
        raw_z = worst_reg.get("z")

    worst_z = round(float(raw_z), 2) if raw_z is not None else None
    worst_severity = worst_reg.get("severity")
    return worst_z, worst_severity


def score_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    """Compute rubric score across delivery dimensions for an analysis result.

    Args:
        result: Analysis result dictionary containing:
            - duration: total audio duration in seconds
            - flaw_regions: list of detected flaw region dictionaries
            - mode: analysis mode ("paired" or "experimental")
            - pauses_examined: optional count of evaluated pause boundaries

    Returns:
        Dictionary conforming to Laya rubric schema:
            - version: rubric configuration version
            - method: plain text formula description
            - method_note: explicit caveat on score interpretation
            - mode: analysis mode
            - reliability_note: baseline comparison context note
            - dimensions: list of dimension score objects
            - overall: unweighted mean of scored dimensions, or null
    """
    duration = float(result.get("duration", 0.0))
    flaw_regions = result.get("flaw_regions", []) or []
    mode = result.get("mode", "experimental")

    if mode == "paired":
        reliability_note = RELIABILITY_NOTE_PAIRED
    else:
        reliability_note = RELIABILITY_NOTE_EXPERIMENTAL

    # 1. Pace (rate flaw regions)
    rate_regions = [r for r in flaw_regions if r.get("type") == "rate"]
    _, rate_flagged_sec = merge_and_clip_regions(rate_regions, duration)
    rate_worst_z, rate_worst_sev = find_worst_deviation(rate_regions)

    if duration > 0:
        rate_score = 100.0 * (1.0 - (rate_flagged_sec / duration))
        rate_score = max(0.0, min(100.0, rate_score))
        rate_score = round(rate_score, 1)
    else:
        rate_score = 100.0 if not rate_regions else 0.0

    pace_dim = {
        "name": "Pace",
        "score": rate_score,
        "scored": True,
        "measured": True,
        "reason": None,
        "region_count": len(rate_regions),
        "flagged_seconds": round(rate_flagged_sec, 1),
        "worst_z": rate_worst_z,
        "worst_severity": rate_worst_sev,
        "criterion": CRITERION_PACE,
    }

    # 2. Pitch variation (pitch flaw regions)
    pitch_regions = [r for r in flaw_regions if r.get("type") == "pitch"]
    _, pitch_flagged_sec = merge_and_clip_regions(pitch_regions, duration)
    pitch_worst_z, pitch_worst_sev = find_worst_deviation(pitch_regions)

    if duration > 0:
        pitch_score = 100.0 * (1.0 - (pitch_flagged_sec / duration))
        pitch_score = max(0.0, min(100.0, pitch_score))
        pitch_score = round(pitch_score, 1)
    else:
        pitch_score = 100.0 if not pitch_regions else 0.0

    pitch_dim = {
        "name": "Pitch variation",
        "score": pitch_score,
        "scored": True,
        "measured": True,
        "reason": None,
        "region_count": len(pitch_regions),
        "flagged_seconds": round(pitch_flagged_sec, 1),
        "worst_z": pitch_worst_z,
        "worst_severity": pitch_worst_sev,
        "criterion": CRITERION_PITCH,
    }

    # 3. Pauses (pause flags)
    pause_regions = [r for r in flaw_regions if r.get("type") == "pause"]
    flagged_pauses = len(pause_regions)
    pause_worst_z, pause_worst_sev = find_worst_deviation(pause_regions)

    pauses_examined = result.get("pauses_examined")
    if isinstance(pauses_examined, (int, float)) and pauses_examined > 0:
        p_examined_int = int(pauses_examined)
        p_score = 100.0 * (1.0 - (flagged_pauses / p_examined_int))
        p_score = max(0.0, min(100.0, p_score))
        p_score = round(p_score, 1)
        pauses_dim = {
            "name": "Pauses",
            "score": p_score,
            "scored": True,
            "measured": True,
            "reason": None,
            "region_count": flagged_pauses,
            "flagged_pauses": flagged_pauses,
            "pauses_examined": p_examined_int,
            "worst_z": pause_worst_z,
            "worst_severity": pause_worst_sev,
            "criterion": CRITERION_PAUSES,
        }
    else:
        pauses_dim = {
            "name": "Pauses",
            "score": None,
            "scored": False,
            "measured": True,
            "reason": REASON_PAUSES_UNSCORED,
            "region_count": flagged_pauses,
            "flagged_pauses": flagged_pauses,
            "pauses_examined": None,
            "worst_z": pause_worst_z,
            "worst_severity": pause_worst_sev,
            "criterion": CRITERION_PAUSES,
        }

    # 4. Loudness (unscored yet)
    loudness_regions = [
        r for r in flaw_regions if r.get("type") in ("loudness", "energy")
    ]
    loud_worst_z, loud_worst_sev = find_worst_deviation(loudness_regions)
    loudness_dim = {
        "name": "Loudness",
        "score": None,
        "scored": False,
        "measured": False,
        "reason": REASON_LOUDNESS_UNSCORED,
        "region_count": len(loudness_regions),
        "flagged_seconds": 0.0,
        "worst_z": loud_worst_z,
        "worst_severity": loud_worst_sev,
        "criterion": CRITERION_LOUDNESS,
    }

    # Dimension grouping
    dimensions = [pace_dim, pitch_dim, pauses_dim, loudness_dim]

    # Overall score: plain average of scored dimensions
    scored_dims = [
        d for d in dimensions if d.get("scored") and d.get("score") is not None
    ]
    if len(scored_dims) >= MIN_SCORED_DIMENSIONS:
        overall_score: Optional[float] = round(
            sum(d["score"] for d in scored_dims) / len(scored_dims), 1
        )
    else:
        overall_score = None

    return {
        "version": RUBRIC_VERSION,
        "method": METHOD_FORMULA_TEXT,
        "method_note": METHOD_NOTE,
        "mode": mode,
        "reliability_note": reliability_note,
        "dimensions": dimensions,
        "overall": overall_score,
    }
