#!/usr/bin/env python3
"""Take comparison module for Laya.

Compares two speech analysis results (take 1 and take 2) and computes
deterministic rubric deltas, detector metrics, flag overlap tracking,
and transcript diffs for Indic languages.
"""

import difflib
from typing import Any, Dict, List, Optional

METHOD_NOTE = (
    "This compares two measurements from two recordings. A lower flagged time "
    "means the detectors flagged less in this take. It does not prove the delivery is better."
)


def regions_overlap_50pct(r1: Dict[str, Any], r2: Dict[str, Any]) -> bool:
    """Check if two regions of the same type overlap in time by >= 50% of the shorter region."""
    t1 = str(r1.get("type", "")).strip().lower()
    t2 = str(r2.get("type", "")).strip().lower()
    if t1 != t2:
        return False

    s1 = float(r1.get("start", 0.0))
    e1 = float(r1.get("end", 0.0))
    s2 = float(r2.get("start", 0.0))
    e2 = float(r2.get("end", 0.0))

    d1 = max(0.0, e1 - s1)
    d2 = max(0.0, e2 - s2)
    shorter = min(d1, d2)

    if shorter <= 0.0:
        return abs(s1 - s2) < 0.15

    overlap = max(0.0, min(e1, e2) - max(s1, s2))
    return (overlap / shorter) >= 0.50


def extract_passage_or_reference_id(take: Dict[str, Any]) -> Optional[str]:
    """Extract reference or practice passage identifier if present."""
    for key in ("passage_id", "reference_passage", "reference_id", "reference_file", "reference"):
        val = take.get(key)
        if val is not None and str(val).strip():
            p = str(val).strip().replace("\\", "/")
            name = p.split("/")[-1]
            if "." in name:
                name = name.rsplit(".", 1)[0]
            return name.lower()
    return None


def compare_takes(take1: Dict[str, Any], take2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two speech analysis result objects and return their comparative metrics.

    Parameters:
        take1: First analysis result dictionary.
        take2: Second analysis result dictionary.

    Returns:
        Dictionary containing comparability flag, reasons, rubric deltas,
        detector flag counts/seconds, region overlap tracking, transcript diffs,
        and method note.
    """
    reasons: List[str] = []

    # 1. Language comparison
    lang1 = str(take1.get("language_mode") or take1.get("language") or "english").strip().lower()
    lang2 = str(take2.get("language_mode") or take2.get("language") or "english").strip().lower()
    if lang1 != lang2:
        reasons.append("Language differs between take 1 and take 2.")

    # 2. Mode comparison (paired vs experimental)
    mode1 = str(take1.get("mode", "")).strip().lower()
    mode2 = str(take2.get("mode", "")).strip().lower()
    if mode1 != mode2:
        reasons.append("Analysis mode differs between take 1 and take 2.")

    # 3. Passage / reference identifier comparison (must match if both carry one)
    pass1 = extract_passage_or_reference_id(take1)
    pass2 = extract_passage_or_reference_id(take2)
    if pass1 is not None and pass2 is not None and pass1 != pass2:
        reasons.append("The two takes use different passages.")

    # 4. Duration comparison (> 20 percent difference)
    dur1 = float(take1.get("duration", 0.0))
    dur2 = float(take2.get("duration", 0.0))
    if dur1 <= 0.0 or dur2 <= 0.0:
        reasons.append("Duration must be greater than zero.")
    else:
        dur_diff_pct = abs(dur2 - dur1) / dur1
        if dur_diff_pct > 0.20:
            reasons.append("Duration differs by more than 20 percent between take 1 and take 2.")

    # 5. Transcript similarity comparison (if both results have a transcript)
    t1_raw = take1.get("transcript")
    t2_raw = take2.get("transcript")
    if t1_raw is not None and t2_raw is not None and str(t1_raw).strip() and str(t2_raw).strip():
        w1 = str(t1_raw).strip().split()
        w2 = str(t2_raw).strip().split()
        if w1 and w2:
            ratio = difflib.SequenceMatcher(None, w1, w2).ratio()
            if ratio < 0.6:
                reasons.append("The two takes look like different speeches, so a comparison would not be meaningful.")

    if reasons:
        return {
            "comparable": False,
            "reasons": reasons,
            "reason": " ".join(reasons),
        }

    # 4. Rubric dimensions scored in BOTH takes
    r1 = take1.get("rubric", {}) or {}
    r2 = take2.get("rubric", {}) or {}

    dims1 = {d.get("name"): d for d in r1.get("dimensions", []) if isinstance(d, dict)}
    dims2 = {d.get("name"): d for d in r2.get("dimensions", []) if isinstance(d, dict)}

    dimensions: List[Dict[str, Any]] = []
    standard_order = ["Pace", "Pitch variation", "Pauses", "Loudness"]
    all_names = [n for n in standard_order if n in dims1 or n in dims2]
    for n in dims1:
        if n not in all_names:
            all_names.append(n)

    for name in all_names:
        d1 = dims1.get(name)
        d2 = dims2.get(name)
        if not d1 or not d2:
            continue
        if d1.get("scored") and d2.get("scored") and d1.get("score") is not None and d2.get("score") is not None:
            s1 = round(float(d1["score"]), 1)
            s2 = round(float(d2["score"]), 1)
            ch = round(s2 - s1, 1)
            dimensions.append({
                "name": name,
                "score_1": s1,
                "score_2": s2,
                "change": ch,
                "signed_change": f"{ch:+.1f}",
            })

    # Overall comparison (only when both are not null)
    overall_1_raw = r1.get("overall")
    overall_2_raw = r2.get("overall")
    overall_dict: Dict[str, Any] = {
        "overall_1": None,
        "overall_2": None,
        "overall_change": None,
        "signed_change": None,
    }

    if overall_1_raw is not None and overall_2_raw is not None:
        o1 = round(float(overall_1_raw), 1)
        o2 = round(float(overall_2_raw), 1)
        och = round(o2 - o1, 1)
        overall_dict = {
            "overall_1": o1,
            "overall_2": o2,
            "overall_change": och,
            "signed_change": f"{och:+.1f}",
        }
    elif overall_1_raw is not None:
        overall_dict["overall_1"] = round(float(overall_1_raw), 1)
    elif overall_2_raw is not None:
        overall_dict["overall_2"] = round(float(overall_2_raw), 1)

    # 5. Per detector (rate, pitch, pause): flagged count and seconds in each take
    detectors: Dict[str, Dict[str, Any]] = {}
    regions1 = take1.get("flaw_regions", []) or []
    regions2 = take2.get("flaw_regions", []) or []

    for det in ["rate", "pitch", "pause"]:
        det_regs1 = [r for r in regions1 if str(r.get("type", "")).strip().lower() == det]
        det_regs2 = [r for r in regions2 if str(r.get("type", "")).strip().lower() == det]

        count_1 = len(det_regs1)
        count_2 = len(det_regs2)

        secs_1 = round(sum(max(0.0, float(r.get("end", 0.0)) - float(r.get("start", 0.0))) for r in det_regs1), 2)
        secs_2 = round(sum(max(0.0, float(r.get("end", 0.0)) - float(r.get("start", 0.0))) for r in det_regs2), 2)

        detectors[det] = {
            "count_1": count_1,
            "count_2": count_2,
            "seconds_1": secs_1,
            "seconds_2": secs_2,
        }

    # 6. Flag overlap tracking
    # For each region in take 1: "still flagged" or "not flagged in take 2"
    take_1_flags: List[Dict[str, Any]] = []
    for r1_item in regions1:
        still = any(regions_overlap_50pct(r1_item, r2_item) for r2_item in regions2)
        status = "still flagged" if still else "not flagged in take 2"
        take_1_flags.append({
            "type": r1_item.get("type"),
            "start": round(float(r1_item.get("start", 0.0)), 2),
            "end": round(float(r1_item.get("end", 0.0)), 2),
            "status": status,
        })

    # New flags in take 2 that overlap nothing in take 1: "new in take 2"
    take_2_new_flags: List[Dict[str, Any]] = []
    for r2_item in regions2:
        overlaps_any = any(regions_overlap_50pct(r1_item, r2_item) for r1_item in regions1)
        if not overlaps_any:
            take_2_new_flags.append({
                "type": r2_item.get("type"),
                "start": round(float(r2_item.get("start", 0.0)), 2),
                "end": round(float(r2_item.get("end", 0.0)), 2),
                "status": "new in take 2",
            })

    # 7. Indic transcript diff (Hindi and Hinglish)
    transcripts_diff: Optional[Dict[str, Any]] = None
    if lang1 in ("hindi", "hinglish"):
        t1 = str(take1.get("transcript", "") or "")
        t2 = str(take2.get("transcript", "") or "")
        w1 = t1.strip().split()
        w2 = t2.strip().split()
        matcher = difflib.SequenceMatcher(None, w1, w2)
        differ_count = sum(
            max(i2 - i1, j2 - j1)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes()
            if tag != "equal"
        )
        transcripts_diff = {
            "transcript_1": t1,
            "transcript_2": t2,
            "words_differ": differ_count,
            "label": f"{differ_count} words differ",
        }

    return {
        "comparable": True,
        "method_note": METHOD_NOTE,
        "dimensions": dimensions,
        "overall": overall_dict,
        "detectors": detectors,
        "take_1_flags": take_1_flags,
        "take_2_new_flags": take_2_new_flags,
        "transcripts": transcripts_diff,
    }

