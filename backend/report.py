#!/usr/bin/env python3
"""AI explanation layer and template fallback generator for Laya.

This module provides plain-language explanations of speech measurement results.
It communicates with any OpenAI-compatible LLM endpoint using credentials from .env,
sends only measurement numbers (never audio or word lists), validates all numbers in
the reply against the ground-truth data, and falls back to template explanations
if validation fails, the service times out, or credentials are absent.
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tips import get_practice_tips_for_regions

REPORTS_DIR = PROJECT_ROOT / "dataset" / "examples" / "reports"
DISK_CACHE_DIR = PROJECT_ROOT / "scratch" / "report_cache"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory report cache
_REPORT_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}

router = APIRouter()


def load_env_vars() -> Dict[str, str]:
    """Load LLM configuration variables from .env files without logging keys."""
    env_paths = [
        PROJECT_ROOT / "backend" / ".env",
        PROJECT_ROOT / ".env",
    ]
    loaded: Dict[str, str] = {}
    for p in env_paths:
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in loaded:
                        loaded[k] = v

    # Fallback to os.environ
    for k in ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"]:
        if k in os.environ and k not in loaded:
            loaded[k] = os.environ[k].strip().strip("'\"")

    return loaded


def compute_cache_key(analysis_data: Dict[str, Any]) -> str:
    """Generate a deterministic SHA-256 cache key from analysis measurements."""
    canonical = {
        "duration": round(float(analysis_data.get("duration", 0)), 2),
        "mode": analysis_data.get("mode"),
        "transcript_source": analysis_data.get("transcript_source"),
        "flaw_regions": [
            {
                "type": r.get("type"),
                "start": round(float(r.get("start", 0)), 3),
                "end": round(float(r.get("end", 0)), 3),
                "z": round(float(r.get("z", 0)), 3),
                "severity": r.get("severity"),
                "observed_value": r.get("observed_value"),
                "unit": r.get("unit"),
                "baseline_mean": r.get("baseline_mean"),
            }
            for r in analysis_data.get("flaw_regions", [])
        ],
    }
    raw = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_allowed_numbers(analysis_data: Dict[str, Any]) -> Dict[float, str]:
    """Map every valid number in the analysis to its originating data field."""
    allowed: Dict[float, str] = {}

    def register(num_val: Any, field_name: str) -> None:
        if num_val is None:
            return
        try:
            f = float(num_val)
            # Store standard roundings and raw value
            allowed[f] = field_name
            allowed[abs(f)] = f"abs({field_name})"
            allowed[round(f, 3)] = f"{field_name} (3-dec)"
            allowed[round(f, 2)] = f"{field_name} (2-dec)"
            allowed[round(f, 1)] = f"{field_name} (1-dec)"
            allowed[float(round(f))] = f"{field_name} (integer round)"
            allowed[float(int(f))] = f"{field_name} (integer truncated)"
        except (ValueError, TypeError):
            pass

    # Overall duration and region count
    dur = analysis_data.get("duration")
    if dur is not None:
        register(dur, "duration")

    regions = analysis_data.get("flaw_regions", [])
    register(len(regions), "region_count")
    register(3.0, "tip_count_standard")
    register(1.0, "index_1")
    register(2.0, "index_2")

    for i, r in enumerate(regions):
        prefix = f"flaw_regions[{i}]"
        register(r.get("start"), f"{prefix}.start")
        register(r.get("end"), f"{prefix}.end")
        if r.get("start") is not None and r.get("end") is not None:
            register(r["end"] - r["start"], f"{prefix}.duration")
        register(r.get("z"), f"{prefix}.z")
        register(r.get("peak_z"), f"{prefix}.peak_z")
        register(r.get("observed_value"), f"{prefix}.observed_value")
        register(r.get("baseline_mean"), f"{prefix}.baseline_mean")
        register(r.get("baseline_spread"), f"{prefix}.baseline_spread")
        register(r.get("threshold"), f"{prefix}.threshold")
        register(r.get("reference_value"), f"{prefix}.reference_value")
        register(r.get("peak_window_start"), f"{prefix}.peak_window_start")
        register(r.get("peak_window_end"), f"{prefix}.peak_window_end")

        # Parse tier number if present (e.g. tier 5 -> 5)
        sev = str(r.get("severity", ""))
        tier_match = re.search(r"tier\s*(\d+)", sev, re.IGNORECASE)
        if tier_match:
            register(float(tier_match.group(1)), f"{prefix}.tier")

        # Parse numbers already embedded in template explanation string
        expl = str(r.get("explanation", ""))
        for token in re.findall(r"[-+]?\b\d+(?:\.\d+)?\b", expl):
            register(token, f"{prefix}.explanation_token")

    return allowed


def find_matching_allowed_field(cand: float, allowed_map: Dict[float, str], tolerance: float = 0.06) -> Optional[Tuple[float, str]]:
    """Check if a candidate number matches any allowed number within rounding tolerance."""
    if cand in allowed_map:
        return cand, allowed_map[cand]
    if abs(cand) in allowed_map:
        return abs(cand), allowed_map[abs(cand)]

    for ref_num, origin in allowed_map.items():
        if abs(cand - ref_num) <= tolerance:
            return ref_num, origin
        if abs(abs(cand) - abs(ref_num)) <= tolerance:
            return ref_num, origin
        if round(ref_num) == cand:
            return ref_num, origin

    return None


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract all numeric tokens from text."""
    # Matches integers and decimals with optional signs, handling unit suffixes like 54.58s
    tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    nums = []
    for t in tokens:
        try:
            nums.append(float(t))
        except ValueError:
            continue
    return nums


def validate_report_numbers(report_data: Dict[str, Any], allowed_map: Dict[float, str]) -> Tuple[bool, List[str], Dict[float, str]]:
    """Verify that every number in the report maps back to an input measurement.

    Returns:
        (is_valid, list_of_hallucinated_numbers, matched_numbers_map)
    """
    matched: Dict[float, str] = {}
    hallucinated: List[str] = []

    # Collect all text strings from report
    texts: List[str] = []
    if isinstance(report_data.get("summary"), str):
        texts.append(report_data["summary"])
    if isinstance(report_data.get("what_happened"), list):
        texts.extend([str(item) for item in report_data["what_happened"]])
    elif isinstance(report_data.get("what_happened"), str):
        texts.append(report_data["what_happened"])
    if isinstance(report_data.get("why_it_matters"), str):
        texts.append(report_data["why_it_matters"])
    if isinstance(report_data.get("caveats"), str):
        texts.append(report_data["caveats"])

    practice_tips = report_data.get("practice_tips")
    if isinstance(practice_tips, dict):
        for tips_list in practice_tips.values():
            if isinstance(tips_list, list):
                texts.extend([str(t) for t in tips_list])
    elif isinstance(practice_tips, list):
        texts.extend([str(t) for t in practice_tips])

    combined_text = " ".join(texts)
    extracted = extract_numbers_from_text(combined_text)

    for num in extracted:
        match = find_matching_allowed_field(num, allowed_map)
        if match is not None:
            ref_val, origin = match
            matched[num] = origin
        else:
            hallucinated.append(str(num))

    is_valid = len(hallucinated) == 0
    return is_valid, hallucinated, matched


def compute_region_meaning(flaw_type: str, z: float) -> str:
    """Compute plain-language meaning of a flagged region based on type and sign of z."""
    ft = (flaw_type or "").lower()
    if ft == "pitch":
        if z < 0:
            return (
                "the pitch variation was too small, so the delivery was flat or monotone; "
                "observed_value is the standard deviation of pitch in semitones inside the window "
                "(how much the pitch moved), not the pitch level."
            )
        else:
            return "the pitch moved far more than the baseline."
    elif ft == "rate":
        if z > 0:
            return "speech was faster than the baseline."
        else:
            return "speech was slower than the baseline."
    elif ft == "pause":
        if z < 0:
            return "the pause between two words was shorter than normal."
        else:
            return "the pause between two words was longer than normal."
    return ""


def build_caveats_text(mode: Optional[str], transcript_source: Optional[str]) -> str:
    """Build standardized plain-language caveats text."""
    if mode == "paired":
        base_text = "Compared with an ideal recording of the same speech"
    else:
        base_text = "Compared with a general baseline, so less reliable"

    if transcript_source == "auto":
        return f"{base_text}. Note: the transcript was generated automatically."
    return base_text


def check_meaning_contradictions(
    report_data: Dict[str, Any],
    flaw_regions: List[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Check whether report text contradicts the delivery finding.

    Returns (is_valid, contradiction_reason).
    """
    texts: List[str] = []
    if isinstance(report_data.get("summary"), str):
        texts.append(report_data["summary"])
    if isinstance(report_data.get("what_happened"), list):
        texts.extend([str(item) for item in report_data["what_happened"]])
    elif isinstance(report_data.get("what_happened"), str):
        texts.append(report_data["what_happened"])
    if isinstance(report_data.get("why_it_matters"), str):
        texts.append(report_data["why_it_matters"])

    combined_text = " ".join(texts).lower()

    for r in flaw_regions:
        flaw_type = str(r.get("type", "")).lower()
        z_val = float(r.get("z", r.get("peak_z", 0.0)))

        if flaw_type == "pitch" and z_val < 0:
            # For flat pitch: reject drop, steady pitch, smooth, reduce variation
            flat_patterns = [
                (r"\bdrop[a-z]*\b", "drop"),
                (r"\bsteady\s+pitch\b", "steady pitch"),
                (r"\bsmooth[a-z]*\b", "smooth"),
                (r"\breduc[a-z]*\s+variation\b", "reduce variation"),
                (r"\bwent\s+down\b", "went down"),
                (r"\bfall(?:ing|en)?\s+pitch\b", "falling pitch"),
            ]
            for pat, name in flat_patterns:
                if re.search(pat, combined_text):
                    return False, f"Contradiction for flat pitch finding: detected '{name}'"

        elif flaw_type == "pitch" and z_val > 0:
            over_patterns = [
                (r"\bmonotone\b", "monotone"),
                (r"\bflat\s+pitch\b", "flat pitch"),
                (r"\black\s+of\s+variation\b", "lack of variation"),
            ]
            for pat, name in over_patterns:
                if re.search(pat, combined_text):
                    return False, f"Contradiction for over-varied pitch finding: detected '{name}'"

        elif flaw_type == "rate" and z_val > 0:
            rate_fast_patterns = [
                (r"\btoo\s+slow\b", "too slow"),
                (r"\bsluggish\b", "sluggish"),
                (r"\bdragging\b", "dragging"),
            ]
            for pat, name in rate_fast_patterns:
                if re.search(pat, combined_text):
                    return False, f"Contradiction for fast rate finding: detected '{name}'"

        elif flaw_type == "rate" and z_val < 0:
            rate_slow_patterns = [
                (r"\btoo\s+fast\b", "too fast"),
                (r"\brushed\b", "rushed"),
                (r"\bhurried\b", "hurried"),
            ]
            for pat, name in rate_slow_patterns:
                if re.search(pat, combined_text):
                    return False, f"Contradiction for slow rate finding: detected '{name}'"

        elif flaw_type == "pause" and z_val < 0:
            pause_short_patterns = [
                (r"\btoo\s+long\b", "too long"),
                (r"\blingering\b", "lingering"),
            ]
            for pat, name in pause_short_patterns:
                if re.search(pat, combined_text):
                    return False, f"Contradiction for short pause finding: detected '{name}'"

        elif flaw_type == "pause" and z_val > 0:
            pause_long_patterns = [
                (r"\btoo\s+short\b", "too short"),
                (r"\bclipped\b", "clipped"),
            ]
            for pat, name in pause_long_patterns:
                if re.search(pat, combined_text):
                    return False, f"Contradiction for long pause finding: detected '{name}'"

    return True, None


def generate_template_report(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a high-quality deterministic plain-language report from template rules."""
    duration = float(analysis_data.get("duration", 0.0))
    mode = analysis_data.get("mode", "single")
    transcript_source = analysis_data.get("transcript_source")
    regions = analysis_data.get("flaw_regions", [])

    # 1. Summary
    if not regions:
        summary = (
            f"Analysis of the {duration:.2f}-second recording showed stable delivery "
            f"with no significant deviation from expected baseline pacing or pitch."
        )
    else:
        types_flagged = sorted(list({r.get("type", "speech") for r in regions}))
        types_text = ", ".join(types_flagged)
        peak_z_vals = [abs(float(r.get("peak_z", r.get("z", 0.0)))) for r in regions]
        max_z = max(peak_z_vals) if peak_z_vals else 0.0
        summary = (
            f"We analyzed {duration:.2f} seconds of speech and identified {len(regions)} "
            f"delivery variation region(s) involving {types_text}, with a peak deviation "
            f"score of z = {max_z:.2f}."
        )

    # 2. What Happened (one short paragraph per flagged region)
    what_happened: List[str] = []
    for i, r in enumerate(regions):
        flaw_type = r.get("type", "delivery")
        s = float(r.get("start", 0.0))
        e = float(r.get("end", 0.0))
        reg_dur = e - s
        z_val = float(r.get("z", r.get("peak_z", 0.0)))
        obs = r.get("observed_value")
        unit = r.get("unit", "")
        sev = r.get("severity", "flagged")
        b_mean = r.get("baseline_mean")

        if flaw_type == "pitch":
            if z_val < 0:
                p = (
                    f"From {s:.2f}s to {e:.2f}s ({reg_dur:.2f} seconds duration), the speaker's vocal "
                    f"inflection was exceptionally flat ({sev}). The pitch variation was too small, "
                    f"with measured pitch standard deviation of {obs} {unit} compared to the baseline "
                    f"average of {b_mean} {unit} (deviation z = {z_val:.2f})."
                )
            else:
                p = (
                    f"From {s:.2f}s to {e:.2f}s ({reg_dur:.2f} seconds duration), the pitch moved far "
                    f"more than the baseline ({sev}). Measured pitch standard deviation reached {obs} {unit} "
                    f"compared to the baseline average of {b_mean} {unit} (deviation z = {z_val:.2f})."
                )
        elif flaw_type == "pause":
            if z_val < 0:
                p = (
                    f"At timestamp {s:.2f}s to {e:.2f}s, the silence interval was shorter than normal "
                    f"at {obs} {unit} ({sev}), deviating from the normal expected pause length of {b_mean} {unit} "
                    f"(deviation z = {z_val:.2f})."
                )
            else:
                p = (
                    f"At timestamp {s:.2f}s to {e:.2f}s, the silence interval was longer than normal "
                    f"at {obs} {unit} ({sev}), deviating from the normal expected pause length of {b_mean} {unit} "
                    f"(deviation z = {z_val:.2f})."
                )
        elif flaw_type == "rate":
            if z_val > 0:
                p = (
                    f"Between {s:.2f}s and {e:.2f}s ({reg_dur:.2f} seconds duration), delivery pace "
                    f"was faster than the baseline at {obs} {unit} ({sev}), contrasting against the baseline speaking rate "
                    f"of {b_mean} {unit} (deviation z = {z_val:.2f})."
                )
            else:
                p = (
                    f"Between {s:.2f}s and {e:.2f}s ({reg_dur:.2f} seconds duration), delivery pace "
                    f"was slower than the baseline at {obs} {unit} ({sev}), contrasting against the baseline speaking rate "
                    f"of {b_mean} {unit} (deviation z = {z_val:.2f})."
                )
        else:
            p = (
                f"Between {s:.2f}s and {e:.2f}s ({reg_dur:.2f} seconds duration), a delivery anomaly "
                f"was detected with observed value {obs} {unit} (deviation z = {z_val:.2f})."
            )
        what_happened.append(p)

    if not what_happened:
        what_happened.append(
            f"Across the full {duration:.2f} seconds, delivery cadence and pitch contour remained "
            f"within normal expected parameters."
        )

    # 3. Why It Matters
    why_it_matters = (
        "Listeners depend on melodic pitch inflection and predictable pause boundaries to absorb "
        "spoken ideas effortlessly. Monotone stretches make it difficult for listeners to identify "
        "key takeaways, while unexpected rate rushes or clipped pauses disrupt comprehension."
    )

    # 4. Practice tips: fixed vetted table, never from model
    practice_tips = get_practice_tips_for_regions(regions)

    # 5. Clear caveats
    caveats = build_caveats_text(mode, transcript_source)

    return {
        "source": "template",
        "summary": summary,
        "what_happened": what_happened,
        "why_it_matters": why_it_matters,
        "practice_tips": practice_tips,
        "caveats": caveats,
    }


def call_llm_provider(
    analysis_data: Dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Send measurements payload to an OpenAI-compatible endpoint with temperature 0."""
    if not api_key:
        return None

    # Sanitize endpoint URL
    clean_base = base_url.rstrip("/")
    if clean_base.endswith("/chat/completions"):
        endpoint_url = clean_base
    else:
        endpoint_url = f"{clean_base}/chat/completions"

    # Restrict transmitted data strictly to flaw_regions and summary values
    # Add a 'meaning' field for every flagged region before sending to model
    flaw_regions_payload = []
    for r in analysis_data.get("flaw_regions", []):
        z_val = float(r.get("z", r.get("peak_z", 0.0)))
        t_val = str(r.get("type", ""))
        flaw_regions_payload.append({
            "type": r.get("type"),
            "start": r.get("start"),
            "end": r.get("end"),
            "duration": round(float(r["end"]) - float(r["start"]), 3) if "start" in r and "end" in r else None,
            "z": r.get("z"),
            "peak_z": r.get("peak_z"),
            "meaning": compute_region_meaning(t_val, z_val),
            "severity": r.get("severity"),
            "observed_value": r.get("observed_value"),
            "unit": r.get("unit"),
            "baseline_mean": r.get("baseline_mean"),
            "baseline_spread": r.get("baseline_spread"),
            "threshold": r.get("threshold"),
            "reference_value": r.get("reference_value"),
            "peak_window_start": r.get("peak_window_start"),
            "peak_window_end": r.get("peak_window_end"),
        })

    transmitted_payload = {
        "duration": analysis_data.get("duration"),
        "mode": analysis_data.get("mode"),
        "transcript_source": analysis_data.get("transcript_source"),
        "region_count": len(analysis_data.get("flaw_regions", [])),
        "flaw_regions": flaw_regions_payload,
    }

    system_prompt = (
        "You are an expert speech coach explaining vocal analytics to a speaker. "
        "Use only numbers, times and types present in the provided measurements data. "
        "Never invent values. Write for someone with no technical background. "
        "No emoji, no long dashes, no hype.\n\n"
        "RULES FOR DESCRIBING FINDINGS:\n"
        "1. For each flagged region, describe the finding strictly using the provided 'meaning' field.\n"
        "2. For a pitch variation finding, observed_value is the standard deviation of pitch in semitones inside the window (how much the pitch moved), NOT the pitch level or frequency.\n"
        "3. NEVER say that pitch 'dropped' or 'went down' for a variation finding. A small standard deviation means the pitch variation was too small and the delivery was flat or monotone.\n"
        "4. Do NOT say 'steady pitch', 'smooth', or 'reduce variation' when describing flat or monotone pitch.\n"
        "5. Write NO advice at all. Do not provide practice tips, exercises, or suggestions.\n\n"
        "Return a strictly valid JSON object with the following keys:\n"
        "- 'summary': a short plain-language summary of the overall findings.\n"
        "- 'what_happened': an array of short paragraphs, exactly one short paragraph per flagged region, describing the time range and measured flaw using the provided meaning field.\n"
        "- 'why_it_matters': a short paragraph explaining why these delivery characteristics matter for audience engagement.\n\n"
        "CRITICAL NUMERIC GROUNDING RULES:\n"
        "1. Every single number, timestamp, and duration mentioned MUST appear in the provided measurements data.\n"
        "2. Do not invent any numbers, frequencies, decibels, or percentages.\n"
        "3. Do not use digits like 1, 2, 3 for list items; write complete sentences.\n"
        "4. No em-dashes and no en-dashes; use hyphens or colons only."
    )

    user_prompt = f"Speech measurements:\n{json.dumps(transmitted_payload, indent=2)}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
        "stream": False,
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(endpoint_url, headers=headers, json=body)
            if resp.status_code != 200:
                return None

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return None

            raw_content = choices[0].get("message", {}).get("content")
            if not raw_content:
                return None

            content = raw_content.strip()
            # Handle markdown code blocks if wrapped by model
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)

            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}") + 1]
            else:
                json_str = content

            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                return None

            return parsed
    except Exception:
        # Fallback on any connection error, timeout, or invalid JSON
        return None


def generate_report(
    analysis_data: Dict[str, Any],
    example_id: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Generate or retrieve a verified plain-language report.

    Checks pre-generated reports, in-memory cache, disk cache, LLM provider,
    and template generator in order.
    """
    # 1. Check pre-generated example report if example_id is provided
    if example_id and not force_refresh:
        clean_id = example_id.replace(".json", "")
        ex_report_path = REPORTS_DIR / f"{clean_id}.json"
        if ex_report_path.exists():
            try:
                with open(ex_report_path, "r", encoding="utf-8") as f:
                    cached_report = json.load(f)
                return cached_report
            except Exception:
                pass

    # 2. Check memory and disk cache
    cache_key = compute_cache_key(analysis_data)
    if not force_refresh:
        if cache_key in _REPORT_MEMORY_CACHE:
            return _REPORT_MEMORY_CACHE[cache_key]

        disk_path = DISK_CACHE_DIR / f"{cache_key}.json"
        if disk_path.exists():
            try:
                with open(disk_path, "r", encoding="utf-8") as f:
                    disk_report = json.load(f)
                _REPORT_MEMORY_CACHE[cache_key] = disk_report
                return disk_report
            except Exception:
                pass

    # 3. Extract ground-truth allowed numbers for validation
    allowed_map = extract_allowed_numbers(analysis_data)

    # 4. Attempt AI report generation if credentials exist
    env_vars = load_env_vars()
    api_key = env_vars.get("LLM_API_KEY", "")
    base_url = env_vars.get("LLM_BASE_URL", "")
    model = env_vars.get("LLM_MODEL", "")

    final_report: Optional[Dict[str, Any]] = None

    if api_key and base_url and model:
        raw_ai = call_llm_provider(
            analysis_data=analysis_data,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=30.0,
        )
        if raw_ai:
            # Meaning check: reject replies contradicting findings
            meaning_ok, reason = check_meaning_contradictions(
                raw_ai, analysis_data.get("flaw_regions", [])
            )
            if not meaning_ok:
                print(f"Meaning check rejected AI reply: {reason}. Falling back to template.")
            else:
                # Attach fixed practice tips and clear caveats
                raw_ai["practice_tips"] = get_practice_tips_for_regions(
                    analysis_data.get("flaw_regions", [])
                )
                raw_ai["caveats"] = build_caveats_text(
                    analysis_data.get("mode"), analysis_data.get("transcript_source")
                )

                # Grounding numeric validation
                is_valid, hallucinations, matched = validate_report_numbers(raw_ai, allowed_map)
                if is_valid:
                    raw_ai["source"] = "ai"
                    final_report = raw_ai
                else:
                    print(f"Numeric validation rejected AI reply: {hallucinations}. Falling back to template.")

    # 5. Fallback to deterministic template explanations
    if not final_report:
        final_report = generate_template_report(analysis_data)
        final_report["source"] = "template"

    # 6. Store in memory and on disk cache
    _REPORT_MEMORY_CACHE[cache_key] = final_report
    try:
        with open(disk_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)
    except Exception:
        pass

    return final_report


@router.post("/api/report")
def api_generate_report(payload: Dict[str, Any]):
    """Endpoint taking analysis JSON and returning a verified report."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    example_id = payload.get("example_id")
    report = generate_report(payload, example_id=example_id)
    return JSONResponse(report)
