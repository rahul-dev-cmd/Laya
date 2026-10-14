#!/usr/bin/env python3
"""Script to regenerate dataset/examples/reports/librivox_01_pitch_tier5.json
using the model configured in backend/.env, with ground-truth numeric validation.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.report import (
    build_caveats_text,
    call_llm_provider,
    check_meaning_contradictions,
    extract_allowed_numbers,
    load_env_vars,
    validate_report_numbers,
)
from backend.tips import get_practice_tips_for_regions


def main():
    example_file = PROJECT_ROOT / "dataset" / "examples" / "librivox_01_pitch_tier5.json"
    report_file = PROJECT_ROOT / "dataset" / "examples" / "reports" / "librivox_01_pitch_tier5.json"

    with open(example_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    env_vars = load_env_vars()
    api_key = env_vars.get("LLM_API_KEY", "")
    base_url = env_vars.get("LLM_BASE_URL", "")
    model = env_vars.get("LLM_MODEL", "")

    print(f"Configured Model: {model}")
    print(f"Configured Base URL: {base_url}")
    print(f"API Key Set: {bool(api_key)}")

    if not api_key or not base_url or not model:
        print("ERROR: Missing LLM credentials in backend/.env")
        sys.exit(1)

    print(f"\nRequesting report from {model}...")
    ai_raw = call_llm_provider(
        analysis_data=data,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=30.0,
    )

    if not ai_raw:
        print("ERROR: Call to LLM provider returned None (failed or timed out).")
        print("Keeping existing report file unchanged.")
        sys.exit(1)

    meaning_ok, reason = check_meaning_contradictions(ai_raw, data.get("flaw_regions", []))
    if not meaning_ok:
        print(f"\nMEANING CHECK FAILED: {reason}")
        print("Keeping existing report file unchanged.")
        sys.exit(1)

    # Attach fixed vetted practice tips and standard caveats
    ai_raw["practice_tips"] = get_practice_tips_for_regions(data.get("flaw_regions", []))
    ai_raw["caveats"] = build_caveats_text(data.get("mode"), data.get("transcript_source"))

    allowed_map = extract_allowed_numbers(data)
    is_valid, hallucinations, matched = validate_report_numbers(ai_raw, allowed_map)

    if not is_valid:
        print(f"\nVALIDATION FAILED: Hallucinated numbers detected: {hallucinations}")
        print("Keeping existing report file unchanged.")
        sys.exit(1)

    # Validation passed: mark source as "ai"
    ai_raw["source"] = "ai"

    # Save to dataset/examples/reports/librivox_01_pitch_tier5.json
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(ai_raw, f, indent=2)

    print(f"\nValidation: PASSED")
    print(f"Report successfully saved to: {report_file}")
    print(f"source: {ai_raw.get('source')}")

    print("\n--- REGENERATED REPORT CONTENT ---")
    print(json.dumps(ai_raw, indent=2))

    print("\n--- NUMERIC GROUNDING AUDIT (EVERY NUMBER AND ITS DATA FIELD) ---")
    for num, field in sorted(matched.items(), key=lambda x: x[0]):
        print(f"  Number: {num:>8}  <---  Data Field: {field}")


if __name__ == "__main__":
    main()
