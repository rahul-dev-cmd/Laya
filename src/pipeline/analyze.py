#!/usr/bin/env python3
"""Analysis pipeline and dashboard data contract for Laya.

This module provides the core analysis entry point:
    analyze(audio_wav, transcript_txt, reference_wav=None)

Pipeline steps:
1. Forced word alignment (Whisper via faster-whisper, or reusing existing alignment).
2. Acoustic feature extraction at 10 ms hop:
   - F0 in semitones relative to speaker's median F0.
   - Energy (RMS in dB) z-scored over the clip.
   - Speech rate (words per second in 3.0s sliding window).
   Computed for participant audio, and for reference audio if provided.
3. Multimodal flaw detection using frozen thresholds:
   - Rate detector (sliding window words per second, |z| >= 2.0).
   - Pitch detector (3.0s sliding window F0 standard deviation + Stage 2 word edge refinement).
   - Pause detector (phrase-boundary silence duration, |z| >= 2.0).
4. Template-based explanation generation embedding measured quantities and timestamps.
5. Returns a structured dashboard JSON contract in "paired" or "experimental" mode.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from align import align_transcript_to_audio
from src.audio.preprocess import load_audio
from src.config import RANDOM_SEED, TARGET_SAMPLE_RATE
from src.detection.config import (
    PITCH_WORD_Z_THRESHOLD,
    PITCH_Z_SCORE_THRESHOLD,
    Z_SCORE_THRESHOLD,
)
from src.detection.pause_detect import (
    compute_pause_baseline,
    detect_pause_flaws,
)
from src.detection.pitch_detect import (
    compute_pitch_baseline,
    detect_pitch_flaw_regions,
)
from src.detection.rate_detect import (
    compute_baseline_stats,
    compute_sliding_window_rates,
    detect_flaw_regions,
    load_alignment_csv,
)
from src.features.extract import (
    compute_energy_contour,
    compute_f0_contour,
    compute_speech_rate_and_pauses,
)
from src.utils.seed import set_seed


def classify_severity(abs_z: float, flaw_type: str = "rate") -> str:
    """Classify anomaly severity from peak absolute z-score into standardized tiers.
    
    Tiers:
    - Tier 1: slight deviation (|z| < 2.3)
    - Tier 2: mild deviation (2.3 <= |z| < 2.8)
    - Tier 3: moderate deviation (2.8 <= |z| < 3.5)
    - Tier 4: high deviation (3.5 <= |z| < 4.2)
    - Tier 5: severe deviation (|z| >= 4.2)
    
    Args:
        abs_z: Absolute value of peak z-score.
        flaw_type: Type of speech flaw ("rate", "pitch", "pause").
        
    Returns:
        Formatted severity string, e.g. "tier 4 (high)".
    """
    if abs_z >= 4.2:
        return "tier 5 (severe)"
    elif abs_z >= 3.5:
        return "tier 4 (high)"
    elif abs_z >= 2.8:
        return "tier 3 (moderate)"
    elif abs_z >= 2.3:
        return "tier 2 (mild)"
    else:
        return "tier 1 (slight)"


def find_or_compute_alignment(
    wav_path: Union[str, Path],
    transcript_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[Dict[str, Any]], Path]:
    """Find existing alignment CSV on disk or run forced alignment.
    
    Preserves all existing alignment files without modification.
    
    Args:
        wav_path: Path to speech audio WAV.
        transcript_path: Path to reference transcript TXT.
        
    Returns:
        Tuple of (aligned_words_list, path_to_alignment_csv).
    """
    wav_path = Path(wav_path).resolve()
    stem = wav_path.stem
    parent = wav_path.parent

    # Candidate locations for precomputed alignment CSVs
    candidates: List[Path] = [
        parent / f"{stem}_alignment.csv",
        PROJECT_ROOT / "dataset" / "corrupted" / f"{stem}_alignment.csv",
        PROJECT_ROOT / "dataset" / "sources" / f"{stem}_alignment.csv",
    ]

    # For pitch corruptions without dedicated alignment CSV, audio timing matches source clip
    if "librivox_01_pitch" in stem:
        candidates.append(PROJECT_ROOT / "dataset" / "sources" / "librivox_01_alignment.csv")
    elif "librivox_02_pitch" in stem:
        candidates.append(PROJECT_ROOT / "dataset" / "sources" / "librivox_02_alignment.csv")

    for cand in candidates:
        if cand.exists():
            words = load_alignment_csv(cand)
            return words, cand

    # If no pre-existing alignment CSV exists, run forced alignment
    if transcript_path is None or not Path(transcript_path).exists():
        raise FileNotFoundError(
            f"No alignment CSV found for '{wav_path.name}' and transcript file was not provided."
        )

    out_csv = parent / f"{stem}_alignment.csv"
    aligned_rows, _ = align_transcript_to_audio(
        wav_path=wav_path,
        transcript_path=Path(transcript_path).resolve(),
        output_csv_path=out_csv,
    )
    words = load_alignment_csv(out_csv)
    return words, out_csv


def extract_curves_10ms(
    audio_path: Union[str, Path],
    alignment_csv_path: Union[str, Path],
    duration: float,
) -> Tuple[List[float], List[Optional[float]], List[float], List[float]]:
    """Extract continuous acoustic curves sampled uniformly at a 10 ms hop (100 Hz).
    
    Curves extracted:
    - F0 in semitones relative to speaker's median F0.
    - Energy (RMS in dB) z-scored across the clip.
    - Speech rate (words per second in 3.0 s sliding window).
    
    Args:
        audio_path: Path to WAV audio file.
        alignment_csv_path: Path to word alignment CSV.
        duration: Total audio clip duration in seconds.
        
    Returns:
        Tuple of (timestamps, f0_semitones, energy_z, speech_rate).
    """
    # 10 ms uniform analysis grid (0.01 s hop)
    times = np.round(np.arange(0.0, duration, 0.01), 3)

    # 1. Cleaned F0 contour in semitones relative to speaker median
    _, f0_st, _ = compute_f0_contour(audio_path, times)
    f0_clean: List[Optional[float]] = [
        None if np.isnan(v) else round(float(v), 3) for v in f0_st
    ]

    # 2. Frame-level RMS energy z-scored across the clip
    audio_arr, _ = load_audio(audio_path, target_sr=TARGET_SAMPLE_RATE)
    _, energy_z = compute_energy_contour(audio_arr)
    # Ensure length matches times grid
    energy_clean: List[float] = [
        round(float(v), 3) for v in energy_z[: len(times)]
    ]
    # Pad if energy array is slightly shorter
    while len(energy_clean) < len(times):
        energy_clean.append(0.0)

    # 3. Speech rate in words per second (3.0 s sliding window)
    speech_rate, _ = compute_speech_rate_and_pauses(
        alignment_csv_path=alignment_csv_path,
        times=times,
        total_duration=duration,
        window_duration=3.0,
    )
    rate_clean: List[float] = [
        round(float(v), 3) for v in speech_rate[: len(times)]
    ]

    timestamps: List[float] = [round(float(t), 3) for t in times]
    return timestamps, f0_clean, energy_clean, rate_clean


def analyze(
    audio_wav: Union[str, Path],
    transcript_txt: Union[str, Path],
    reference_wav: Optional[Union[str, Path]] = None,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Run full speech analysis and temporal flaw detection for dashboard visualization.
    
    Args:
        audio_wav: Path to participant speech recording WAV file.
        transcript_txt: Path to reference transcript text file.
        reference_wav: Optional path to paired reference speech WAV file.
                       If provided, mode is "paired"; if None, mode is "experimental".
        seed: Random seed for deterministic reproducibility (default: 42).
        
    Returns:
        Dictionary conforming to Laya dashboard data contract:
        - duration: float
        - words: list of {"word": str, "start": float, "end": float}
        - curves: {
            "participant": {time, f0_semitones, energy_z, speech_rate},
            "reference": {time, f0_semitones, energy_z, speech_rate} or None,
            "timestamps", "f0_semitones", "energy_z", "speech_rate" (convenience)
          }
        - flaw_regions: list of {type, start, end, peak_z, severity, explanation}
        - mode: "paired" or "experimental"
    """
    set_seed(seed)

    audio_path = Path(audio_wav).resolve()
    transcript_path = Path(transcript_txt).resolve()
    is_paired = reference_wav is not None
    mode = "paired" if is_paired else "experimental"

    if not audio_path.exists():
        raise FileNotFoundError(f"Participant audio file not found: {audio_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    # 1. Get participant audio metadata and word alignment
    info = sf.info(str(audio_path))
    participant_duration = round(float(info.duration), 3)

    participant_words_raw, participant_align_csv = find_or_compute_alignment(
        audio_path, transcript_path
    )
    words_list: List[Dict[str, Any]] = [
        {
            "word": w["word"],
            "start": round(float(w["start"]), 3),
            "end": round(float(w["end"]), 3),
        }
        for w in participant_words_raw
    ]

    # 2. Extract participant 10 ms curves
    (
        p_times,
        p_f0,
        p_energy,
        p_rate,
    ) = extract_curves_10ms(audio_path, participant_align_csv, participant_duration)

    participant_curves = {
        "time": p_times,
        "f0_semitones": p_f0,
        "energy_z": p_energy,
        "speech_rate": p_rate,
    }

    # 3. Handle reference audio (if paired mode)
    ref_curves: Optional[Dict[str, Any]] = None
    ref_words_raw: Optional[List[Dict[str, Any]]] = None
    ref_align_csv: Optional[Path] = None
    ref_duration: Optional[float] = None
    ref_audio_path: Optional[Path] = None

    if is_paired and reference_wav is not None:
        ref_audio_path = Path(reference_wav).resolve()
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")

        ref_info = sf.info(str(ref_audio_path))
        ref_duration = round(float(ref_info.duration), 3)

        ref_words_raw, ref_align_csv = find_or_compute_alignment(
            ref_audio_path, transcript_path
        )
        r_times, r_f0, r_energy, r_rate = extract_curves_10ms(
            ref_audio_path, ref_align_csv, ref_duration
        )
        ref_curves = {
            "time": r_times,
            "f0_semitones": r_f0,
            "energy_z": r_energy,
            "speech_rate": r_rate,
        }

    curves_payload = {
        "timestamps": p_times,
        "f0_semitones": p_f0,
        "energy_z": p_energy,
        "speech_rate": p_rate,
        "participant": participant_curves,
        "reference": ref_curves,
    }

    # 4. Flaw Detectors
    flaw_regions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 4a. Rate Flaw Detection
    # ------------------------------------------------------------------
    if is_paired and ref_align_csv is not None and ref_duration is not None:
        b_mean, b_std, orig_words = compute_baseline_stats(
            ref_align_csv, ref_duration
        )
        detected_rate = detect_flaw_regions(
            test_words=participant_words_raw,
            total_duration=participant_duration,
            baseline_mean=b_mean,
            baseline_std=b_std,
            orig_words=orig_words,
            z_threshold=Z_SCORE_THRESHOLD,
        )
        ref_rate_wins = compute_sliding_window_rates(orig_words, ref_duration)
    else:
        # Experimental Mode B: compare against speaker's own internal distribution
        b_mean, b_std, orig_words = compute_baseline_stats(
            participant_align_csv, participant_duration
        )
        detected_rate = detect_flaw_regions(
            test_words=participant_words_raw,
            total_duration=participant_duration,
            baseline_mean=b_mean,
            baseline_std=b_std,
            orig_words=participant_words_raw,
            z_threshold=Z_SCORE_THRESHOLD,
        )
        ref_rate_wins = None

    test_rate_wins = compute_sliding_window_rates(
        participant_words_raw, participant_duration
    )

    for r in detected_rate:
        pk_z = round(float(r["z_score"]), 3)
        s_sec = round(float(r["start_seconds"]), 3)
        e_sec = round(float(r["end_seconds"]), 3)
        direction = r.get("direction", "fast")
        sev = classify_severity(abs(pk_z), "rate")

        span_s, span_e = r["window_span_seconds"]
        cand_wins = [
            w for w in test_rate_wins
            if w["start"] >= span_s - 1e-4 and w["end"] <= span_e + 1e-4
        ]
        if not cand_wins:
            cand_wins = test_rate_wins
        peak_w = min(cand_wins, key=lambda w: abs(((w["rate"] - b_mean) / b_std) - pk_z))

        peak_win_s = round(float(peak_w["start"]), 3)
        peak_win_e = round(float(peak_w["end"]), 3)
        obs_val = round(float(peak_w["rate"]), 2)
        unit = "words/s"
        base_mean_val = round(float(b_mean), 4)
        base_spread_val = round(float(b_std), 4)
        thresh_val = float(Z_SCORE_THRESHOLD) if direction == "fast" else -float(Z_SCORE_THRESHOLD)

        ref_val = None
        if is_paired and ref_rate_wins:
            m_ref = [rw for rw in ref_rate_wins if abs(rw["start"] - peak_win_s) < 1e-4]
            if m_ref:
                ref_val = round(float(m_ref[0]["rate"]), 2)

        recomputed_z = round((obs_val - base_mean_val) / base_spread_val, 3)
        check = bool(abs(recomputed_z - pk_z) <= 0.01)

        ref_str = f"{ref_val:.2f} {unit}" if ref_val is not None else "null"
        action = "accelerated" if direction == "fast" else "decelerated"
        expl = (
            f"Speech rate {action} to {obs_val:.2f} {unit} in peak window [{peak_win_s:.2f}s-{peak_win_e:.2f}s] "
            f"(reference: {ref_str}; baseline mean: {base_mean_val:.2f} {unit}, spread: {base_spread_val:.2f} {unit}, threshold: {thresh_val:+.2f}; "
            f"z = {recomputed_z:+.2f}, peak z = {pk_z:+.2f}) between {s_sec:.2f}s and {e_sec:.2f}s."
        )

        flaw_regions.append({
            "type": "rate",
            "start": s_sec,
            "end": e_sec,
            "peak_z": pk_z,
            "severity": sev,
            "peak_window_start": peak_win_s,
            "peak_window_end": peak_win_e,
            "observed_value": obs_val,
            "unit": unit,
            "baseline_mean": base_mean_val,
            "baseline_spread": base_spread_val,
            "threshold": thresh_val,
            "reference_value": ref_val,
            "z": recomputed_z,
            "check": check,
            "null_reason": None,
            "explanation": expl,
        })

    # ------------------------------------------------------------------
    # 4b. Pitch Flaw Detection
    # ------------------------------------------------------------------
    if is_paired and ref_audio_path is not None:
        pitch_base = compute_pitch_baseline(
            ref_audio_path, alignment_words=ref_words_raw
        )
        detected_pitch, test_wins, flagged_wins, _ = detect_pitch_flaw_regions(
            test_wav_path=audio_path,
            baseline_stats=pitch_base,
            alignment_words=participant_words_raw,
            z_threshold=PITCH_Z_SCORE_THRESHOLD,
            word_z_threshold=PITCH_WORD_Z_THRESHOLD,
            disable_contrastive_gate=False,
        )
    else:
        # Experimental Mode B: speaker internal intonation spread
        pitch_base = compute_pitch_baseline(
            audio_path, alignment_words=participant_words_raw
        )
        detected_pitch, test_wins, flagged_wins, _ = detect_pitch_flaw_regions(
            test_wav_path=audio_path,
            baseline_stats=pitch_base,
            alignment_words=participant_words_raw,
            z_threshold=PITCH_Z_SCORE_THRESHOLD,
            word_z_threshold=PITCH_WORD_Z_THRESHOLD,
            disable_contrastive_gate=True,
        )
    b_pitch_mean = pitch_base["mean_f0_sd"]
    b_pitch_spread = pitch_base["std_f0_sd"]

    for p in detected_pitch:
        pk_z = round(float(p["z_score"]), 3)
        s_sec = round(float(p["start_seconds"]), 3)
        e_sec = round(float(p["end_seconds"]), 3)
        sev = classify_severity(abs(pk_z), "pitch")

        span_s, span_e = p["window_span_seconds"]
        cand_wins = [
            w for w in flagged_wins
            if w["start"] >= span_s - 1e-4 and w["end"] <= span_e + 1e-4
        ]
        if not cand_wins:
            cand_wins = test_wins
        peak_w = min(cand_wins, key=lambda w: abs(w.get("z_score", 0.0) - pk_z))

        peak_win_s = round(float(peak_w["start"]), 3)
        peak_win_e = round(float(peak_w["end"]), 3)
        obs_val = round(float(peak_w["f0_sd"]), 3)
        unit = "semitones"
        base_mean_val = round(float(b_pitch_mean), 4)
        base_spread_val = round(float(b_pitch_spread), 4)
        thresh_val = float(PITCH_Z_SCORE_THRESHOLD)

        ref_val = None
        if is_paired and "windows" in pitch_base:
            m_ref = [bw for bw in pitch_base["windows"] if abs(bw["start"] - peak_win_s) < 1e-4]
            if m_ref and not np.isnan(m_ref[0]["f0_sd"]):
                ref_val = round(float(m_ref[0]["f0_sd"]), 3)

        recomputed_z = round((obs_val - base_mean_val) / base_spread_val, 3)
        check = bool(abs(recomputed_z - pk_z) <= 0.01)

        ref_str = f"{ref_val:.3f} {unit}" if ref_val is not None else "null"
        expl = (
            f"Pitch intonation is monotone / flat with observed F0 SD of {obs_val:.3f} {unit} in peak window [{peak_win_s:.2f}s-{peak_win_e:.2f}s] "
            f"(reference: {ref_str}; baseline mean: {base_mean_val:.3f} {unit}, spread: {base_spread_val:.3f} {unit}, threshold: {thresh_val:+.2f}; "
            f"z = {recomputed_z:+.3f}, peak z = {pk_z:+.3f}) between {s_sec:.2f}s and {e_sec:.2f}s."
        )

        flaw_regions.append({
            "type": "pitch",
            "start": s_sec,
            "end": e_sec,
            "peak_z": pk_z,
            "severity": sev,
            "peak_window_start": peak_win_s,
            "peak_window_end": peak_win_e,
            "observed_value": obs_val,
            "unit": unit,
            "baseline_mean": base_mean_val,
            "baseline_spread": base_spread_val,
            "threshold": thresh_val,
            "reference_value": ref_val,
            "z": recomputed_z,
            "check": check,
            "null_reason": None,
            "explanation": expl,
        })

    # ------------------------------------------------------------------
    # 4c. Pause Flaw Detection
    # ------------------------------------------------------------------
    if is_paired and ref_align_csv is not None:
        pause_base = compute_pause_baseline(ref_align_csv)
        pause_res = detect_pause_flaws(
            test_alignment_csv_path=participant_align_csv,
            baseline_stats=pause_base,
            z_threshold=Z_SCORE_THRESHOLD,
            eval_mode="A",
        )
    else:
        # Experimental Mode B: speaker population norm
        pause_base = compute_pause_baseline(participant_align_csv)
        pause_res = detect_pause_flaws(
            test_alignment_csv_path=participant_align_csv,
            baseline_stats=pause_base,
            z_threshold=Z_SCORE_THRESHOLD,
            eval_mode="B",
        )

    for b in pause_res.get("flagged_boundaries", []):
        pk_z = round(float(b["active_z_score"]), 3)
        s_sec = round(float(b["start_seconds"]), 3)
        e_sec = round(float(b["end_seconds"]), 3)
        w_before = b["word_before"]
        w_after = b["word_after"]
        t_dur = round(float(b["test_duration_seconds"]), 2)
        base_dur = round(float(b["baseline_duration_seconds"]), 2)
        direction = b["direction"]
        sev = classify_severity(abs(pk_z), "pause")

        obs_val = t_dur
        unit = "s"
        base_mean_val = base_dur if is_paired else round(float(pause_base["median_seconds"]), 2)
        base_spread_val = round(float(pause_base["std_seconds"]), 4) if is_paired else round(float(pause_base["scaled_mad_seconds"]), 4)
        thresh_val = -float(Z_SCORE_THRESHOLD) if direction == "too short" else float(Z_SCORE_THRESHOLD)
        ref_val = base_dur if is_paired else None

        recomputed_z = round((obs_val - base_mean_val) / base_spread_val, 3)
        check = bool(abs(recomputed_z - pk_z) <= 0.01)

        ref_str = f"{ref_val:.2f} {unit}" if ref_val is not None else "null"
        action = "shortened" if direction == "too short" else "extended"
        expl = (
            f"Pause between '{w_before}' and '{w_after}' {action} to {obs_val:.2f} {unit} "
            f"(reference: {ref_str}; baseline mean: {base_mean_val:.2f} {unit}, spread: {base_spread_val:.2f} {unit}, threshold: {thresh_val:+.2f}; "
            f"z = {recomputed_z:+.2f}, peak z = {pk_z:+.2f}; peak window: null [boundary measurement]) between {s_sec:.2f}s and {e_sec:.2f}s."
        )

        flaw_regions.append({
            "type": "pause",
            "start": s_sec,
            "end": e_sec,
            "peak_z": pk_z,
            "severity": sev,
            "peak_window_start": None,
            "peak_window_end": None,
            "observed_value": obs_val,
            "unit": unit,
            "baseline_mean": base_mean_val,
            "baseline_spread": base_spread_val,
            "threshold": thresh_val,
            "reference_value": ref_val,
            "z": recomputed_z,
            "check": check,
            "null_reason": "Pause flaws are discrete boundary duration measurements between words, not 3.0 s sliding windows.",
            "explanation": expl,
        })

    # Sort all detected flaw regions chronologically by start timestamp
    flaw_regions.sort(key=lambda x: x["start"])

    # 5. Assemble final dashboard JSON object
    result = {
        "mode": mode,
        "duration": participant_duration,
        "audio_file": str(audio_path).replace("\\", "/"),
        "reference_file": str(ref_audio_path).replace("\\", "/") if ref_audio_path else None,
        "words": words_list,
        "curves": curves_payload,
        "flaw_regions": flaw_regions,
    }

    return result


def main() -> None:
    """Command-line interface for running speech analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Run Laya multimodal speech analysis and generate dashboard contract JSON."
    )
    parser.add_argument("--audio", required=True, help="Path to participant audio WAV file")
    parser.add_argument("--transcript", required=True, help="Path to transcript TXT file")
    parser.add_argument("--reference", default=None, help="Path to reference audio WAV file")
    parser.add_argument("--output", default=None, help="Path to save output JSON file")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: 42)")

    args = parser.parse_args()

    result = analyze(
        audio_wav=args.audio,
        transcript_txt=args.transcript,
        reference_wav=args.reference,
        seed=args.seed,
    )

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[Analyze] Analysis successfully saved to: {out_path}")
    else:
        # Print summary
        print(f"[Analyze] Mode: {result['mode']} | Duration: {result['duration']:.2f}s")
        print(f"[Analyze] Words aligned: {len(result['words'])}")
        print(f"[Analyze] Detected flaw regions: {len(result['flaw_regions'])}")
        for i, reg in enumerate(result["flaw_regions"]):
            print(f"  [{i+1}] {reg['type'].upper()} ({reg['severity']}): {reg['explanation']}")


if __name__ == "__main__":
    main()
