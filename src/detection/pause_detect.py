#!/usr/bin/env python3
"""Pause flaw detection and evaluation module for Second Take.

This module implements temporal grounding and anomaly detection for phrase-boundary pauses:
1. Baseline computation:
   - Identifies all natural phrase boundaries (pauses >= 150 ms) in the uncorrupted clip.
   - Computes baseline pause distribution: mean, std, median, and MAD.
2. Two-mode pause z-scoring:
   - Mode A (Contrastive word-boundary): z-score of the pause duration difference relative
     to the identical word boundary in the original take:
     z_A = (test_pause - baseline_pause) / baseline_std
   - Mode B (Speaker population norm): z-score of the test pause duration relative to
     the speaker's median and scaled MAD across all natural pauses:
     z_B = (test_pause - speaker_median) / (1.4826 * speaker_mad)
3. Flagging & Direction:
   - Flags boundaries where |z| >= 2.0.
   - Direction: "too short" (rushed) if z <= -2.0, "too long" (draggy) if z >= +2.0.
4. Threshold compliance:
   - All thresholds defined in src/detection/config.py and printed at the start of each run.
   - Does NOT read label JSON during detection.
5. Boundary-level evaluation:
   - Reports per file: detected, direction correct, peak z, start error, end error,
     and boundary-level counts of TP, FP, and FN.
   - Runs full 11-point ladder on clip 01, freezes thresholds, and evaluates 7 files on clip 02.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.config import (
    PAUSE_MIN_DURATION_SECONDS,
    Z_SCORE_THRESHOLD,
    MAD_SCALE_FACTOR,
    PRIMARY_MODE,
    print_detection_config,
)

RANDOM_SEED: int = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_alignment_csv(csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load aligned words from an alignment CSV file."""
    words: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
                w = row["word"].strip()
                if end > start:
                    words.append({
                        "index": idx,
                        "word": w,
                        "start": start,
                        "end": end,
                    })
            except (ValueError, KeyError):
                continue
    return words


def compute_pause_baseline(
    alignment_csv_path: Union[str, Path],
    min_pause_sec: float = PAUSE_MIN_DURATION_SECONDS,
) -> Dict[str, Any]:
    """Compute baseline pause distribution from the uncorrupted audio alignment.
    
    Identifies all natural pauses >= min_pause_sec and computes mean, standard deviation,
    median, and Median Absolute Deviation (MAD).
    """
    words = load_alignment_csv(alignment_csv_path)
    natural_pauses: List[Dict[str, Any]] = []

    for i in range(len(words) - 1):
        gap = round(words[i + 1]["start"] - words[i]["end"], 4)
        if gap >= min_pause_sec:
            natural_pauses.append({
                "boundary_index": i,
                "word_before": words[i]["word"],
                "word_after": words[i + 1]["word"],
                "start_seconds": words[i]["end"],
                "end_seconds": words[i + 1]["start"],
                "duration_seconds": gap,
            })

    durations = np.array([p["duration_seconds"] for p in natural_pauses], dtype=np.float64)

    mean_dur = float(np.mean(durations)) if len(durations) > 0 else 0.0
    std_dur = float(np.std(durations)) if len(durations) > 0 else 0.0
    median_dur = float(np.median(durations)) if len(durations) > 0 else 0.0
    mad_dur = float(np.median(np.abs(durations - median_dur))) if len(durations) > 0 else 0.0
    scaled_mad = mad_dur * MAD_SCALE_FACTOR

    # Map of word-boundary index -> baseline pause dict
    boundary_map = {p["boundary_index"]: p for p in natural_pauses}

    return {
        "alignment_csv": str(alignment_csv_path),
        "total_words": len(words),
        "words": words,
        "natural_pauses": natural_pauses,
        "boundary_map": boundary_map,
        "count": len(natural_pauses),
        "mean_seconds": round(mean_dur, 4),
        "std_seconds": round(std_dur, 4),
        "median_seconds": round(median_dur, 4),
        "mad_seconds": round(mad_dur, 4),
        "scaled_mad_seconds": round(scaled_mad, 4),
    }


def detect_pause_flaws(
    test_alignment_csv_path: Union[str, Path],
    baseline_stats: Dict[str, Any],
    z_threshold: float = Z_SCORE_THRESHOLD,
    eval_mode: str = PRIMARY_MODE,
) -> Dict[str, Any]:
    """Detect pause flaws in test audio by z-scoring each phrase boundary against baseline.
    
    IMPORTANT: This function does NOT read label JSONs. Detection is performed strictly
    from the test alignment timestamps and precomputed baseline statistics.
    """
    test_words = load_alignment_csv(test_alignment_csv_path)
    base_b_map = baseline_stats["boundary_map"]
    b_std = baseline_stats["std_seconds"]
    b_median = baseline_stats["median_seconds"]
    b_scaled_mad = baseline_stats["scaled_mad_seconds"]

    evaluated_boundaries: List[Dict[str, Any]] = []
    flagged_boundaries: List[Dict[str, Any]] = []

    # Iterate over all word boundaries present in the clip
    for i in range(len(test_words) - 1):
        w_curr = test_words[i]
        w_next = test_words[i + 1]
        t_gap = round(max(0.0, w_next["start"] - w_curr["end"]), 4)

        is_base_phrase = (i in base_b_map)
        base_dur = base_b_map[i]["duration_seconds"] if is_base_phrase else 0.0

        # Only evaluate boundaries that were natural phrase boundaries in baseline
        # or exhibit a substantial pause in test audio
        if not is_base_phrase and t_gap < PAUSE_MIN_DURATION_SECONDS:
            continue

        # Mode A: Contrastive z-score against identical boundary in baseline
        diff_A = round(t_gap - base_dur, 4)
        z_A = round(diff_A / b_std, 3) if b_std > 0 else 0.0

        # Mode B: Non-contrastive z-score against speaker median & MAD
        diff_B = round(t_gap - b_median, 4)
        z_B = round(diff_B / b_scaled_mad, 3) if b_scaled_mad > 0 else 0.0

        # Primary mode selection
        active_z = z_A if eval_mode == "A" else z_B

        # Direction classification
        if active_z <= -z_threshold:
            direction = "too short"
        elif active_z >= z_threshold:
            direction = "too long"
        else:
            direction = "normal"

        is_flagged = abs(active_z) >= z_threshold

        boundary_info = {
            "boundary_index": i,
            "word_before": w_curr["word"],
            "word_after": w_next["word"],
            "start_seconds": w_curr["end"],
            "end_seconds": w_next["start"],
            "test_duration_seconds": t_gap,
            "baseline_duration_seconds": base_dur,
            "z_score_mode_A": z_A,
            "z_score_mode_B": z_B,
            "active_z_score": active_z,
            "direction": direction,
            "is_flagged": is_flagged,
        }
        evaluated_boundaries.append(boundary_info)

        if is_flagged:
            flagged_boundaries.append(boundary_info)

    # Peak z-score across evaluated boundaries
    peak_z = 0.0
    if flagged_boundaries:
        peak_z = max(flagged_boundaries, key=lambda b: abs(b["active_z_score"]))["active_z_score"]
    elif evaluated_boundaries:
        peak_z = max(evaluated_boundaries, key=lambda b: abs(b["active_z_score"]))["active_z_score"]

    return {
        "test_alignment_csv": str(test_alignment_csv_path),
        "total_evaluated_boundaries": len(evaluated_boundaries),
        "evaluated_boundaries": evaluated_boundaries,
        "flagged_boundaries": flagged_boundaries,
        "flagged_count": len(flagged_boundaries),
        "peak_z": peak_z,
    }


def score_against_label_json(
    detection_result: Dict[str, Any],
    label_json_path: Union[str, Path],
) -> Dict[str, Any]:
    """Score detected boundaries against ground-truth label JSON.
    
    Reports:
    - detected (bool)
    - direction_correct (bool)
    - peak_z (float)
    - start_error (float or None)
    - end_error (float or None)
    - boundary-level counts: TP, FP, FN
    """
    with open(label_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    is_control = (meta.get("direction") == "control" or meta.get("tier") == "control")
    target_boundary = meta.get("target_boundary", {})
    target_idx = target_boundary.get("word_before_index")
    gt_direction = meta.get("direction", "control")

    flaw_reg = meta.get("flaw_region", {})
    true_start = float(flaw_reg.get("start_seconds", 0.0))
    true_end = float(flaw_reg.get("end_seconds", 0.0))

    flagged = detection_result["flagged_boundaries"]
    flagged_indices = [b["boundary_index"] for b in flagged]

    detected = False
    direction_correct = False
    start_error = None
    end_error = None
    target_flagged_entry = None

    if is_control:
        # Control clip: Any flagged boundary is a False Positive
        tp = 0
        fp = len(flagged)
        fn = 0
        detected = False
        direction_correct = True  # Neutral / clean is expected
    else:
        # Corrupted clip
        if target_idx is not None and target_idx in flagged_indices:
            detected = True
            target_flagged_entry = next(b for b in flagged if b["boundary_index"] == target_idx)
            rep_dir = target_flagged_entry["direction"]
            # Match direction
            if gt_direction == "rushed" and rep_dir == "too short":
                direction_correct = True
            elif gt_direction == "draggy" and rep_dir == "too long":
                direction_correct = True
            else:
                direction_correct = False

            det_start = target_flagged_entry["start_seconds"]
            det_end = target_flagged_entry["end_seconds"]
            start_error = round(abs(det_start - true_start), 4)
            end_error = round(abs(det_end - true_end), 4)

            tp = 1
            fp = len(flagged) - 1
            fn = 0
        else:
            detected = False
            direction_correct = False
            tp = 0
            fp = len(flagged)
            fn = 1

    peak_z = detection_result["peak_z"]

    # Retrieve target boundary entry from evaluated_boundaries
    target_eval_entry = None
    if target_idx is not None:
        target_eval_entry = next((b for b in detection_result["evaluated_boundaries"] if b["boundary_index"] == target_idx), None)

    target_z_A = target_eval_entry["z_score_mode_A"] if target_eval_entry else 0.0
    target_z_B = target_eval_entry["z_score_mode_B"] if target_eval_entry else 0.0

    return {
        "file": Path(label_json_path).name,
        "tier": meta.get("tier"),
        "direction": gt_direction,
        "multiplier": meta.get("multiplier", 1.0),
        "detected": detected,
        "direction_correct": direction_correct,
        "peak_z": peak_z,
        "target_z_mode_A": target_z_A,
        "target_z_mode_B": target_z_B,
        "start_error": start_error,
        "end_error": end_error,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "target_boundary_index": target_idx,
        "target_flagged_entry": target_flagged_entry,
        "all_flagged_boundaries": flagged,
    }


def run_pause_evaluation(save_results: bool = True) -> Dict[str, Any]:
    """Run full pause flaw ladder evaluation across Clip 01 and Clip 02."""
    set_seed(RANDOM_SEED)

    sources_dir = PROJECT_ROOT / "dataset" / "sources"
    corrupted_dir = PROJECT_ROOT / "dataset" / "corrupted"

    # 1. Print natural pauses statistics for both clips
    a01 = sources_dir / "librivox_01_alignment.csv"
    a02 = sources_dir / "librivox_02_alignment.csv"

    base01 = compute_pause_baseline(a01)
    base02 = compute_pause_baseline(a02)

    print("\n" + "=" * 90)
    print("NATURAL PAUSES (>= 150 ms) ACOUSTIC BASELINE STATISTICS")
    print("=" * 90)
    print(f"Clip 01 (librivox_01):")
    print(f"  Total natural pauses: {base01['count']}")
    print(f"  Mean duration:        {base01['mean_seconds']:.4f} s ({base01['mean_seconds']*1000:.1f} ms)")
    print(f"  SD duration:          {base01['std_seconds']:.4f} s ({base01['std_seconds']*1000:.1f} ms)")
    print(f"  Median duration:      {base01['median_seconds']:.4f} s | MAD: {base01['mad_seconds']:.4f} s (scaled: {base01['scaled_mad_seconds']:.4f} s)")
    print()
    print(f"Clip 02 (librivox_02):")
    print(f"  Total natural pauses: {base02['count']}")
    print(f"  Mean duration:        {base02['mean_seconds']:.4f} s ({base02['mean_seconds']*1000:.1f} ms)")
    print(f"  SD duration:          {base02['std_seconds']:.4f} s ({base02['std_seconds']*1000:.1f} ms)")
    print(f"  Median duration:      {base02['median_seconds']:.4f} s | MAD: {base02['mad_seconds']:.4f} s (scaled: {base02['scaled_mad_seconds']:.4f} s)")
    print("=" * 90)

    # 2. Print active detection thresholds
    print()
    print_detection_config()

    # 3. Clip 01 Full Ladder Evaluation (11 files)
    print("\n" + "=" * 130)
    print("CLIP 01: FULL PAUSE FLAW LADDER EVALUATION (Both Directions, Tiers 1-5 + Control)")
    print("=" * 130)
    header = f"{'File':32s} | {'Tier':7s} | {'Mult':5s} | {'Det':4s} | {'Dir OK':6s} | {'Peak z':8s} | {'Mode A z':8s} | {'Mode B z':8s} | {'Start Err':9s} | {'End Err':9s} | {'TP':2s} | {'FP':2s} | {'FN':2s}"
    print(header)
    print("-" * len(header))

    clip01_targets = [
        # Rushed 1-5
        "librivox_01_pause_rushed_tier1",
        "librivox_01_pause_rushed_tier2",
        "librivox_01_pause_rushed_tier3",
        "librivox_01_pause_rushed_tier4",
        "librivox_01_pause_rushed_tier5",
        # Draggy 1-5
        "librivox_01_pause_draggy_tier1",
        "librivox_01_pause_draggy_tier2",
        "librivox_01_pause_draggy_tier3",
        "librivox_01_pause_draggy_tier4",
        "librivox_01_pause_draggy_tier5",
        # Control
        "librivox_01_pause_control",
    ]

    clip01_results: List[Dict[str, Any]] = []

    for stem in clip01_targets:
        csv_path = corrupted_dir / f"{stem}_alignment.csv"
        json_path = corrupted_dir / f"{stem}.json"

        det = detect_pause_flaws(csv_path, base01)
        score = score_against_label_json(det, json_path)
        clip01_results.append(score)

        det_str = "YES" if score["detected"] else "NO"
        dir_str = "YES" if score["direction_correct"] else "NO"
        s_err_str = f"{score['start_error']:.4f}s" if score["start_error"] is not None else "N/A"
        e_err_str = f"{score['end_error']:.4f}s" if score["end_error"] is not None else "N/A"
        tier_str = str(score["tier"])
        mult_str = f"{score['multiplier']:.2f}"

        print(f"{stem:32s} | {tier_str:7s} | {mult_str:5s} | {det_str:4s} | {dir_str:6s} | {score['peak_z']:+8.3f} | {score['target_z_mode_A']:+8.3f} | {score['target_z_mode_B']:+8.3f} | {s_err_str:9s} | {e_err_str:9s} | {score['true_positives']:2d} | {score['false_positives']:2d} | {score['false_negatives']:2d}")

    # 4. Clip 02 Evaluation (7 files) - Freeze thresholds, no tuning!
    print("\n" + "=" * 130)
    print("CLIP 02: FROZEN THRESHOLD EVALUATION (New Region, Tiers 2, 3, 5 + Control)")
    print("=" * 130)
    print(header)
    print("-" * len(header))

    clip02_targets = [
        "librivox_02_pause_rushed_tier2",
        "librivox_02_pause_rushed_tier3",
        "librivox_02_pause_rushed_tier5",
        "librivox_02_pause_draggy_tier2",
        "librivox_02_pause_draggy_tier3",
        "librivox_02_pause_draggy_tier5",
        "librivox_02_pause_control",
    ]

    clip02_results: List[Dict[str, Any]] = []

    for stem in clip02_targets:
        csv_path = corrupted_dir / f"{stem}_alignment.csv"
        json_path = corrupted_dir / f"{stem}.json"

        det = detect_pause_flaws(csv_path, base02)
        score = score_against_label_json(det, json_path)
        clip02_results.append(score)

        det_str = "YES" if score["detected"] else "NO"
        dir_str = "YES" if score["direction_correct"] else "NO"
        s_err_str = f"{score['start_error']:.4f}s" if score["start_error"] is not None else "N/A"
        e_err_str = f"{score['end_error']:.4f}s" if score["end_error"] is not None else "N/A"
        tier_str = str(score["tier"])
        mult_str = f"{score['multiplier']:.2f}"

        print(f"{stem:32s} | {tier_str:7s} | {mult_str:5s} | {det_str:4s} | {dir_str:6s} | {score['peak_z']:+8.3f} | {score['target_z_mode_A']:+8.3f} | {score['target_z_mode_B']:+8.3f} | {s_err_str:9s} | {e_err_str:9s} | {score['true_positives']:2d} | {score['false_positives']:2d} | {score['false_negatives']:2d}")

    # 5. Summary Statistics: Means & Ranges
    def compute_summary_stats(results: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        flawed = [r for r in results if r["direction"] != "control"]
        detected_flaws = [r for r in flawed if r["detected"]]

        det_rate = len(detected_flaws) / len(flawed) if flawed else 0.0
        dir_rate = sum(1 for r in detected_flaws if r["direction_correct"]) / len(detected_flaws) if detected_flaws else 0.0

        all_peak_z = [r["peak_z"] for r in results]
        s_errors = [r["start_error"] for r in detected_flaws if r["start_error"] is not None]
        e_errors = [r["end_error"] for r in detected_flaws if r["end_error"] is not None]

        total_tp = sum(r["true_positives"] for r in results)
        total_fp = sum(r["false_positives"] for r in results)
        total_fn = sum(r["false_negatives"] for r in results)

        return {
            "dataset": name,
            "total_files": len(results),
            "flawed_files": len(flawed),
            "detection_rate": round(det_rate, 4),
            "direction_accuracy": round(dir_rate, 4),
            "peak_z_mean": round(float(np.mean(all_peak_z)), 3),
            "peak_z_range": [round(float(min(all_peak_z)), 3), round(float(max(all_peak_z)), 3)],
            "start_error_mean": round(float(np.mean(s_errors)), 4) if s_errors else 0.0,
            "start_error_range": [round(float(min(s_errors)), 4), round(float(max(s_errors)), 4)] if s_errors else [0.0, 0.0],
            "end_error_mean": round(float(np.mean(e_errors)), 4) if e_errors else 0.0,
            "end_error_range": [round(float(min(e_errors)), 4), round(float(max(e_errors)), 4)] if e_errors else [0.0, 0.0],
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        }

    stats01 = compute_summary_stats(clip01_results, "Clip 01 (Full Ladder)")
    stats02 = compute_summary_stats(clip02_results, "Clip 02 (Generalization)")

    print("\n" + "=" * 90)
    print("SUMMARY METRICS: MEANS AND RANGES")
    print("=" * 90)
    for st in [stats01, stats02]:
        print(f"Dataset: {st['dataset']}")
        print(f"  Flaw Detection Rate:      {st['detection_rate']*100:.1f}% ({st['total_tp']}/{st['flawed_files']} flaws detected)")
        print(f"  Direction Accuracy:       {st['direction_accuracy']*100:.1f}%")
        print(f"  Peak z Mean:              {st['peak_z_mean']:+.3f} (Range: [{st['peak_z_range'][0]:+.3f}, {st['peak_z_range'][1]:+.3f}])")
        print(f"  Start Error Mean:         {st['start_error_mean']:.4f} s (Range: [{st['start_error_range'][0]:.4f}s, {st['start_error_range'][1]:.4f}s])")
        print(f"  End Error Mean:           {st['end_error_mean']:.4f} s (Range: [{st['end_error_range'][0]:.4f}s, {st['end_error_range'][1]:.4f}s])")
        print(f"  Boundary-level Totals:    TP = {st['total_tp']}, FP = {st['total_fp']}, FN = {st['total_fn']}")
        print()
    print("=" * 90)

    all_data = {
        "baseline_clip01": base01,
        "baseline_clip02": base02,
        "clip01_results": clip01_results,
        "clip02_results": clip02_results,
        "clip01_summary": stats01,
        "clip02_summary": stats02,
    }

    if save_results:
        out_json = corrupted_dir / "pause_ladder_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2)
        print(f"Comprehensive results successfully saved to: {out_json}")

    return all_data


if __name__ == "__main__":
    run_pause_evaluation(save_results=True)
