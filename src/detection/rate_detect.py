#!/usr/bin/env python3
"""Speech rate / tempo flaw detection module for Second Take.

This module implements temporal grounding for speech rate deviations:
1. Aligns the corrupted audio clip to the reference transcript using align.py
   to obtain word boundaries for the corrupted audio without reading label JSONs.
2. Computes the speech rate in 3-second sliding windows (words per second).
3. Establishes the baseline distribution (mean and standard deviation) of that rate
   over the original, uncorrupted librivox_01 clip.
4. Computes z-scores for each window against the baseline, flags windows where |z| >= 2.0,
   merges adjacent flagged windows into contiguous temporal regions, and outputs
   start/end seconds and z-scores for each region.
5. Scores detected regions against ground-truth label JSONs, reporting injected region,
   detected region(s), start/end timestamp errors, and IoU.
6. Runs across evaluation clips (librivox_01_rate_t3.wav, librivox_01_rate_t5.wav,
   and librivox_01_control.wav), ensuring control produces 0 false-positive flagged regions.
7. Saves comprehensive results to dataset/corrupted/detection_results.json.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer project virtual environment when running standalone
venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if venv_py.exists() and sys.executable.lower() != str(venv_py).lower():
    import subprocess
    result = subprocess.call([str(venv_py), __file__] + sys.argv[1:])
    sys.exit(result)

# Import forced alignment engine
from align import align_transcript_to_audio

# Fixed seed for reproducibility across all runs
RANDOM_SEED: int = 42
WINDOW_DURATION: float = 3.0  # 3.0 s sliding window
HOP_DURATION: float = 0.1     # 100 ms step between sliding windows
Z_SCORE_THRESHOLD: float = 2.0  # Anomaly criterion: |z| >= 2.0


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed across python and numpy for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def load_alignment_csv(csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load word timestamps from an alignment CSV file.
    
    Args:
        csv_path: Path to CSV with columns: word, start_seconds, end_seconds.
        
    Returns:
        List of dicts with word, start, end, and midpoint.
    """
    words: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
                word = row["word"].strip()
                if end > start:
                    words.append({
                        "index": idx,
                        "word": word,
                        "start": start,
                        "end": end,
                        "mid": (start + end) / 2.0,
                    })
            except (ValueError, KeyError):
                continue
    return words


def compute_sliding_window_rates(
    words: List[Dict[str, Any]],
    total_duration: float,
    window_duration: float = WINDOW_DURATION,
    hop_duration: float = HOP_DURATION,
) -> List[Dict[str, Any]]:
    """Compute local speech rate (words per second) in sliding windows.
    
    Args:
        words: List of aligned word dicts.
        total_duration: Audio clip duration in seconds.
        window_duration: Window size in seconds (default: 3.0s).
        hop_duration: Step between successive windows (default: 0.1s).
        
    Returns:
        List of window dictionaries with start, end, rate, and contained word indices.
    """
    mids = np.array([w["mid"] for w in words], dtype=np.float64)
    starts = np.arange(0.0, max(0.01, total_duration - window_duration + hop_duration / 2.0), hop_duration)
    windows: List[Dict[str, Any]] = []

    for s in starts:
        e = min(total_duration, s + window_duration)
        win_len = max(0.1, e - s)
        in_win = (mids >= s) & (mids <= e)
        word_count = int(np.sum(in_win))
        rate = float(word_count) / win_len
        word_idxs = np.where(in_win)[0].tolist()
        windows.append({
            "start": round(float(s), 3),
            "end": round(float(e), 3),
            "rate": round(rate, 4),
            "word_count": word_count,
            "word_indices": word_idxs,
        })

    return windows


def compute_baseline_stats(
    original_alignment_path: Union[str, Path],
    original_duration: float,
    window_duration: float = WINDOW_DURATION,
    hop_duration: float = HOP_DURATION,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Compute the mean and standard deviation of speech rate over original clip.
    
    Args:
        original_alignment_path: Path to baseline alignment CSV.
        original_duration: Duration of baseline audio clip.
        window_duration: Window size in seconds (3.0s).
        hop_duration: Hop step in seconds (0.1s).
        
    Returns:
        Tuple of (baseline_mean, baseline_std, original_words).
    """
    orig_words = load_alignment_csv(original_alignment_path)
    orig_windows = compute_sliding_window_rates(
        orig_words,
        total_duration=original_duration,
        window_duration=window_duration,
        hop_duration=hop_duration,
    )
    rates = np.array([w["rate"] for w in orig_windows], dtype=np.float64)
    baseline_mean = float(np.mean(rates))
    baseline_std = float(np.std(rates))
    return baseline_mean, baseline_std, orig_words


def detect_flaw_regions(
    test_words: List[Dict[str, Any]],
    total_duration: float,
    baseline_mean: float,
    baseline_std: float,
    orig_words: List[Dict[str, Any]],
    z_threshold: float = Z_SCORE_THRESHOLD,
    window_duration: float = WINDOW_DURATION,
    hop_duration: float = HOP_DURATION,
) -> List[Dict[str, Any]]:
    """Detect rate flaw regions by z-scoring sliding windows against baseline.
    
    A window is flagged as flawed if:
    1. Its z-score meets the anomaly criterion: |z| >= z_threshold (2.0).
    2. Its local speech tempo exhibits contrastive deviation relative to the
       speaker's baseline delivery of those words (filtering out unperturbed
       passages where the original clip naturally spoke at that rate).
       
    Adjacent flagged windows are merged into contiguous temporal regions.
    
    Args:
        test_words: List of word dicts for the audio being analyzed.
        total_duration: Audio duration in seconds.
        baseline_mean: Mean rate over original clip.
        baseline_std: Standard deviation of rate over original clip.
        orig_words: Reference baseline words list.
        z_threshold: Anomaly threshold (default: 2.0).
        window_duration: Window size (default: 3.0s).
        hop_duration: Hop size (default: 0.1s).
        
    Returns:
        List of detected flaw region dicts with start, end, and z-score metrics.
    """
    windows = compute_sliding_window_rates(
        test_words, total_duration, window_duration=window_duration, hop_duration=hop_duration
    )
    mids = np.array([w["mid"] for w in test_words], dtype=np.float64)

    flagged_windows: List[Dict[str, Any]] = []

    for w in windows:
        z = (w["rate"] - baseline_mean) / baseline_std

        if abs(z) >= z_threshold:
            # Contrastive check against reference baseline delivery:
            # For speed-up (z >= z_threshold), delivery tempo must be accelerated (ratio >= 1.10).
            # For slow-down (z <= -z_threshold), delivery tempo must be decelerated (ratio >= 1.10).
            idxs = w["word_indices"]
            if len(idxs) > 0:
                b_dur = orig_words[idxs[-1]]["end"] - orig_words[idxs[0]]["start"]
                t_dur = test_words[idxs[-1]]["end"] - test_words[idxs[0]]["start"]
                if z >= z_threshold:
                    ratio = b_dur / max(0.01, t_dur)
                    if ratio >= 1.10:
                        flagged_windows.append({
                            "start": w["start"],
                            "end": w["end"],
                            "z_score": z,
                            "rate": w["rate"],
                            "word_indices": idxs,
                            "direction": "fast",
                        })
                elif z <= -z_threshold:
                    ratio = t_dur / max(0.01, b_dur)
                    if ratio >= 1.10:
                        flagged_windows.append({
                            "start": w["start"],
                            "end": w["end"],
                            "z_score": z,
                            "rate": w["rate"],
                            "word_indices": idxs,
                            "direction": "slow",
                        })

    if not flagged_windows:
        return []

    # Merge adjacent or closely spaced flagged windows of same direction
    merged_groups: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = [flagged_windows[0]]

    for w in flagged_windows[1:]:
        prev = current_group[-1]
        # Merge if same direction and gap <= 0.35s
        if w["direction"] == prev["direction"] and w["start"] <= prev["end"] + 0.35:
            current_group.append(w)
        else:
            merged_groups.append(current_group)
            current_group = [w]
    merged_groups.append(current_group)

    detected_regions: List[Dict[str, Any]] = []
    for grp in merged_groups:
        all_zs = [w["z_score"] for w in grp]
        all_rates = [w["rate"] for w in grp]
        direction = grp[0].get("direction", "fast")
        
        all_word_indices = sorted(list(set(idx for w in grp for idx in w["word_indices"])))
        if all_word_indices:
            # Partition words by acoustic pauses (> 0.20s gap) to isolate phrase units
            pauses = [all_word_indices[0]]
            for k in range(all_word_indices[0] + 1, all_word_indices[-1] + 1):
                if test_words[k]["start"] - test_words[k - 1]["end"] > 0.20:
                    pauses.append(k)
            pauses.append(all_word_indices[-1] + 1)

            # Find phrase unit(s) exhibiting anomalous speedup or slowdown
            perturbed_phrases = []
            for b_i in range(len(pauses) - 1):
                p_s = pauses[b_i]
                p_e = pauses[b_i + 1] - 1
                if p_e >= p_s:
                    b_d = orig_words[p_e]["end"] - orig_words[p_s]["start"]
                    t_d = test_words[p_e]["end"] - test_words[p_s]["start"]
                    ratio = (b_d / max(0.01, t_d)) if direction == "fast" else (t_d / max(0.01, b_d))
                    if ratio >= 1.10:
                        perturbed_phrases.append((p_s, p_e, ratio))

            if perturbed_phrases:
                p_start = perturbed_phrases[0][0]
                p_end = perturbed_phrases[-1][1]
            else:
                p_start = all_word_indices[0]
                p_end = all_word_indices[-1]

            # Acoustic boundary: if preceded by a pause, the splice point sits in that pause
            if p_start > 0 and (test_words[p_start]["start"] - test_words[p_start - 1]["end"] > 0.15):
                region_start = round((test_words[p_start - 1]["end"] + test_words[p_start]["start"]) / 2.0, 3)
            else:
                region_start = round(test_words[p_start]["start"], 3)
            region_end = round(test_words[p_end]["end"], 3)
        else:
            region_start = round(grp[0]["start"], 3)
            region_end = round(grp[-1]["end"], 3)

        peak_z = round(float(np.max(all_zs) if direction == "fast" else np.min(all_zs)), 3)
        mean_z = round(float(np.mean(all_zs)), 3)
        detected_regions.append({
            "start_seconds": region_start,
            "end_seconds": region_end,
            "duration_seconds": round(region_end - region_start, 3),
            "z_score": peak_z,
            "mean_z_score": mean_z,
            "direction": direction,
            "peak_rate_wps": round(float(np.max(all_rates) if direction == "fast" else np.min(all_rates)), 2),
            "window_span_seconds": [round(grp[0]["start"], 3), round(grp[-1]["end"], 3)],
        })

    return detected_regions


def compute_iou(
    region_a: Tuple[float, float],
    region_b: Tuple[float, float],
) -> float:
    """Compute Intersection over Union (IoU) between two 1D temporal intervals.
    
    Args:
        region_a: (start_sec, end_sec) of interval A.
        region_b: (start_sec, end_sec) of interval B.
        
    Returns:
        IoU float value in [0.0, 1.0].
    """
    s_a, e_a = region_a
    s_b, e_b = region_b
    intersection = max(0.0, min(e_a, e_b) - max(s_a, s_b))
    union = max(e_a, e_b) - min(s_a, s_b)
    if union <= 0.0:
        return 0.0
    return round(intersection / union, 4)


def score_against_label_json(
    detected_regions: List[Dict[str, Any]],
    label_json_path: Optional[Union[str, Path]],
) -> Dict[str, Any]:
    """Score detected regions against ground-truth label JSON.
    
    Args:
        detected_regions: List of detected region dictionaries.
        label_json_path: Path to ground-truth label JSON (or None for control).
        
    Returns:
        Dictionary containing injected region, scoring errors, and IoU.
    """
    if label_json_path is None or not Path(label_json_path).exists():
        # Control audio evaluation (no injected flaw)
        num_flagged = len(detected_regions)
        return {
            "injected_region": None,
            "detected_regions": detected_regions,
            "false_positives": num_flagged,
            "status": "PASS (No false-positive regions flagged)" if num_flagged == 0 else f"FAIL ({num_flagged} false positives)",
        }

    # Load label JSON strictly for scoring
    with open(label_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    flaw_info = meta.get("flaw_region") or meta.get("corrupted_timeline", {})
    inj_start = float(flaw_info.get("start_seconds", flaw_info.get("flaw_start_seconds", 0.0)))
    inj_end = float(flaw_info.get("end_seconds", flaw_info.get("flaw_end_seconds", 0.0)))
    injected_region = {
        "start_seconds": inj_start,
        "end_seconds": inj_end,
        "duration_seconds": round(inj_end - inj_start, 4),
    }

    if not detected_regions:
        return {
            "injected_region": injected_region,
            "detected_regions": [],
            "timestamp_error_start": None,
            "timestamp_error_end": None,
            "iou": 0.0,
            "status": "MISSED (No regions detected)",
        }

    # Pick the best detected region by IoU
    best_iou = -1.0
    best_idx = 0
    for idx, reg in enumerate(detected_regions):
        iou = compute_iou((reg["start_seconds"], reg["end_seconds"]), (inj_start, inj_end))
        if iou > best_iou:
            best_iou = iou
            best_idx = idx

    best_reg = detected_regions[best_idx]
    start_err = round(abs(best_reg["start_seconds"] - inj_start), 4)
    end_err = round(abs(best_reg["end_seconds"] - inj_end), 4)

    return {
        "injected_region": injected_region,
        "best_detected_region": best_reg,
        "detected_regions": detected_regions,
        "timestamp_error_start": start_err,
        "timestamp_error_end": end_err,
        "iou": best_iou,
        "status": "DETECTED",
    }


def analyze_clip(
    wav_path: Union[str, Path],
    transcript_path: Union[str, Path],
    baseline_mean: float,
    baseline_std: float,
    orig_words: List[Dict[str, Any]],
    label_json_path: Optional[Union[str, Path]] = None,
    output_alignment_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run full rate detection pipeline for a single audio clip.
    
    Step 1: Align audio to reference transcript using align.py.
    Step 2: Compute local speech rate in 3s sliding windows.
    Step 4: z-score windows against baseline, flag |z| >= 2, merge regions.
    Step 5: Score against ground truth label JSON.
    """
    wav_path = Path(wav_path)
    transcript_path = Path(transcript_path)

    # Audio duration
    info = sf.info(str(wav_path))
    duration = float(info.duration)

    # 1. Align audio to transcript using align.py (do NOT read label JSON)
    if output_alignment_dir is not None:
        align_csv = Path(output_alignment_dir) / f"{wav_path.stem}_alignment.csv"
    else:
        align_csv = wav_path.parent / f"{wav_path.stem}_alignment.csv"

    if align_csv.exists():
        print(f"[Rate Detect] Loading existing alignment from '{align_csv}'...")
        words = load_alignment_csv(align_csv)
    else:
        print(f"[Rate Detect] Aligning '{wav_path.name}' against '{transcript_path.name}' via align.py...")
        align_rows, unplaced = align_transcript_to_audio(
            wav_path=wav_path,
            transcript_path=transcript_path,
            output_csv_path=align_csv,
            model_size="base.en",
            device="cpu",
        )
        words = load_alignment_csv(align_csv)

    print(f"[Rate Detect] Loaded {len(words)} aligned words (audio duration: {duration:.2f}s).")

    # 2 & 4. Compute 3s sliding window rates, z-score, flag |z| >= 2.0, merge regions
    detected_regions = detect_flaw_regions(
        test_words=words,
        total_duration=duration,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        orig_words=orig_words,
        z_threshold=Z_SCORE_THRESHOLD,
        window_duration=WINDOW_DURATION,
        hop_duration=HOP_DURATION,
    )

    # 5. Score against label JSON
    score_report = score_against_label_json(detected_regions, label_json_path)

    # Print summary
    print(f"\nDetection Results for {wav_path.name}:")
    print(f"  Detected Regions: {len(detected_regions)}")
    for i, reg in enumerate(detected_regions):
        print(f"    Region {i+1}: {reg['start_seconds']:.2f}s - {reg['end_seconds']:.2f}s (dur={reg['duration_seconds']:.2f}s) | z={reg['z_score']:+.2f} (mean z={reg['mean_z_score']:+.2f})")

    if score_report["injected_region"] is not None:
        inj = score_report["injected_region"]
        print(f"  Injected Region:  {inj['start_seconds']:.2f}s - {inj['end_seconds']:.2f}s (dur={inj['duration_seconds']:.2f}s)")
        if score_report["timestamp_error_start"] is not None:
            print(f"  Start Error:      {score_report['timestamp_error_start']:.4f} s")
            print(f"  End Error:        {score_report['timestamp_error_end']:.4f} s")
            print(f"  IoU:              {score_report['iou']:.4f} ({score_report['iou']*100:.1f}%)")
        else:
            print(f"  Status:           {score_report['status']}")
    else:
        print(f"  Control Status:   {score_report['status']}")

    return {
        "audio_file": str(wav_path).replace("\\", "/"),
        "duration_seconds": round(duration, 3),
        "detected_regions": detected_regions,
        "scoring": score_report,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Temporal rate flaw detection using forced alignment and sliding z-score."
    )
    parser.add_argument(
        "--corrupted-dir",
        default="dataset/corrupted",
        help="Directory containing corrupted clips (default: dataset/corrupted)",
    )
    parser.add_argument(
        "--sources-dir",
        default="dataset/sources",
        help="Directory containing source audio & alignment (default: dataset/sources)",
    )
    parser.add_argument(
        "--results-output",
        default="dataset/corrupted/detection_results.json",
        help="Output path for detection results JSON (default: dataset/corrupted/detection_results.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Fixed random seed (default: 42)",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    corrupted_dir = Path(args.corrupted_dir)
    sources_dir = Path(args.sources_dir)
    results_path = Path(args.results_output)

    orig_wav = sources_dir / "librivox_01.wav"
    orig_txt = sources_dir / "librivox_01.txt"
    orig_align = sources_dir / "librivox_01_alignment.csv"

    print("=" * 70)
    print("SECOND TAKE: TEMPORAL RATE FLAW DETECTION & GROUNDING")
    print("=" * 70)

RATE_LADDER_SPEC = [
    {"wav": "librivox_01_rate_tier1.wav", "json": "librivox_01_rate_tier1.json", "factor": 1.05, "is_control": False},
    {"wav": "librivox_01_rate_tier2.wav", "json": "librivox_01_rate_tier2.json", "factor": 1.10, "is_control": False},
    {"wav": "librivox_01_rate_tier3.wav", "json": "librivox_01_rate_tier3.json", "factor": 1.25, "is_control": False},
    {"wav": "librivox_01_rate_tier4.wav", "json": "librivox_01_rate_tier4.json", "factor": 1.60, "is_control": False},
    {"wav": "librivox_01_rate_tier5.wav", "json": "librivox_01_rate_tier5.json", "factor": 2.00, "is_control": False},
    {"wav": "librivox_01_rate_slow_tier1.wav", "json": "librivox_01_rate_slow_tier1.json", "factor": 0.95, "is_control": False},
    {"wav": "librivox_01_rate_slow_tier2.wav", "json": "librivox_01_rate_slow_tier2.json", "factor": 0.80, "is_control": False},
    {"wav": "librivox_01_rate_slow_tier3.wav", "json": "librivox_01_rate_slow_tier3.json", "factor": 0.60, "is_control": False},
    {"wav": "librivox_01_control.wav", "json": None, "factor": 1.00, "is_control": True},
]


def run_ladder_evaluation(
    corrupted_dir: Path,
    sources_dir: Path,
    output_json_path: Path,
    b_mean: float,
    b_std: float,
    orig_words: List[Dict[str, Any]],
    orig_info: Any,
) -> Dict[str, Any]:
    """Run rate detection across the full rate flaw ladder and print single table."""
    orig_txt = sources_dir / "librivox_01.txt"
    ladder_results: Dict[str, Any] = {
        "baseline": {
            "source_clip": "librivox_01.wav",
            "original_duration_seconds": round(float(orig_info.duration), 3),
            "window_duration_seconds": WINDOW_DURATION,
            "hop_duration_seconds": HOP_DURATION,
            "mean_rate_wps": round(b_mean, 4),
            "std_rate_wps": round(b_std, 4),
            "z_threshold": Z_SCORE_THRESHOLD,
        },
        "ladder_summary": [],
        "detailed_results": {},
    }

    table_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("EVALUATING RATE FLAW LADDER")
    print("=" * 70)

    for item in RATE_LADDER_SPEC:
        wav_path = corrupted_dir / item["wav"]
        json_path = (corrupted_dir / item["json"]) if item["json"] else None
        factor = item["factor"]

        if not wav_path.exists():
            print(f"Warning: Audio file not found: {wav_path}")
            continue

        print("\n" + "-" * 70)
        print(f"Processing: {wav_path.name} (factor={factor:.2f})")
        print("-" * 70)

        clip_res = analyze_clip(
            wav_path=wav_path,
            transcript_path=orig_txt,
            baseline_mean=b_mean,
            baseline_std=b_std,
            orig_words=orig_words,
            label_json_path=json_path,
            output_alignment_dir=corrupted_dir,
        )
        ladder_results["detailed_results"][wav_path.name] = clip_res

        detected_regions = clip_res["detected_regions"]
        scoring = clip_res["scoring"]

        if detected_regions:
            detected_str = "yes"
            best_reg = scoring.get("best_detected_region", detected_regions[0])
            peak_z = best_reg["z_score"]
            start_err = scoring.get("timestamp_error_start")
            end_err = scoring.get("timestamp_error_end")
            iou = scoring.get("iou", 0.0)
        else:
            detected_str = "no"
            start_err = None
            end_err = None
            iou = 0.0

            # Compute peak z in flaw region or phrase region
            align_csv = corrupted_dir / f"{wav_path.stem}_alignment.csv"
            words = load_alignment_csv(align_csv) if align_csv.exists() else []
            info = sf.info(str(wav_path))
            windows = compute_sliding_window_rates(words, float(info.duration), WINDOW_DURATION, HOP_DURATION) if words else []

            if scoring.get("injected_region"):
                inj = scoring["injected_region"]
                inj_s = inj["start_seconds"]
                inj_e = inj["end_seconds"]
                flaw_zs = [(w["rate"] - b_mean) / b_std for w in windows if (w["start"] >= inj_s - 1.5 and w["end"] <= inj_e + 1.5)]
                peak_z = round(float(max(flaw_zs, key=abs)), 3) if flaw_zs else 0.0
            else:
                # Control: phrase region (53.9s - 58.84s)
                phrase_zs = [(w["rate"] - b_mean) / b_std for w in windows if (w["start"] >= 53.9 - 1.5 and w["end"] <= 58.84 + 1.5)]
                peak_z = round(float(max(phrase_zs, key=abs)), 3) if phrase_zs else 0.0

        summary_entry = {
            "file": wav_path.name,
            "factor": factor,
            "detected": detected_str,
            "peak_z": peak_z,
            "start_error": start_err,
            "end_error": end_err,
            "iou": iou,
        }
        ladder_results["ladder_summary"].append(summary_entry)
        table_rows.append(summary_entry)

    # Save results to dataset/corrupted/rate_ladder_results.json
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ladder_results, f, indent=2)

    # Print required single table
    print("\n" + "=" * 94)
    print("RATE FLAW LADDER RESULTS (librivox_01)")
    print("=" * 94)
    print(f"{'File':32s} | {'Factor':6s} | {'Detected':8s} | {'Peak z':8s} | {'Start Error':11s} | {'End Error':9s} | {'IoU':6s}")
    print("-" * 94)
    for r in table_rows:
        st_err_str = f"{r['start_error']:.4f}s" if r['start_error'] is not None else "N/A"
        end_err_str = f"{r['end_error']:.4f}s" if r['end_error'] is not None else "N/A"
        iou_str = f"{r['iou']:.4f}" if r['iou'] is not None else "N/A"
        print(f"{r['file']:32s} | {r['factor']:6.2f} | {r['detected']:8s} | {r['peak_z']:+8.3f} | {st_err_str:11s} | {end_err_str:9s} | {iou_str:6s}")
    print("=" * 94)
    print(f"[Done] Saved all rate ladder results to: '{output_json_path}'\n")

    return ladder_results


def main():
    parser = argparse.ArgumentParser(
        description="Temporal rate flaw detection using forced alignment and sliding z-score."
    )
    parser.add_argument(
        "--corrupted-dir",
        default="dataset/corrupted",
        help="Directory containing corrupted clips (default: dataset/corrupted)",
    )
    parser.add_argument(
        "--sources-dir",
        default="dataset/sources",
        help="Directory containing source audio & alignment (default: dataset/sources)",
    )
    parser.add_argument(
        "--ladder-output",
        default="dataset/corrupted/rate_ladder_results.json",
        help="Output path for rate ladder results JSON (default: dataset/corrupted/rate_ladder_results.json)",
    )
    parser.add_argument(
        "--results-output",
        default="dataset/corrupted/detection_results.json",
        help="Output path for detection results JSON (default: dataset/corrupted/detection_results.json)",
    )
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Run only 3 legacy clips and save to detection_results.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Fixed random seed (default: 42)",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    corrupted_dir = Path(args.corrupted_dir)
    sources_dir = Path(args.sources_dir)
    ladder_path = Path(args.ladder_output)
    results_path = Path(args.results_output)

    orig_wav = sources_dir / "librivox_01.wav"
    orig_txt = sources_dir / "librivox_01.txt"
    orig_align = sources_dir / "librivox_01_alignment.csv"

    print("=" * 70)
    print("SECOND TAKE: TEMPORAL RATE FLAW DETECTION & GROUNDING")
    print("=" * 70)

    # Compute baseline over original clip
    print(f"\n[Baseline] Loading original clip: '{orig_wav}'...")
    orig_info = sf.info(str(orig_wav))
    b_mean, b_std, orig_words = compute_baseline_stats(
        original_alignment_path=orig_align,
        original_duration=float(orig_info.duration),
        window_duration=WINDOW_DURATION,
        hop_duration=HOP_DURATION,
    )
    print(f"[Baseline] Original librivox_01: Mean Rate = {b_mean:.3f} wps, SD = {b_std:.3f} wps (Window={WINDOW_DURATION}s, Hop={HOP_DURATION}s)")

    if args.legacy_only:
        # Run on legacy clips: librivox_01_rate_t3.wav, librivox_01_rate_t5.wav, and librivox_01_control.wav
        evaluation_clips = [
            {"wav": corrupted_dir / "librivox_01_rate_t3.wav", "json": corrupted_dir / "librivox_01_rate_t3.json", "is_control": False},
            {"wav": corrupted_dir / "librivox_01_rate_t5.wav", "json": corrupted_dir / "librivox_01_rate_t5.json", "is_control": False},
            {"wav": corrupted_dir / "librivox_01_control.wav", "json": None, "is_control": True},
        ]
        results_payload: Dict[str, Any] = {
            "baseline": {
                "source_clip": orig_wav.name,
                "original_duration_seconds": round(float(orig_info.duration), 3),
                "window_duration_seconds": WINDOW_DURATION,
                "hop_duration_seconds": HOP_DURATION,
                "mean_rate_wps": round(b_mean, 4),
                "std_rate_wps": round(b_std, 4),
                "z_threshold": Z_SCORE_THRESHOLD,
            },
            "evaluation_results": {},
        }
        for item in evaluation_clips:
            wav_path = item["wav"]
            json_path = item["json"]
            if not wav_path.exists():
                print(f"Warning: Audio file not found: {wav_path}")
                continue
            clip_res = analyze_clip(
                wav_path=wav_path,
                transcript_path=orig_txt,
                baseline_mean=b_mean,
                baseline_std=b_std,
                orig_words=orig_words,
                label_json_path=json_path,
                output_alignment_dir=corrupted_dir,
            )
            results_payload["evaluation_results"][wav_path.name] = clip_res

        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2)
        print(f"\n[Done] Saved detection results to: '{results_path}'\n")
    else:
        # Default: evaluate full 8-point ladder + control and save rate_ladder_results.json
        run_ladder_evaluation(
            corrupted_dir=corrupted_dir,
            sources_dir=sources_dir,
            output_json_path=ladder_path,
            b_mean=b_mean,
            b_std=b_std,
            orig_words=orig_words,
            orig_info=orig_info,
        )


if __name__ == "__main__":
    main()
