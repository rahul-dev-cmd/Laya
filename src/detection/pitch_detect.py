#!/usr/bin/env python3
"""Pitch / intonation flaw detection module for Second Take.

This module implements two-stage temporal anomaly detection for pitch dynamic range compression
(monotone / flat delivery):
1. Stage 1 (Window-level detection):
   - Computes cleaned F0 contour (via Parselmouth / Praat) converted to semitones relative to
     the speaker's median F0.
   - Computes the standard deviation of F0 in 3-second sliding windows (100 ms hop).
   - Establishes the baseline distribution (mean and standard deviation of window F0 SD)
     over the original, uncorrupted librivox_01 clip.
   - Flags windows with z <= -2.0 (flat delivery) and contrastive drop relative to baseline delivery,
     merging overlapping flagged windows into candidate regions.
2. Stage 2 (Per-word pitch spread edge placement):
   - For each aligned word of the corrupted audio, computes its voiced F0 range in semitones (max - min).
   - Z-scores each word's range deviation against the same word's baseline range from the original clip:
     z_word = (test_range - baseline_range) / baseline_std.
   - Inside and around each flagged region, finds the runs of consecutive words whose pitch range is
     clearly compressed (z_word <= -1.5).
   - Uses the first and last run edges as the refined start and end timestamps.
   - Does NOT use pauses and does NOT read the label JSON during detection.
3. Scoring & Evaluation:
   - Evaluates both Window-level IoU (raw merged windows) and Refined IoU (per-word edge placement).
   - Reports start error, end error, and counts control-like (no-flaw) flagged windows across the clip.
   - Updates dataset/corrupted/pitch_ladder_results.json and prints comprehensive summary tables.
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

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer project's virtual environment when running standalone
venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if venv_py.exists() and sys.executable.lower() != str(venv_py).lower():
    import subprocess
    result = subprocess.call([str(venv_py), __file__] + sys.argv[1:])
    sys.exit(result)

from src.features.extract import compute_f0_contour

# Reproducibility seed & sliding window parameters
RANDOM_SEED: int = 42
WINDOW_DURATION: float = 3.0       # 3.0 s sliding window
HOP_DURATION: float = 0.1          # 100 ms step between sliding windows
Z_SCORE_THRESHOLD: float = -2.0    # Window anomaly criterion: z <= -2.0 (flat intonation)
WORD_Z_THRESHOLD: float = -1.5     # Word pitch spread compression criterion: z <= -1.5
MIN_VOICED_FRAMES: int = 30        # At least 300 ms voiced speech in a 3s window

PITCH_LADDER_SPEC = [
    {"wav": "librivox_01_pitch_tier1.wav", "json": "librivox_01_pitch_tier1.json", "k": 0.85, "tier": 1},
    {"wav": "librivox_01_pitch_tier2.wav", "json": "librivox_01_pitch_tier2.json", "k": 0.65, "tier": 2},
    {"wav": "librivox_01_pitch_tier3.wav", "json": "librivox_01_pitch_tier3.json", "k": 0.45, "tier": 3},
    {"wav": "librivox_01_pitch_tier4.wav", "json": "librivox_01_pitch_tier4.json", "k": 0.25, "tier": 4},
    {"wav": "librivox_01_pitch_tier5.wav", "json": "librivox_01_pitch_tier5.json", "k": 0.00, "tier": 5},
    {"wav": "librivox_01_pitch_control.wav", "json": "librivox_01_pitch_control.json", "k": 1.00, "tier": "control"},
]

MIDPHRASE_SPEC = [
    {"wav": "librivox_01_pitch_midphrase_1.wav", "json": "librivox_01_pitch_midphrase_1.json", "k": 0.25, "id": 1, "words": "16-26"},
    {"wav": "librivox_01_pitch_midphrase_2.wav", "json": "librivox_01_pitch_midphrase_2.json", "k": 0.25, "id": 2, "words": "59-70"},
    {"wav": "librivox_01_pitch_midphrase_3.wav", "json": "librivox_01_pitch_midphrase_3.json", "k": 0.25, "id": 3, "words": "140-151"},
    {"wav": "librivox_01_pitch_midphrase_4.wav", "json": "librivox_01_pitch_midphrase_4.json", "k": 0.25, "id": 4, "words": "169-183"},
    {"wav": "librivox_01_pitch_midphrase_5.wav", "json": "librivox_01_pitch_midphrase_5.json", "k": 0.25, "id": 5, "words": "222-233"},
    {"wav": "librivox_01_pitch_midphrase_6.wav", "json": "librivox_01_pitch_midphrase_6.json", "k": 0.25, "id": 6, "words": "265-276"},
]


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed across python and numpy for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def load_alignment_csv(csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load word timestamps from an alignment CSV file."""
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
                        "mid": (start + end) / 2.0,
                    })
            except (ValueError, KeyError):
                continue
    return words


def compute_f0_semitones(
    wav_path: Union[str, Path],
    speaker_median_f0: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Extract cleaned F0 contour and convert to semitones relative to speaker median.
    
    Args:
        wav_path: Path to audio WAV file.
        speaker_median_f0: Optional speaker median F0 in Hz. If None, uses clip median.
        
    Returns:
        Tuple of (frame_times, f0_semitones, clip_median_f0, duration).
    """
    info = sf.info(str(wav_path))
    duration = float(info.duration)
    # Analysis frames: 10 ms hop matching project standards
    frame_times = np.arange(0.0125, duration, 0.01)
    
    f0_hz, f0_st, clip_median = compute_f0_contour(wav_path, frame_times)
    
    # If speaker median is provided, compute semitones relative to speaker baseline median
    ref_median = speaker_median_f0 if speaker_median_f0 is not None else clip_median
    if ref_median > 0:
        voiced = ~np.isnan(f0_hz) & (f0_hz > 0.0)
        f0_semitones = np.full_like(f0_hz, np.nan)
        f0_semitones[voiced] = 12.0 * np.log2(f0_hz[voiced] / ref_median)
    else:
        f0_semitones = f0_st
        
    return frame_times, f0_semitones, clip_median, duration


def compute_sliding_window_f0_sd(
    f0_semitones: np.ndarray,
    frame_times: np.ndarray,
    total_duration: float,
    window_duration: float = WINDOW_DURATION,
    hop_duration: float = HOP_DURATION,
    min_voiced_frames: int = MIN_VOICED_FRAMES,
) -> List[Dict[str, Any]]:
    """Compute local F0 standard deviation in sliding windows.
    
    Args:
        f0_semitones: 1D array of F0 in semitones (NaN for unvoiced).
        frame_times: 1D array of frame center timestamps.
        total_duration: Total audio duration in seconds.
        window_duration: Window length in seconds (default: 3.0s).
        hop_duration: Step between sliding windows (default: 0.1s).
        min_voiced_frames: Minimum voiced frames required in a window (default: 30).
        
    Returns:
        List of window dictionaries with start, end, f0_sd, and voiced_frames count.
    """
    starts = np.arange(0.0, max(0.01, total_duration - window_duration + hop_duration / 2.0), hop_duration)
    windows: List[Dict[str, Any]] = []
    
    for s in starts:
        e = min(total_duration, s + window_duration)
        mask = (frame_times >= s) & (frame_times <= e)
        win_vals = f0_semitones[mask]
        voiced_vals = win_vals[~np.isnan(win_vals)]
        voiced_count = len(voiced_vals)
        
        if voiced_count >= min_voiced_frames:
            sd = float(np.std(voiced_vals))
        else:
            sd = float("nan")
            
        windows.append({
            "start": round(float(s), 3),
            "end": round(float(e), 3),
            "f0_sd": sd,
            "voiced_frames": voiced_count,
        })
        
    return windows


def compute_word_f0_ranges(
    f0_semitones: np.ndarray,
    frame_times: np.ndarray,
    words: List[Dict[str, Any]],
) -> np.ndarray:
    """Compute F0 range in semitones (max - min) for each aligned word.
    
    Args:
        f0_semitones: 1D array of F0 in semitones.
        frame_times: 1D array of frame center timestamps.
        words: List of aligned word dicts.
        
    Returns:
        1D numpy array of word F0 ranges in semitones (0.0 if < 2 voiced frames).
    """
    ranges = []
    for w in words:
        s = float(w["start"])
        e = float(w["end"])
        mask = (frame_times >= s) & (frame_times <= e)
        v = f0_semitones[mask][~np.isnan(f0_semitones[mask])]
        if len(v) >= 2:
            ranges.append(float(np.max(v) - np.min(v)))
        else:
            ranges.append(0.0)
    return np.array(ranges, dtype=np.float64)


def compute_pitch_baseline(
    baseline_wav_path: Union[str, Path],
    alignment_words: Optional[List[Dict[str, Any]]] = None,
    window_duration: float = WINDOW_DURATION,
    hop_duration: float = HOP_DURATION,
) -> Dict[str, Any]:
    """Establish the speaker baseline distribution from the original uncorrupted clip.
    
    Args:
        baseline_wav_path: Path to dataset/sources/librivox_01.wav.
        alignment_words: Optional list of aligned words for baseline word ranges.
        window_duration: Window size (default: 3.0s).
        hop_duration: Hop size (default: 0.1s).
        
    Returns:
        Baseline dictionary with mean, std, speaker_median_f0, per-window baseline SDs,
        and per-word baseline F0 ranges.
    """
    frame_times, f0_st, speaker_med, dur = compute_f0_semitones(baseline_wav_path)
    windows = compute_sliding_window_f0_sd(
        f0_st, frame_times, dur, window_duration=window_duration, hop_duration=hop_duration
    )
    
    sds = np.array([w["f0_sd"] for w in windows], dtype=np.float64)
    valid_mask = ~np.isnan(sds)
    valid_sds = sds[valid_mask]
    
    baseline_mean = float(np.mean(valid_sds))
    baseline_std = float(np.std(valid_sds))
    
    base_word_ranges = None
    if alignment_words:
        base_word_ranges = compute_word_f0_ranges(f0_st, frame_times, alignment_words)
    
    return {
        "wav_path": str(baseline_wav_path),
        "duration": dur,
        "speaker_median_f0": round(speaker_med, 2),
        "mean_f0_sd": round(baseline_mean, 4),
        "std_f0_sd": round(baseline_std, 4),
        "windows": windows,
        "frame_times": frame_times,
        "f0_semitones": f0_st,
        "word_ranges": base_word_ranges,
    }


def find_consecutive_runs(indices: List[int]) -> List[List[int]]:
    """Group sorted list of integer indices into runs of strictly consecutive values."""
    if not indices:
        return []
    runs = []
    curr = [indices[0]]
    for idx in indices[1:]:
        if idx == curr[-1] + 1:
            curr.append(idx)
        else:
            runs.append(curr)
            curr = [idx]
    runs.append(curr)
    return runs


def detect_pitch_flaw_regions(
    test_wav_path: Union[str, Path],
    baseline_stats: Dict[str, Any],
    alignment_words: Optional[List[Dict[str, Any]]] = None,
    z_threshold: float = Z_SCORE_THRESHOLD,
    word_z_threshold: float = WORD_Z_THRESHOLD,
    window_duration: float = WINDOW_DURATION,
    hop_duration: float = HOP_DURATION,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """Two-stage pitch anomaly detector: window flagging followed by per-word edge placement.
    
    Stage 1 (Window flagging):
    1. Computes cleaned F0 in semitones relative to speaker's baseline median F0.
    2. Slides 3.0s windows (100ms hop) and computes local F0 standard deviation.
    3. Flags windows where z <= -2.0 that exhibit contrastive compression relative
       to baseline delivery of that passage.
    4. Merges overlapping / adjacent flagged windows into candidate flaw regions.
    
    Stage 2 (Per-word pitch spread edge placement):
    1. Computes F0 range in semitones for each aligned word of the corrupted audio.
    2. Z-scores each word against its own baseline range: z = (test_range - base_range) / baseline_std.
    3. Inside and around each flagged region, finds runs of consecutive words with z <= -1.5.
    4. Places refined start at the first run's start, and refined end at the last run's end.
    5. Does NOT use pauses and does NOT read the label JSON.
       
    Args:
        test_wav_path: Path to corrupted or control WAV file.
        baseline_stats: Precomputed baseline stats from original clip.
        alignment_words: List of aligned words for Stage 2 per-word edge placement.
        z_threshold: Window anomaly cutoff (default: -2.0).
        word_z_threshold: Per-word range compression cutoff (default: -1.5).
        window_duration: Window size (default: 3.0s).
        hop_duration: Hop size (default: 0.1s).
        
    Returns:
        Tuple of (detected_regions, all_windows, flagged_windows, peak_z).
    """
    b_mean = baseline_stats["mean_f0_sd"]
    b_std = baseline_stats["std_f0_sd"]
    spk_med = baseline_stats["speaker_median_f0"]
    b_wins = baseline_stats["windows"]
    b_sds = [w["f0_sd"] for w in b_wins]
    base_word_ranges = baseline_stats.get("word_ranges")
    
    # 1. Compute test F0 contour
    c_times, test_f0_st, _, dur = compute_f0_semitones(test_wav_path, speaker_median_f0=spk_med)
    
    # 2. Stage 1: 3.0s sliding window F0 standard deviations
    test_wins = compute_sliding_window_f0_sd(
        test_f0_st, c_times, dur, window_duration=window_duration, hop_duration=hop_duration
    )
    
    flagged_windows: List[Dict[str, Any]] = []
    
    for idx, w in enumerate(test_wins):
        tsd = w["f0_sd"]
        if np.isnan(tsd):
            continue
            
        z = (tsd - b_mean) / b_std
        w["z_score"] = round(float(z), 3)
        
        if z <= z_threshold:
            osd = b_sds[idx] if idx < len(b_sds) else np.nan
            # Contrastive check against baseline delivery:
            # (at least 15% reduction in SD and >= 0.4 semitones drop)
            if not np.isnan(osd) and (tsd <= osd * 0.85) and ((osd - tsd) >= 0.4):
                flagged_windows.append(w)

    if not flagged_windows:
        return [], test_wins, [], 0.0

    # Merge overlapping flagged windows: w['start'] <= curr_end + 0.35
    merged_groups: List[List[Dict[str, Any]]] = []
    curr = [flagged_windows[0]]
    curr_end = flagged_windows[0]["end"]

    for w in flagged_windows[1:]:
        if w["start"] <= curr_end + 0.35:
            curr.append(w)
            curr_end = max(curr_end, w["end"])
        else:
            merged_groups.append(curr)
            curr = [w]
            curr_end = w["end"]
    merged_groups.append(curr)

    # Compute test word F0 ranges if alignment is provided
    test_word_ranges = None
    if alignment_words and base_word_ranges is not None:
        test_word_ranges = compute_word_f0_ranges(test_f0_st, c_times, alignment_words)

    detected_regions: List[Dict[str, Any]] = []

    for grp in merged_groups:
        all_zs = [w["z_score"] for w in grp]
        peak_z = round(float(min(all_zs)), 3)
        mean_z = round(float(np.mean(all_zs)), 3)
        
        # Raw merged window bounds
        raw_start = round(grp[0]["start"], 3)
        raw_end = round(max(w["end"] for w in grp), 3)
        
        # Stage 2: per-word pitch spread edge placement
        if alignment_words and test_word_ranges is not None:
            # Search words overlapping or around flagged window region (within 0.5s margin)
            search_words = [
                i for i, w in enumerate(alignment_words)
                if float(w["end"]) >= raw_start - 0.5 and float(w["start"]) <= raw_end + 0.5
            ]
            
            comp_indices = []
            for i in search_words:
                diff = test_word_ranges[i] - base_word_ranges[i]
                z_w = diff / b_std
                if z_w <= word_z_threshold:
                    comp_indices.append(i)
                    
            runs = find_consecutive_runs(comp_indices)
            if runs:
                first_run = runs[0]
                last_run = runs[-1]
                region_start = round(float(alignment_words[first_run[0]]["start"]), 3)
                region_end = round(float(alignment_words[last_run[-1]]["end"]), 3)
            else:
                region_start = raw_start
                region_end = raw_end
        else:
            region_start = raw_start
            region_end = raw_end
            
        detected_regions.append({
            "start_seconds": region_start,
            "end_seconds": region_end,
            "duration_seconds": round(region_end - region_start, 3),
            "z_score": peak_z,
            "mean_z_score": mean_z,
            "flagged_window_count": len(grp),
            "window_span_seconds": [raw_start, raw_end],
        })

    overall_peak_z = min(r["z_score"] for r in detected_regions) if detected_regions else 0.0
    return detected_regions, test_wins, flagged_windows, overall_peak_z


def compute_iou(
    region_a: Tuple[float, float],
    region_b: Tuple[float, float],
) -> float:
    """Compute Intersection over Union (IoU) between two 1D intervals."""
    s_a, e_a = region_a
    s_b, e_b = region_b
    intersection = max(0.0, min(e_a, e_b) - max(s_a, s_b))
    union = max(e_a, e_b) - min(s_a, s_b)
    if union <= 0.0:
        return 0.0
    return round(intersection / union, 4)


def count_no_flaw_flagged_windows(
    flagged_windows: List[Dict[str, Any]],
    flaw_start: float,
    flaw_end: float,
) -> int:
    """Count how many flagged windows do not overlap the injected flaw region at all."""
    count = 0
    for w in flagged_windows:
        if w["end"] <= flaw_start or w["start"] >= flaw_end:
            count += 1
    return count


def score_against_label_json(
    detected_regions: List[Dict[str, Any]],
    label_json_path: Optional[Union[str, Path]],
    flagged_windows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score detected regions against ground-truth label JSON.
    
    Computes both:
    - window_iou: IoU of the raw merged window span.
    - refined_iou: IoU after per-word pitch spread edge placement.
    
    Args:
        detected_regions: List of detected region dictionaries.
        label_json_path: Path to ground-truth label JSON.
        flagged_windows: Optional list of all raw flagged windows across the clip.
        
    Returns:
        Dictionary containing injected region, scoring errors, both IoU metrics,
        and no-flaw flagged window count.
    """
    if label_json_path is None or not Path(label_json_path).exists():
        num_flagged = len(detected_regions)
        no_flaw_count = len(flagged_windows or [])
        return {
            "injected_region": None,
            "detected_regions": detected_regions,
            "false_positives": num_flagged,
            "no_flaw_flagged_windows": no_flaw_count,
            "status": "PASS (No false-positive regions flagged)" if num_flagged == 0 else f"FAIL ({num_flagged} false positives)",
        }

    with open(label_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    flaw_info = meta.get("flaw_region") or meta.get("original_region", {})
    inj_start = float(flaw_info.get("start_seconds", 53.90))
    inj_end = float(flaw_info.get("end_seconds", 58.84))
    
    injected_region = {
        "start_seconds": inj_start,
        "end_seconds": inj_end,
        "duration_seconds": round(inj_end - inj_start, 4),
    }

    no_flaw_count = count_no_flaw_flagged_windows(flagged_windows or [], inj_start, inj_end)

    if not detected_regions:
        return {
            "injected_region": injected_region,
            "detected_regions": [],
            "timestamp_error_start": None,
            "timestamp_error_end": None,
            "window_iou": 0.0,
            "refined_iou": 0.0,
            "iou": 0.0,
            "no_flaw_flagged_windows": no_flaw_count,
            "status": "MISSED (No regions detected)",
        }

    # Best detected region by refined IoU (or window IoU fallback)
    best_iou = -1.0
    best_idx = 0
    for idx, reg in enumerate(detected_regions):
        r_iou = compute_iou((reg["start_seconds"], reg["end_seconds"]), (inj_start, inj_end))
        if r_iou > best_iou:
            best_iou = r_iou
            best_idx = idx

    best_reg = detected_regions[best_idx]
    raw_span = tuple(best_reg.get("window_span_seconds", [best_reg["start_seconds"], best_reg["end_seconds"]]))
    
    window_iou = compute_iou(raw_span, (inj_start, inj_end))
    refined_iou = compute_iou((best_reg["start_seconds"], best_reg["end_seconds"]), (inj_start, inj_end))
    
    start_err = round(abs(best_reg["start_seconds"] - inj_start), 4)
    end_err = round(abs(best_reg["end_seconds"] - inj_end), 4)

    return {
        "injected_region": injected_region,
        "best_detected_region": best_reg,
        "detected_regions": detected_regions,
        "timestamp_error_start": start_err,
        "timestamp_error_end": end_err,
        "window_iou": window_iou,
        "refined_iou": refined_iou,
        "iou": refined_iou,
        "no_flaw_flagged_windows": no_flaw_count,
        "status": "DETECTED",
    }


def evaluate_pitch_ladder(
    sources_dir: Path,
    corrupted_dir: Path,
    output_json_path: Path,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Run pitch detection across all tiers and control, output single table, and save JSON."""
    set_seed(seed)
    
    baseline_wav = sources_dir / "librivox_01.wav"
    alignment_csv = sources_dir / "librivox_01_alignment.csv"
    
    if not baseline_wav.exists():
        raise FileNotFoundError(f"Baseline audio file not found: {baseline_wav}")
        
    print("\n" + "=" * 110)
    print("ESTABLISHING PITCH BASELINE (librivox_01.wav)")
    print("=" * 110)
    
    words = load_alignment_csv(alignment_csv) if alignment_csv.exists() else []
    baseline_stats = compute_pitch_baseline(baseline_wav, alignment_words=words)
    
    b_mean = baseline_stats["mean_f0_sd"]
    b_std = baseline_stats["std_f0_sd"]
    spk_med = baseline_stats["speaker_median_f0"]
    print(f"Speaker Median F0: {spk_med:.2f} Hz")
    print(f"Baseline Window F0 SD: Mean = {b_mean:.4f} semitones, Std = {b_std:.4f} semitones")
    
    ladder_results: Dict[str, Any] = {
        "baseline": {
            "source_clip": "librivox_01.wav",
            "original_duration_seconds": baseline_stats["duration"],
            "window_duration_seconds": WINDOW_DURATION,
            "hop_duration_seconds": HOP_DURATION,
            "speaker_median_f0_hz": spk_med,
            "baseline_mean_f0_sd_semitones": b_mean,
            "baseline_std_f0_sd_semitones": b_std,
            "z_threshold": Z_SCORE_THRESHOLD,
            "word_z_threshold": WORD_Z_THRESHOLD,
        },
        "ladder_summary": [],
        "detailed_results": {},
    }
    
    table_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 110)
    print("EVALUATING PITCH FLAW LADDER (librivox_01)")
    print("=" * 110)

    for item in PITCH_LADDER_SPEC:
        wav_path = corrupted_dir / item["wav"]
        json_path = corrupted_dir / item["json"]
        k = item["k"]

        if not wav_path.exists():
            print(f"Warning: Audio file not found: {wav_path}")
            continue

        # Detect flaws WITHOUT reading label JSON
        detected_regions, all_wins, flagged_wins, _ = detect_pitch_flaw_regions(
            test_wav_path=wav_path,
            baseline_stats=baseline_stats,
            alignment_words=words,
        )

        # Score against label JSON
        scoring = score_against_label_json(detected_regions, json_path, flagged_windows=flagged_wins)

        ladder_results["detailed_results"][wav_path.name] = {
            "audio_file": str(wav_path).replace("\\", "/"),
            "k": k,
            "detected_regions": detected_regions,
            "scoring": scoring,
        }

        if detected_regions:
            detected_str = "yes"
            best_reg = scoring.get("best_detected_region", detected_regions[0])
            peak_z = best_reg["z_score"]
            start_err = scoring.get("timestamp_error_start")
            end_err = scoring.get("timestamp_error_end")
            w_iou = scoring.get("window_iou", 0.0)
            r_iou = scoring.get("refined_iou", 0.0)
            no_flaw_count = scoring.get("no_flaw_flagged_windows", 0)
        else:
            detected_str = "no"
            start_err = None
            end_err = None
            w_iou = 0.0
            r_iou = 0.0
            no_flaw_count = scoring.get("no_flaw_flagged_windows", 0)

            # Compute phrase z-score for undetected / control clips (53.90s - 58.84s)
            clip_times, test_f0_st, _, _ = compute_f0_semitones(wav_path, speaker_median_f0=spk_med)
            p_mask = (clip_times >= 53.90) & (clip_times <= 58.84)
            pv = test_f0_st[p_mask][~np.isnan(test_f0_st[p_mask])]
            phrase_sd = float(np.std(pv)) if len(pv) > 0 else 0.0
            peak_z = round(float((phrase_sd - b_mean) / b_std), 3)

        summary_entry = {
            "file": wav_path.name,
            "k": k,
            "detected": detected_str,
            "peak_z": peak_z,
            "window_iou": w_iou,
            "refined_iou": r_iou,
            "start_error": start_err,
            "end_error": end_err,
            "no_flaw_flagged_windows": no_flaw_count,
        }
        ladder_results["ladder_summary"].append(summary_entry)
        table_rows.append(summary_entry)

    # Save to dataset/corrupted/pitch_ladder_results.json
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ladder_results, f, indent=2)
    print(f"\n[Saved] Detailed pitch ladder results written to: {output_json_path}")

    # Print summary table
    print("\n" + "=" * 118)
    print("PITCH FLAW LADDER RESULTS (librivox_01)")
    print("=" * 118)
    header = (
        f"{'File':30s} | {'k':5s} | {'Detected':8s} | {'Peak z':8s} | "
        f"{'Window IoU':10s} | {'Refined IoU':11s} | {'Start Error':11s} | {'End Error':9s} | {'No-Flaw Flagged':15s}"
    )
    print(header)
    print("-" * len(header))
    for row in table_rows:
        s_err_str = f"{row['start_error']:.4f}s" if row['start_error'] is not None else "N/A"
        e_err_str = f"{row['end_error']:.4f}s" if row['end_error'] is not None else "N/A"
        z_str = f"{row['peak_z']:+.3f}"
        print(
            f"{row['file']:30s} | {row['k']:<5.2f} | {row['detected']:8s} | {z_str:8s} | "
            f"{row['window_iou']:<10.4f} | {row['refined_iou']:<11.4f} | {s_err_str:11s} | {e_err_str:9s} | {row['no_flaw_flagged_windows']:<15d}"
        )
    print("=" * 118 + "\n")

    return ladder_results


def evaluate_midphrase_tests(
    sources_dir: Path,
    corrupted_dir: Path,
    seed: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """Evaluate 6 mid-phrase flaw test clips (continuous speech, not bounded by pauses)."""
    set_seed(seed)
    
    baseline_wav = sources_dir / "librivox_01.wav"
    alignment_csv = sources_dir / "librivox_01_alignment.csv"
    
    words = load_alignment_csv(alignment_csv) if alignment_csv.exists() else []
    baseline_stats = compute_pitch_baseline(baseline_wav, alignment_words=words)

    table_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 132)
    print("MID-PHRASE PITCH FLAW TEST RESULTS (k=0.25, continuous speech, NOT bounded by pauses)")
    print("=" * 132)

    for item in MIDPHRASE_SPEC:
        wav_path = corrupted_dir / item["wav"]
        json_path = corrupted_dir / item["json"]
        k = item["k"]

        if not wav_path.exists():
            print(f"Warning: Audio file not found: {wav_path}")
            continue

        detected_regions, all_wins, flagged_wins, peak_z = detect_pitch_flaw_regions(
            test_wav_path=wav_path,
            baseline_stats=baseline_stats,
            alignment_words=words,
        )

        scoring = score_against_label_json(detected_regions, json_path, flagged_windows=flagged_wins)
        inj = scoring.get("injected_region", {})
        inj_str = f"[{inj.get('start_seconds', 0.0):.2f}s, {inj.get('end_seconds', 0.0):.2f}s]"

        if detected_regions:
            detected_str = "yes"
            best_reg = scoring.get("best_detected_region", detected_regions[0])
            peak_z = best_reg["z_score"]
            start_err = scoring.get("timestamp_error_start")
            end_err = scoring.get("timestamp_error_end")
            w_iou = scoring.get("window_iou", 0.0)
            r_iou = scoring.get("refined_iou", 0.0)
            no_flaw_count = scoring.get("no_flaw_flagged_windows", 0)
        else:
            detected_str = "no"
            start_err = None
            end_err = None
            w_iou = 0.0
            r_iou = 0.0
            no_flaw_count = scoring.get("no_flaw_flagged_windows", 0)

        row_entry = {
            "file": wav_path.name,
            "words": item["words"],
            "ground_truth": inj_str,
            "k": k,
            "detected": detected_str,
            "peak_z": peak_z,
            "window_iou": w_iou,
            "refined_iou": r_iou,
            "start_error": start_err,
            "end_error": end_err,
            "no_flaw_flagged_windows": no_flaw_count,
        }
        table_rows.append(row_entry)

    header = (
        f"{'File':34s} | {'Words':7s} | {'Ground Truth':16s} | {'Detected':8s} | {'Peak z':8s} | "
        f"{'Window IoU':10s} | {'Refined IoU':11s} | {'Start Error':11s} | {'End Error':9s} | {'No-Flaw Flagged':15s}"
    )
    print(header)
    print("-" * len(header))
    for row in table_rows:
        s_err_str = f"{row['start_error']:.4f}s" if row['start_error'] is not None else "N/A"
        e_err_str = f"{row['end_error']:.4f}s" if row['end_error'] is not None else "N/A"
        z_str = f"{row['peak_z']:+.3f}"
        print(
            f"{row['file']:34s} | {row['words']:7s} | {row['ground_truth']:16s} | {row['detected']:8s} | {z_str:8s} | "
            f"{row['window_iou']:<10.4f} | {row['refined_iou']:<11.4f} | {s_err_str:11s} | {e_err_str:9s} | {row['no_flaw_flagged_windows']:<15d}"
        )
    print("=" * 132 + "\n")

    return table_rows


def main():
    parser = argparse.ArgumentParser(
        description="Detect pitch dynamic range compression (flat / monotone delivery) flaws in speech clips."
    )
    parser.add_argument(
        "--sources-dir",
        default="dataset/sources",
        help="Path to dataset sources directory (default: dataset/sources)",
    )
    parser.add_argument(
        "--corrupted-dir",
        default="dataset/corrupted",
        help="Path to corrupted audio directory (default: dataset/corrupted)",
    )
    parser.add_argument(
        "--output-json",
        default="dataset/corrupted/pitch_ladder_results.json",
        help="Output JSON path (default: dataset/corrupted/pitch_ladder_results.json)",
    )
    parser.add_argument(
        "--evaluate-ladder",
        action="store_true",
        default=True,
        help="Evaluate full pitch flaw ladder (5 tiers + control)",
    )
    parser.add_argument(
        "--evaluate-midphrase",
        action="store_true",
        default=True,
        help="Evaluate 6 mid-phrase flaw test clips (not bounded by pauses)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    sources_dir = Path(args.sources_dir)
    corrupted_dir = Path(args.corrupted_dir)
    output_json = Path(args.output_json)

    # 1. Pitch flaw ladder evaluation
    evaluate_pitch_ladder(
        sources_dir=sources_dir,
        corrupted_dir=corrupted_dir,
        output_json_path=output_json,
        seed=args.seed,
    )

    # 2. Mid-phrase flaw test evaluation
    evaluate_midphrase_tests(
        sources_dir=sources_dir,
        corrupted_dir=corrupted_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
