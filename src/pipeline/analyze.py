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
import difflib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faster_whisper
import numpy as np
import soundfile as sf

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from align import align_transcript_to_audio, clean_token
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


_CACHED_WHISPER_MODELS: Dict[Tuple[str, str, str], faster_whisper.WhisperModel] = {}


def get_whisper_model(
    model_size: str = "small.en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> faster_whisper.WhisperModel:
    """Get or instantiate cached Whisper model."""
    global _CACHED_WHISPER_MODELS
    key = (model_size, device, compute_type)
    if key not in _CACHED_WHISPER_MODELS:
        _CACHED_WHISPER_MODELS[key] = faster_whisper.WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
    return _CACHED_WHISPER_MODELS[key]


def transcribe_audio_whisper(
    wav_path: Union[str, Path],
    model_size: str = "small.en",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    seed: int = RANDOM_SEED,
    initial_prompt: str = "Um, uh, so, I mean, like, you know.",
    condition_on_previous_text: bool = False,
) -> Tuple[List[Dict[str, Any]], str, Path]:
    """Transcribe audio with faster-whisper and word timestamps.
    
    Returns:
        Tuple of (words_list, full_transcript_text, alignment_csv_path).
    """
    wav_path = Path(wav_path).resolve()
    set_seed(seed)
    model = get_whisper_model(model_size=model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(wav_path),
        word_timestamps=True,
        beam_size=beam_size,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
    )

    whisper_words: List[Dict[str, Any]] = []
    for s in segments:
        if s.words:
            for w in s.words:
                clean_w = w.word.strip()
                if clean_w:
                    conf = (
                        round(float(w.probability), 3)
                        if hasattr(w, "probability") and w.probability is not None
                        else 1.0
                    )
                    whisper_words.append({
                        "word": clean_w,
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3),
                        "confidence": conf,
                    })

    transcript_text = " ".join(w["word"] for w in whisper_words)

    # Save alignment CSV so downstream feature extractors can read it
    out_csv = wav_path.parent / f"{wav_path.stem}_auto_alignment.csv"
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "start_seconds", "end_seconds", "confidence"])
            for w in whisper_words:
                writer.writerow([w["word"], f"{w['start']:.3f}", f"{w['end']:.3f}", f"{w['confidence']:.3f}"])
    except (PermissionError, OSError):
        out_csv = Path(tempfile.gettempdir()) / f"{wav_path.stem}_auto_alignment.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "start_seconds", "end_seconds", "confidence"])
            for w in whisper_words:
                writer.writerow([w["word"], f"{w['start']:.3f}", f"{w['end']:.3f}", f"{w['confidence']:.3f}"])

    words_loaded = []
    for idx, w in enumerate(whisper_words):
        words_loaded.append({
            "index": idx,
            "word": w["word"],
            "start": w["start"],
            "end": w["end"],
            "confidence": w["confidence"],
            "mid": (w["start"] + w["end"]) / 2.0,
        })
    return words_loaded, transcript_text, out_csv


def map_words_difflib(
    participant_words: List[Dict[str, Any]],
    reference_words: List[Dict[str, Any]],
) -> Tuple[Dict[int, int], float]:
    """Map participant word indices to reference word indices via difflib.
    
    Returns:
        Tuple of (p_to_r_map, word_match_rate).
    """
    p_clean = [clean_token(w["word"]) for w in participant_words if clean_token(w["word"])]
    r_clean = [clean_token(w["word"]) for w in reference_words if clean_token(w["word"])]

    p_indices = [i for i, w in enumerate(participant_words) if clean_token(w["word"])]
    r_indices = [j for j, w in enumerate(reference_words) if clean_token(w["word"])]

    matcher = difflib.SequenceMatcher(None, p_clean, r_clean)
    p_to_r: Dict[int, int] = {}
    matched_count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                p_idx = p_indices[i1 + offset]
                r_idx = r_indices[j1 + offset]
                p_to_r[p_idx] = r_idx
                matched_count += 1

    total_tokens = len(p_clean)
    match_rate = round(float(matched_count) / max(total_tokens, 1), 4)
    return p_to_r, match_rate


def get_reference_time_for_participant_window(
    p_win_start: float,
    p_win_end: float,
    participant_words: List[Dict[str, Any]],
    reference_words: List[Dict[str, Any]],
    p_to_r: Dict[int, int],
) -> Optional[Tuple[float, float]]:
    """Map participant time window to reference time window by word position."""
    if not p_to_r or not participant_words or not reference_words:
        return None

    overlapping = [
        i for i, w in enumerate(participant_words)
        if w["end"] > p_win_start and w["start"] < p_win_end
    ]
    if not overlapping:
        mid = (p_win_start + p_win_end) / 2.0
        closest_i = min(
            range(len(participant_words)),
            key=lambda i: abs((participant_words[i]["start"] + participant_words[i]["end"]) / 2.0 - mid),
        )
        overlapping = [closest_i]

    mapped_r = [p_to_r[i] for i in overlapping if i in p_to_r]
    if not mapped_r:
        return None

    r_start = reference_words[min(mapped_r)]["start"]
    r_end = reference_words[max(mapped_r)]["end"]
    return r_start, r_end


def build_mapped_orig_words(
    participant_words: List[Dict[str, Any]],
    orig_words: List[Dict[str, Any]],
    p_to_r: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Build word-position mapped reference words matching participant word indices."""
    if not participant_words or not orig_words:
        return orig_words

    mapped = []
    last_r_idx = 0
    for i, pw in enumerate(participant_words):
        if i in p_to_r:
            r_idx = p_to_r[i]
            last_r_idx = r_idx
            mapped.append(dict(orig_words[r_idx]))
        else:
            clamped_idx = min(last_r_idx, len(orig_words) - 1)
            mapped.append(dict(orig_words[clamped_idx]))
    return mapped


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
    model_size: str = "small.en",
) -> Tuple[List[Dict[str, Any]], Path]:
    """Find existing alignment CSV on disk or run forced alignment.
    
    Preserves all existing alignment files without modification.
    
    Args:
        wav_path: Path to speech audio WAV.
        transcript_path: Path to reference transcript TXT.
        model_size: Whisper model size for forced alignment.
        
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
        model_size=model_size,
    )
    words = load_alignment_csv(out_csv)
    # If aligned_rows has confidence, attach it to words
    for i, w in enumerate(words):
        if i < len(aligned_rows) and "confidence" in aligned_rows[i]:
            w["confidence"] = aligned_rows[i]["confidence"]
        else:
            w["confidence"] = 1.0
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
    transcript_txt: Optional[Union[str, Path]] = None,
    reference_wav: Optional[Union[str, Path]] = None,
    seed: int = RANDOM_SEED,
    whisper_model: str = "small.en",
    transcript_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full speech analysis and temporal flaw detection for dashboard visualization.
    
    Args:
        audio_wav: Path to participant speech recording WAV file.
        transcript_txt: Optional path to reference transcript text file.
                        If None or empty, transcribed automatically via Whisper.
        reference_wav: Optional path to paired reference speech WAV file.
                       If provided, mode is "paired"; if None, mode is "experimental".
        seed: Random seed for deterministic reproducibility (default: 42).
        whisper_model: Whisper model size for automatic transcription (default: small.en).
        transcript_source: Optional label for transcript provenance (auto, provided, edited).
        
    Returns:
        Dictionary conforming to Laya dashboard data contract:
        - duration: float
        - transcript_source: "provided", "auto", or "edited"
        - transcript: str
        - word_match_rate: Optional[float]
        - words: list of {"word": str, "start": float, "end": float, "confidence": float}
        - curves: {
            "participant": {time, f0_semitones, energy_z, speech_rate},
            "reference": {time, f0_semitones, energy_z, speech_rate} or None,
            "timestamps", "f0_semitones", "energy_z", "speech_rate" (convenience)
          }
        - flaw_regions: list of {type, start, end, peak_z, severity, explanation, ...}
        - mode: "paired" or "experimental"
        - mode_note: Optional[str]
    """
    set_seed(seed)

    audio_path = Path(audio_wav).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Participant audio file not found: {audio_path}")

    # Determine whether a non-empty transcript file was supplied
    has_transcript = False
    transcript_path: Optional[Path] = None
    if transcript_txt is not None and str(transcript_txt).strip() != "":
        cand_t = Path(transcript_txt).resolve()
        if cand_t.exists() and cand_t.is_file():
            with open(cand_t, "r", encoding="utf-8") as f:
                t_content = f.read().strip()
            if t_content:
                has_transcript = True
                transcript_path = cand_t

    # Determine active transcript source
    if transcript_source is not None:
        active_transcript_source = transcript_source
    elif has_transcript and transcript_path is not None:
        active_transcript_source = "provided"
    else:
        active_transcript_source = "auto"

    # 1. Get participant audio metadata and word alignment
    info = sf.info(str(audio_path))
    participant_duration = round(float(info.duration), 3)

    if has_transcript and transcript_path is not None:
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()
        participant_words_raw, participant_align_csv = find_or_compute_alignment(
            audio_path, transcript_path, model_size=whisper_model
        )
    else:
        (
            participant_words_raw,
            transcript_text,
            participant_align_csv,
        ) = transcribe_audio_whisper(
            audio_path,
            model_size=whisper_model,
            seed=seed,
        )

    words_list: List[Dict[str, Any]] = [
        {
            "word": w["word"],
            "start": round(float(w["start"]), 3),
            "end": round(float(w["end"]), 3),
            "confidence": round(float(w.get("confidence", 1.0)), 3),
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
    p_to_r: Dict[int, int] = {}
    word_match_rate: Optional[float] = None
    mode_note: Optional[str] = None
    is_paired = False
    mode = "experimental"

    if reference_wav is not None:
        ref_audio_path = Path(reference_wav).resolve()
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")

        ref_info = sf.info(str(ref_audio_path))
        ref_duration = round(float(ref_info.duration), 3)

        if has_transcript and transcript_path is not None:
            ref_words_raw, ref_align_csv = find_or_compute_alignment(
                ref_audio_path, transcript_path, model_size=whisper_model
            )
            p_to_r, word_match_rate = map_words_difflib(
                participant_words_raw, ref_words_raw
            )
            is_paired = True
            mode = "paired"
        else:
            # Paired mode without a transcript: transcribe reference as well
            (
                ref_words_raw,
                _,
                ref_align_csv,
            ) = transcribe_audio_whisper(
                ref_audio_path,
                model_size=whisper_model,
                seed=seed,
            )
            # Map participant words to reference words by sequence alignment (difflib)
            p_to_r, word_match_rate = map_words_difflib(
                participant_words_raw, ref_words_raw
            )
            if word_match_rate < 0.60:
                is_paired = False
                mode = "experimental"
                mode_note = (
                    f"Word match rate was {word_match_rate * 100:.1f}% (< 60.0%). "
                    f"Switched from Paired to Experimental mode."
                )
            else:
                is_paired = True
                mode = "paired"
                mode_note = None

        if ref_align_csv is not None and ref_duration is not None:
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
        # Map reference words to participant word positions for contrastive check
        mapped_orig = build_mapped_orig_words(participant_words_raw, orig_words, p_to_r)
        detected_rate = detect_flaw_regions(
            test_words=participant_words_raw,
            total_duration=participant_duration,
            baseline_mean=b_mean,
            baseline_std=b_std,
            orig_words=mapped_orig,
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
            if p_to_r and ref_words_raw:
                ref_time = get_reference_time_for_participant_window(
                    peak_win_s, peak_win_e, participant_words_raw, ref_words_raw, p_to_r
                )
                if ref_time is not None:
                    r_s, r_e = ref_time
                    r_mid = (r_s + r_e) / 2.0
                    closest_rw = min(ref_rate_wins, key=lambda rw: abs((rw["start"] + rw["end"]) / 2.0 - r_mid))
                    ref_val = round(float(closest_rw["rate"]), 2)
            if ref_val is None:
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
            if p_to_r and ref_words_raw:
                ref_time = get_reference_time_for_participant_window(
                    peak_win_s, peak_win_e, participant_words_raw, ref_words_raw, p_to_r
                )
                if ref_time is not None:
                    r_s, r_e = ref_time
                    r_mid = (r_s + r_e) / 2.0
                    cand_bw = [bw for bw in pitch_base["windows"] if not np.isnan(bw["f0_sd"])]
                    if cand_bw:
                        closest_bw = min(cand_bw, key=lambda bw: abs((bw["start"] + bw["end"]) / 2.0 - r_mid))
                        ref_val = round(float(closest_bw["f0_sd"]), 3)
            if ref_val is None:
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
        "mode_note": mode_note,
        "duration": participant_duration,
        "audio_file": str(audio_path).replace("\\", "/"),
        "reference_file": str(ref_audio_path).replace("\\", "/") if ref_audio_path else None,
        "transcript_source": active_transcript_source,
        "transcript": transcript_text,
        "word_match_rate": word_match_rate,
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
    parser.add_argument("--transcript", default=None, help="Path to transcript TXT file (optional)")
    parser.add_argument("--reference", default=None, help="Path to reference audio WAV file")
    parser.add_argument("--output", default=None, help="Path to save output JSON file")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: 42)")
    parser.add_argument(
        "--model",
        default="small.en",
        choices=["tiny.en", "base.en", "small.en", "medium.en"],
        help="Whisper model size (default: small.en)",
    )
    parser.add_argument(
        "--transcript-source",
        default=None,
        help="Transcript provenance label (provided, auto, edited)",
    )

    args = parser.parse_args()

    result = analyze(
        audio_wav=args.audio,
        transcript_txt=args.transcript,
        reference_wav=args.reference,
        seed=args.seed,
        whisper_model=args.model,
        transcript_source=args.transcript_source,
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
