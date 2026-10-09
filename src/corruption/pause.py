#!/usr/bin/env python3
"""Pause flaw corruption module for Second Take.

This module implements silence scaling at phrase boundaries:
1. Identifies natural phrase boundaries (pauses >= 150 ms) within a chosen 3-4 s region.
2. Scales pause duration in two directions:
   - "rushed" (shorter): tiers 1-5 (multipliers: 0.85, 0.65, 0.40, 0.20, 0.00)
   - "draggy" (longer): tiers 1-5 (multipliers: 1.15, 1.40, 1.80, 2.50, 3.00)
   - "control" (neutral): multiplier 1.00
3. Lengthened pauses are filled with the recording's own ambient room noise (not digital silence).
4. Uses 10 ms linear crossfades at all audio cut splices to prevent acoustic clicks.
5. Saves label JSON with flaw region specified in the corrupted file's shifted timeline.
6. Verifies by re-running Whisper forced alignment and printing max difference against label timestamps.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.config import (
    PAUSE_MIN_DURATION_SECONDS,
    CROSSFADE_MS,
    RUSHED_TIERS,
    DRAGGY_TIERS,
    CONTROL_MULTIPLIER,
    CLIP_01_CHOSEN_WORDS,
    CLIP_01_TARGET_PAUSE,
    CLIP_02_CHOSEN_WORDS,
    CLIP_02_TARGET_PAUSE,
)
from align import align_transcript_to_audio

RANDOM_SEED: int = 42
TARGET_SAMPLE_RATE: int = 16000


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed across python and numpy for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def load_alignment_words(alignment_csv_path: Path) -> List[Dict[str, Any]]:
    """Load word timestamps from an alignment CSV file."""
    words: List[Dict[str, Any]] = []
    with open(alignment_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
                w = row["word"].strip()
                if end > start:
                    words.append({"index": idx, "word": w, "start": start, "end": end})
            except (ValueError, KeyError):
                continue
    return words


def extract_room_noise(
    audio: np.ndarray,
    start_sample: int,
    end_sample: int,
    needed_samples: int,
    fade_samples: int,
) -> np.ndarray:
    """Extract ambient room noise from within a pause and tile/extend if needed."""
    # Guard margins away from speech edges
    interior_start = start_sample + fade_samples
    interior_end = end_sample - fade_samples

    if interior_end > interior_start + fade_samples:
        noise_chunk = audio[interior_start:interior_end]
    else:
        noise_chunk = audio[start_sample:end_sample]

    if len(noise_chunk) == 0:
        return np.zeros(needed_samples, dtype=np.float32)

    if needed_samples <= len(noise_chunk):
        return noise_chunk[:needed_samples].copy()

    # Tile with 10 ms crossfade to prevent looping clicks
    out = [noise_chunk]
    curr_len = len(noise_chunk)
    w_fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

    while curr_len < needed_samples:
        prev = out[-1]
        next_chunk = noise_chunk.copy()
        # Crossfade tail of prev with head of next
        overlap = prev[-fade_samples:] * (1.0 - w_fade) + next_chunk[:fade_samples] * w_fade
        out[-1] = prev[:-fade_samples]
        out.append(overlap)
        out.append(next_chunk[fade_samples:])
        curr_len = sum(len(x) for x in out)

    concatenated = np.concatenate(out)
    return concatenated[:needed_samples]


def crossfade_splice_pause(
    audio: np.ndarray,
    p_start_sample: int,
    p_end_sample: int,
    multiplier: float,
    sample_rate: int = TARGET_SAMPLE_RATE,
    crossfade_ms: float = CROSSFADE_MS,
) -> Tuple[np.ndarray, int, int]:
    """Splice audio scaling the pause between p_start_sample and p_end_sample.
    
    Returns:
        Tuple of (spliced_audio, new_pause_start_sample, new_pause_end_sample)
    """
    fade_samples = int(round((crossfade_ms / 1000.0) * sample_rate))
    orig_pause_samples = p_end_sample - p_start_sample
    new_pause_samples = int(round(multiplier * orig_pause_samples))

    if multiplier == 1.0:
        # No modification
        return audio.copy(), p_start_sample, p_end_sample

    w = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

    if multiplier == 0.0:
        # Tier 5 rushed: Complete pause removal.
        # Crossfade the end of preceding word with the start of following word.
        part_before = audio[:p_start_sample - fade_samples]
        cross = audio[p_start_sample - fade_samples : p_start_sample] * (1.0 - w) + \
                audio[p_end_sample : p_end_sample + fade_samples] * w
        part_after = audio[p_end_sample + fade_samples:]
        spliced = np.concatenate([part_before, cross, part_after])
        new_start = p_start_sample - fade_samples // 2
        new_end = new_start  # 0 duration pause
        return spliced, new_start, new_end

    elif multiplier < 1.0:
        # Rushed (shorter pause): Keep first new_pause_samples of the pause
        # Crossfade the end of shortened pause into following audio
        part_before = audio[:p_start_sample]
        kept_pause = audio[p_start_sample : p_start_sample + new_pause_samples]

        pause_body = kept_pause[:-fade_samples] if len(kept_pause) > fade_samples else np.array([], dtype=np.float32)
        pause_tail = kept_pause[-fade_samples:] if len(kept_pause) >= fade_samples else kept_pause

        # If kept_pause is smaller than fade_samples, adapt fade length
        cur_fade = min(len(pause_tail), fade_samples)
        cur_w = np.linspace(0.0, 1.0, cur_fade, dtype=np.float32)

        cross = pause_tail[-cur_fade:] * (1.0 - cur_w) + audio[p_end_sample : p_end_sample + cur_fade] * cur_w
        part_after = audio[p_end_sample + cur_fade:]

        spliced = np.concatenate([part_before, pause_body, cross, part_after])
        new_start = p_start_sample
        new_end = p_start_sample + new_pause_samples
        return spliced, new_start, new_end

    else:
        # Draggy (longer pause): Extend with recording's own room noise
        added_samples = new_pause_samples - orig_pause_samples
        room_noise = extract_room_noise(
            audio, p_start_sample, p_end_sample, added_samples + fade_samples, fade_samples
        )

        # Crossfade added noise into the pause body
        part_before = audio[:p_start_sample]
        orig_pause = audio[p_start_sample:p_end_sample]

        # Insert room noise in the middle of pause with crossfade
        insert_idx = len(orig_pause) // 2
        pause_h1 = orig_pause[:insert_idx]
        pause_h2 = orig_pause[insert_idx:]

        # Crossfade into added noise
        cross1 = pause_h1[-fade_samples:] * (1.0 - w) + room_noise[:fade_samples] * w
        noise_body = room_noise[fade_samples : added_samples]
        cross2 = room_noise[added_samples : added_samples + fade_samples] * (1.0 - w) + pause_h2[:fade_samples] * w

        spliced = np.concatenate([
            part_before,
            pause_h1[:-fade_samples],
            cross1,
            noise_body,
            cross2,
            pause_h2[fade_samples:],
            audio[p_end_sample:],
        ])
        new_start = p_start_sample
        new_end = p_start_sample + new_pause_samples
        return spliced, new_start, new_end


def apply_pause_corruption(
    source_wav_path: Path,
    source_alignment_csv: Path,
    output_dir: Path,
    direction: str,
    tier: Any,
    multiplier: float,
    target_pause_words: Tuple[int, int],
    chosen_region_words: Tuple[int, int],
    stem: str,
    seed: int = RANDOM_SEED,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Apply pause duration scaling, generate shifted alignment CSV, and save label JSON."""
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(str(source_wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    words = load_alignment_words(source_alignment_csv)
    w_before_idx, w_after_idx = target_pause_words
    w_before = words[w_before_idx]
    w_after = words[w_after_idx]

    p_orig_start = float(w_before["end"])
    p_orig_end = float(w_after["start"])
    orig_pause_dur = round(p_orig_end - p_orig_start, 4)

    p_start_samp = int(round(p_orig_start * sr))
    p_end_samp = int(round(p_orig_end * sr))

    # Spliced audio
    spliced_audio, new_start_samp, new_end_samp = crossfade_splice_pause(
        audio, p_start_samp, p_end_samp, multiplier, sample_rate=sr
    )

    new_pause_start = round(new_start_samp / sr, 4)
    new_pause_end = round(new_end_samp / sr, 4)
    new_pause_dur = round(new_pause_end - new_pause_start, 4)
    time_shift = round(new_pause_dur - orig_pause_dur, 4)

    # Output paths
    out_wav_path = output_dir / f"{stem}.wav"
    out_json_path = output_dir / f"{stem}.json"
    out_csv_path = output_dir / f"{stem}_alignment.csv"

    sf.write(str(out_wav_path), spliced_audio, sr, subtype="PCM_16")

    # Shifted word timestamps in corrupted timeline
    corrupted_words: List[Dict[str, Any]] = []
    for w in words:
        idx = w["index"]
        if idx <= w_before_idx:
            corrupted_words.append({
                "word": w["word"],
                "start_seconds": round(float(w["start"]), 4),
                "end_seconds": round(float(w["end"]), 4),
            })
        else:
            corrupted_words.append({
                "word": w["word"],
                "start_seconds": round(float(w["start"]) + time_shift, 4),
                "end_seconds": round(float(w["end"]) + time_shift, 4),
            })

    # Save corrupted alignment CSV
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "start_seconds", "end_seconds"])
        writer.writeheader()
        writer.writerows(corrupted_words)

    # Chosen 3-4s region in corrupted timeline
    ch_start_idx, ch_end_idx = chosen_region_words
    ch_start_sec = corrupted_words[ch_start_idx]["start_seconds"]
    ch_end_sec = corrupted_words[ch_end_idx]["end_seconds"]
    ch_dur = round(ch_end_sec - ch_start_sec, 4)

    # Label JSON metadata
    label_meta = {
        "source_clip": source_wav_path.name,
        "source_audio_path": str(source_wav_path).replace("\\", "/"),
        "flaw_type": "pause",
        "direction": direction,
        "tier": tier,
        "multiplier": multiplier,
        "target_boundary": {
            "word_before_index": w_before_idx,
            "word_before": w_before["word"],
            "word_after_index": w_after_idx,
            "word_after": w_after["word"],
            "original_pause_seconds": orig_pause_dur,
            "corrupted_pause_seconds": new_pause_dur,
        },
        "original_region": {
            "start_seconds": p_orig_start,
            "end_seconds": p_orig_end,
            "duration_seconds": orig_pause_dur,
        },
        "flaw_region": {
            "start_seconds": new_pause_start,
            "end_seconds": new_pause_end,
            "duration_seconds": new_pause_dur,
        },
        "corrupted_timeline": {
            "flaw_start_seconds": new_pause_start,
            "flaw_end_seconds": new_pause_end,
            "time_shift_seconds": time_shift,
        },
        "chosen_3_to_4s_region": {
            "start_word_index": ch_start_idx,
            "end_word_index": ch_end_idx,
            "phrase_text": " ".join([w["word"] for w in words[ch_start_idx : ch_end_idx + 1]]),
            "corrupted_start_seconds": ch_start_sec,
            "corrupted_end_seconds": ch_end_sec,
            "duration_seconds": ch_dur,
        },
        "crossfade_details": {
            "crossfade_ms": CROSSFADE_MS,
            "crossfade_samples": int(round((CROSSFADE_MS / 1000.0) * sr)),
        },
        "audio_info": {
            "sample_rate": sr,
            "original_duration_seconds": round(len(audio) / sr, 4),
            "corrupted_duration_seconds": round(len(spliced_audio) / sr, 4),
        },
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(label_meta, f, indent=2)

    return out_wav_path, out_json_path, label_meta


def verify_alignment_with_whisper(
    corrupted_wav_path: Path,
    transcript_path: Path,
    label_json_path: Path,
) -> float:
    """Verify corrupted timeline by re-running Whisper alignment and computing max difference."""
    with open(label_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Run Whisper forced alignment into a temporary scratch CSV
    scratch_csv = PROJECT_ROOT / "scratch" / f"verify_{corrupted_wav_path.stem}.csv"
    scratch_csv.parent.mkdir(parents=True, exist_ok=True)

    aligned_whisper, _ = align_transcript_to_audio(
        wav_path=corrupted_wav_path,
        transcript_path=transcript_path,
        output_csv_path=scratch_csv,
        device="cpu",
        compute_type="int8",
    )

    # Load calculated corrupted alignment
    calc_csv = corrupted_wav_path.parent / f"{corrupted_wav_path.stem}_alignment.csv"
    calculated_words = load_alignment_words(calc_csv)

    # Compare word start/end times
    max_diff = 0.0
    for w_calc, w_whisp in zip(calculated_words, aligned_whisper):
        s_diff = abs(float(w_calc["start"]) - float(w_whisp["start_seconds"]))
        e_diff = abs(float(w_calc["end"]) - float(w_whisp["end_seconds"]))
        max_diff = max(max_diff, s_diff, e_diff)

    if scratch_csv.exists():
        scratch_csv.unlink()

    return round(max_diff, 4)


def generate_pause_ladder(verify: bool = True) -> None:
    """Generate the full pause flaw ladder for Clip 01 and Clip 02."""
    sources_dir = PROJECT_ROOT / "dataset" / "sources"
    corrupted_dir = PROJECT_ROOT / "dataset" / "corrupted"

    w01 = sources_dir / "librivox_01.wav"
    a01 = sources_dir / "librivox_01_alignment.csv"
    t01 = sources_dir / "librivox_01.txt"

    w02 = sources_dir / "librivox_02.wav"
    a02 = sources_dir / "librivox_02_alignment.csv"
    t02 = sources_dir / "librivox_02.txt"

    print("Generating Clip 01 pause ladder (11 files)...")
    clip01_specs = []
    # Rushed 1-5
    for tier in range(1, 6):
        clip01_specs.append(("rushed", tier, RUSHED_TIERS[tier], f"librivox_01_pause_rushed_tier{tier}"))
    # Draggy 1-5
    for tier in range(1, 6):
        clip01_specs.append(("draggy", tier, DRAGGY_TIERS[tier], f"librivox_01_pause_draggy_tier{tier}"))
    # Control
    clip01_specs.append(("control", "control", CONTROL_MULTIPLIER, "librivox_01_pause_control"))

    for direction, tier, mult, stem in clip01_specs:
        wav_p, json_p, meta = apply_pause_corruption(
            source_wav_path=w01,
            source_alignment_csv=a01,
            output_dir=corrupted_dir,
            direction=direction,
            tier=tier,
            multiplier=mult,
            target_pause_words=CLIP_01_TARGET_PAUSE,
            chosen_region_words=CLIP_01_CHOSEN_WORDS,
            stem=stem,
        )
        print(f"  Created: {wav_p.name} -> Flaw region: [{meta['flaw_region']['start_seconds']:.3f}s, {meta['flaw_region']['end_seconds']:.3f}s] (shift: {meta['corrupted_timeline']['time_shift_seconds']:+.3f}s)")

    print("\nGenerating Clip 02 pause files (7 files)...")
    clip02_specs = [
        ("rushed", 2, RUSHED_TIERS[2], "librivox_02_pause_rushed_tier2"),
        ("rushed", 3, RUSHED_TIERS[3], "librivox_02_pause_rushed_tier3"),
        ("rushed", 5, RUSHED_TIERS[5], "librivox_02_pause_rushed_tier5"),
        ("draggy", 2, DRAGGY_TIERS[2], "librivox_02_pause_draggy_tier2"),
        ("draggy", 3, DRAGGY_TIERS[3], "librivox_02_pause_draggy_tier3"),
        ("draggy", 5, DRAGGY_TIERS[5], "librivox_02_pause_draggy_tier5"),
        ("control", "control", CONTROL_MULTIPLIER, "librivox_02_pause_control"),
    ]

    for direction, tier, mult, stem in clip02_specs:
        wav_p, json_p, meta = apply_pause_corruption(
            source_wav_path=w02,
            source_alignment_csv=a02,
            output_dir=corrupted_dir,
            direction=direction,
            tier=tier,
            multiplier=mult,
            target_pause_words=CLIP_02_TARGET_PAUSE,
            chosen_region_words=CLIP_02_CHOSEN_WORDS,
            stem=stem,
        )
        print(f"  Created: {wav_p.name} -> Flaw region: [{meta['flaw_region']['start_seconds']:.3f}s, {meta['flaw_region']['end_seconds']:.3f}s] (shift: {meta['corrupted_timeline']['time_shift_seconds']:+.3f}s)")

    if verify:
        print("\n" + "=" * 70)
        print("VERIFICATION: Re-running Whisper alignment on corrupted audio...")
        print("=" * 70)
        # Verify a representative subset or selected files to confirm timing alignment
        sample_verify = [
            ("librivox_01_pause_rushed_tier3", t01),
            ("librivox_01_pause_draggy_tier3", t01),
            ("librivox_02_pause_rushed_tier3", t02),
        ]
        for stem, t_path in sample_verify:
            wav_path = corrupted_dir / f"{stem}.wav"
            json_path = corrupted_dir / f"{stem}.json"
            print(f"  Re-aligning {wav_path.name}...")
            max_d = verify_alignment_with_whisper(wav_path, t_path, json_path)
            print(f"  --> Max difference between label times and re-aligned times: {max_d * 1000:.1f} ms ({max_d:.4f} s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pause flaw corruption dataset.")
    parser.add_argument("--verify", action="store_true", help="Re-run Whisper alignment to verify timing.")
    args = parser.parse_args()
    generate_pause_ladder(verify=args.verify)
