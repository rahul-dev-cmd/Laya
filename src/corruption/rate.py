"""Rate / Speech-tempo corruption module for Second Take.

This module implements local articulation rate corruption:
1. Selects a continuous phrase (~4 to 6 seconds) from the middle of the audio clip,
   anchored at word boundaries and bounded by pauses (>150 ms) on both sides.
2. Speeds up only that phrase by a factor (e.g. 1.25 for Tier 3) with pitch preserved
   using phase-vocoder time stretching (librosa.effects.time_stretch).
3. Leaves all surrounding audio unchanged, crossfading the joins by 10 ms to prevent
   acoustic discontinuities or splice clicks.
4. Produces:
   - Corrupted WAV: dataset/corrupted/librivox_01_rate_t3.wav
   - Control WAV: dataset/corrupted/librivox_01_control.wav (factor 1.0 resynthesis)
   - Metadata JSON: dataset/corrupted/librivox_01_rate_t3.json (flaw region in corrupted timeline)
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

# Reproducibility seed & default constants
RANDOM_SEED: int = 42
TARGET_SAMPLE_RATE: int = 16000
DEFAULT_CROSSFADE_MS: float = 10.0


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed across python and numpy for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def load_alignment_words(alignment_csv_path: Path) -> List[Dict[str, Any]]:
    """Load word timestamps from alignment CSV file.
    
    Args:
        alignment_csv_path: Path to CSV with columns word, start_seconds, end_seconds.
        
    Returns:
        List of dicts: [{'word': str, 'start': float, 'end': float, 'index': int}, ...]
    """
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


def find_middle_phrase(
    words: List[Dict[str, Any]],
    total_duration: float,
    min_dur: float = 4.0,
    max_dur: float = 6.0,
    min_pause_sec: float = 0.30,
) -> Tuple[int, int, Dict[str, Any]]:
    """Find a 4-6 second phrase near the middle of the clip bounded by strong pauses.
    
    Args:
        words: List of aligned word entries.
        total_duration: Total audio clip duration in seconds.
        min_dur: Minimum phrase duration in seconds (default: 4.0).
        max_dur: Maximum phrase duration in seconds (default: 6.0).
        min_pause_sec: Minimum pause duration for boundary silence (default: 0.30).
        
    Returns:
        Tuple of (start_word_idx, end_word_idx, phrase_info_dict).
    """
    mid_time = total_duration / 2.0
    candidates = []

    for i in range(len(words)):
        # Check pause before word i (or beginning of file)
        pause_before = (words[i]["start"] >= min_pause_sec) if i == 0 else (
            words[i]["start"] - words[i - 1]["end"] >= min_pause_sec
        )
        if not pause_before:
            continue

        for j in range(i, len(words)):
            dur = words[j]["end"] - words[i]["start"]
            if dur > max_dur + 0.5:
                break

            # Check pause after word j (or end of file)
            pause_after = (total_duration - words[j]["end"] >= min_pause_sec) if j == len(words) - 1 else (
                words[j + 1]["start"] - words[j]["end"] >= min_pause_sec
            )

            if min_dur <= dur <= max_dur and pause_after:
                phrase_mid = (words[i]["start"] + words[j]["end"]) / 2.0
                dist_from_clip_mid = abs(phrase_mid - mid_time)
                # Score combines proximity to clip middle and proximity to 5.0s ideal duration
                score = dist_from_clip_mid + 2.0 * abs(dur - 5.0)
                candidates.append((score, i, j, dur))

    if not candidates:
        # Fall back to 0.15s if no candidates found with 0.30s pauses
        return find_middle_phrase(
            words, total_duration, min_dur=min_dur, max_dur=max_dur, min_pause_sec=0.15
        )

    # Pick candidate with best score
    candidates.sort(key=lambda c: c[0])
    _, best_i, best_j, best_dur = candidates[0]

    selected_words = [words[k]["word"] for k in range(best_i, best_j + 1)]
    start_sec = words[best_i]["start"]
    end_sec = words[best_j]["end"]
    gap_before = words[best_i]["start"] if best_i == 0 else (words[best_i]["start"] - words[best_i - 1]["end"])
    gap_after = (total_duration - words[best_j]["end"]) if best_j == len(words) - 1 else (
        words[best_j + 1]["start"] - words[best_j]["end"]
    )

    phrase_info = {
        "start_word_index": best_i,
        "end_word_index": best_j,
        "words": selected_words,
        "phrase_text": " ".join(selected_words),
        "start_seconds": start_sec,
        "end_seconds": end_sec,
        "duration_seconds": round(best_dur, 4),
        "pause_before_seconds": round(gap_before, 4),
        "pause_after_seconds": round(gap_after, 4),
    }

    return best_i, best_j, phrase_info


def crossfade_splice(
    audio: np.ndarray,
    processed_phrase: np.ndarray,
    start_sample: int,
    end_sample: int,
    fade_samples: int,
) -> np.ndarray:
    """Splice processed phrase into audio and crossfade both joins.
    
    The audio preceding start_sample and following end_sample is unchanged,
    except for a linear crossfade of length fade_samples across each join to
    prevent clicks.
    
    Args:
        audio: 1D full audio array.
        processed_phrase: 1D processed (time-stretched) phrase audio array.
        start_sample: Sample index where original phrase begins.
        end_sample: Sample index where original phrase ends.
        fade_samples: Number of samples over which to linearly crossfade (e.g. 160 for 10 ms).
        
    Returns:
        1D spliced audio array.
    """
    if fade_samples <= 0:
        # Hard splice without crossfade
        return np.concatenate([audio[:start_sample], processed_phrase, audio[end_sample:]])

    w = np.linspace(0.0, 1.0, fade_samples)

    # 1. Audio preceding the first join (up to start_sample - fade_samples)
    part_before = audio[:start_sample - fade_samples]

    # 2. Left join: crossfade preceding pause (fading out) with phrase onset (fading in)
    join1 = audio[start_sample - fade_samples : start_sample] * (1.0 - w) + processed_phrase[:fade_samples] * w

    # 3. Phrase body (untouched stretched audio)
    phrase_body = processed_phrase[fade_samples : -fade_samples]

    # 4. Right join: crossfade phrase offset (fading out) with following pause (fading in)
    join2 = processed_phrase[-fade_samples:] * (1.0 - w) + audio[end_sample : end_sample + fade_samples] * w

    # 5. Audio following the second join
    part_after = audio[end_sample + fade_samples:]

    return np.concatenate([part_before, join1, phrase_body, join2, part_after])


def apply_rate_corruption(
    audio_path: Path,
    alignment_path: Path,
    output_dir: Path,
    factor: float = 1.25,
    tier: Any = 3,
    crossfade_ms: float = DEFAULT_CROSSFADE_MS,
    min_pause_sec: float = 0.35,
    phrase_start_word: Optional[int] = None,
    phrase_end_word: Optional[int] = None,
    output_stem: Optional[str] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[Path, Path, Path]:
    """Apply rate corruption (speed-up or slow-down) and generate control audio and metadata JSON.
    
    Args:
        audio_path: Path to source WAV audio file.
        alignment_path: Path to word alignment CSV file.
        output_dir: Directory where corrupted audio, control, and JSON are written.
        factor: Time-stretch factor (e.g. 1.25 for speed up, 0.80 for slow down).
        tier: Severity tier level or name (default: 3).
        crossfade_ms: Crossfade duration in milliseconds at the joins (default: 10.0 ms).
        min_pause_sec: Minimum pause threshold for phrase boundaries (default: 0.35s).
        phrase_start_word: Optional word index to start phrase (overrides automatic search).
        phrase_end_word: Optional word index to end phrase (inclusive).
        output_stem: Optional custom file stem for outputs (e.g. 'librivox_01_rate_tier1').
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (corrupted_wav_path, control_wav_path, json_path).
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load source audio
    print(f"[Rate Corruption] Loading audio from '{audio_path}'...")
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Ensure mono
    total_duration = len(audio) / float(sr)
    fade_samples = int(round((crossfade_ms / 1000.0) * sr))

    # 2. Load word alignment and pick middle phrase
    print(f"[Rate Corruption] Loading alignment from '{alignment_path}'...")
    words = load_alignment_words(alignment_path)
    if phrase_start_word is not None and phrase_end_word is not None:
        selected_words = [words[k]["word"] for k in range(phrase_start_word, phrase_end_word + 1)]
        start_sec = words[phrase_start_word]["start"]
        end_sec = words[phrase_end_word]["end"]
        gap_before = words[phrase_start_word]["start"] if phrase_start_word == 0 else (
            words[phrase_start_word]["start"] - words[phrase_start_word - 1]["end"]
        )
        gap_after = (total_duration - words[phrase_end_word]["end"]) if phrase_end_word == len(words) - 1 else (
            words[phrase_end_word + 1]["start"] - words[phrase_end_word]["end"]
        )
        phrase_info = {
            "start_word_index": phrase_start_word,
            "end_word_index": phrase_end_word,
            "words": selected_words,
            "phrase_text": " ".join(selected_words),
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "duration_seconds": round(end_sec - start_sec, 4),
            "pause_before_seconds": round(gap_before, 4),
            "pause_after_seconds": round(gap_after, 4),
        }
    else:
        # Default to the consistent phrase: words 174-191 ("However this may be...")
        phrase_start_word = 174
        phrase_end_word = 191
        selected_words = [words[k]["word"] for k in range(phrase_start_word, phrase_end_word + 1)]
        start_sec = words[phrase_start_word]["start"]
        end_sec = words[phrase_end_word]["end"]
        gap_before = words[phrase_start_word]["start"] - words[phrase_start_word - 1]["end"]
        gap_after = words[phrase_end_word + 1]["start"] - words[phrase_end_word]["end"]
        phrase_info = {
            "start_word_index": phrase_start_word,
            "end_word_index": phrase_end_word,
            "words": selected_words,
            "phrase_text": " ".join(selected_words),
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "duration_seconds": round(end_sec - start_sec, 4),
            "pause_before_seconds": round(gap_before, 4),
            "pause_after_seconds": round(gap_after, 4),
        }

    start_sec = phrase_info["start_seconds"]
    end_sec = phrase_info["end_seconds"]
    orig_phrase_dur = phrase_info["duration_seconds"]
    start_sample = int(round(start_sec * sr))
    end_sample = int(round(end_sec * sr))

    print(f"[Rate Corruption] Selected phrase (bounded by pauses):")
    print(f"  Words: \"{phrase_info['phrase_text']}\"")
    print(f"  Original timeline: {start_sec:.2f}s to {end_sec:.2f}s (duration: {orig_phrase_dur:.2f}s)")
    print(f"  Bounding pauses: {phrase_info['pause_before_seconds']*1000:.0f}ms before, {phrase_info['pause_after_seconds']*1000:.0f}ms after")

    # 3. Extract original phrase audio
    phrase_audio = audio[start_sample:end_sample]

    # 4. Apply pitch-preserving time stretch via phase vocoder
    direction_desc = "speed-up" if factor > 1.0 else ("slow-down" if factor < 1.0 else "neutral")
    print(f"[Rate Corruption] Time-stretching phrase by factor {factor:.2f} ({direction_desc}, Tier {tier})...")
    phrase_stretched = librosa.effects.time_stretch(phrase_audio, rate=factor)

    # 5. Generate control phrase (factor 1.0) with identical phase-vocoder processing
    phrase_control = librosa.effects.time_stretch(phrase_audio, rate=1.0)

    # 6. Splice with crossfade joins
    print(f"[Rate Corruption] Splicing into audio with {crossfade_ms:.1f}ms crossfade joins...")
    audio_corrupted = crossfade_splice(
        audio, phrase_stretched, start_sample, end_sample, fade_samples
    )
    audio_control = crossfade_splice(
        audio, phrase_control, start_sample, end_sample, fade_samples
    )

    # 7. Compute corrupted timeline coordinates
    corrupted_phrase_dur = round(orig_phrase_dur / factor, 4)
    flaw_start_sec = round(start_sec, 4)
    flaw_end_sec = round(start_sec + corrupted_phrase_dur, 4)
    time_shift_sec = round(corrupted_phrase_dur - orig_phrase_dur, 4)

    # Exact audio sample boundaries for precision inspection:
    exact_flaw_start_sample = start_sample - fade_samples
    exact_flaw_end_sample = exact_flaw_start_sample + len(phrase_stretched)
    exact_flaw_start_sec = round(exact_flaw_start_sample / float(sr), 4)
    exact_flaw_end_sec = round(exact_flaw_end_sample / float(sr), 4)

    # 8. Save outputs
    stem = output_stem if output_stem is not None else f"librivox_01_rate_t{tier}"
    corrupted_wav_path = output_dir / f"{stem}.wav"
    control_wav_path = output_dir / "librivox_01_control.wav"
    json_path = output_dir / f"{stem}.json"

    print(f"[Rate Corruption] Saving corrupted audio to '{corrupted_wav_path}'...")
    sf.write(str(corrupted_wav_path), audio_corrupted, sr, subtype="PCM_16")

    if not control_wav_path.exists():
        print(f"[Rate Corruption] Saving control audio to '{control_wav_path}'...")
        sf.write(str(control_wav_path), audio_control, sr, subtype="PCM_16")
    else:
        print(f"[Rate Corruption] Preserving existing control audio at '{control_wav_path}'...")

    # 9. Write metadata JSON
    metadata = {
        "source_clip": audio_path.name,
        "source_audio_path": str(audio_path).replace("\\", "/"),
        "flaw_type": "rate",
        "direction": "speed_up" if factor > 1.0 else ("slow_down" if factor < 1.0 else "neutral"),
        "tier": tier,
        "factor": factor,
        "words": phrase_info["words"],
        "phrase_text": phrase_info["phrase_text"],
        "phrase_word_count": len(phrase_info["words"]),
        "original_region": {
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "duration_seconds": orig_phrase_dur,
        },
        "flaw_region": {
            "start_seconds": flaw_start_sec,
            "end_seconds": flaw_end_sec,
            "duration_seconds": corrupted_phrase_dur,
        },
        "corrupted_timeline": {
            "flaw_start_seconds": flaw_start_sec,
            "flaw_end_seconds": flaw_end_sec,
            "time_shift_seconds": time_shift_sec,
            "subsequent_audio_shifted_earlier_by_seconds": abs(time_shift_sec) if time_shift_sec < 0 else 0.0,
            "subsequent_audio_shifted_later_by_seconds": time_shift_sec if time_shift_sec > 0 else 0.0,
        },
        "crossfade_details": {
            "crossfade_ms": crossfade_ms,
            "crossfade_samples": fade_samples,
            "exact_sample_flaw_start_seconds": exact_flaw_start_sec,
            "exact_sample_flaw_end_seconds": exact_flaw_end_sec,
        },
        "audio_info": {
            "sample_rate": sr,
            "original_duration_seconds": round(total_duration, 4),
            "corrupted_duration_seconds": round(len(audio_corrupted) / float(sr), 4),
            "control_duration_seconds": round(len(audio_control) / float(sr), 4),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Rate Corruption] Saved metadata JSON to '{json_path}'")
    print(f"  Flaw Region (corrupted timeline): {flaw_start_sec:.2f}s to {flaw_end_sec:.2f}s (dur={corrupted_phrase_dur:.2f}s)")
    if time_shift_sec < 0:
        print(f"  Subsequent audio shifted earlier by: {abs(time_shift_sec):.3f}s")
    else:
        print(f"  Subsequent audio shifted later by: {time_shift_sec:.3f}s")

    return corrupted_wav_path, control_wav_path, json_path


RATE_LADDER_SPEC = [
    # Speed-up tiers
    {"stem": "librivox_01_rate_tier1", "factor": 1.05, "tier": 1, "desc": "Speed-up Tier 1 (1.05x)"},
    {"stem": "librivox_01_rate_tier2", "factor": 1.10, "tier": 2, "desc": "Speed-up Tier 2 (1.10x)"},
    {"stem": "librivox_01_rate_tier3", "factor": 1.25, "tier": 3, "desc": "Speed-up Tier 3 (1.25x)"},
    {"stem": "librivox_01_rate_tier4", "factor": 1.60, "tier": 4, "desc": "Speed-up Tier 4 (1.60x)"},
    {"stem": "librivox_01_rate_tier5", "factor": 2.00, "tier": 5, "desc": "Speed-up Tier 5 (2.00x)"},
    # Slow-down tiers
    {"stem": "librivox_01_rate_slow_tier1", "factor": 0.95, "tier": "slow_1", "desc": "Slow-down Tier 1 (0.95x)"},
    {"stem": "librivox_01_rate_slow_tier2", "factor": 0.80, "tier": "slow_2", "desc": "Slow-down Tier 2 (0.80x)"},
    {"stem": "librivox_01_rate_slow_tier3", "factor": 0.60, "tier": "slow_3", "desc": "Slow-down Tier 3 (0.60x)"},
]


def build_full_ladder(
    audio_path: Path,
    alignment_path: Path,
    output_dir: Path,
    seed: int = RANDOM_SEED,
) -> List[Tuple[Path, Path, Path]]:
    """Build the complete 8-point rate flaw ladder (5 speed-up + 3 slow-down).
    
    Preserves any existing files that should not be overwritten.
    """
    results = []
    print("\n" + "=" * 70)
    print("BUILDING FULL RATE-FLAW LADDER FOR LIBRIVOX_01")
    print("=" * 70)
    for spec in RATE_LADDER_SPEC:
        print(f"\n>>> Generating {spec['desc']}: factor={spec['factor']}, stem={spec['stem']}...")
        w, c, j = apply_rate_corruption(
            audio_path=audio_path,
            alignment_path=alignment_path,
            output_dir=output_dir,
            factor=spec["factor"],
            tier=spec["tier"],
            phrase_start_word=174,
            phrase_end_word=191,
            output_stem=spec["stem"],
            seed=seed,
        )
        results.append((w, c, j))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build rate corruption (speed-up / slow-down with pitch preserved) for speech clips."
    )
    parser.add_argument(
        "--audio",
        default="dataset/sources/librivox_01.wav",
        help="Path to WAV audio file (default: dataset/sources/librivox_01.wav)",
    )
    parser.add_argument(
        "--alignment",
        default="dataset/sources/librivox_01_alignment.csv",
        help="Path to alignment CSV (default: dataset/sources/librivox_01_alignment.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/corrupted",
        help="Destination directory for corrupted outputs (default: dataset/corrupted)",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=1.25,
        help="Time-stretch factor (default: 1.25)",
    )
    parser.add_argument(
        "--tier",
        default=3,
        help="Severity tier level or name (default: 3)",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Custom output filename stem (without extension)",
    )
    parser.add_argument(
        "--build-ladder",
        action="store_true",
        help="Generate full 8-point rate flaw ladder (5 speed-up tiers + 3 slow-down tiers)",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=float,
        default=DEFAULT_CROSSFADE_MS,
        help="Crossfade duration in ms at splice joins (default: 10.0 ms)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    alignment_path = Path(args.alignment)
    out_dir = Path(args.output_dir)

    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}", file=sys.stderr)
        sys.exit(1)

    if not alignment_path.exists():
        print(f"Error: Alignment file not found at {alignment_path}", file=sys.stderr)
        sys.exit(1)

    if args.build_ladder:
        build_full_ladder(
            audio_path=audio_path,
            alignment_path=alignment_path,
            output_dir=out_dir,
            seed=args.seed,
        )
    else:
        print("\n" + "=" * 60)
        print(f"SECOND TAKE: RATE CORRUPTION (Factor {args.factor})")
        print("=" * 60)

        apply_rate_corruption(
            audio_path=audio_path,
            alignment_path=alignment_path,
            output_dir=out_dir,
            factor=args.factor,
            tier=args.tier,
            output_stem=args.output_stem,
            crossfade_ms=args.crossfade_ms,
            seed=args.seed,
        )

    print("\n[Done] Rate corruption processing completed successfully!")


if __name__ == "__main__":
    main()
