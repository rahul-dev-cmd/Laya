#!/usr/bin/env python3
"""Pitch / intonation flaw corruption module for Second Take.

This module implements pitch dynamic range compression (monotone delivery flaw):
1. Targets the same phrase as the rate ladder (words 174-191 in librivox_01.wav,
   "However this may be it is certain that he soon became domesticated in the family of Colonel Syme").
2. Decomposes the phrase using PyWorld vocoder (F0, spectral envelope, aperiodicity).
3. Scales the phrase's voiced F0 deviations from its own median by range factor k:
   F0_new = median * (F0 / median) ** k  (log-domain scaling).
   - k = 1.0: original pitch dynamics (control)
   - k = 0.0: fully monotone / flat pitch
4. Resynthesizes the phrase, keeping duration, timing, and loudness (RMS) identical.
5. Slices and crossfades the joins by 10 ms to prevent clicks.
6. Produces:
   - 5 tiers: dataset/corrupted/librivox_01_pitch_tier1..5.wav + .json (k = 0.85, 0.65, 0.45, 0.25, 0.0)
   - Control: dataset/corrupted/librivox_01_pitch_control.wav + .json (k = 1.0)
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyworld as pw
import soundfile as sf

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer virtual environment python
venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if venv_py.exists() and sys.executable.lower() != str(venv_py).lower():
    import subprocess
    result = subprocess.call([str(venv_py), __file__] + sys.argv[1:])
    sys.exit(result)

# Reproducibility seed & default constants
RANDOM_SEED: int = 42
TARGET_SAMPLE_RATE: int = 16000
DEFAULT_CROSSFADE_MS: float = 10.0

PITCH_LADDER_SPEC = [
    {"stem": "librivox_01_pitch_tier1", "k": 0.85, "tier": 1, "desc": "Pitch Tier 1 (k=0.85, Subtle flatness)"},
    {"stem": "librivox_01_pitch_tier2", "k": 0.65, "tier": 2, "desc": "Pitch Tier 2 (k=0.65, Mild flatness)"},
    {"stem": "librivox_01_pitch_tier3", "k": 0.45, "tier": 3, "desc": "Pitch Tier 3 (k=0.45, Moderate flatness)"},
    {"stem": "librivox_01_pitch_tier4", "k": 0.25, "tier": 4, "desc": "Pitch Tier 4 (k=0.25, Strong flatness)"},
    {"stem": "librivox_01_pitch_tier5", "k": 0.00, "tier": 5, "desc": "Pitch Tier 5 (k=0.00, Fully monotone)"},
    {"stem": "librivox_01_pitch_control", "k": 1.00, "tier": "control", "desc": "Pitch Control (k=1.00, Resynthesized neutral)"},
]


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
        processed_phrase: 1D processed phrase audio array.
        start_sample: Sample index where original phrase begins.
        end_sample: Sample index where original phrase ends.
        fade_samples: Number of samples over which to linearly crossfade (e.g. 160 for 10 ms).
        
    Returns:
        1D spliced audio array.
    """
    if fade_samples <= 0:
        return np.concatenate([audio[:start_sample], processed_phrase, audio[end_sample:]])

    w = np.linspace(0.0, 1.0, fade_samples)

    # 1. Audio preceding the first join (up to start_sample - fade_samples)
    part_before = audio[:start_sample - fade_samples]

    # 2. Left join: crossfade preceding pause (fading out) with phrase onset (fading in)
    join1 = audio[start_sample - fade_samples : start_sample] * (1.0 - w) + processed_phrase[:fade_samples] * w

    # 3. Phrase body (untouched modified audio)
    phrase_body = processed_phrase[fade_samples : -fade_samples]

    # 4. Right join: crossfade phrase offset (fading out) with following pause (fading in)
    join2 = processed_phrase[-fade_samples:] * (1.0 - w) + audio[end_sample : end_sample + fade_samples] * w

    # 5. Audio following the second join
    part_after = audio[end_sample + fade_samples:]

    return np.concatenate([part_before, join1, phrase_body, join2, part_after])


def scale_f0_contour(
    f0: np.ndarray,
    k: float,
) -> Tuple[np.ndarray, float]:
    """Scale voiced F0 deviations from median by factor k in the log domain.
    
    Formula: F0_new = median * (F0 / median) ** k for voiced frames.
    - k = 1.0: exact original F0
    - k = 0.0: constant median F0 (monotone)
    - unvoiced frames remain 0.0.
    
    Args:
        f0: 1D array of F0 values in Hz from PyWorld.
        k: Range scaling factor in [0.0, 1.0].
        
    Returns:
        Tuple of (scaled_f0, phrase_median_f0).
    """
    voiced = f0 > 0.0
    if not np.any(voiced):
        return f0.copy(), 0.0

    med_f0 = float(np.median(f0[voiced]))
    f0_scaled = np.zeros_like(f0)

    if k <= 1e-6:
        # Fully monotone: all voiced frames set to median F0
        f0_scaled[voiced] = med_f0
    else:
        # Log-domain range scaling
        ratio = f0[voiced] / med_f0
        f0_scaled[voiced] = med_f0 * np.power(ratio, k)

    return f0_scaled, med_f0


def apply_pitch_corruption(
    audio_path: Path,
    alignment_path: Path,
    output_dir: Path,
    k: float = 0.0,
    tier: Any = 5,
    phrase_start_word: int = 174,
    phrase_end_word: int = 191,
    crossfade_ms: float = DEFAULT_CROSSFADE_MS,
    output_stem: Optional[str] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[Path, Path]:
    """Apply pitch dynamic range compression to the specified phrase using PyWorld.
    
    Args:
        audio_path: Path to source WAV audio file.
        alignment_path: Path to word alignment CSV file.
        output_dir: Directory where corrupted audio and JSON are written.
        k: Pitch range scaling factor (0.0 = monotone, 1.0 = neutral control).
        tier: Severity tier level or name (1..5 or 'control').
        phrase_start_word: First word index of phrase (default: 174).
        phrase_end_word: Last word index of phrase inclusive (default: 191).
        crossfade_ms: Crossfade duration in ms at joins (default: 10.0 ms).
        output_stem: Output file stem (e.g. 'librivox_01_pitch_tier1').
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (wav_path, json_path).
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load source audio
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    total_duration = len(audio) / float(sr)
    fade_samples = int(round((crossfade_ms / 1000.0) * sr))

    # 2. Load word alignment
    words = load_alignment_words(alignment_path)
    selected_words = [words[i]["word"] for i in range(phrase_start_word, phrase_end_word + 1)]
    start_sec = words[phrase_start_word]["start"]
    end_sec = words[phrase_end_word]["end"]
    orig_phrase_dur = round(end_sec - start_sec, 4)

    start_sample = int(round(start_sec * sr))
    end_sample = int(round(end_sec * sr))
    phrase_audio = audio[start_sample:end_sample].astype(np.float64)

    # 3. Analyze phrase with PyWorld vocoder
    # Using harvest for precise F0 estimation + cheaptrick + d4c
    _f0, time_axis = pw.harvest(phrase_audio, sr, frame_period=5.0)
    f0 = pw.stonemask(phrase_audio, _f0, time_axis, sr)
    sp = pw.cheaptrick(phrase_audio, f0, time_axis, sr)
    ap = pw.d4c(phrase_audio, f0, time_axis, sr)

    # 4. Scale F0 deviations from median by factor k
    f0_mod, phrase_med_f0 = scale_f0_contour(f0, k=k)

    # 5. Resynthesize phrase with PyWorld
    synth_phrase = pw.synthesize(f0_mod, sp, ap, sr)

    # Match exact length
    target_len = len(phrase_audio)
    if len(synth_phrase) > target_len:
        synth_phrase = synth_phrase[:target_len]
    elif len(synth_phrase) < target_len:
        synth_phrase = np.pad(synth_phrase, (0, target_len - len(synth_phrase)), "constant")

    # Match exact loudness (RMS energy) of the original phrase
    orig_rms = float(np.sqrt(np.mean(phrase_audio ** 2)))
    syn_rms = float(np.sqrt(np.mean(synth_phrase ** 2)))
    if syn_rms > 1e-8:
        synth_phrase = synth_phrase * (orig_rms / syn_rms)

    synth_phrase_f32 = synth_phrase.astype(np.float32)

    # 6. Splice with 10 ms linear crossfade
    audio_corrupted = crossfade_splice(
        audio, synth_phrase_f32, start_sample, end_sample, fade_samples
    )

    # 7. Output paths
    stem = output_stem if output_stem is not None else f"librivox_01_pitch_t{tier}"
    wav_path = output_dir / f"{stem}.wav"
    json_path = output_dir / f"{stem}.json"

    # Save audio
    sf.write(str(wav_path), audio_corrupted, sr, subtype="PCM_16")

    # 8. Metadata JSON
    metadata = {
        "source_clip": audio_path.name,
        "source_audio_path": str(audio_path).replace("\\", "/"),
        "flaw_type": "pitch",
        "description": "Pitch dynamic range compression (flat / monotone intonation flaw)",
        "tier": tier,
        "k": round(float(k), 4),
        "is_monotone": bool(k == 0.0),
        "phrase_median_f0_hz": round(phrase_med_f0, 2),
        "words": selected_words,
        "phrase_text": " ".join(selected_words),
        "phrase_word_count": len(selected_words),
        "original_region": {
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "duration_seconds": orig_phrase_dur,
        },
        "flaw_region": {
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "duration_seconds": orig_phrase_dur,
        },
        "corrupted_timeline": {
            "flaw_start_seconds": start_sec,
            "flaw_end_seconds": end_sec,
            "time_shift_seconds": 0.0,
        },
        "crossfade_details": {
            "crossfade_ms": crossfade_ms,
            "crossfade_samples": fade_samples,
        },
        "audio_info": {
            "sample_rate": sr,
            "original_duration_seconds": round(total_duration, 4),
            "corrupted_duration_seconds": round(len(audio_corrupted) / float(sr), 4),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Pitch Corruption] Saved {wav_path.name} (k={k:.2f}, tier={tier}) | Flaw: [{start_sec:.2f}s, {end_sec:.2f}s]")
    return wav_path, json_path


def build_pitch_ladder(
    audio_path: Path,
    alignment_path: Path,
    output_dir: Path,
    seed: int = RANDOM_SEED,
) -> List[Tuple[Path, Path]]:
    """Generate all 5 pitch tiers and the control audio."""
    print("\n" + "=" * 70)
    print("BUILDING PITCH-FLAW LADDER FOR LIBRIVOX_01")
    print("=" * 70)
    results = []
    for spec in PITCH_LADDER_SPEC:
        print(f"\n>>> Generating {spec['desc']}...")
        w, j = apply_pitch_corruption(
            audio_path=audio_path,
            alignment_path=alignment_path,
            output_dir=output_dir,
            k=spec["k"],
            tier=spec["tier"],
            output_stem=spec["stem"],
            seed=seed,
        )
        results.append((w, j))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build pitch corruption (flat / monotone intonation via PyWorld) for speech clips."
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
        help="Destination directory for outputs (default: dataset/corrupted)",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.0,
        help="Pitch range factor k (0.0 = monotone, 1.0 = neutral)",
    )
    parser.add_argument(
        "--tier",
        default=5,
        help="Severity tier level or name (default: 5)",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Custom output filename stem (without extension)",
    )
    parser.add_argument(
        "--build-ladder",
        action="store_true",
        help="Generate full pitch flaw ladder (5 tiers + control)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    alignment_path = Path(args.alignment)
    out_dir = Path(args.output_dir)

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if not alignment_path.exists():
        print(f"Error: Alignment file not found: {alignment_path}", file=sys.stderr)
        sys.exit(1)

    if args.build_ladder:
        build_pitch_ladder(
            audio_path=audio_path,
            alignment_path=alignment_path,
            output_dir=out_dir,
            seed=args.seed,
        )
    else:
        apply_pitch_corruption(
            audio_path=audio_path,
            alignment_path=alignment_path,
            output_dir=out_dir,
            k=args.k,
            tier=args.tier,
            output_stem=args.output_stem,
            seed=args.seed,
        )

    print("\n[Done] Pitch corruption completed successfully!")


if __name__ == "__main__":
    main()
